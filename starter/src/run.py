"""Run the dev phase, then the held-out test phase, for the ingestion lab."""
import argparse
import csv
import hashlib
import json
import os
import platform
import time
from dataclasses import replace
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import MyEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_pdf
from .store import PgVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STARTER_ROOT = PROJECT_ROOT / "starter"
DATA_ROOT = PROJECT_ROOT / "data"
MANIFEST_PATH = DATA_ROOT / "manifest.csv"
QUESTIONS_PATH = PROJECT_ROOT / "benchmark" / "questions.jsonl"
RESULTS_ROOT = PROJECT_ROOT / "results"
FILTER_FIELDS = {"chapter"}
STRATEGIES = {
    "A": "A_fixed_256_words_overlap_50",
    "B": "B_paragraph_256_words",
    "C": "C_paragraph_256_words_filter",
}

load_dotenv(STARTER_ROOT / ".env")


def _fingerprint(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _load_manifest() -> list[dict[str, Any]]:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Corpus manifest is missing: {MANIFEST_PATH}")
    with MANIFEST_PATH.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError("Corpus manifest contains no documents")

    seen_ids: set[str] = set()
    for row in rows:
        source_id = row["source_id"].strip()
        if not source_id or source_id in seen_ids:
            raise ValueError(f"Missing or duplicate source_id in {MANIFEST_PATH}")
        seen_ids.add(source_id)
        path = (PROJECT_ROOT / row["source_path"]).resolve()
        if DATA_ROOT.resolve() not in path.parents or not path.is_file():
            raise FileNotFoundError(f"Invalid or missing corpus file for {source_id}: {path}")
        row["resolved_path"] = path
        row["content_page_numbers"] = {
            int(page.strip()) for page in row["content_page_numbers"].split(",") if page.strip()
        }
        if not row["content_page_numbers"]:
            raise ValueError(f"No content pages recorded for {source_id}")
        if not row["title"].strip() or not row["license_or_permission"].strip():
            raise ValueError(f"Missing title or rights information for {source_id}")
    return rows


def _load_corpus(manifest: list[dict[str, Any]]) -> tuple[list[Any], dict[str, Any]]:
    pages = []
    summary: dict[str, Any] = {"documents": [], "content_pages": 0, "content_characters": 0}
    for row in manifest:
        document_pages = load_pdf(row["resolved_path"], row["source_id"], row["source_path"])
        physical_page_count = len(document_pages)
        if max(row["content_page_numbers"]) > physical_page_count:
            raise ValueError(f"Manifest page number exceeds PDF length for {row['source_id']}")

        included = []
        for page in document_pages:
            if page.page not in row["content_page_numbers"]:
                continue
            if page.metadata["is_empty"]:
                raise ValueError(f"Manifest includes an empty page: {row['source_id']} page {page.page}")
            page.metadata.update({
                "chapter": row["title"],
                "source_url": row["source_url"],
                "license": row["license_or_permission"],
            })
            included.append(page)
        if len(included) != int(row["content_page_count"]):
            raise ValueError(
                f"Manifest page count mismatch for {row['source_id']}: "
                f"declared {row['content_page_count']}, found {len(included)}"
            )
        pages.extend(included)
        characters = sum(len(page.text) for page in included)
        summary["content_pages"] += len(included)
        summary["content_characters"] += characters
        summary["documents"].append({
            "source_id": row["source_id"],
            "title": row["title"],
            "physical_pages": physical_page_count,
            "content_pages": len(included),
            "content_characters": characters,
            "manifest_extracted_characters": int(row["extracted_characters"]),
        })

    if summary["content_pages"] < 30:
        raise ValueError(f"Corpus has {summary['content_pages']} content pages; at least 30 are required")
    return pages, summary


def _load_questions() -> list[dict[str, Any]]:
    if not QUESTIONS_PATH.is_file():
        raise FileNotFoundError(f"Benchmark file is missing: {QUESTIONS_PATH}")
    with QUESTIONS_PATH.open(encoding="utf-8") as file:
        questions = [json.loads(line) for line in file if line.strip()]
    ids = [question.get("id") for question in questions]
    if len(questions) < 20 or len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError("Benchmark needs at least 20 questions with unique, non-empty IDs")
    if any(question.get("split") not in {"dev", "test"} for question in questions):
        raise ValueError("Every benchmark question must use the dev or test split")
    for split in ("dev", "test"):
        if sum(question.get("split") == split for question in questions) < (15 if split == "dev" else 5):
            raise ValueError("Benchmark needs at least 15 dev and 5 test questions")
    filtered = [question for question in questions if question.get("filter")]
    if len(filtered) < 3 or not any(question.get("split") == "test" for question in filtered):
        raise ValueError("Benchmark needs at least 3 metadata-filter questions, including one test question")

    manifest = _load_manifest()
    manifest_ids = {row["source_id"] for row in manifest}
    titles = {row["title"] for row in manifest}
    for question in questions:
        required = ("question", "expected_answer", "gold_source_id", "gold_page", "evidence_quote")
        if any(not question.get(field) for field in required):
            raise ValueError(f"Question {question['id']} is missing required gold fields")
        if question["gold_source_id"] not in manifest_ids:
            raise ValueError(f"Question {question['id']} points to an unknown source")
        if question.get("filter"):
            if not set(question["filter"]).issubset(FILTER_FIELDS):
                raise ValueError(f"Question {question['id']} uses an unsupported metadata filter")
            if any(value not in titles for value in question["filter"].values()):
                raise ValueError(f"Question {question['id']} has a filter value absent from the corpus")
            if any(value.casefold() not in question["question"].casefold()
                   for value in question["filter"].values()):
                raise ValueError(f"Question {question['id']} filter is not stated in its query context")
        if question.get("status") != "verified":
            raise ValueError(
                f"Question {question['id']} is not verified. Check its answer, page, and evidence, "
                "then set status to 'verified'."
            )
    return questions


def _check_gold_evidence(questions: list[dict[str, Any]], pages: list[Any]) -> None:
    page_text = {(page.document_id, page.page): page.text for page in pages}
    for question in questions:
        text = page_text.get((question["gold_source_id"], int(question["gold_page"])))
        quote = " ".join(question["evidence_quote"].split())
        normalized_text = " ".join((text or "").split())
        if text is None or quote.casefold() not in normalized_text.casefold():
            raise ValueError(f"Question {question['id']} evidence quote is not found on its gold page")


def _build_chunks(pages: list[Any]) -> dict[str, list[Any]]:
    chunks: dict[str, list[Any]] = {name: [] for name in STRATEGIES.values()}
    for page in pages:
        chunks[STRATEGIES["A"]].extend(fixed_size(page, size=256, overlap=50))
        paragraph_chunks = by_paragraph(page, max_size=256)
        chunks[STRATEGIES["B"]].extend(paragraph_chunks)
        chunks[STRATEGIES["C"]].extend(
            replace(
                chunk,
                chunk_id=hashlib.sha256(f"C\0{chunk.chunk_id}".encode("utf-8")).hexdigest(),
            )
            for chunk in paragraph_chunks
        )
    if any(not items for items in chunks.values()):
        raise ValueError("A strategy produced no chunks")
    return chunks


def _index_name(run_id: str, strategy: str) -> str:
    index_name = f"{run_id}:{strategy}"
    if len(index_name) > 50:
        raise ValueError("Run-specific strategy index name exceeds the database column limit")
    return index_name


def _chunks_for_index(run_id: str, strategy: str, chunks: list[Any]) -> list[Any]:
    return [
        replace(
            chunk,
            chunk_id=hashlib.sha256(
                f"{run_id}\0{strategy}\0{chunk.chunk_id}".encode("utf-8")
            ).hexdigest(),
        )
        for chunk in chunks
    ]


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _trace_summary(traces: list[dict[str, Any]]) -> dict[str, Any]:
    grouped = {}
    for strategy in STRATEGIES.values():
        grouped[strategy] = {}
        for split in ("dev", "test"):
            selected = [
                trace for trace in traces
                if trace["strategy"] == strategy and trace["split"] == split
            ]
            if selected:
                metrics = summarize(selected)
                filtered = [trace for trace in selected if trace.get("filter_compliant") is not None]
                if filtered:
                    metrics["filter_compliance"] = {
                        "compliant_questions": sum(trace["filter_compliant"] for trace in filtered),
                        "filtered_questions": len(filtered),
                    }
                grouped[strategy][split] = metrics
    return grouped


def _filter_compliant(hits: list[dict[str, Any]], requested_filter: dict[str, Any] | None) -> bool | None:
    if not requested_filter:
        return None
    return bool(hits) and all(
        all((hit.get("metadata") or {}).get(key) == value for key, value in requested_filter.items())
        for hit in hits
    )


def _run_dev() -> None:
    manifest = _load_manifest()
    questions = _load_questions()
    pages, corpus_summary = _load_corpus(manifest)
    _check_gold_evidence(questions, pages)
    chunking_started = time.perf_counter()
    chunks_by_strategy = _build_chunks(pages)
    chunking_seconds = time.perf_counter() - chunking_started

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is required in starter/.env")
    model_name = os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"
    embedder = MyEmbedder(model_name=model_name)
    store = PgVectorStore(database_url=database_url, dimensions=embedder.dimensions)
    store.create_schema()

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = RESULTS_ROOT / f"data-ingestion-{run_id}"
    run_dir.mkdir(parents=True, exist_ok=False)
    input_files = [MANIFEST_PATH, QUESTIONS_PATH] + [row["resolved_path"] for row in manifest]
    input_fingerprint = _fingerprint(input_files)
    index_names = {
        strategy: _index_name(run_id, strategy)
        for strategy in STRATEGIES.values()
    }

    strategy_stats = {}
    vectors_by_strategy = {}
    for label in ("A", "B"):
        strategy = STRATEGIES[label]
        chunks = _chunks_for_index(run_id, strategy, chunks_by_strategy[strategy])
        print(f"Embedding {len(chunks)} chunks for {strategy} with {model_name}...")
        started = time.perf_counter()
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        embedding_seconds = time.perf_counter() - started
        store_started = time.perf_counter()
        store.upsert(strategy=index_names[strategy], chunks=chunks, vectors=vectors)
        store_seconds = time.perf_counter() - store_started
        if store.count(index_names[strategy]) != len(chunks):
            raise RuntimeError(f"Stored chunk count mismatch for strategy {strategy}")
        vectors_by_strategy[strategy] = vectors
        strategy_stats[strategy] = {
            "chunk_count": len(chunks),
            "embedding_seconds": embedding_seconds,
            "store_seconds": store_seconds,
            "chunking_seconds": chunking_seconds,
            "mean_chunk_words": sum(len(chunk.text.split()) for chunk in chunks) / len(chunks),
        }

    # B and C differ only by applying the question-provided chapter filter at retrieval.
    store_started = time.perf_counter()
    chunks_c = _chunks_for_index(run_id, STRATEGIES["C"], chunks_by_strategy[STRATEGIES["C"]])
    store.upsert(
        strategy=index_names[STRATEGIES["C"]],
        chunks=chunks_c,
        vectors=vectors_by_strategy[STRATEGIES["B"]],
    )
    store_seconds = time.perf_counter() - store_started
    if store.count(index_names[STRATEGIES["C"]]) != len(chunks_c):
        raise RuntimeError(f"Stored chunk count mismatch for strategy {STRATEGIES['C']}")
    strategy_stats[STRATEGIES["C"]] = {
        **strategy_stats[STRATEGIES["B"]],
        "embedding_reused": True,
        "embedding_seconds": 0.0,
        "store_seconds": store_seconds,
    }

    dev_questions = [question for question in questions if question["split"] == "dev"]
    evaluation_started = time.perf_counter()
    query_vectors = [embedder.embed_query(question["question"]) for question in dev_questions]
    traces = []
    for label, strategy in STRATEGIES.items():
        for question, query_vector in zip(dev_questions, query_vectors):
            requested_filter = question.get("filter")
            applied_filter = requested_filter if label == "C" else None
            hits = store.search(
                strategy=index_names[strategy],
                query_vector=query_vector,
                top_k=3,
                filters=applied_filter,
            )
            traces.append({
                "strategy": strategy,
                "split": question["split"],
                "question": question["question"],
                "requested_filter": requested_filter,
                "applied_filter": applied_filter,
                "filter_compliant": _filter_compliant(hits, requested_filter),
                "gold_source_id": question["gold_source_id"],
                "gold_page": question["gold_page"],
                "evidence_quote": question["evidence_quote"],
                **evaluate_query(question, hits),
            })
    evaluation_seconds = time.perf_counter() - evaluation_started

    with (run_dir / "dev_traces.jsonl").open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace, ensure_ascii=False, default=_json_default) + "\n")
    metadata = {
        "run_id": run_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "phase": "dev",
        "corpus": corpus_summary,
        "embedding_model": model_name,
        "embedding_sdk_version": version("google-genai"),
        "embedding_dimensions": embedder.dimensions,
        "pypdf_version": version("pypdf"),
        "psycopg_version": version("psycopg"),
        "python_version": platform.python_version(),
        "vector_store": store.backend_info(),
        "metric": "cosine distance",
        "search_type": "exact",
        "strategies": STRATEGIES,
        "index_names": index_names,
        "strategy_stats": strategy_stats,
        "dev_evaluation_seconds": evaluation_seconds,
        "dev_metrics": _trace_summary(traces),
        "input_fingerprint": input_fingerprint,
        "api_cost_usd": None,
        "api_cost_note": "Check provider billing; cost is not inferred by this runner.",
    }
    (run_dir / "run_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Dev phase complete. Review dev_traces.jsonl and select using dev results only.")
    print(f"Run ID: {run_id}")
    print(f"Results: {run_dir}")
    print(
        f"Test command: python -m src.run --phase test --run-id {run_id} "
        "--selected-strategy <A|B|C> --selection-criterion '<dev-based rule>'"
    )


def _run_test(run_id: str, selected_strategy: str, selection_criterion: str) -> None:
    questions = _load_questions()
    manifest = _load_manifest()
    pages, _ = _load_corpus(manifest)
    _check_gold_evidence(questions, pages)

    run_dir = RESULTS_ROOT / f"data-ingestion-{run_id}"
    metadata_path = run_dir / "run_metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"No dev run metadata found for run ID {run_id}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    input_files = [MANIFEST_PATH, QUESTIONS_PATH] + [row["resolved_path"] for row in manifest]
    if metadata["input_fingerprint"] != _fingerprint(input_files):
        raise ValueError("Manifest or benchmark changed after dev; do not treat test as a holdout")
    if metadata["embedding_model"] != (os.getenv("EMBEDDING_MODEL") or "gemini-embedding-001"):
        raise ValueError("Embedding model changed since the dev run")
    if metadata["embedding_sdk_version"] != version("google-genai"):
        raise ValueError("google-genai version changed since the dev run")
    if metadata["pypdf_version"] != version("pypdf"):
        raise ValueError("pypdf version changed since the dev run")
    if metadata["psycopg_version"] != version("psycopg"):
        raise ValueError("psycopg version changed since the dev run")

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL is required in starter/.env")
    embedder = MyEmbedder(model_name=metadata["embedding_model"])
    store = PgVectorStore(database_url=database_url, dimensions=embedder.dimensions)
    if metadata["vector_store"] != store.backend_info():
        raise ValueError("PostgreSQL or pgvector version changed since the dev run")
    index_names = metadata["index_names"]
    for strategy in STRATEGIES.values():
        if store.count(index_names[strategy]) == 0:
            raise ValueError(f"Missing stored index for strategy {strategy}; run the dev phase again")

    test_questions = [question for question in questions if question["split"] == "test"]
    evaluation_started = time.perf_counter()
    print(f"Embedding {len(test_questions)} held-out test questions...")
    query_vectors = [embedder.embed_query(question["question"]) for question in test_questions]
    traces = []
    for label, strategy in STRATEGIES.items():
        for question, query_vector in zip(test_questions, query_vectors):
            requested_filter = question.get("filter")
            applied_filter = requested_filter if label == "C" else None
            hits = store.search(
                strategy=index_names[strategy],
                query_vector=query_vector,
                top_k=3,
                filters=applied_filter,
            )
            traces.append({
                "strategy": strategy,
                "split": question["split"],
                "question": question["question"],
                "requested_filter": requested_filter,
                "applied_filter": applied_filter,
                "filter_compliant": _filter_compliant(hits, requested_filter),
                "gold_source_id": question["gold_source_id"],
                "gold_page": question["gold_page"],
                "evidence_quote": question["evidence_quote"],
                **evaluate_query(question, hits),
            })
    evaluation_seconds = time.perf_counter() - evaluation_started

    with (run_dir / "test_traces.jsonl").open("w", encoding="utf-8") as file:
        for trace in traces:
            file.write(json.dumps(trace, ensure_ascii=False, default=_json_default) + "\n")
    metadata.update({
        "test_completed_at": datetime.now(timezone.utc).isoformat(),
        "selected_strategy_on_dev": STRATEGIES[selected_strategy],
        "selection_criterion": selection_criterion,
        "test_evaluation_seconds": evaluation_seconds,
        "test_metrics": _trace_summary(traces),
    })
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    print(f"Test phase complete for dev-selected strategy {STRATEGIES[selected_strategy]}.")
    print(f"Results: {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Day 06 ingestion benchmark.")
    parser.add_argument("--phase", choices=("dev", "test"), required=True)
    parser.add_argument("--run-id", help="Run ID printed at the end of the dev phase")
    parser.add_argument("--selected-strategy", choices=tuple(STRATEGIES), help="Chosen from dev results")
    parser.add_argument("--selection-criterion", help="Criterion chosen from dev results before test")
    args = parser.parse_args()

    if args.phase == "dev":
        if args.run_id or args.selected_strategy or args.selection_criterion:
            parser.error("test-phase arguments are only valid for --phase test")
        _run_dev()
        return
    if not args.run_id or not args.selected_strategy or not args.selection_criterion:
        parser.error(
            "--phase test requires --run-id, --selected-strategy, and --selection-criterion"
        )
    _run_test(args.run_id, args.selected_strategy, args.selection_criterion)


if __name__ == "__main__":
    main()
