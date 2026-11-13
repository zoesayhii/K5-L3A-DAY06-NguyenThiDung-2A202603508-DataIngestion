"""TODO: evaluate retrieved evidence and preserve per-query traces."""
import re
import unicodedata
from typing import Any


def _normalize_evidence(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def evaluate_query(question: dict[str, Any], hits: list[dict[str, Any]]) -> dict[str, Any]:
    """Check source, page, and evidence text for Hit@1 and Evidence Hit@3.

    Accept any documented alternative gold evidence. Human grounding and chunk
    coherence must still be labeled manually.
    """
    gold_sources = question.get("gold_sources") or [{
        "source_id": question.get("gold_source_id", question.get("source")),
        "page": question.get("gold_page", question.get("page")),
    }]
    evidence_quotes = question.get("evidence_quotes") or [question.get("evidence_quote", "")]
    normalized_quotes = [_normalize_evidence(quote) for quote in evidence_quotes if quote]

    def is_evidence_hit(hit: dict[str, Any]) -> bool:
        source_id = hit.get("document_id", hit.get("source"))
        page = hit.get("page")
        matching_source = any(
            gold.get("source_id", gold.get("source")) == source_id
            and (gold.get("page") is None or gold.get("page") == page)
            for gold in gold_sources
        )
        normalized_chunk = _normalize_evidence(hit.get("text", ""))
        return matching_source and any(quote in normalized_chunk for quote in normalized_quotes)

    hit_1 = bool(hits) and is_evidence_hit(hits[0])
    hit_3 = any(is_evidence_hit(hit) for hit in hits[:3])

    return {
        "question_id": question.get("id"),
        "hit_1": hit_1,
        "hit_3": hit_3,
        "hits": hits
    }

def summarize(traces: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate counts and rates; callers group by strategy and split."""
    if not traces:
        return {
            "total": 0,
            "hit_1_count": 0,
            "hit_3_count": 0,
            "hit_1_accuracy": 0.0,
            "hit_3_accuracy": 0.0,
        }

    total = len(traces)
    hit_1_count = sum(1 for t in traces if t.get("hit_1"))
    hit_3_count = sum(1 for t in traces if t.get("hit_3"))

    return {
        "total": total,
        "hit_1_count": hit_1_count,
        "hit_3_count": hit_3_count,
        "hit_1_accuracy": hit_1_count / total,
        "hit_3_accuracy": hit_3_count / total
    }