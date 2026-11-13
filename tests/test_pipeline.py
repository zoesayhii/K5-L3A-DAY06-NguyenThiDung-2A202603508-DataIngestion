import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from starter.src.chunking import by_paragraph, fixed_size
from starter.src.evaluation import evaluate_query, summarize
from starter.src.models import DocumentPage
from starter.src.store import PgVectorStore
import starter.src.run as runner
from starter.src.run import (
    QUESTIONS_PATH,
    STRATEGIES,
    _build_chunks,
    _chunks_for_index,
    _filter_compliant,
    _index_name,
    _load_corpus,
    _load_manifest,
    _load_questions,
)


class ChunkingTests(unittest.TestCase):
    def setUp(self):
        self.page = DocumentPage(
            document_id="doc",
            source="doc.pdf",
            page=1,
            text="one two three four five six",
            metadata={"chapter": "Test"},
        )

    def test_fixed_size_overlap_and_stable_ids(self):
        first = fixed_size(self.page, size=4, overlap=1)
        second = fixed_size(self.page, size=4, overlap=1)

        self.assertEqual([chunk.text for chunk in first], ["one two three four", "four five six"])
        self.assertEqual([chunk.chunk_id for chunk in first], [chunk.chunk_id for chunk in second])

    def test_rejects_invalid_chunk_sizes(self):
        with self.assertRaises(ValueError):
            fixed_size(self.page, size=3, overlap=3)
        with self.assertRaises(ValueError):
            by_paragraph(self.page, max_size=0)

    def test_paragraph_chunking_preserves_paragraph_boundaries(self):
        page = DocumentPage("doc", "doc.pdf", 1, "one two\n\nthree four", {})
        chunks = by_paragraph(page, max_size=2)
        self.assertEqual([chunk.text for chunk in chunks], ["one two", "three four"])


class EvaluationTests(unittest.TestCase):
    def test_hit_requires_gold_page_and_evidence(self):
        question = {
            "id": "q1",
            "gold_source_id": "doc",
            "gold_page": 2,
            "evidence_quote": "important evidence",
        }
        unrelated_same_page = {"document_id": "doc", "page": 2, "text": "other text"}
        wrong_page = {"document_id": "doc", "page": 1, "text": "important evidence"}
        correct = {"document_id": "doc", "page": 2, "text": "Some  important\nevidence here."}

        self.assertFalse(evaluate_query(question, [unrelated_same_page])["hit_3"])
        self.assertFalse(evaluate_query(question, [wrong_page])["hit_3"])
        self.assertTrue(evaluate_query(question, [unrelated_same_page, correct])["hit_3"])
        self.assertFalse(evaluate_query(question, [unrelated_same_page, correct])["hit_1"])

    def test_summary_reports_counts_and_rates(self):
        result = summarize([{"hit_1": True, "hit_3": True}, {"hit_1": False, "hit_3": True}])
        self.assertEqual(result["hit_1_count"], 1)
        self.assertEqual(result["hit_3_count"], 2)
        self.assertEqual(result["hit_1_accuracy"], 0.5)
        self.assertEqual(result["hit_3_accuracy"], 1.0)


class StoreTests(unittest.TestCase):
    def test_vector_serialization_validates_values_and_dimensions(self):
        store = PgVectorStore("unused", dimensions=2)
        self.assertEqual(store._vector_text([0.25, -0.5]), "[0.25,-0.5]")
        with self.assertRaises(ValueError):
            store._vector_text([0.25])
        with self.assertRaises(ValueError):
            store._vector_text([float("nan"), 0.5])


class LabArtifactsTests(unittest.TestCase):
    def test_manifest_has_at_least_thirty_content_pages(self):
        pages, summary = _load_corpus(_load_manifest())
        self.assertEqual(summary["content_pages"], 38)
        self.assertEqual(len(pages), 38)

    def test_verified_evidence_quotes_exist_on_declared_pages(self):
        pages, _ = _load_corpus(_load_manifest())
        page_text = {(page.document_id, page.page): " ".join(page.text.split()) for page in pages}
        questions = [json.loads(line) for line in QUESTIONS_PATH.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(questions), 20)
        self.assertEqual(sum(question["split"] == "dev" for question in questions), 15)
        self.assertEqual(sum(question["split"] == "test" for question in questions), 5)
        self.assertTrue(all(question["status"] == "verified" for question in questions))
        filtered = [question for question in questions if question.get("filter")]
        self.assertGreaterEqual(len(filtered), 3)
        self.assertTrue(any(question["split"] == "test" for question in filtered))
        for question in questions:
            quote = " ".join(question["evidence_quote"].split())
            text = page_text[(question["gold_source_id"], question["gold_page"])]
            self.assertIn(quote.casefold(), text.casefold(), question["id"])
            if question.get("filter"):
                self.assertIn(question["filter"]["chapter"].casefold(), question["question"].casefold())

    def test_runner_blocks_unreviewed_draft_questions(self):
        draft_questions = []
        for index in range(20):
            question = {
                "id": f"draft-{index}",
                "split": "dev" if index < 15 else "test",
                "question": "Cancer Biology: What does the draft benchmark ask?",
                "expected_answer": "A test answer",
                "gold_source_id": "cancer-biology",
                "gold_page": 1,
                "evidence_quote": "A test quote",
                "filter": {"chapter": "Cancer Biology"} if index < 3 or index == 15 else None,
                "status": "draft",
            }
            draft_questions.append(question)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "questions.jsonl"
            path.write_text(
                "\n".join(json.dumps(question) for question in draft_questions) + "\n",
                encoding="utf-8",
            )
            with patch.object(runner, "QUESTIONS_PATH", path):
                with self.assertRaisesRegex(ValueError, "is not verified"):
                    _load_questions()

    def test_b_and_c_use_same_chunks_with_distinct_index_ids(self):
        pages, _ = _load_corpus(_load_manifest())
        chunks = _build_chunks(pages)
        strategy_b = chunks[STRATEGIES["B"]]
        strategy_c = chunks[STRATEGIES["C"]]

        self.assertEqual([chunk.text for chunk in strategy_b], [chunk.text for chunk in strategy_c])
        self.assertTrue(set(chunk.chunk_id for chunk in strategy_b).isdisjoint(
            chunk.chunk_id for chunk in strategy_c
        ))

    def test_filter_compliance_checks_every_retrieved_hit(self):
        hits = [
            {"metadata": {"chapter": "Cancer Biology"}},
            {"metadata": {"chapter": "Cancer Biology"}},
        ]
        self.assertTrue(_filter_compliant(hits, {"chapter": "Cancer Biology"}))
        self.assertFalse(_filter_compliant(hits + [{"metadata": {"chapter": "Other"}}],
                                           {"chapter": "Cancer Biology"}))
        self.assertIsNone(_filter_compliant(hits, None))

    def test_run_specific_indexes_and_chunk_ids_are_isolated(self):
        strategy = STRATEGIES["A"]
        first_id = _index_name("20261114T000000Z", strategy)
        second_id = _index_name("20261114T000001Z", strategy)
        self.assertNotEqual(first_id, second_id)

        page = DocumentPage("doc", "doc.pdf", 1, "one two", {})
        chunk = fixed_size(page, size=2, overlap=0)[0]
        first_chunk = _chunks_for_index("20261114T000000Z", strategy, [chunk])[0]
        second_chunk = _chunks_for_index("20261114T000001Z", strategy, [chunk])[0]
        self.assertNotEqual(first_chunk.chunk_id, second_chunk.chunk_id)


if __name__ == "__main__":
    unittest.main()
