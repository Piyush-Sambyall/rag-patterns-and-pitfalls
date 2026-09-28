"""
Retrieval quality regression test.

Guards against accuracy regressions in retrieval changes (e.g. tuning
the BM25/TF-IDF hybrid weights) by checking top-1 retrieval accuracy
against a small hand-labeled eval set: each query is paired with the
corpus document that should be the top-ranked result.

This is intentionally a small, hand-built eval set (not a substitute
for a large IR benchmark) -- its job is to catch an obvious regression
before it ships, not to make a general claim about retrieval quality.
"""

import unittest
from pathlib import Path

from src.chunker import load_and_chunk_corpus
from src.vector_store import TfidfVectorStore


# Fixture corpus checked into the test suite -- see test_pipeline.py for why
# this isn't data/corpus/ (which ships empty by design).
CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "corpus"

EVAL_SET = [
    ("what is embedding drift?", "03_embeddings.txt"),
    ("what is HNSW?", "02_vector_databases.txt"),
    ("what is multi-hop retrieval?", "06_future_agentic_rag.txt"),
    ("what is agentic RAG?", "06_future_agentic_rag.txt"),
    ("what is LlamaIndex?", "04_frameworks_and_tools.txt"),
    ("what is context window limit?", "05_rag_challenges.txt"),
    ("what is ranking error?", "05_rag_challenges.txt"),
    ("what is contrastive objective in embedding training?", "03_embeddings.txt"),
    ("what is self-correcting RAG?", "06_future_agentic_rag.txt"),
    ("what does the term hallucination mean for language models?", "01_rag_overview.txt"),
    ("what is customer support assist?", "07_applications.txt"),
    ("how does RAG help with domain customization?", "08_benefits.txt"),
]

# Measured top-1 accuracy at the time hybrid retrieval was added: 10/11 (~91%).
# The floor below is set a bit under that so small, harmless score-magnitude
# shifts don't break the build -- only a real regression should.
MIN_ACCEPTABLE_ACCURACY = 0.8


class TestRetrievalQuality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        chunks = load_and_chunk_corpus(CORPUS_DIR)
        cls.store = TfidfVectorStore()
        cls.store.build(chunks)

    def test_top1_accuracy_meets_floor(self):
        correct = 0
        misses = []
        for query, expected_source in EVAL_SET:
            results = self.store.search(query, top_k=1)
            got = results[0].chunk.source if results else None
            if got == expected_source:
                correct += 1
            else:
                misses.append((query, expected_source, got))

        accuracy = correct / len(EVAL_SET)
        self.assertGreaterEqual(
            accuracy,
            MIN_ACCEPTABLE_ACCURACY,
            f"Top-1 accuracy {accuracy:.2f} fell below the {MIN_ACCEPTABLE_ACCURACY} "
            f"floor. Misses: {misses}",
        )

    def test_off_topic_query_returns_nothing_confident(self):
        """An unrelated query shouldn't retrieve anything above a sane relevance floor."""
        results = self.store.search("what's the weather like today in Jammu?", top_k=4)
        for r in results:
            self.assertLess(
                r.score, 0.15, "Off-topic query scored suspiciously high -- "
                "check that hybrid boosting hasn't broken the relevance floor."
            )


if __name__ == "__main__":
    unittest.main()
