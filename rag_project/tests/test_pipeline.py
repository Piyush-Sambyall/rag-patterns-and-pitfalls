"""
Offline tests -- no API key or network access required.
Run with:  python -m unittest discover -v
"""

import unittest
from pathlib import Path

from src.chunker import chunk_text, load_and_chunk_corpus
from src.generator import ExtractiveGenerator, build_augmented_prompt
from src.pipeline import RAGPipeline
from src.retriever import Retriever
from src.vector_store import TfidfVectorStore

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


class TestChunker(unittest.TestCase):
    def test_chunk_text_splits_long_text(self):
        text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        chunks = chunk_text(text, source="synthetic.txt", max_words=20)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text.split()), 40)  # allows overlap slack

    def test_chunk_text_empty_input(self):
        self.assertEqual(chunk_text("", source="empty.txt"), [])

    def test_load_and_chunk_corpus(self):
        chunks = load_and_chunk_corpus(CORPUS_DIR)
        self.assertGreater(len(chunks), 0)
        sources = {c.source for c in chunks}
        self.assertIn("01_rag_overview.txt", sources)


class TestVectorStoreAndRetriever(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_and_chunk_corpus(CORPUS_DIR)
        cls.store = TfidfVectorStore()
        cls.store.build(cls.chunks)

    def test_search_returns_relevant_chunk(self):
        results = self.store.search("what is a vector database", top_k=3)
        self.assertGreater(len(results), 0)
        top_sources = {r.chunk.source for r in results}
        self.assertIn("02_vector_databases.txt", top_sources)

    def test_retriever_rank_deduplicates_sources(self):
        retriever = Retriever(self.store)
        candidates, ranked = retriever.retrieve_and_rank("RAG failure modes and hallucination")
        self.assertLessEqual(len(ranked), retriever.config.final_k)
        self.assertGreater(len(candidates), 0)


class TestGeneratorAndPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.chunks = load_and_chunk_corpus(CORPUS_DIR)
        cls.store = TfidfVectorStore()
        cls.store.build(cls.chunks)

    def test_augmented_prompt_contains_context_and_question(self):
        results = self.store.search("embedding drift", top_k=2)
        prompt = build_augmented_prompt("What is embedding drift?", results)
        self.assertIn("Question: What is embedding drift?", prompt)
        self.assertIn("[1]", prompt)

    def test_extractive_generator_runs_offline(self):
        results = self.store.search("agentic RAG multi-hop", top_k=2)
        generator = ExtractiveGenerator()
        answer = generator.generate("What is agentic RAG?", results)
        self.assertIsInstance(answer, str)
        self.assertGreater(len(answer), 0)

    def test_pipeline_end_to_end_offline(self):
        pipeline = RAGPipeline(self.store, generator=ExtractiveGenerator())
        result = pipeline.answer("How does RAG reduce hallucination?")
        self.assertTrue(result.used_rag)
        self.assertGreater(len(result.ranked), 0)
        self.assertGreater(len(result.answer), 0)
        self.assertGreater(len(result.sources), 0)


if __name__ == "__main__":
    unittest.main()
