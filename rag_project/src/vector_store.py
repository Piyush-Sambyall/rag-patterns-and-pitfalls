"""
vector_store.py
----------------
A small, self-contained vector store built on scikit-learn's TF-IDF
vectorizer + cosine similarity.

Why TF-IDF instead of a neural embedding model? Two reasons, both
deliberate for this project:

1. It runs fully offline with no model download, so the demo works the
   moment `pip install` finishes -- no waiting on a 400MB sentence
   transformer, no GPU, no internet dependency for the retrieval half
   of the pipeline.
2. It is transparent: you can print the vocabulary and the weights and
   *see* why a chunk was retrieved, which is genuinely useful when you
   are explaining the pipeline in a seminar.

Swapping this for FAISS + a sentence-transformers encoder is a drop-in
change -- see the "Swapping in real embeddings" section of the README.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunker import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class TfidfVectorStore:
    """In-memory TF-IDF index with cosine-similarity search."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_df=0.9,
        )
        self.chunks: list[Chunk] = []
        self._matrix = None  # sparse doc-term matrix, built at index time

    def build(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("Cannot build a vector store from zero chunks.")
        self.chunks = chunks
        texts = [c.text for c in chunks]
        self._matrix = self.vectorizer.fit_transform(texts)

    def search(self, query: str, top_k: int = 4) -> list[ScoredChunk]:
        if self._matrix is None:
            raise RuntimeError("Vector store is empty -- call build() first.")

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self._matrix)[0]

        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [
            ScoredChunk(chunk=self.chunks[i], score=float(similarities[i]))
            for i in top_indices
            if similarities[i] > 0.0
        ]
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "chunks": self.chunks,
                    "matrix": self._matrix,
                },
                f,
            )

    @classmethod
    def load(cls, path: str | Path) -> "TfidfVectorStore":
        with open(path, "rb") as f:
            data = pickle.load(f)
        store = cls()
        store.vectorizer = data["vectorizer"]
        store.chunks = data["chunks"]
        store._matrix = data["matrix"]
        return store
