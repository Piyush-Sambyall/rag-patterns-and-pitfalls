"""
vector_store.py
----------------
A small, self-contained vector store combining two classic sparse
retrieval signals: TF-IDF cosine similarity and BM25.

Why TF-IDF/BM25 instead of a neural embedding model? Two reasons, both
deliberate for this project:

1. It runs fully offline with no model download, so the demo works the
   moment `pip install` finishes -- no waiting on a 400MB sentence
   transformer, no GPU, no internet dependency for the retrieval half
   of the pipeline.
2. It is transparent: you can print the vocabulary and the weights and
   *see* why a chunk was retrieved, which is genuinely useful when you
   are explaining the pipeline in a seminar.

Why hybrid (TF-IDF + BM25) instead of either alone: TF-IDF cosine
captures broad topical overlap well but doesn't model term-frequency
saturation or document length; BM25 does, and rewards exact keyword
matches more directly. Combining them (BM25 re-weights the TF-IDF
ranking rather than replacing it) improves ranking accuracy without
changing the score's scale enough to break the relevance-floor
filtering in retriever.py -- see `search()` below for exactly how
they're combined and why.

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

from .bm25 import STOP_WORDS, BM25Index
from .chunker import Chunk


@dataclass
class ScoredChunk:
    chunk: Chunk
    score: float


class TfidfVectorStore:
    """In-memory hybrid TF-IDF + BM25 index with cosine-similarity search."""

    def __init__(self) -> None:
        self.vectorizer = TfidfVectorizer(
            stop_words=list(STOP_WORDS),  # shared with bm25.py -- see its comment
            ngram_range=(1, 2),
            max_df=0.9,
            sublinear_tf=True,  # log-scaled term frequency: a term appearing 10x
            # shouldn't count as "10x more relevant" than appearing once; this
            # softens that curve and measurably improves ranking on longer chunks.
        )
        self.bm25 = BM25Index()
        self.chunks: list[Chunk] = []
        self._matrix = None  # sparse doc-term matrix, built at index time

    def build(self, chunks: list[Chunk]) -> None:
        """
        Builds the index from `chunks`. An empty list is valid -- it leaves
        the store in a deliberate "no data yet" state (chunks=[],
        matrix=None) rather than raising, since a fresh run with an empty
        data/corpus/ is the expected starting point, not an error.
        """
        self.chunks = list(chunks)
        if not chunks:
            self._matrix = None
            return
        texts = [c.text for c in chunks]
        self._matrix = self.vectorizer.fit_transform(texts)
        self.bm25.build(texts)

    def is_empty(self) -> bool:
        """True if nothing has been indexed yet."""
        return self._matrix is None or not self.chunks

    def add_documents(self, new_chunks: list[Chunk]) -> None:
        """
        Merges additional chunks (e.g. from an uploaded PDF) into the
        index at runtime. TF-IDF/BM25 statistics depend on the whole
        corpus, so this re-fits both over ALL chunks (existing + new)
        rather than trying to incrementally patch them -- cheap enough
        at this corpus size (fast even at a few thousand chunks) and it
        keeps the similarity scores correct.
        """
        if not new_chunks:
            return
        combined = self.chunks + new_chunks
        self.build(combined)

    def source_names(self) -> list[str]:
        """Distinct source documents currently indexed, in first-seen order."""
        seen: list[str] = []
        for c in self.chunks:
            if c.source not in seen:
                seen.append(c.source)
        return seen

    def search(self, query: str, top_k: int = 4) -> list[ScoredChunk]:
        if self.is_empty():
            return []

        query_vec = self.vectorizer.transform([query])
        cosine_scores = cosine_similarity(query_vec, self._matrix)[0]
        bm25_scores = self.bm25.score(query)

        # BM25 re-weights the cosine ranking rather than replacing its scale:
        # normalize BM25 to [0, 1] across this query's results, then use it as
        # up to a 30% boost multiplier on the cosine score. A chunk with zero
        # keyword overlap keeps its raw cosine score (multiplier floor 0.7x);
        # a chunk that also matches BM25's exact-term signal gets boosted
        # toward its full cosine score (multiplier up to 1.0x). Combined score
        # is therefore always <= cosine score, so a genuinely irrelevant
        # query (near-zero cosine for everything) still gets filtered out by
        # the relevance floor in retriever.py -- hybrid re-ranking improves
        # ordering among plausible candidates without weakening that guard.
        max_bm25 = max(bm25_scores) if bm25_scores else 0.0
        combined_scores = np.array(
            [
                cosine_scores[i] * (0.7 + 0.3 * (bm25_scores[i] / max_bm25 if max_bm25 > 0 else 0.0))
                for i in range(len(self.chunks))
            ]
        )

        top_indices = np.argsort(combined_scores)[::-1][:top_k]
        results = [
            ScoredChunk(chunk=self.chunks[i], score=float(combined_scores[i]))
            for i in top_indices
            if combined_scores[i] > 0.0
        ]
        return results

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(
                {
                    "vectorizer": self.vectorizer,
                    "bm25": self.bm25,
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
        # backward-compatible: an index saved before hybrid retrieval was
        # added won't have a "bm25" key -- rebuild it from the chunk text
        # instead of forcing a full re-run of build_index.py.
        if "bm25" in data:
            store.bm25 = data["bm25"]
        else:
            store.bm25 = BM25Index()
            store.bm25.build([c.text for c in store.chunks])
        return store
