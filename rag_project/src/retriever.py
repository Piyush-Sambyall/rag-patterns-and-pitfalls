"""
retriever.py
------------
Wraps the vector store with the two pipeline stages that sit between
"raw similarity search" and "stuffing text into a prompt":

  retrieval -> ranking

`retrieve()` pulls a generous candidate pool from the vector store.
`rank()` then re-orders / filters that pool with two cheap, explainable
heuristics:

  1. A minimum-similarity floor, so an off-topic query doesn't get
     force-fed the "least bad" chunk as if it were relevant.
  2. Source diversity (a simplified MMR / max-marginal-relevance idea):
     once a source document has contributed a chunk, its other chunks
     are pushed down the ranking, so the context window isn't wasted on
     five near-duplicate chunks from the same file.
"""

from __future__ import annotations

from dataclasses import dataclass

from .vector_store import ScoredChunk, TfidfVectorStore


@dataclass
class RetrieverConfig:
    candidate_pool: int = 8       # how many chunks to pull from the index
    final_k: int = 4              # how many chunks survive ranking
    min_similarity: float = 0.05  # below this, a chunk is treated as noise
    diversity_penalty: float = 0.15  # score penalty per repeat source


class Retriever:
    def __init__(self, store: TfidfVectorStore, config: RetrieverConfig | None = None):
        self.store = store
        self.config = config or RetrieverConfig()

    def retrieve(self, query: str) -> list[ScoredChunk]:
        """Stage 1: pull raw candidates from the vector store."""
        return self.store.search(query, top_k=self.config.candidate_pool)

    def rank(self, candidates: list[ScoredChunk]) -> list[ScoredChunk]:
        """Stage 2: filter by relevance floor, then re-rank for source diversity."""
        filtered = [c for c in candidates if c.score >= self.config.min_similarity]

        seen_sources: dict[str, int] = {}
        adjusted: list[ScoredChunk] = []
        for candidate in filtered:
            source = candidate.chunk.source
            repeats = seen_sources.get(source, 0)
            penalty = repeats * self.config.diversity_penalty
            adjusted_score = max(candidate.score - penalty, 0.0)
            adjusted.append(ScoredChunk(chunk=candidate.chunk, score=adjusted_score))
            seen_sources[source] = repeats + 1

        adjusted.sort(key=lambda c: c.score, reverse=True)
        return adjusted[: self.config.final_k]

    def retrieve_and_rank(self, query: str) -> tuple[list[ScoredChunk], list[ScoredChunk]]:
        """Convenience wrapper returning (raw_candidates, ranked_final)."""
        candidates = self.retrieve(query)
        ranked = self.rank(candidates)
        return candidates, ranked
