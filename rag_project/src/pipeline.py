"""
pipeline.py
-----------
Ties every module together into the six-stage pipeline used throughout
the seminar deck:

    query -> retrieval -> ranking -> augmentation -> generation -> output

Each stage is a distinct, inspectable step on the returned PipelineResult,
so a caller (CLI, tests, a notebook) can print or log exactly what
happened at each hop -- which is the whole pedagogical point of building
this instead of just calling an API.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .generator import BaseGenerator, NoContextGenerator, build_augmented_prompt, build_generator
from .retriever import Retriever
from .vector_store import ScoredChunk, TfidfVectorStore


@dataclass
class PipelineResult:
    query: str
    candidates: list[ScoredChunk]        # stage: retrieval
    ranked: list[ScoredChunk]            # stage: ranking
    augmented_prompt: str                # stage: augmentation
    answer: str                          # stage: generation
    generator_name: str
    used_rag: bool
    latency_seconds: float
    sources: list[str] = field(default_factory=list)


class RAGPipeline:
    def __init__(self, store: TfidfVectorStore, generator: BaseGenerator | None = None):
        self.store = store
        self.retriever = Retriever(store)
        self.generator = generator or build_generator()

    def answer(self, query: str) -> PipelineResult:
        """Runs the full retrieval-augmented pipeline for one query."""
        start = time.perf_counter()

        # 1-2: retrieval + ranking
        candidates, ranked = self.retriever.retrieve_and_rank(query)

        # 3: augmentation
        augmented_prompt = build_augmented_prompt(query, ranked)

        # 4: generation
        answer_text = self.generator.generate(query, ranked)

        elapsed = time.perf_counter() - start

        return PipelineResult(
            query=query,
            candidates=candidates,
            ranked=ranked,
            augmented_prompt=augmented_prompt,
            answer=answer_text,
            generator_name=self.generator.name,
            used_rag=True,
            latency_seconds=elapsed,
            sources=sorted({sc.chunk.source for sc in ranked}),
        )

    def answer_without_rag(self, query: str, api_key: str | None = None) -> PipelineResult:
        """
        Comparison mode: skips retrieval entirely and asks the LLM to answer
        from parametric knowledge only, so the two modes can be shown
        side by side (this mirrors the published web demo's compare view).
        """
        start = time.perf_counter()
        no_context_generator = NoContextGenerator(api_key=api_key)
        answer_text = no_context_generator.generate(query, [])
        elapsed = time.perf_counter() - start

        return PipelineResult(
            query=query,
            candidates=[],
            ranked=[],
            augmented_prompt=query,
            answer=answer_text,
            generator_name=no_context_generator.name,
            used_rag=False,
            latency_seconds=elapsed,
            sources=[],
        )
