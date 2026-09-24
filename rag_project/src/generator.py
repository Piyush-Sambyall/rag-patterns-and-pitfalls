"""
generator.py
------------
The "generation" stage of the pipeline: turns (query + retrieved context)
into a final answer.

Two implementations are provided:

- ClaudeGenerator: calls the Anthropic API (Claude) with the augmented
  prompt. This is the real LLM-backed generator.
- ExtractiveGenerator: a zero-dependency, offline fallback that just
  returns the highest-scoring retrieved sentences. It exists so the
  whole pipeline is runnable and demonstrable even with no API key and
  no internet -- useful for a live seminar where Wi-Fi is not
  guaranteed.

`build_generator()` picks whichever one is usable based on whether an
API key is configured.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod

from .vector_store import ScoredChunk

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Cite passages "
    "inline like [1], [2] where you use them. If the context does not "
    "contain the answer, say so explicitly instead of guessing."
)


def build_augmented_prompt(query: str, ranked_chunks: list[ScoredChunk]) -> str:
    """Stage: augmentation. Assembles the numbered-context prompt Claude will see."""
    if not ranked_chunks:
        context_block = "(no relevant context was retrieved)"
    else:
        context_block = "\n\n".join(
            f"[{i + 1}] (source: {sc.chunk.source}, score: {sc.score:.3f})\n{sc.chunk.text}"
            for i, sc in enumerate(ranked_chunks)
        )

    return (
        f"Context passages:\n{context_block}\n\n"
        f"Question: {query}\n\n"
        "Answer using only the context above, with inline citations like [1]."
    )


class BaseGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, query: str, ranked_chunks: list[ScoredChunk]) -> str:
        ...


class ClaudeGenerator(BaseGenerator):
    """Calls the Anthropic Messages API with the augmented, retrieval-grounded prompt."""

    name = "claude"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic  # imported here so the package is optional at install time

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, query: str, ranked_chunks: list[ScoredChunk]) -> str:
        prompt = build_augmented_prompt(query, ranked_chunks)
        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


class ExtractiveGenerator(BaseGenerator):
    """
    Offline fallback with no external calls: stitches together the
    top-scoring retrieved sentences into a plain, citation-tagged answer.
    Not as fluent as an LLM, but it proves the retrieval half of the
    pipeline works end to end even with zero network access.
    """

    name = "extractive (offline fallback)"

    def generate(self, query: str, ranked_chunks: list[ScoredChunk]) -> str:
        if not ranked_chunks:
            return (
                "No relevant passages were retrieved for this question, "
                "so no grounded answer can be given."
            )

        lines = [
            f"[{i + 1}] {sc.chunk.text}" for i, sc in enumerate(ranked_chunks)
        ]
        body = "\n".join(lines)
        return (
            "(offline extractive mode -- set ANTHROPIC_API_KEY for a fluent, "
            "synthesized answer)\n\n"
            f"Most relevant retrieved passages for \"{query}\":\n{body}"
        )


class NoContextGenerator(BaseGenerator):
    """Used for the no-RAG comparison mode: answers from parametric knowledge only."""

    name = "no-rag (parametric only)"

    def __init__(self, model: str = "claude-sonnet-4-6", api_key: str | None = None):
        import anthropic

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def generate(self, query: str, ranked_chunks: list[ScoredChunk]) -> str:
        response = self.client.messages.create(
            model=self.model,
            max_tokens=600,
            system=(
                "Answer the user's question directly from your own knowledge. "
                "Do not mention that you lack access to external documents."
            ),
            messages=[{"role": "user", "content": query}],
        )
        return "".join(block.text for block in response.content if block.type == "text")


def build_generator() -> BaseGenerator:
    """Picks ClaudeGenerator if ANTHROPIC_API_KEY is set, else the offline fallback."""
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if api_key:
        try:
            return ClaudeGenerator(api_key=api_key)
        except ImportError:
            pass  # anthropic package not installed -- fall through to offline mode
    return ExtractiveGenerator()
