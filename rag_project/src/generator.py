from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from abc import ABC, abstractmethod

from .vector_store import ScoredChunk

SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the user's question "
    "using ONLY the numbered context passages provided. Cite passages "
    "inline like [1], [2] where you use them. If the context does not "
    "contain the answer, say so explicitly instead of guessing."
)

NO_CONTEXT_SYSTEM_PROMPT = (
    "Answer the user's question directly from your own knowledge. Do not "
    "mention that you lack access to external documents."
)

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_TIMEOUT = float(os.environ.get("OLLAMA_TIMEOUT", "30"))


def build_augmented_prompt(query: str, ranked_chunks: list[ScoredChunk]) -> str:
    """Stage: augmentation. Assembles the numbered-context prompt the generator will see."""
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


def _ollama_chat(host: str, model: str, system: str, user: str, timeout: float) -> str:
    """POSTs a chat request to a local Ollama server's /api/chat endpoint."""
    payload = {
        "model": model,
        "stream": False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    request = urllib.request.Request(
        f"{host.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return data.get("message", {}).get("content", "").strip()


def ollama_is_reachable(host: str = OLLAMA_HOST, timeout: float = 1.5) -> bool:
    """Cheap connectivity check used by build_generator() to decide fallback."""
    try:
        urllib.request.urlopen(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        return True
    except (urllib.error.URLError, OSError):
        return False


class BaseGenerator(ABC):
    name: str = "base"

    @abstractmethod
    def generate(self, query: str, ranked_chunks: list[ScoredChunk]) -> str:
        ...

class OllamaGenerator(BaseGenerator):
    """Calls a local, open-source Ollama server with the augmented, retrieval-grounded prompt."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        timeout: float = OLLAMA_TIMEOUT,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.name = f"ollama ({model})"

    def generate(self, query: str, ranked_chunks: list[ScoredChunk]) -> str:
        prompt = build_augmented_prompt(query, ranked_chunks)
        try:
            return _ollama_chat(self.host, self.model, SYSTEM_PROMPT, prompt, self.timeout)
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host} with model '{self.model}'. "
                "Is `ollama serve` running, and has the model been pulled "
                f"(`ollama pull {self.model}`)? Original error: {exc}"
            ) from exc


class ExtractiveGenerator(BaseGenerator):
    """
    Offline fallback with no external calls: stitches together the
    top-scoring retrieved sentences into a plain, citation-tagged answer.
    Not as fluent as an LLM, but it proves the retrieval half of the
    pipeline works end to end even with no model server running.
    """

    name = "extractive (offline fallback)"

    def generate(self, query: str, ranked_chunks: list[ScoredChunk]) -> str:
        if not ranked_chunks:
            return (
                "No relevant passages were retrieved for this question, "
                "so no grounded answer can be given."
            )

        lines = [
            f"- **[{i + 1}]** {sc.chunk.text}" for i, sc in enumerate(ranked_chunks)
        ]
        body = "\n\n".join(lines)
        return (
            "(offline extractive mode -- start `ollama serve` with a pulled "
            "model for a fluent, synthesized answer)\n\n"
            f"Most relevant retrieved passages for \"{query}\":\n\n{body}"
        )


class NoContextGenerator(BaseGenerator):
    """Used for the no-RAG comparison mode: answers from parametric knowledge only, via Ollama."""

    def __init__(
        self,
        model: str = OLLAMA_MODEL,
        host: str = OLLAMA_HOST,
        timeout: float = OLLAMA_TIMEOUT,
    ):
        self.model = model
        self.host = host
        self.timeout = timeout
        self.name = f"ollama ({model}, no-rag baseline)"

    def generate(self, query: str, ranked_chunks: list[ScoredChunk]) -> str:
        try:
            return _ollama_chat(
                self.host, self.model, NO_CONTEXT_SYSTEM_PROMPT, query, self.timeout
            )
        except (urllib.error.URLError, OSError) as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self.host} with model '{self.model}' "
                f"for the no-RAG comparison. Original error: {exc}"
            ) from exc


def build_generator() -> BaseGenerator:
    """Uses a local Ollama model if a server is reachable right now, else the offline fallback."""
    if ollama_is_reachable():
        return OllamaGenerator()
    return ExtractiveGenerator()


"""
generator.py

The "generation" stage of the pipeline: turns (query + retrieved context)
into a final answer.

Two implementations are provided:

- OllamaGenerator: calls a locally-running Ollama server. Ollama is
  open-source software that serves open-weight models (Llama, Mistral,
  Gemma, Qwen, ...) on your own machine over a local HTTP API -- there is
  no external company, account, or API key involved, which is why it
  replaces the earlier Anthropic-backed generator in this build.
- ExtractiveGenerator: a zero-dependency, offline fallback that just
  returns the highest-scoring retrieved sentences. It exists so the
  whole pipeline is runnable and demonstrable even with no LLM running --
  useful for a live seminar where the Ollama server or Wi-Fi is not
  guaranteed.

`build_generator()` picks whichever one is usable, by checking whether an
Ollama server is actually reachable right now.

Setup for real LLM-backed generation:
    1. Install Ollama:      https://ollama.com
    2. Pull an open model:  ollama pull llama3.2
    3. Leave it running:    ollama serve   (usually starts automatically)
No API key, no signup, no external network calls at generation time --
everything happens against http://localhost:11434 on your own machine.

"""
