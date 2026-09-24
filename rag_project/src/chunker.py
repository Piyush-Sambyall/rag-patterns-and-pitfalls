"""
chunker.py
----------
Splits raw documents into overlapping chunks that are small enough to be
retrieved individually and fed to an LLM as context.

This is the first, unglamorous step of any RAG pipeline: garbage chunking
(too big, too small, split mid-sentence) quietly wrecks retrieval quality
later on, no matter how good the embeddings or the LLM are.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chunk:
    """A single retrievable unit of text plus its provenance."""

    id: str
    text: str
    source: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def _split_into_sentences(text: str) -> list[str]:
    """Naive sentence splitter (good enough for clean prose corpora)."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    # Split on sentence-ending punctuation followed by whitespace + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_text(
    text: str,
    source: str,
    max_words: int = 90,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Groups sentences into chunks of up to `max_words` words, carrying
    `overlap_sentences` sentences over into the next chunk so that context
    at a chunk boundary isn't lost entirely.
    """
    sentences = _split_into_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    current: list[str] = []
    current_words = 0
    chunk_index = 0

    for sentence in sentences:
        sentence_words = len(sentence.split())
        if current and current_words + sentence_words > max_words:
            chunk_text_ = " ".join(current)
            chunks.append(
                Chunk(
                    id=f"{source}::{chunk_index}",
                    text=chunk_text_,
                    source=source,
                    chunk_index=chunk_index,
                )
            )
            chunk_index += 1
            # carry the tail of the previous chunk forward for continuity
            carry = current[-overlap_sentences:] if overlap_sentences else []
            current = list(carry)
            current_words = sum(len(s.split()) for s in current)

        current.append(sentence)
        current_words += sentence_words

    if current:
        chunks.append(
            Chunk(
                id=f"{source}::{chunk_index}",
                text=" ".join(current),
                source=source,
                chunk_index=chunk_index,
            )
        )

    return chunks


def load_and_chunk_corpus(corpus_dir: str | Path, max_words: int = 90) -> list[Chunk]:
    """Reads every .txt file in `corpus_dir` and returns all chunks."""
    corpus_dir = Path(corpus_dir)
    all_chunks: list[Chunk] = []

    for file_path in sorted(corpus_dir.glob("*.txt")):
        raw_text = file_path.read_text(encoding="utf-8")
        file_chunks = chunk_text(raw_text, source=file_path.name, max_words=max_words)
        all_chunks.extend(file_chunks)

    return all_chunks
