"""
build_index.py
---------------
Ingests every .txt file in data/corpus/, chunks it, builds a TF-IDF
vector index, and saves it to index/store.pkl.

Run this once (or whenever the corpus changes) before using cli.py:

    python -m src.build_index
"""

from __future__ import annotations

from pathlib import Path

from .chunker import load_and_chunk_corpus
from .vector_store import TfidfVectorStore

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"
INDEX_PATH = Path(__file__).resolve().parent.parent / "index" / "store.pkl"


def main() -> None:
    print(f"Loading corpus from: {CORPUS_DIR}")
    chunks = load_and_chunk_corpus(CORPUS_DIR)
    print(f"Loaded {len(chunks)} chunks from "
          f"{len({c.source for c in chunks})} source documents.")

    store = TfidfVectorStore()
    store.build(chunks)
    store.save(INDEX_PATH)
    print(f"Index built and saved to: {INDEX_PATH}")


if __name__ == "__main__":
    main()
