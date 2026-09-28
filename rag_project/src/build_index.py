from __future__ import annotations

from pathlib import Path

from .chunker import load_and_chunk_corpus

CORPUS_DIR = Path(__file__).resolve().parent.parent / "data" / "corpus"


def main() -> None:
    print(f"Scanning corpus at: {CORPUS_DIR}")
    chunks = load_and_chunk_corpus(CORPUS_DIR)
    sources = sorted({c.source for c in chunks})

    if not sources:
        print(
            "No .txt files found. This is expected on a fresh checkout"
            "drop your own .txt files into data/corpus/ and run this again "
            "if you want to confirm they're picked up correctly."
        )
        return

    print(f"Found {len(chunks)} chunks across {len(sources)} document(s):")
    for name in sources:
        count = sum(1 for c in chunks if c.source == name)
        print(f"  - {name}: {count} chunk(s)")
    print(
        "\nNothing was saved -- cli.py and app_streamlit.py will build this "
        "same index fresh, in memory, when you run them."
    )


if __name__ == "__main__":
    main()


"""
build_index.py
---------------
Optional sanity-check utility: reports how many .txt files and chunks are
currently in data/corpus/ without building or saving anything.

Nothing in this project persists an index to disk anymore -- cli.py and
app_streamlit.py both build the index fresh, in memory, every time they
start, directly from data/corpus/. This script exists only so you can
quickly check your corpus is being read the way you expect:

    python -m src.build_index
"""
