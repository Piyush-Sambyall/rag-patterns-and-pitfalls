"""
cli.py
------
Interactive command-line demo for the RAG pipeline.

Usage (from the project root, after `pip install -r requirements.txt`):

    python cli.py

Every run builds the index fresh, in memory, from whatever .txt files are
currently in data/corpus/ -- nothing is cached to disk, so there is never
stale or leftover data from a previous session. Add your own .txt files
to data/corpus/ before running it.

Commands inside the REPL:
    <any question>      -> runs the full RAG pipeline (retrieve, rank,
                            augment, generate) and prints every stage
    :compare <question>  -> runs the same question WITH and WITHOUT RAG,
                            side by side, so you can see the difference
    :quit                -> exit
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from src.chunker import load_and_chunk_corpus
from src.pipeline import RAGPipeline
from src.vector_store import TfidfVectorStore

CORPUS_DIR = Path(__file__).resolve().parent / "data" / "corpus"

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
    HAS_RICH = True
except ImportError:  # rich is a nicety, not a requirement
    console = None
    HAS_RICH = False


def _print(text: str, style: str = "") -> None:
    if HAS_RICH:
        console.print(text, style=style)
    else:
        print(text)


def _print_result(result, label: str) -> None:
    if HAS_RICH:
        table = Table(title=f"Retrieved & ranked context ({label})", show_lines=True)
        table.add_column("#", width=3)
        table.add_column("Source")
        table.add_column("Score", width=8)
        table.add_column("Chunk (truncated)")
        for i, sc in enumerate(result.ranked, start=1):
            snippet = sc.chunk.text[:100] + ("..." if len(sc.chunk.text) > 100 else "")
            table.add_row(str(i), sc.chunk.source, f"{sc.score:.3f}", snippet)
        if result.ranked:
            console.print(table)
        console.print(
            Panel(
                result.answer,
                title=f"Answer  ·  generator: {result.generator_name}  ·  "
                f"{result.latency_seconds:.2f}s  ·  RAG={'on' if result.used_rag else 'off'}",
                border_style="green" if result.used_rag else "yellow",
            )
        )
    else:
        print(f"\n--- {label} ---")
        print(f"Generator: {result.generator_name} | RAG: {result.used_rag} | "
              f"Latency: {result.latency_seconds:.2f}s")
        if result.ranked:
            print("Retrieved & ranked chunks:")
            for i, sc in enumerate(result.ranked, start=1):
                print(f"  [{i}] ({sc.chunk.source}, score={sc.score:.3f}) "
                      f"{sc.chunk.text[:100]}...")
        print("\nAnswer:")
        print(result.answer)


def load_pipeline() -> RAGPipeline:
    """Builds the index fresh, in memory, from whatever is in data/corpus/
    right now. Nothing is loaded from a previous run."""
    chunks = load_and_chunk_corpus(CORPUS_DIR)
    store = TfidfVectorStore()
    store.build(chunks)  # empty chunks is fine -- store stays in a "no data" state
    return RAGPipeline(store)


def main() -> None:
    load_dotenv()  # loads OLLAMA_HOST / OLLAMA_MODEL from .env if present
    pipeline = load_pipeline()

    if pipeline.store.is_empty():
        _print(
            f"No documents found in {CORPUS_DIR}.\n"
            "Add some .txt files there and restart -- every question will "
            "return \"no relevant passages\" until you do.\n",
            style="bold yellow",
        )
    else:
        num_sources = len(pipeline.store.source_names())
        _print(
            f"RAG pipeline ready. Indexed {len(pipeline.store.chunks)} chunks "
            f"from {num_sources} document(s) in {CORPUS_DIR}.\n"
            "Type a question, ':compare <question>' for a RAG-vs-no-RAG "
            "comparison, or ':quit' to exit.\n",
            style="bold cyan",
        )

    while True:
        try:
            user_input = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input in (":quit", ":q", "exit"):
            break

        if user_input.startswith(":compare "):
            question = user_input[len(":compare "):].strip()
            rag_result = pipeline.answer(question)
            _print_result(rag_result, "WITH RAG")
            try:
                no_rag_result = pipeline.answer_without_rag(question)
                _print_result(no_rag_result, "WITHOUT RAG (parametric only)")
            except Exception as exc:  # e.g. missing API key for the no-RAG call
                _print(f"(no-RAG comparison unavailable: {exc})", style="dim")
            continue

        result = pipeline.answer(user_input)
        _print_result(result, "RAG")


if __name__ == "__main__":
    main()
