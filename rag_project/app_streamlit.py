"""
app_streamlit.py
-----------------
A browser-based interface for the RAG pipeline, built for live seminar
demos. Every stage of the pipeline (retrieval -> ranking -> augmentation
-> generation) is shown, not hidden behind a single "answer" box, so the
audience can see *why* the model answered the way it did.

Run with:

    streamlit run app_streamlit.py

(from the rag_project/ root, with the venv activated and the index
already built via `python -m src.build_index`).
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chunker import load_and_chunk_corpus
from src.generator import ExtractiveGenerator, NoContextGenerator, build_generator
from src.pipeline import RAGPipeline
from src.vector_store import TfidfVectorStore

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "data" / "corpus"
INDEX_PATH = ROOT / "index" / "store.pkl"

load_dotenv()

st.set_page_config(
    page_title="RAG Pipeline Demo",
    page_icon="🔎",
    layout="wide",
)


# --------------------------------------------------------------------------
# Cached loaders — Streamlit reruns the whole script on every interaction,
# so the index and pipeline are cached to avoid rebuilding them each time.
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_store() -> TfidfVectorStore | None:
    if not INDEX_PATH.exists():
        return None
    return TfidfVectorStore.load(INDEX_PATH)


@st.cache_resource(show_spinner=False)
def get_pipeline(_store: TfidfVectorStore) -> RAGPipeline:
    return RAGPipeline(_store)


def build_index_now() -> None:
    chunks = load_and_chunk_corpus(CORPUS_DIR)
    store = TfidfVectorStore()
    store.build(chunks)
    store.save(INDEX_PATH)
    get_store.clear()
    get_pipeline.clear()


# --------------------------------------------------------------------------
# Sidebar — pipeline status, corpus stats, generator status
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("🔎 RAG Pipeline")
    st.caption("Retrieval-Augmented Generation — Architectures and Failure Modes")
    st.caption("Piyush Sambyal · 2023A6R055 · MIET Jammu")

    st.markdown("---")
    st.subheader("Pipeline stages")
    st.markdown(
        "1. **Query**\n"
        "2. **Retrieval** — TF-IDF + cosine similarity\n"
        "3. **Ranking** — relevance floor + source diversity\n"
        "4. **Augmentation** — numbered context prompt\n"
        "5. **Generation** — Claude, or offline fallback\n"
        "6. **Output**"
    )

    st.markdown("---")
    st.subheader("Generator status")
    api_key_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if api_key_set:
        st.success("ANTHROPIC_API_KEY detected — using Claude for generation.")
    else:
        st.warning(
            "No ANTHROPIC_API_KEY found — using the offline extractive "
            "fallback. Add a key to your `.env` file for fluent, "
            "synthesized answers."
        )

    st.markdown("---")
    with st.expander("Retrieval settings"):
        candidate_pool = st.slider("Candidate pool (raw retrieval)", 4, 12, 8)
        final_k = st.slider("Final chunks after ranking", 1, 6, 4)
        min_similarity = st.slider("Minimum similarity floor", 0.0, 0.3, 0.05, 0.01)


# --------------------------------------------------------------------------
# Index bootstrap
# --------------------------------------------------------------------------

store = get_store()

if store is None:
    st.warning("No index found yet. Build it once from the demo corpus below.")
    if st.button("⚙️ Build index from data/corpus/", type="primary"):
        with st.spinner("Chunking corpus and building TF-IDF index..."):
            build_index_now()
        st.rerun()
    st.stop()

pipeline = get_pipeline(store)
pipeline.retriever.config.candidate_pool = candidate_pool
pipeline.retriever.config.final_k = final_k
pipeline.retriever.config.min_similarity = min_similarity

st.markdown("---")


# --------------------------------------------------------------------------
# Main query area
# --------------------------------------------------------------------------

st.header("Ask the RAG pipeline a question")

col_query, col_toggle = st.columns([4, 1])
with col_query:
    query = st.text_input(
        "Question",
        placeholder="e.g. Why does RAG reduce hallucination compared to pure LLM generation?",
        label_visibility="collapsed",
    )
with col_toggle:
    compare_mode = st.toggle("Compare vs. no-RAG", value=False)

run_clicked = st.button("Run pipeline ▶", type="primary", disabled=not query)

if run_clicked and query:
    with st.spinner("Retrieving, ranking, and generating..."):
        rag_result = pipeline.answer(query)
        no_rag_result = None
        no_rag_error = None
        if compare_mode:
            try:
                no_rag_result = pipeline.answer_without_rag(query)
            except Exception as exc:  # missing anthropic package or API key
                no_rag_error = str(exc)

    if compare_mode:
        col_rag, col_norag = st.columns(2)
    else:
        col_rag, col_norag = st.container(), None

    with col_rag:
        st.subheader("✅ With RAG")
        st.caption(
            f"Generator: `{rag_result.generator_name}` · "
            f"Latency: {rag_result.latency_seconds:.3f}s · "
            f"Sources used: {', '.join(rag_result.sources) or 'none'}"
        )
        st.markdown("**Retrieved & ranked context**")
        if rag_result.ranked:
            for i, sc in enumerate(rag_result.ranked, start=1):
                with st.expander(
                    f"[{i}] {sc.chunk.source} — similarity {sc.score:.3f}"
                ):
                    st.write(sc.chunk.text)
        else:
            st.info("No context passed the similarity floor for this query.")

        with st.expander("🔧 Full augmented prompt sent to the generator"):
            st.code(rag_result.augmented_prompt, language="text")

        st.markdown("**Answer**")
        st.success(rag_result.answer)

    if compare_mode and col_norag is not None:
        with col_norag:
            st.subheader("🚫 Without RAG (parametric only)")
            if no_rag_error:
                st.error(
                    "No-RAG comparison needs the `anthropic` package and "
                    f"an ANTHROPIC_API_KEY: {no_rag_error}"
                )
            elif no_rag_result:
                st.caption(
                    f"Generator: `{no_rag_result.generator_name}` · "
                    f"Latency: {no_rag_result.latency_seconds:.3f}s"
                )
                st.markdown("**Answer (no retrieved context)**")
                st.warning(no_rag_result.answer)

st.markdown("---")
with st.expander("📚 Corpus documents indexed"):
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        st.markdown(f"**{path.name}**")
        st.caption(path.read_text(encoding="utf-8")[:220] + "...")
