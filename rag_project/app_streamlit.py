"""
app_streamlit.py
-----------------
A browser-based interface for the RAG pipeline, built for live seminar
demos. Every stage of the pipeline (retrieval -> ranking -> augmentation
-> generation) is shown, not hidden behind a single "answer" box, so the
audience can see *why* the model answered the way it did.

Also supports uploading your own PDFs, images, and videos: their content
is extracted (PDF text, OCR for images, speech transcription for videos),
chunked, and merged into the retrieval index for the current session, so
you can ask questions grounded in a document you just uploaded, on top
of (or instead of) the built-in RAG-topic corpus.

Run with:

    streamlit run app_streamlit.py

(from the rag_project/ root, with the venv activated and the base index
already built via `python -m src.build_index`).
"""

from __future__ import annotations

import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from src.chunker import load_and_chunk_corpus
from src.file_router import ALL_SUPPORTED_EXTENSIONS
from src.file_router import chunk_uploaded_file
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
# Cached loaders
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_base_store() -> TfidfVectorStore | None:
    """The persisted index built from data/corpus/ (the built-in RAG-topic docs)."""
    if not INDEX_PATH.exists():
        return None
    return TfidfVectorStore.load(INDEX_PATH)


def build_index_now() -> None:
    chunks = load_and_chunk_corpus(CORPUS_DIR)
    store = TfidfVectorStore()
    store.build(chunks)
    store.save(INDEX_PATH)
    get_base_store.clear()


@st.cache_data(show_spinner=False)
def extract_and_chunk_upload(file_bytes: bytes, filename: str):
    """
    Cached on (file content, filename) so a Streamlit rerun -- triggered by
    any widget interaction, not just a new upload -- doesn't re-extract or
    re-transcribe the same file every time. Routes to the right loader
    (PDF text / image OCR / video transcript) by extension.
    """
    return chunk_uploaded_file(file_bytes, filename)


# --------------------------------------------------------------------------
# Sidebar — pipeline status, generator status, retrieval settings, PDF upload
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

    st.markdown("---")
    st.subheader("📎 Add your own files")
    uploaded_files = st.file_uploader(
        "Upload PDFs, images, or videos to add to the knowledge base",
        type=sorted(ext.lstrip(".") for ext in ALL_SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )
    st.caption(
        "PDF → text extraction · Image → OCR · Video → speech transcript. "
        "Extracted content is chunked and merged into retrieval for this "
        "session only."
    )
    st.caption(
        "Video transcription needs the optional `faster-whisper` package "
        "and downloads a small model the first time it runs (needs "
        "internet once). Images with no text, or a video with no speech, "
        "won't add any chunks."
    )


# --------------------------------------------------------------------------
# Index bootstrap
# --------------------------------------------------------------------------

base_store = get_base_store()

if base_store is None:
    st.warning("No base index found yet. Build it once from the demo corpus below.")
    if st.button("⚙️ Build index from data/corpus/", type="primary"):
        with st.spinner("Chunking corpus and building TF-IDF index..."):
            build_index_now()
        st.rerun()
    st.stop()


# --------------------------------------------------------------------------
# Merge uploaded files into a working (in-memory) store for this run.
# The persisted base_store on disk is never modified by uploads.
# --------------------------------------------------------------------------

uploaded_chunks = []
empty_uploads = []      # parsed fine, but no extractable content (blank image, silent video)
failed_uploads = []     # a real error (missing dependency, ffmpeg failure, corrupt file)

if uploaded_files:
    for uploaded in uploaded_files:
        try:
            chunks = extract_and_chunk_upload(uploaded.getvalue(), uploaded.name)
        except ImportError as exc:
            failed_uploads.append((uploaded.name, str(exc)))
            continue
        except Exception as exc:  # ffmpeg failure, corrupt PDF/image, etc.
            failed_uploads.append((uploaded.name, str(exc)))
            continue

        if chunks:
            uploaded_chunks.extend(chunks)
        else:
            empty_uploads.append(uploaded.name)

if uploaded_chunks:
    combined_chunks = base_store.chunks + uploaded_chunks
    working_store = TfidfVectorStore()
    working_store.build(combined_chunks)
    uploaded_sources = sorted({c.source for c in uploaded_chunks})
    st.sidebar.success(
        f"Added {len(uploaded_chunks)} chunks from {len(uploaded_sources)} "
        f"uploaded file(s): {', '.join(uploaded_sources)}"
    )
else:
    working_store = base_store

if empty_uploads:
    st.sidebar.warning(
        f"No extractable content found in: {', '.join(empty_uploads)} "
        "(e.g. an image with no text, or a video with no speech)."
    )

if failed_uploads:
    for name, error in failed_uploads:
        st.sidebar.error(f"Couldn't process '{name}': {error}")

pipeline = RAGPipeline(working_store)
pipeline.retriever.config.candidate_pool = candidate_pool
pipeline.retriever.config.final_k = final_k
pipeline.retriever.config.min_similarity = min_similarity

with st.sidebar.expander(
    f"Indexed sources ({len({c.source for c in working_store.chunks})})"
):
    for name in sorted({c.source for c in working_store.chunks}):
        tag = " (uploaded)" if any(uc.source == name for uc in uploaded_chunks) else ""
        st.caption(name + tag)

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
with st.expander("📚 Built-in corpus documents"):
    for path in sorted(CORPUS_DIR.glob("*.txt")):
        st.markdown(f"**{path.name}**")
        st.caption(path.read_text(encoding="utf-8")[:220] + "...")
