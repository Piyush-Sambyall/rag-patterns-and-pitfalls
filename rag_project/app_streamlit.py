"""
app_streamlit.py
A browser-based interface for the RAG pipeline, built for live seminar
demos. Every stage of the pipeline (retrieval -> ranking -> augmentation
-> generation) is shown, not hidden behind a single "answer" box, so the
audience can see *why* the model answered the way it did -- including a
visual trace of the actual query moving through each stage.

Also supports uploading your own PDFs, images, and videos: their content
is extracted (PDF text, OCR for images, speech transcription for videos),
chunked, and merged into the retrieval index for the current session, so
you can ask questions grounded in a document you just uploaded, on top
of (or instead of) the built-in RAG-topic corpus.

Run with: streamlit run app_streamlit.py

(from the rag_project/ root, with the venv activated). No index-build step
is needed: the base index is built fresh, in memory, from whatever .txt
files are in data/corpus/ at the moment the app starts. Nothing is cached
to disk, so restarting the app (or the whole process) always starts clean
- there is no leftover data from a previous run to worry about.
"""

from __future__ import annotations

import html
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

load_dotenv()

st.set_page_config(
    page_title="RAG Pipeline",
    layout="wide",
)

def inject_theme() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

        :root {
            --navy: #2b2d52;
            --navy-light: #3a3d72;
            --teal: #3f8f8a;
            --teal-light: #5db3ac;
            --text-soft: #d6d8f0;
            --text-dim: #9a9cc0;
        }

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        .stApp {
            background: linear-gradient(120deg, #0f1024, #1c1f4a, #133337, #1c1f4a, #0f1024);
            background-size: 400% 400%;
            animation: ragGradientShift 22s ease infinite;
        }

        @keyframes ragGradientShift {
            0%   { background-position: 0% 50%; }
            50%  { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #14152c, #191b3a);
            border-right: 1px solid rgba(255,255,255,0.08);
        }

        h1, h2, h3, p, span, label, .stMarkdown {
            color: var(--text-soft);
        }

        section[data-testid="stSidebar"] .stMarkdown p,
        section[data-testid="stSidebar"] label {
            color: var(--text-dim);
        }

        .stTextInput input, .stTextArea textarea {
            background: rgba(255,255,255,0.06) !important;
            color: var(--text-soft) !important;
            border: 1px solid rgba(255,255,255,0.14) !important;
        }

        .stButton button {
            background: linear-gradient(135deg, var(--teal), var(--navy-light));
            color: #ffffff;
            border: none;
            font-weight: 600;
        }
        .stButton button:hover {
            background: linear-gradient(135deg, var(--teal-light), var(--navy));
            color: #ffffff;
        }

        div[data-testid="stExpander"] {
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            border-radius: 10px;
        }

        hr { border-color: rgba(255,255,255,0.12) !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_flow_diagram(
    query: str,
    num_candidates: int,
    top_score: float,
    num_augmented: int,
    generator_name: str,
    sources: list[str],
) -> str:
    """
    Builds the pipeline-trace visual: six stages connected by animated,
    directional connectors, with the query's own real numbers filling in
    each stage as it completes. This is what actually shows "the direction
    a given question took" through the system, using this run's real data
    - not a static diagram.
    """
    q = html.escape(query)
    steps = [
        ("Step 1", "Query", f'"{q}"' if len(q) < 40 else f'"{q[:37]}..."'),
        ("Step 2", "Retrieval", f"{num_candidates} candidates found"),
        ("Step 3", "Ranking", f"top match {top_score:.3f}" if top_score else "no confident match"),
        ("Step 4", "Augmentation", f"{num_augmented} chunks injected"),
        ("Step 5", "Generation", html.escape(generator_name)),
        ("Step 6", "Output", "answer returned"),
    ]

    nodes_html = ""
    for i, (step_num, label, detail) in enumerate(steps):
        active = "active" if (num_augmented > 0 or i == 0) else ""
        nodes_html += (
            f'<div class="rag-node {active}">'
            f'<div class="rag-step-num">{step_num}</div>'
            f'<div class="rag-step-label">{label}</div>'
            f'<div class="rag-step-detail">{detail}</div>'
            f"</div>"
        )
        if i < len(steps) - 1:
            delay = i * 0.3
            nodes_html += (
                f'<div class="rag-connector"><div class="rag-pulse" '
                f'style="animation-delay:{delay}s"></div></div>'
            )

    chips = "".join(
        f'<span class="rag-source-chip">{html.escape(s)}</span>' for s in sources
    ) or '<span class="rag-source-dim">none passed the relevance floor</span>'

    return f"""
    <style>
    .rag-flow {{ display: flex; align-items: stretch; gap: 0; margin: 10px 0 22px 0; }}
    .rag-node {{
        flex: 1; background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.10); border-radius: 12px;
        padding: 14px 10px; text-align: center;
    }}
    .rag-node.active {{
        background: linear-gradient(160deg, #3f8f8a, #3a3d72);
        border-color: #5db3ac;
    }}
    .rag-step-num {{ font-size: 10px; color: #9a9cc0; letter-spacing: 1px; text-transform: uppercase; }}
    .rag-node.active .rag-step-num {{ color: rgba(255,255,255,0.75); }}
    .rag-step-label {{ font-size: 13px; font-weight: 600; color: #fff; margin: 3px 0 6px 0; }}
    .rag-step-detail {{ font-size: 10.5px; color: #9a9cc0; line-height: 1.35; word-break: break-word; }}
    .rag-node.active .rag-step-detail {{ color: rgba(255,255,255,0.85); }}
    .rag-connector {{ width: 22px; display: flex; align-items: center; justify-content: center; position: relative; }}
    .rag-connector::before {{
        content: ""; position: absolute; left: 0; right: 0; top: 50%; height: 2px;
        background: linear-gradient(90deg, #5db3ac, #3f8f8a); transform: translateY(-50%);
    }}
    .rag-pulse {{
        position: absolute; top: 50%; left: 0; width: 6px; height: 6px; margin-top: -3px;
        border-radius: 50%; background: #ffffff; box-shadow: 0 0 8px 2px #5db3ac;
        animation: ragTravel 2.2s linear infinite;
    }}
    @keyframes ragTravel {{
        0% {{ left: 0%; opacity: 0; }} 10% {{ opacity: 1; }} 90% {{ opacity: 1; }} 100% {{ left: 100%; opacity: 0; }}
    }}
    .rag-result-card {{
        background: rgba(255,255,255,0.05); border: 1px solid rgba(255,255,255,0.10);
        border-radius: 12px; padding: 14px 18px; margin-top: 4px;
    }}
    .rag-result-card .rag-label {{
        font-size: 10.5px; letter-spacing: 1px; text-transform: uppercase;
        color: #5db3ac; font-weight: 600; margin-bottom: 8px;
    }}
    .rag-source-chip {{
        display: inline-block; background: rgba(93,179,172,0.15);
        border: 1px solid rgba(93,179,172,0.4); color: #5db3ac;
        font-size: 12px; padding: 4px 10px; border-radius: 999px; margin: 3px 4px 3px 0;
    }}
    .rag-source-dim {{ color: #9a9cc0; font-size: 12px; font-style: italic; }}
    </style>
    <div class="rag-flow">{nodes_html}</div>
    <div class="rag-result-card">
        <div class="rag-label">Sources this answer is grounded in</div>
        {chips}
    </div>
    """


inject_theme()


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_base_store() -> TfidfVectorStore:
    """
    Builds the index fresh, in memory, from whatever .txt files are
    currently in data/corpus/. Nothing is cached to disk -- `st.cache_resource`
    only holds this for the lifetime of the running app process, so a
    restart always rebuilds from scratch. An empty data/corpus/ (the
    shipped default) is valid and produces an empty store, not an error.
    """
    chunks = load_and_chunk_corpus(CORPUS_DIR)
    store = TfidfVectorStore()
    store.build(chunks)
    return store


def reset_base_store() -> None:
    """Forces a rebuild from data/corpus/ on the next call to get_base_store()."""
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
# Sidebar — pipeline status, generator status, retrieval settings, upload
# --------------------------------------------------------------------------

with st.sidebar:
    st.title("RAG Pipeline")
    st.caption("Retrieval-Augmented Generation — Architectures and Failure Modes")
    st.caption("Piyush Sambyal · 2023A6R055 · MIET Jammu")

    st.markdown("---")
    st.subheader("Pipeline stages")
    st.markdown(
        "1. **Query**\n"
        "2. **Retrieval** — hybrid TF-IDF + BM25\n"
        "3. **Ranking** — relevance floor + source diversity\n"
        "4. **Augmentation** — numbered context prompt\n"
        "5. **Generation** — local open-source model via Ollama, or offline fallback\n"
        "6. **Output**"
    )

    

    st.markdown("---")
    with st.expander("Retrieval settings"):
        candidate_pool = st.slider("Candidate pool (raw retrieval)", 4, 12, 8)
        final_k = st.slider("Final chunks after ranking", 1, 6, 4)
        min_similarity = st.slider("Minimum similarity floor", 0.0, 0.3, 0.05, 0.01)

    st.markdown("---")
    st.subheader("Add your own files")
    uploaded_files = st.file_uploader(
        "Upload PDFs, images, or videos to add to the knowledge base",
        type=sorted(ext.lstrip(".") for ext in ALL_SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
        label_visibility="collapsed",
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
# Index bootstrap -- always fresh, never loaded from a previous run
# --------------------------------------------------------------------------

base_store = get_base_store()

if base_store.is_empty():
    st.info(
        f"No documents in {CORPUS_DIR} yet. Drop your own `.txt` files there "
        "and click the button below, or use the uploader in the sidebar to "
        "add PDFs/images/videos for this session."
    )
    if st.button("Rescan data/corpus/", type="primary"):
        reset_base_store()
        st.rerun()


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
        placeholder="What's your Query",
        label_visibility="collapsed",
    )
with col_toggle:
    compare_mode = st.toggle("Compare vs. no-RAG", value=False)

run_clicked = st.button("Run pipeline", type="primary", disabled=not query)

if run_clicked and query:
    with st.spinner("Retrieving, ranking, and generating..."):
        rag_result = pipeline.answer(query)
        no_rag_result = None
        no_rag_error = None
        if compare_mode:
            try:
                no_rag_result = pipeline.answer_without_rag(query)
            except Exception as exc:  # Ollama unreachable, model not pulled, etc.
                no_rag_error = str(exc)

    # The directional flow trace, built from this run's real numbers.
    candidates_for_trace = pipeline.retriever.retrieve(query)
    top_score = rag_result.ranked[0].score if rag_result.ranked else 0.0
    st.markdown(
        render_flow_diagram(
            query=query,
            num_candidates=len(candidates_for_trace),
            top_score=top_score,
            num_augmented=len(rag_result.ranked),
            generator_name=rag_result.generator_name,
            sources=rag_result.sources,
        ),
        unsafe_allow_html=True,
    )

    if compare_mode:
        col_rag, col_norag = st.columns(2)
    else:
        col_rag, col_norag = st.container(), None

    with col_rag:
        st.subheader("With RAG")
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

        with st.expander("Full augmented prompt sent to the generator"):
            st.code(rag_result.augmented_prompt, language="text")

        st.markdown("**Answer**")
        st.success(rag_result.answer)

    if compare_mode and col_norag is not None:
        with col_norag:
            st.subheader("Without RAG (parametric only)")
            if no_rag_error:
                st.error(
                    f"No-RAG comparison needs a running Ollama server: {no_rag_error}"
                )
            elif no_rag_result:
                st.caption(
                    f"Generator: `{no_rag_result.generator_name}` · "
                    f"Latency: {no_rag_result.latency_seconds:.3f}s"
                )
                st.markdown("**Answer (no retrieved context)**")
                st.warning(no_rag_result.answer)

st.markdown("---")
corpus_files = sorted(CORPUS_DIR.glob("*.txt"))
with st.expander(f"data/corpus/ documents ({len(corpus_files)})"):
    if not corpus_files:
        st.caption(
            "Empty — no default data is shipped. Drop .txt files into "
            "data/corpus/ and click \"Rescan data/corpus/\" above, or use "
            "the uploader in the sidebar for a this-session-only source."
        )
    for path in corpus_files:
        st.markdown(f"**{path.name}**")
        st.caption(path.read_text(encoding="utf-8")[:220] + "...")



