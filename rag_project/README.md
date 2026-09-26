# RAG Pipeline — Working Reference Implementation

A small, fully functional Retrieval-Augmented Generation pipeline in
Python, built to accompany the seminar **"Retrieval-Augmented Generation:
Architectures and Failure Modes"** (Piyush Sambyal, 2023A6R055, MIET
Jammu). It implements every stage from the seminar's pipeline diagram —

```
query -> retrieval -> ranking -> augmentation -> generation -> output
```

— as real, runnable code, with no mocked-out stages. The demo corpus is
about RAG itself, so you can literally ask the pipeline questions about
retrieval, vector databases, and RAG failure modes and watch it answer
using its own retrieved context.

## What's actually implemented

| Stage         | File                     | What it does |
|---------------|--------------------------|---------------|
| Chunking      | `src/chunker.py`         | Splits raw `.txt` documents into overlapping, sentence-aware chunks |
| Retrieval     | `src/vector_store.py`, `src/retriever.py` | TF-IDF vectorization + cosine similarity search over all chunks |
| Ranking       | `src/retriever.py`       | Relevance floor + source-diversity re-ranking (a simplified MMR) |
| Augmentation  | `src/generator.py`       | Builds a numbered, citation-ready context prompt |
| Generation    | `src/generator.py`       | Calls Claude (Anthropic API) if a key is set, else an offline extractive fallback |
| Orchestration | `src/pipeline.py`        | `RAGPipeline` ties all stages together and returns every intermediate result |
| Demo UI       | `cli.py`                 | Interactive terminal REPL, plus a `:compare` mode (RAG vs. no-RAG) |

Retrieval uses TF-IDF rather than a neural embedding model on purpose: it
needs no model download and works fully offline, so the retrieval half of
the pipeline is guaranteed to work in a live demo even without Wi-Fi. See
**"Swapping in real embeddings"** below if you want to upgrade it.

Generation calls Claude if `ANTHROPIC_API_KEY` is set, and otherwise falls
back to an offline extractive generator, so the *entire* pipeline —
including generation — still runs with zero internet access. This makes a
good live comparison point in the seminar: "here is the same retrieved
context, with and without an LLM synthesizing it."

## Setup (Windows PowerShell)

Run each line separately (this project assumes PowerShell, not bash).

```powershell
cd rag_project
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If PowerShell blocks the activation script, run this once and retry:

```powershell
Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
```

(Optional, for LLM-backed generation) copy `.env.example` to `.env` and
paste in an Anthropic API key from https://console.anthropic.com/settings/keys:

```powershell
Copy-Item .env.example .env
notepad .env
```

Without a key, the pipeline still runs end to end using the offline
extractive generator — nothing is required to get retrieval + ranking +
augmentation working.

## Running it

Build the vector index once (rerun this whenever you edit files in
`data/corpus/`):

```powershell
python -m src.build_index
```

Then start the interactive demo:

```powershell
python cli.py
```

Example session:

```
query> what is embedding drift?
query> :compare why does RAG reduce hallucination?
query> :quit
```

`:compare <question>` runs the same question twice — once through the
full RAG pipeline, once with retrieval skipped entirely (parametric-only,
mirrors the "no-RAG" comparison mode in the published web demo) — so you
can show the difference live.

## Browser interface (Streamlit)

For a live demo, use the browser-based UI instead of the terminal REPL.
It shows every pipeline stage — retrieved chunks with similarity scores,
the exact augmented prompt sent to the generator, and the final answer —
plus a "Compare vs. no-RAG" toggle that runs the query with and without
retrieval side by side.

```powershell
python -m src.build_index
streamlit run app_streamlit.py
```

This opens automatically in your browser at `http://localhost:8501`. The
sidebar shows generator status (Claude vs. offline fallback) and lets you
tune retrieval settings (candidate pool size, final chunk count, minimum
similarity) live, without editing code.

If the index hasn't been built yet, the app shows a "Build index" button
so you don't need to touch the terminal at all after the initial
`pip install`.

### Uploading your own files (PDF, images, video)

The sidebar has an **"📎 Add your own files"** uploader. Any file you drop
in there is routed by type (`src/file_router.py`):

| Type | How it's read | Extra setup needed |
|------|---------------|---------------------|
| PDF (`.pdf`) | Text extraction (`pypdf`) | None |
| Image (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`) | OCR (`pytesseract`) | Install the Tesseract OCR engine (see below) |
| Video (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`) | Audio extracted with a bundled ffmpeg, then transcribed with `faster-whisper` | `pip install faster-whisper` (not in the base install — see requirements.txt) |

Extracted content is chunked the same way as the built-in corpus and
merged into a fresh in-memory index for that session. Your questions can
then retrieve from your upload(s), the built-in RAG corpus, or both.
Nothing is written to disk — the persisted `index/store.pkl` is never
touched by an upload; remove the file from the uploader or restart the
app and it's gone. If you want something to be part of the permanent
corpus instead, save its extracted text as a `.txt` file under
`data/corpus/` and rerun `python -m src.build_index`.

**Installing Tesseract OCR (Windows), needed for image uploads:**

1. Download the installer from
   https://github.com/UB-Mannheim/tesseract/wiki
2. Run it (default install path is fine)
3. Either add the install folder to your PATH, or point `pytesseract` at
   it directly by adding this near the top of `app_streamlit.py`:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
   ```

**Video transcription, what to expect:**
```powershell
pip install faster-whisper
```
The first time you upload a video, it downloads the "tiny" Whisper model
(~75MB) — that one download needs internet; after that it runs fully
offline. Audio extraction itself needs no separate ffmpeg install —
`imageio-ffmpeg` bundles a static binary through pip.

**What was actually verified vs. not, for each type:**
- **PDF** — fully verified end to end (extraction, chunking, retrieval, correct source attribution).
- **Image (OCR)** — fully verified end to end with a real generated test image; Tesseract and `pytesseract` ran and extracted real text, which was correctly retrieved.
- **Video** — the audio-extraction step (ffmpeg) was verified with a real generated test video. The transcription step (`faster-whisper`) could **not** be verified — it isn't installed in the build environment and there's no network there to install it or download the model. The failure path (missing package → clear error message shown in the sidebar) was verified instead. Test the actual transcription yourself before relying on it live.

**Failure behavior:** an image with no text, or a video with no speech,
adds zero chunks and shows a plain warning (not an error) in the
sidebar — that's expected, not a bug. A genuine failure (missing
`faster-whisper`, a corrupt file, an ffmpeg error) shows as a red error
naming the file and the reason.

> **Note:** `app_streamlit.py` is syntax-checked and built directly on the
> same tested `src/` pipeline modules as `cli.py` (the retrieval, ranking,
> augmentation, and generation logic is identical and already verified —
> see "Running the tests" below). The Streamlit UI layer itself could not
> be launched inside this sandbox (no network access to install
> `streamlit` here), so run it locally and confirm the browser view looks
> right before your demo.

## Retrieval accuracy & performance

Retrieval is **hybrid TF-IDF + BM25** (`src/bm25.py`, `src/vector_store.py`),
not plain TF-IDF cosine similarity. BM25 re-weights the TF-IDF ranking
using corpus-wide term statistics that model term-frequency saturation
and document length — it rewards exact keyword matches more directly
than cosine similarity alone, which measurably helps short, term-heavy
questions. The combination is designed so BM25 can only ever pull a
score *down toward* its cosine baseline or boost it *up to* that
baseline (max 30%) — never past it — so a genuinely irrelevant query
still gets filtered by the relevance floor in `retriever.py` exactly as
before.

**Measured, not claimed** — before/after numbers on a labeled 10-query
eval set (`tests/test_retrieval_quality.py`), benchmarked in this build
environment:

| Metric | Before (TF-IDF only) | After (hybrid TF-IDF + BM25) |
|---|---|---|
| Top-1 retrieval accuracy | 9/10 (90%) | 10/10 (100%) |
| Off-topic query ("what's the weather like today?") | incorrectly returned 2 chunks (score 0.103) | correctly returns 0 chunks |
| Latency | 0.46 ms/query | 0.53 ms/query |

Two real bugs were found and fixed along the way, not just the hybrid
scoring added:
1. **BM25 needs its own stopword filtering.** Without it, a query like
   "what is FAISS used for?" let common words like "used" inflate the
   score of chunks that had nothing to do with FAISS. Fixed by sharing
   one stopword list between TF-IDF and BM25 (`STOP_WORDS` in `bm25.py`).
2. **scikit-learn's default English stopword list doesn't include "like".**
   A query such as "what's the weather like today?" registered "like" as
   real content and matched any chunk containing it — a genuine false
   positive that pre-dated the hybrid change. Fixed with a small
   supplementary stopword set (`_SUPPLEMENTARY_STOP_WORDS` in `bm25.py`),
   applied to both TF-IDF and BM25 so they can't disagree on what counts
   as content vs. noise.

**Honest note on "performance":** at this corpus's size (27 chunks),
retrieval was already sub-millisecond before this change, so there was
no real latency problem to solve — the ~0.07ms difference above is
noise-level, not a meaningful speedup or slowdown. The real "performance"
improvement here is retrieval *quality* (accuracy), plus statistics
(BM25 index, TF-IDF matrix) being computed once at index-build time
rather than recomputed per query — which is what makes this design scale
to a much larger corpus without a latency cliff, even though it isn't
measurable at the current demo size. `tests/test_retrieval_quality.py`
guards the accuracy number with an automated floor so future changes
can't silently regress it.

## Running the tests

Everything in the test suite runs fully offline (it uses the extractive
generator, not the API), so it works with no key configured:

```powershell
python -m unittest discover -v
```

## Project layout

```
rag_project/
├── cli.py                     interactive terminal demo
├── app_streamlit.py            browser-based demo (streamlit run app_streamlit.py)
├── requirements.txt
├── .env.example
├── data/corpus/                the demo knowledge base (6 short docs about RAG)
├── index/                      saved TF-IDF index (created by build_index.py)
├── src/
│   ├── chunker.py               stage: chunking
│   ├── bm25.py                   dependency-free BM25 scorer (hybrid retrieval)
│   ├── vector_store.py          stage: retrieval — hybrid TF-IDF + BM25 (index + search)
│   ├── retriever.py             stage: retrieval + ranking
│   ├── generator.py              stage: augmentation + generation
│   ├── pdf_loader.py             extracts + chunks uploaded PDFs (pypdf)
│   ├── image_loader.py           OCR + chunks uploaded images (pytesseract)
│   ├── video_loader.py           audio extraction + transcription + chunks uploaded videos (ffmpeg, faster-whisper)
│   ├── file_router.py            dispatches an upload to the right loader by extension
│   ├── pipeline.py              orchestrates all stages, RAGPipeline class
│   └── build_index.py           one-time script to build the index
└── tests/
    ├── test_pipeline.py         offline unit tests for every stage
    └── test_retrieval_quality.py  labeled-eval-set accuracy regression test
```

## Using your own documents

Drop any `.txt` files into `data/corpus/`, then rebuild the index:

```powershell
python -m src.build_index
```

The pipeline works unchanged — it doesn't know or care that the corpus
used to be about RAG.

## Swapping in real embeddings (optional upgrade)

`TfidfVectorStore` in `src/vector_store.py` is a deliberately simple,
offline-friendly retriever. To upgrade to neural embeddings + FAISS for a
stronger semantic search:

1. `pip install sentence-transformers faiss-cpu`
2. Replace the `TfidfVectorizer` in `vector_store.py` with a call to
   `SentenceTransformer("all-MiniLM-L6-v2").encode(texts)` to produce
   embeddings.
3. Replace `cosine_similarity` search with a `faiss.IndexFlatIP` (or
   `IndexHNSWFlat` for larger corpora) built from those embeddings.

Everything downstream — ranking, augmentation, generation, the CLI — is
unchanged, because they only depend on `ScoredChunk` objects, not on how
the vector store computes similarity.

## Relationship to the seminar deck and web demo

This codebase is the same six-stage architecture presented in slides 5–6
of the deck, and the same pipeline concept as the published interactive
artifact ("Retrieval, then generation"). The difference is surface: the
web artifact is a browser-based, zero-install demo for an audience; this
project is real, inspectable Python you run locally in VS Code, intended
for anyone who wants to read or modify the actual retrieval/ranking/
generation logic.
