cdc# RAG Pipeline — Working Reference Implementation

A small, fully functional Retrieval-Augmented Generation pipeline in
Python, built to accompany the seminar **"Retrieval-Augmented Generation:
Architectures and Failure Modes"** (Piyush Sambyal, 2023A6R055, MIET
Jammu). It implements every stage from the seminar's pipeline diagram —

```
query -> retrieval -> ranking -> augmentation -> generation -> output
```

— as real, runnable code, with no mocked-out stages. **There is no bundled
demo corpus.** `data/corpus/` ships empty on purpose — drop in your own
`.txt` files and the pipeline answers questions grounded in *your* data.

## What's actually implemented

| Stage         | File                     | What it does |
|---------------|--------------------------|---------------|
| Chunking      | `src/chunker.py`         | Splits raw `.txt` documents into overlapping, sentence-aware chunks |
| Retrieval     | `src/vector_store.py`, `src/retriever.py` | TF-IDF vectorization + cosine similarity search over all chunks |
| Ranking       | `src/retriever.py`       | Relevance floor + source-diversity re-ranking (a simplified MMR) |
| Augmentation  | `src/generator.py`       | Builds a numbered, citation-ready context prompt |
| Generation    | `src/generator.py`       | Calls a local, open-source model via Ollama if a server is reachable, else an offline extractive fallback |
| Orchestration | `src/pipeline.py`        | `RAGPipeline` ties all stages together and returns every intermediate result |
| Demo UI       | `cli.py`                 | Interactive terminal REPL, plus a `:compare` mode (RAG vs. no-RAG) |

Retrieval uses TF-IDF rather than a neural embedding model on purpose: it
needs no model download and works fully offline, so the retrieval half of
the pipeline is guaranteed to work in a live demo even without Wi-Fi. See
**"Swapping in real embeddings"** below if you want to upgrade it.

**Generation is open-source and fully local.** It calls a locally-running
[Ollama](https://ollama.com) server if one is reachable, and otherwise
falls back to an offline extractive generator — so the *entire* pipeline,
generation included, runs with no external API, no account, and no key,
ever. Setup:

```powershell
# 1. Install Ollama from https://ollama.com, then:
ollama pull llama3.2
# 2. Leave it running (it usually starts automatically as a background service)
ollama serve
```

Without Ollama running, the pipeline still runs end to end using the
offline extractive fallback — nothing is required to get retrieval +
ranking + augmentation working.

## Fresh by default — no leftover data between runs

Nothing is cached to disk. Every time you start `cli.py` or
`streamlit run app_streamlit.py`, the index is rebuilt from scratch, in
memory, from whatever `.txt` files currently sit in `data/corpus/`. There
is no saved index file to go stale, and no way for a previous session's
data to silently carry over — add or remove files, restart, and the
pipeline reflects exactly what's there now, nothing more. Uploads made
through the Streamlit sidebar are even more short-lived: they exist only
for that browser session and are never written to `data/corpus/` at all.

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

(Optional) copy `.env.example` to `.env` if you want to point at a
non-default Ollama host/model/timeout:

```powershell
Copy-Item .env.example .env
notepad .env
```

## Running it

Add your own `.txt` files to `data/corpus/` (see `data/corpus/README.md`),
then start the interactive demo — no separate build step needed, the
index is built fresh each time it starts:

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

The interface has a dark, animated-gradient theme with no emoji icons —
plain text labels throughout. After you run a query, a **pipeline trace**
renders above the results: six connected stages (Query → Retrieval →
Ranking → Augmentation → Generation → Output) showing that specific
query's own real numbers at each stage (candidate count, top similarity
score, chunks injected, which generator ran), with the actual source
documents the answer is grounded in listed underneath. It's built from
that run's real `PipelineResult`, not a static picture — ask a different
question and the numbers change.

```powershell
streamlit run app_streamlit.py
```

No separate index-build step is needed — the app builds the index fresh
from `data/corpus/` the moment it starts. This opens automatically in your
browser at `http://localhost:8501`. The sidebar shows generator status
(Ollama vs. offline fallback) and lets you tune retrieval settings
(candidate pool size, final chunk count, minimum similarity) live, without
editing code.

`data/corpus/` ships empty, so on first run the app shows a note that
there's nothing indexed yet — drop `.txt` files in and click "Rescan
data/corpus/", or just use the sidebar uploader for that session.

**What was verified vs. not:** the pipeline-trace function
(`render_flow_diagram`) was imported directly from this exact file and
run against a real query, then rendered to an image to confirm the HTML
it generates is correct and matches the real `PipelineResult` — that part
is genuinely checked, not assumed. The surrounding CSS theme (gradient
animation, dark styling of Streamlit's own widgets) could only be
verified as valid CSS, not seen rendered inside a real Streamlit session,
since `streamlit` itself has no network access to install in this build
environment. Confirm the overall look once when you first run it.

### Uploading your own files (PDF, images, video)

The sidebar has an **"📎 Add your own files"** uploader. Any file you drop
in there is routed by type (`src/file_router.py`):

| Type | How it's read | Extra setup needed |
|------|---------------|---------------------|
| PDF (`.pdf`) | Text extraction (`pypdf`) | None |
| Image (`.png`, `.jpg`, `.jpeg`, `.bmp`, `.tiff`, `.webp`) | OCR (`pytesseract`) | Install the Tesseract OCR engine (see below) |
| Video (`.mp4`, `.mov`, `.mkv`, `.avi`, `.webm`, `.m4v`) | Audio extracted with a bundled ffmpeg, then transcribed with `faster-whisper` | `pip install faster-whisper` (not in the base install — see requirements.txt) |

Extracted content is chunked the same way as `data/corpus/` and merged
into an in-memory index for that session. Your questions can then
retrieve from your upload(s), whatever's in `data/corpus/`, or both.
Nothing is ever written to disk from an upload — remove the file from the
uploader, or just close/restart the app, and it's gone, no trace left
behind. If you want something to persist across restarts instead, save
its extracted text as a `.txt` file under `data/corpus/` — that folder
(and only that folder) is what survives between runs.

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

**Honest note on "performance":** at this corpus's size (36 chunks),
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

The test suite runs against a small fixture corpus checked into
`tests/fixtures/corpus/` (not `data/corpus/`, which ships empty by
design) and uses the offline extractive generator, so it needs no Ollama
server running and no network access:

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
├── data/corpus/                ships empty -- drop your own .txt files here (see its README.md)
├── src/
│   ├── chunker.py               stage: chunking
│   ├── bm25.py                   dependency-free BM25 scorer (hybrid retrieval)
│   ├── vector_store.py          stage: retrieval — hybrid TF-IDF + BM25 (index + search)
│   ├── retriever.py             stage: retrieval + ranking
│   ├── generator.py              stage: augmentation + generation (Ollama, or offline fallback)
│   ├── pdf_loader.py             extracts + chunks uploaded PDFs (pypdf)
│   ├── image_loader.py           OCR + chunks uploaded images (pytesseract)
│   ├── video_loader.py           audio extraction + transcription + chunks uploaded videos (ffmpeg, faster-whisper)
│   ├── file_router.py            dispatches an upload to the right loader by extension
│   ├── pipeline.py              orchestrates all stages, RAGPipeline class
│   └── build_index.py           optional corpus sanity-check (no index is ever saved to disk)
└── tests/
    ├── fixtures/corpus/          small fixture corpus used only by the tests below
    ├── test_pipeline.py         offline unit tests for every stage
    └── test_retrieval_quality.py  labeled-eval-set accuracy regression test
```

## Using your own documents

Drop any `.txt` files into `data/corpus/` (see the placeholder
`data/corpus/README.md` there) and just run `cli.py` or the Streamlit app
— no build step needed, the index is built fresh from whatever's in that
folder every time the app starts. The pipeline works unchanged regardless
of what the corpus is about.

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

This codebase is the same six-stage architecture presented in slides 4–5
of the deck ("How It Works — Architecture & Components" and "The RAG
Pipeline -- End-to-End Flow"), and the same pipeline concept as the
published interactive artifact ("Retrieval, then generation"). The
difference is surface: the web artifact is a browser-based, zero-install
demo for an audience; this project is real, inspectable Python you run
locally in VS Code, intended for anyone who wants to read or modify the
actual retrieval/ranking/generation logic — pointed at whatever documents
you drop into `data/corpus/`, not a fixed demo topic.

cd path\to\rag_project
venv\Scripts\Activate.ps1
streamlit run app_streamlit.py