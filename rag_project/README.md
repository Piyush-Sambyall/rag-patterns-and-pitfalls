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

## Running the tests

Everything in the test suite runs fully offline (it uses the extractive
generator, not the API), so it works with no key configured:

```powershell
python -m unittest discover -v
```

## Project layout

```
rag_project/
├── cli.py                     interactive demo entry point
├── requirements.txt
├── .env.example
├── data/corpus/                the demo knowledge base (6 short docs about RAG)
├── index/                      saved TF-IDF index (created by build_index.py)
├── src/
│   ├── chunker.py               stage: chunking
│   ├── vector_store.py          stage: retrieval (index + search)
│   ├── retriever.py             stage: retrieval + ranking
│   ├── generator.py              stage: augmentation + generation
│   ├── pipeline.py              orchestrates all stages, RAGPipeline class
│   └── build_index.py           one-time script to build the index
└── tests/
    └── test_pipeline.py         offline unit tests for every stage
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
