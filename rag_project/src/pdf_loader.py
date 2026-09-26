"""
pdf_loader.py
-------------
Extracts text from user-uploaded PDFs so it can be chunked and added to
the retrieval index alongside (or instead of) the built-in corpus.

Uses `pypdf`, a pure-Python library with no external binary dependency
(no poppler/ghostscript install needed) -- important for a one-command
`pip install` setup on Windows. The trade-off: it extracts text that's
actually embedded in the PDF; it does not OCR scanned/image-only pages.
If a PDF returns no text, that's almost always why.
"""

from __future__ import annotations

import io

from pypdf import PdfReader

from .chunker import Chunk, chunk_text


def extract_pdf_text(file_bytes: bytes) -> str:
    """
    Extracts all text from a PDF's pages, in order, joined with blank
    lines between pages. Returns an empty string if the PDF has no
    extractable text layer (e.g. a scan with no OCR) -- callers should
    check for that and tell the user, rather than silently indexing
    nothing.
    """
    reader = PdfReader(io.BytesIO(file_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(t for t in pages_text if t.strip())


def chunk_pdf(file_bytes: bytes, filename: str, max_words: int = 90) -> list[Chunk]:
    """Extracts text from an uploaded PDF and chunks it, same shape as the corpus loader."""
    text = extract_pdf_text(file_bytes)
    if not text.strip():
        return []
    return chunk_text(text, source=filename, max_words=max_words)
