from __future__ import annotations

import io

from PIL import Image
import pytesseract

from .chunker import Chunk, chunk_text


def extract_image_text(file_bytes: bytes) -> str:
    """
    Runs OCR on an image and returns whatever text was found. Returns an
    empty string for images with no text (e.g. a plain photo) -- callers
    should treat that as "nothing to index", not an error.
    """
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image)


def chunk_image(file_bytes: bytes, filename: str, max_words: int = 90) -> list[Chunk]:
    """Extracts OCR text from an uploaded image and chunks it, same shape as the PDF loader."""
    text = extract_image_text(file_bytes)
    if not text.strip():
        return []
    return chunk_text(text, source=filename, max_words=max_words)

"""
image_loader.py

Extracts text from an uploaded image via OCR (Tesseract, through the
`pytesseract` wrapper), so a screenshot, a photo of a whiteboard, or a
scanned page can be chunked and retrieved just like a PDF or corpus doc.

Requires the Tesseract OCR engine to be installed on the system --
`pytesseract` is only a thin wrapper around it, not an OCR engine itself.
See the README for the Windows install step; this is the one piece of
this project that needs something outside `pip install`.
"""
