from __future__ import annotations

from pathlib import Path

from .chunker import Chunk
from .image_loader import chunk_image
from .pdf_loader import chunk_pdf
from .video_loader import chunk_video

PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}

ALL_SUPPORTED_EXTENSIONS = PDF_EXTENSIONS | IMAGE_EXTENSIONS | VIDEO_EXTENSIONS


def chunk_uploaded_file(file_bytes: bytes, filename: str, max_words: int = 90) -> list[Chunk]:
    """
    Extracts and chunks an uploaded file's content based on its extension.
    Raises ValueError for an unsupported extension, so the caller can show
    a clear message instead of guessing what went wrong.
    """
    suffix = Path(filename).suffix.lower()

    if suffix in PDF_EXTENSIONS:
        return chunk_pdf(file_bytes, filename, max_words=max_words)
    if suffix in IMAGE_EXTENSIONS:
        return chunk_image(file_bytes, filename, max_words=max_words)
    if suffix in VIDEO_EXTENSIONS:
        return chunk_video(file_bytes, filename, max_words=max_words)

    raise ValueError(
        f"Unsupported file type '{suffix}'. Supported: "
        f"{', '.join(sorted(ALL_SUPPORTED_EXTENSIONS))}"
    )

"""
file_router.py
---------------
Single entry point the UI calls for any upload, regardless of type.
Dispatches to the right loader (pdf_loader / image_loader / video_loader)
by file extension, so app_streamlit.py doesn't need to know the details
of any one format.
"""
