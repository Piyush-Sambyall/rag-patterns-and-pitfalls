"""
video_loader.py
----------------
Extracts a spoken-word transcript from an uploaded video, so a recorded
lecture, a demo walkthrough, or any talking-head clip can be chunked and
retrieved just like a PDF, image, or corpus doc.

Two steps, kept separate so each is independently testable:

  1. extract_audio_wav()  -- pulls the audio track out of the video and
     saves it as a 16kHz mono WAV. Uses `imageio-ffmpeg`, which bundles a
     static ffmpeg binary through pip -- no separate system install of
     ffmpeg is required, unlike most ffmpeg-based tools.

  2. transcribe_audio()   -- runs that WAV through `faster-whisper`
     (the "tiny" model, CPU, int8) to produce a text transcript.
     faster-whisper is NOT bundled with the base install: it's a fairly
     large dependency, and it downloads its model weights (~75MB for
     "tiny") the first time it runs, which needs an internet connection
     once. See the README for exactly what to expect on first run.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

import imageio_ffmpeg

from .chunker import Chunk, chunk_text


def extract_audio_wav(video_bytes: bytes, suffix: str = ".mp4") -> Path:
    """
    Writes `video_bytes` to a temp file, runs it through ffmpeg, and
    returns the path to a temp 16kHz mono WAV file (the format
    faster-whisper expects). Caller is responsible for deleting both
    temp files when done.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as video_file:
        video_file.write(video_bytes)
        video_path = Path(video_file.name)

    audio_path = video_path.with_suffix(".wav")

    result = subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-i", str(video_path),
            "-ar", "16000",
            "-ac", "1",
            "-vn",
            str(audio_path),
        ],
        capture_output=True,
        text=True,
    )

    video_path.unlink(missing_ok=True)

    if result.returncode != 0 or not audio_path.exists():
        raise RuntimeError(
            f"ffmpeg failed to extract audio (return code {result.returncode}). "
            f"stderr: {result.stderr[-500:]}"
        )

    return audio_path


def transcribe_audio(wav_path: Path, model_size: str = "tiny") -> str:
    """
    Transcribes a WAV file with faster-whisper. Imports faster_whisper
    lazily so the rest of the project works without it installed --
    only PDF/text/image ingestion is affected if it's missing.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ImportError(
            "Video transcription requires the 'faster-whisper' package. "
            "Install it with: pip install faster-whisper"
        ) from exc

    model = WhisperModel(model_size, device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(wav_path))
    return " ".join(segment.text.strip() for segment in segments)


def chunk_video(file_bytes: bytes, filename: str, max_words: int = 90) -> list[Chunk]:
    """Full pipeline: extract audio -> transcribe -> chunk, tagged with the video's filename."""
    suffix = Path(filename).suffix or ".mp4"
    audio_path = extract_audio_wav(file_bytes, suffix=suffix)
    try:
        transcript = transcribe_audio(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)

    if not transcript.strip():
        return []
    return chunk_text(transcript, source=filename, max_words=max_words)
