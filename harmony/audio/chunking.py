"""Split waveforms into overlapping chunks."""

from __future__ import annotations

import numpy as np


def chunk_audio(
    waveform: np.ndarray,
    sample_rate: int,
    *,
    chunk_seconds: int = 10,
    overlap_seconds: int = 2,
    min_chunk_seconds: int = 1,
) -> list[tuple[np.ndarray, int, int]]:
    """Split audio into chunks.

    Returns list of (chunk_waveform, start_ms, end_ms).
    """
    chunk_samples = chunk_seconds * sample_rate
    step_samples = (chunk_seconds - overlap_seconds) * sample_rate
    min_samples = min_chunk_seconds * sample_rate

    chunks: list[tuple[np.ndarray, int, int]] = []
    start = 0

    while start < len(waveform):
        end = start + chunk_samples
        chunk = waveform[start:end]
        if len(chunk) < min_samples:
            break
        start_ms = int(start / sample_rate * 1000)
        end_ms = int(min(end, len(waveform)) / sample_rate * 1000)
        chunks.append((chunk, start_ms, end_ms))
        start += step_samples

    return chunks
