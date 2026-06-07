"""Benchmark embedding throughput for a single audio file."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from harmony.audio.chunking import chunk_audio
from harmony.audio.loader import load_audio
from harmony.audio.resample import resample
from harmony.config import Config
from harmony.embedding.base import Embedder
from harmony.embedding.factory import create_embedder
from harmony.embedding.pooling import mean_pool


@dataclass(frozen=True)
class EncodeBenchmarkResult:
    path: str
    duration_s: float
    source_sample_rate: int
    target_sample_rate: int
    chunks: int
    batches: int
    batch_size: int
    load_ms: float
    resample_ms: float
    chunk_ms: float
    model_load_ms: float
    embed_ms: float
    total_ms: float
    device: str
    model: str
    vector_dim: int


def benchmark_encode(
    path: Path | str,
    config: Config,
    *,
    embedder: Embedder | None = None,
) -> EncodeBenchmarkResult:
    """Time load → resample → chunk → embed for one audio file."""
    path = Path(path)
    owns_embedder = embedder is None
    if embedder is None:
        embedder = create_embedder(config)

    total_start = time.perf_counter()

    load_start = time.perf_counter()
    waveform, file_sr = load_audio(
        path,
        mono=config.audio.mono,
        config=config.audio,
    )
    load_ms = (time.perf_counter() - load_start) * 1000
    duration_s = len(waveform) / file_sr

    target_sr = config.audio.target_sample_rate
    resample_ms = 0.0
    sample_rate = file_sr
    if file_sr != target_sr:
        resample_start = time.perf_counter()
        waveform = resample(waveform, file_sr, target_sr)
        resample_ms = (time.perf_counter() - resample_start) * 1000
        sample_rate = target_sr

    chunk_start = time.perf_counter()
    chunks = chunk_audio(
        waveform,
        sample_rate,
        chunk_seconds=config.audio.chunk_seconds,
        overlap_seconds=config.audio.overlap_seconds,
        min_chunk_seconds=config.audio.min_chunk_seconds,
    )
    chunk_ms = (time.perf_counter() - chunk_start) * 1000
    if not chunks:
        raise ValueError(f"No embeddable audio in {path}")

    chunk_waveforms = [c[0] for c in chunks]
    batch_size = max(1, config.embedding.batch_size)
    batches = (len(chunk_waveforms) + batch_size - 1) // batch_size

    model_load_ms = _time_model_load(embedder)

    session_factory = getattr(embedder, "session", None)
    embed_start = time.perf_counter()
    if session_factory:
        with session_factory():
            chunk_embeddings = _embed_chunks_batched(
                embedder,
                chunk_waveforms,
                sample_rate,
                batch_size=batch_size,
            )
    else:
        chunk_embeddings = _embed_chunks_batched(
            embedder,
            chunk_waveforms,
            sample_rate,
            batch_size=batch_size,
        )
    embed_ms = (time.perf_counter() - embed_start) * 1000

    track_vector = mean_pool(chunk_embeddings)
    total_ms = (time.perf_counter() - total_start) * 1000

    if owns_embedder and hasattr(embedder, "unload"):
        embedder.unload()

    return EncodeBenchmarkResult(
        path=str(path),
        duration_s=duration_s,
        source_sample_rate=file_sr,
        target_sample_rate=target_sr,
        chunks=len(chunk_waveforms),
        batches=batches,
        batch_size=batch_size,
        load_ms=load_ms,
        resample_ms=resample_ms,
        chunk_ms=chunk_ms,
        model_load_ms=model_load_ms,
        embed_ms=embed_ms,
        total_ms=total_ms,
        device=embedder.device,
        model=embedder.name,
        vector_dim=int(track_vector.shape[0]),
    )


def _time_model_load(embedder: Embedder) -> float:
    """Load model weights before timing inference, if the embedder supports it."""
    preload = getattr(embedder, "preload", None)
    is_loaded = getattr(embedder, "is_loaded", None)
    if preload is None or is_loaded is None or is_loaded:
        return 0.0

    start = time.perf_counter()
    preload()
    return (time.perf_counter() - start) * 1000


def _embed_chunks_batched(
    embedder: Embedder,
    chunk_waveforms: list[np.ndarray],
    sample_rate: int,
    *,
    batch_size: int,
) -> np.ndarray:
    parts: list[np.ndarray] = []
    for start in range(0, len(chunk_waveforms), batch_size):
        batch = chunk_waveforms[start : start + batch_size]
        if hasattr(embedder, "embed_audio_batch"):
            part = embedder.embed_audio_batch(batch, sample_rate=sample_rate)  # type: ignore[attr-defined]
        else:
            part = np.stack(
                [embedder.embed_audio(w, sample_rate) for w in batch],
            )
        parts.append(part)
    return np.vstack(parts)
