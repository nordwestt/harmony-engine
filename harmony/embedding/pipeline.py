"""Embed tracks: load audio, chunk, run model, persist vectors."""

from __future__ import annotations

import logging
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np

from harmony.audio.chunking import chunk_audio
from harmony.audio.loader import load_audio
from harmony.config import Config
from harmony.embedding.base import Embedder
from harmony.embedding.pooling import mean_pool
from harmony.models import Track, utcnow
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore

logger = logging.getLogger(__name__)


class TrackEmbeddingPipeline:
    def __init__(
        self,
        config: Config,
        store: MetadataStore,
        vectors: VectorStore,
        embedder: Embedder,
    ) -> None:
        self.config = config
        self.store = store
        self.vectors = vectors
        self.embedder = embedder

    def embed_track(self, track: Track) -> np.ndarray:
        """Load, chunk, embed, and persist a single track vector."""
        waveform, sample_rate = load_audio(
            track.primary_path,
            mono=self.config.audio.mono,
        )
        duration_ms = int(len(waveform) / sample_rate * 1000)

        chunks = chunk_audio(
            waveform,
            sample_rate,
            chunk_seconds=self.config.audio.chunk_seconds,
            overlap_seconds=self.config.audio.overlap_seconds,
            min_chunk_seconds=self.config.audio.min_chunk_seconds,
        )

        if not chunks:
            raise ValueError(f"No embeddable audio in {track.primary_path}")

        chunk_waveforms = [c[0] for c in chunks]
        chunk_embeddings = self._embed_chunks_batched(chunk_waveforms, sample_rate)
        track_vector = mean_pool(chunk_embeddings)

        version = self.config.embedding_version()
        self.vectors.save_track_vector(track.track_id, track_vector, version)
        self.store.mark_track_embedded(track.track_id, duration_ms=duration_ms, version=version)
        return track_vector

    def embed_pending(
        self,
        *,
        on_progress: Callable[[int, int, str], None] | None = None,
    ) -> tuple[int, int]:
        """Embed all active tracks missing vectors at the current version.

        Returns (embedded_count, failed_count).
        """
        pending = self.store.list_tracks_pending_embedding()
        if not pending:
            return 0, 0

        total = len(pending)
        print(f"Embedding {total} track(s)…", file=sys.stderr)

        embedded = 0
        failed = 0
        iterator: object = pending

        try:
            from tqdm import tqdm

            iterator = tqdm(pending, unit="track", file=sys.stderr)
        except ImportError:
            pass

        for i, track in enumerate(iterator, start=1):
            label = f"{track.artist} — {track.title}"
            try:
                self.embed_track(track)
                embedded += 1
                logger.info("Embedded %s — %s", track.artist, track.title)
                if on_progress:
                    on_progress(i, total, label)
            except Exception:
                failed += 1
                logger.exception("Failed to embed %s", track.primary_path)
                self.store.mark_track_failed(track.track_id, utcnow())
                print(f"Failed: {label} ({track.primary_path})", file=sys.stderr)

        return embedded, failed

    def _embed_chunks_batched(
        self,
        chunk_waveforms: list[np.ndarray],
        sample_rate: int,
    ) -> np.ndarray:
        batch_size = max(1, self.config.embedding.batch_size)
        parts: list[np.ndarray] = []

        for start in range(0, len(chunk_waveforms), batch_size):
            batch = chunk_waveforms[start : start + batch_size]
            if hasattr(self.embedder, "embed_audio_batch"):
                part = self.embedder.embed_audio_batch(batch, sample_rate=sample_rate)  # type: ignore[attr-defined]
            else:
                part = np.stack(
                    [self.embedder.embed_audio(w, sample_rate) for w in batch],
                )
            parts.append(part)

        return np.vstack(parts)
