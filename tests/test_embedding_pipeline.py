"""Tests for the track embedding pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harmony.config import Config
from harmony.embedding.pipeline import TrackEmbeddingPipeline
from harmony.models import Track, TrackStatus, utcnow
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore


class FakeEmbedder:
    name = "fake"
    dimension = 4

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def embed_audio_batch(
        self,
        waveforms: list[np.ndarray],
        *,
        sample_rate: int,
    ) -> np.ndarray:
        return np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (len(waveforms), 1))

    def embed_text(self, text: str) -> np.ndarray:
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))


@pytest.fixture
def wav_file(tmp_path: Path) -> Path:
    pytest.importorskip("soundfile")
    import soundfile as sf

    sr = 24000
    t = np.linspace(0, 12, sr * 12, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone.wav"
    sf.write(path, audio, sr)
    return path


def test_embed_track_persists_vector(tmp_path: Path, wav_file: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    vectors = VectorStore(cfg)
    pipeline = TrackEmbeddingPipeline(cfg, store, vectors, FakeEmbedder())

    track = Track(
        track_id="track-1",
        content_hash="abc",
        status=TrackStatus.ACTIVE,
        primary_path=str(wav_file),
        duration_ms=0,
        title="Tone",
        artist="Test",
        album="Test",
        embedding_version=cfg.embedding_version(),
    )

    vector = pipeline.embed_track(track)
    assert vector.shape == (4,)
    assert vectors.load_track_vector("track-1") is not None
    store.close()


def test_embed_pending_processes_unindexed_track(tmp_path: Path, wav_file: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    store.conn.execute(
        """
        INSERT INTO tracks (
            track_id, content_hash, status, primary_path,
            duration_ms, title, artist, album, embedding_version,
            indexed_at, last_seen_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
        """,
        (
            "track-1",
            "hash",
            "active",
            str(wav_file),
            0,
            "Tone",
            "Test",
            "Album",
            cfg.embedding_version(),
            utcnow().isoformat(),
            utcnow().isoformat(),
            utcnow().isoformat(),
        ),
    )
    store.conn.commit()

    pipeline = TrackEmbeddingPipeline(cfg, store, VectorStore(cfg), FakeEmbedder())
    embedded, failed = pipeline.embed_pending(reembed=True)
    assert embedded == 1
    assert failed == 0
    assert store.count_embedded_tracks() == 1
    store.close()
