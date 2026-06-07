"""Tests for the track embedding pipeline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harmony.config import Config
from harmony.embedding.pipeline import TrackEmbeddingPipeline
from harmony.models import Track, TrackStatus, track_id_from_content_hash, utcnow

TRACK_ID = track_id_from_content_hash("test-track")
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore
from tests.fake_embedder import FakeEmbedder


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
        track_id=TRACK_ID,
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
    assert vectors.load_track_vector(TRACK_ID) is not None
    store.close()


def test_embed_track_resamples_once_before_chunking(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("soundfile")
    import soundfile as sf

    resample_calls: list[tuple[int, int, int]] = []

    def _track_resample(waveform: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
        resample_calls.append((len(waveform), source_sr, target_sr))
        ratio = target_sr / source_sr
        out_len = int(len(waveform) * ratio)
        return np.linspace(0.0, 1.0, out_len, dtype=np.float32)

    monkeypatch.setattr("harmony.embedding.pipeline.resample", _track_resample)

    sr = 44100
    t = np.linspace(0, 12, sr * 12, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    path = tmp_path / "tone-44k.wav"
    sf.write(path, audio, sr)

    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    vectors = VectorStore(cfg)
    pipeline = TrackEmbeddingPipeline(cfg, store, vectors, FakeEmbedder())

    track = Track(
        track_id=TRACK_ID,
        content_hash="abc",
        status=TrackStatus.ACTIVE,
        primary_path=str(path),
        duration_ms=0,
        title="Tone",
        artist="Test",
        album="Test",
        embedding_version=cfg.embedding_version(),
    )

    pipeline.embed_track(track)
    assert len(resample_calls) == 1
    assert resample_calls[0] == (sr * 12, 44100, 24000)
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
            TRACK_ID,
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
    embedded, failed, _ids = pipeline.embed_pending(reembed=True)
    assert embedded == 1
    assert failed == 0
    assert store.count_embedded_tracks() == 1
    store.close()
