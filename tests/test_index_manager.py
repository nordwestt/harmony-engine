"""Tests for track index manager."""

from pathlib import Path

import numpy as np

from harmony.config import Config
from harmony.index.manager import TrackIndexManager
from harmony.models import TrackStatus, track_id_from_content_hash, utcnow

TRACK_A = track_id_from_content_hash("h1")
TRACK_B = track_id_from_content_hash("h2")
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore


def test_rebuild_loads_vectors_into_index(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    vectors = VectorStore(cfg)
    version = cfg.embedding_version()

    store.conn.execute(
        """
        INSERT INTO tracks (
            track_id, content_hash, status, primary_path,
            duration_ms, title, artist, album, embedding_version,
            indexed_at, last_seen_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            TRACK_A,
            "h1",
            "active",
            "/music/a.flac",
            1000,
            "A",
            "Artist",
            "Album",
            version,
            utcnow().isoformat(),
            utcnow().isoformat(),
            utcnow().isoformat(),
            utcnow().isoformat(),
        ),
    )
    store.conn.commit()

    vectors.save_track_vector(TRACK_A, np.array([1.0, 0.0, 0.0], dtype=np.float32), version)
    vectors.save_track_vector(TRACK_B, np.array([0.0, 1.0, 0.0], dtype=np.float32), version)

    manager = TrackIndexManager(cfg, store, vectors)
    count = manager.rebuild()
    assert count == 1

    index = manager.ensure_loaded()
    ids, scores = index.search(np.array([1.0, 0.0, 0.0]), k=1)
    assert ids == [TRACK_A]
    assert scores[0] > 0.9
    store.close()
