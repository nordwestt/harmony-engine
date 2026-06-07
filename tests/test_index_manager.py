"""Tests for track index manager."""

from pathlib import Path

import numpy as np

from harmony.config import Config
from harmony.index.manager import TrackIndexManager
from harmony.models import TrackStatus, utcnow
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
            "a",
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

    vectors.save_track_vector("a", np.array([1.0, 0.0, 0.0], dtype=np.float32), version)
    vectors.save_track_vector("b", np.array([0.0, 1.0, 0.0], dtype=np.float32), version)

    manager = TrackIndexManager(cfg, store, vectors)
    count = manager.rebuild()
    assert count == 1

    index = manager.ensure_loaded()
    ids, scores = index.search(np.array([1.0, 0.0, 0.0]), k=1)
    assert ids == ["a"]
    assert scores[0] > 0.9
    store.close()
