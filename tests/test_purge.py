"""Tests for library purge."""

from pathlib import Path

import numpy as np

from harmony.config import Config
from harmony.engine import Engine
from harmony.models import TrackStatus, utcnow
from harmony.scanner.filesystem import FilesystemScanner
from harmony.storage.metadata import MetadataStore
from harmony.storage.purge import LibraryPurge
from harmony.storage.sync import LibrarySync
from harmony.storage.vectors import VectorStore


def _insert_track(store: MetadataStore, track_id: str, path: str, status: str) -> None:
    version = store.config.embedding_version()
    now = utcnow().isoformat()
    store.conn.execute(
        """
        INSERT INTO tracks (
            track_id, content_hash, status, primary_path,
            duration_ms, title, artist, album, embedding_version,
            indexed_at, last_seen_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            track_id,
            f"hash-{track_id}",
            status,
            path,
            0,
            track_id,
            "Artist",
            "Album",
            version,
            now,
            now,
            now,
            now,
        ),
    )
    store.conn.commit()


def test_prune_missing_deletes_gone_tracks(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    vectors = VectorStore(cfg)

    _insert_track(store, "keep", "/music/keep.flac", "active")
    _insert_track(store, "gone", "/music/gone.flac", "missing")
    vectors.save_track_vector("gone", np.array([1.0, 0.0], dtype=np.float32))

    purge = LibraryPurge(cfg, store, vectors)
    purged = purge.prune_missing()

    assert purged == ["gone"]
    assert store.count_tracks_by_status().get("active", 0) == 1
    assert vectors.load_track_vector("gone") is None
    store.close()


def test_index_prune_after_library_shrink(tmp_path: Path) -> None:
    music = tmp_path / "music"
    music.mkdir()
    keep = music / "keep.flac"
    keep.write_bytes(b"audio-keep")

    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    sync = LibrarySync(cfg, store)

    sync.reconcile(FilesystemScanner([music]))
    _insert_track(store, "gone", "/music/gone.flac", "active")
    store.set_tracks_status(["gone"], TrackStatus.MISSING)

    engine = Engine(cfg.data_dir)
    report = engine.index(paths=[str(music)], embed=False, prune=True)
    engine.close()

    assert report.purged == 1
    assert store.count_tracks_by_status().get("active", 0) == 1
    store.close()
