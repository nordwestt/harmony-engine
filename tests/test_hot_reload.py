"""Tests for live index updates without restarting the engine."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from harmony.config import Config
from harmony.engine import Engine
from harmony.models import utcnow
from harmony.retrieval.search import SearchEngine
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore


class FakeEmbedder:
    name = "fake"
    dimension = 3

    def embed_text(self, text: str) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def embed_text_batch(self, texts: list[str]) -> np.ndarray:
        return np.tile(np.array([1.0, 0.0, 0.0], dtype=np.float32), (len(texts), 1))


def _insert_track(
    store: MetadataStore,
    *,
    track_id: str,
    version: str,
    vector: np.ndarray,
    vectors: VectorStore,
) -> None:
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
            "active",
            f"/music/{track_id}.flac",
            1000,
            track_id.upper(),
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
    vectors.save_track_vector(track_id, vector, version)


def test_search_sees_incremental_index_update(tmp_path: Path) -> None:
    """SearchEngine resolves the live index after upsert without recreating Engine."""
    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    vectors = VectorStore(cfg)
    version = cfg.embedding_version()

    _insert_track(
        store,
        track_id="a",
        version=version,
        vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        vectors=vectors,
    )
    _insert_track(
        store,
        track_id="b",
        version=version,
        vector=np.array([0.0, 1.0, 0.0], dtype=np.float32),
        vectors=vectors,
    )

    engine = Engine(cfg.data_dir)
    engine._embedder = FakeEmbedder()  # type: ignore[assignment]
    manager = engine._get_index_manager()
    manager.rebuild()

    search = engine._get_search()
    result = search.search_by_track("a", k=10)
    assert len(result.items) == 1
    assert result.items[0].track_id == "b"

    _insert_track(
        store,
        track_id="c",
        version=version,
        vector=np.array([0.9, 0.1, 0.0], dtype=np.float32),
        vectors=vectors,
    )
    manager.upsert_track("c", vectors.load_track_vector("c", version))

    result2 = search.search_by_track("a", k=10)
    track_ids = {item.track_id for item in result2.items}
    assert track_ids == {"b", "c"}
    store.close()
    engine.close()


def test_invalidate_search_clears_cached_engine(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    vectors = VectorStore(cfg)
    version = cfg.embedding_version()

    _insert_track(
        store,
        track_id="a",
        version=version,
        vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        vectors=vectors,
    )

    engine = Engine(cfg.data_dir)
    engine._embedder = FakeEmbedder()  # type: ignore[assignment]
    engine._get_index_manager().rebuild()
    first = engine._get_search()
    engine._invalidate_search()
    second = engine._get_search()
    assert first is not second
    assert isinstance(second, SearchEngine)
    store.close()
    engine.close()
