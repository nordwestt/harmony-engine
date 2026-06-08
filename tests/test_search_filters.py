"""Integration tests for artist/album filters in search."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from harmony.config import Config
from harmony.engine import Engine
from harmony.models import track_id_from_content_hash, utcnow
from harmony.retrieval.filters import Filters
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore
from tests.fake_embedder import FakeEmbedder

TRACK_A = track_id_from_content_hash("hash-a")
TRACK_B = track_id_from_content_hash("hash-b")
TRACK_C = track_id_from_content_hash("hash-c")


def _insert_track(
    store: MetadataStore,
    *,
    track_id: str,
    version: str,
    vector: np.ndarray,
    vectors: VectorStore,
    artist: str = "Artist",
    album: str = "Album",
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
            artist,
            album,
            version,
            now,
            now,
            now,
            now,
        ),
    )
    store.conn.commit()
    vectors.save_track_vector(track_id, vector, version)


def _setup_engine(tmp_path: Path) -> tuple[Engine, MetadataStore]:
    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    vectors = VectorStore(cfg)
    version = cfg.embedding_version()

    _insert_track(
        store,
        track_id=TRACK_A,
        version=version,
        vector=np.array([1.0, 0.0, 0.0], dtype=np.float32),
        vectors=vectors,
        artist="Radiohead",
        album="OK Computer",
    )
    _insert_track(
        store,
        track_id=TRACK_B,
        version=version,
        vector=np.array([0.9, 0.1, 0.0], dtype=np.float32),
        vectors=vectors,
        artist="Michael Jackson",
        album="Thriller",
    )
    _insert_track(
        store,
        track_id=TRACK_C,
        version=version,
        vector=np.array([0.8, 0.2, 0.0], dtype=np.float32),
        vectors=vectors,
        artist="Beck",
        album="Odelay",
    )

    engine = Engine(cfg.data_dir)
    engine._embedder = FakeEmbedder(dimension=3)
    engine._get_index_manager().rebuild()
    return engine, store


def test_text_search_filters_by_artist(tmp_path: Path) -> None:
    engine, store = _setup_engine(tmp_path)
    try:
        result = engine.search_by_text(
            "test query",
            k=10,
            filters=Filters(artists=["Radiohead", "Michael Jackson"]),
        )
        artists = {item.metadata.artist for item in result.items}
        assert artists == {"Radiohead", "Michael Jackson"}
    finally:
        store.close()
        engine.close()


def test_track_search_filters_by_album(tmp_path: Path) -> None:
    engine, store = _setup_engine(tmp_path)
    try:
        result = engine.search_by_track(
            TRACK_A,
            k=10,
            filters=Filters(albums=["Thriller"]),
        )
        assert len(result.items) == 1
        assert result.items[0].track_id == TRACK_B
        assert result.items[0].metadata.album == "Thriller"
    finally:
        store.close()
        engine.close()


def test_combined_artist_and_album_filter(tmp_path: Path) -> None:
    engine, store = _setup_engine(tmp_path)
    try:
        result = engine.search_by_text(
            "test query",
            k=10,
            filters=Filters(artists=["Radiohead"], albums=["OK Computer"]),
        )
        assert len(result.items) == 1
        assert result.items[0].track_id == TRACK_A
    finally:
        store.close()
        engine.close()
