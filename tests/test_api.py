"""Tests for the HTTP API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harmony.api.app import create_app


def test_health_and_init(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    data_dir = tmp_path / "data"
    app = create_app(data_dir, preload_on_serve=False)

    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok"}
        v1_health = client.get("/v1/health").json()
        assert v1_health["status"] == "ok"
        assert v1_health["message"] == "Ready"
        assert (data_dir / "config.yaml").exists()

        init_resp = client.post("/v1/init")
        assert init_resp.status_code == 200
        assert init_resp.json()["data_dir"] == str(data_dir)


def test_v1_health_reports_model_loading(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from harmony.engine import Engine
    from tests.fake_embedder import FakeEmbedder

    engine = Engine(tmp_path / "data")
    embedder = FakeEmbedder()
    embedder.is_loading = True
    engine._embedder = embedder

    app = create_app(tmp_path / "data", preload_on_serve=False, engine=engine)
    client = TestClient(app)

    body = client.get("/v1/health").json()
    assert body["status"] == "starting"
    assert "Hugging Face" in body["message"]

    ready = client.get("/v1/ready").json()
    assert ready["model_loading"] is True
    assert "Hugging Face" in ready["message"]


def test_ready_without_index(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    app = create_app(tmp_path / "data", preload_on_serve=False)
    client = TestClient(app)
    ready = client.get("/v1/ready").json()
    assert ready["ready"] is False
    assert ready["index_size"] == 0


def test_library_tracks_empty(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    app = create_app(tmp_path / "data", preload_on_serve=False)
    client = TestClient(app)
    resp = client.get("/v1/library/tracks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_index_uses_env_paths_when_omitted(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    music_dir = tmp_path / "music"
    monkeypatch.setenv("HARMONY_INDEX_PATHS", str(music_dir))

    captured: list[list[str]] = []

    class FakeScanner:
        def __init__(self, paths, config=None) -> None:
            captured.append([str(p) for p in paths])

        def scan(self):
            return iter([])

    monkeypatch.setattr("harmony.engine.FilesystemScanner", FakeScanner)

    app = create_app(tmp_path / "data", preload_on_serve=False)
    client = TestClient(app)

    resp = client.post("/v1/index", json={"embed": False})
    assert resp.status_code == 200
    assert captured == [[str(music_dir)]]


def test_purge_requires_flags(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    app = create_app(tmp_path / "data", preload_on_serve=False)
    client = TestClient(app)
    resp = client.post("/v1/library/purge", json={})
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


def test_search_text_rejects_empty_query(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    client = TestClient(create_app(tmp_path / "data", preload_on_serve=False))
    resp = client.post("/v1/search/text", json={"query": "   "})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


def test_search_text_rejects_oversized_query(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    client = TestClient(create_app(tmp_path / "data", preload_on_serve=False))
    resp = client.post("/v1/search/text", json={"query": "x" * 513})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


def test_search_track_rejects_invalid_track_id(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    client = TestClient(create_app(tmp_path / "data", preload_on_serve=False))
    resp = client.post("/v1/search/track", json={"track_id": "../../secrets"})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


def test_search_text_empty_index_returns_503(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from harmony.engine import Engine
    from tests.fake_embedder import FakeEmbedder

    engine = Engine(tmp_path / "data")
    embedder = FakeEmbedder()
    embedder.is_loaded = True
    engine._embedder = embedder

    client = TestClient(create_app(tmp_path / "data", preload_on_serve=False, engine=engine))
    resp = client.post("/v1/search/text", json={"query": "jazz"})
    assert resp.status_code == 503
    body = resp.json()
    assert body["code"] == "index_not_ready"


def test_search_text_model_load_error_returns_503(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from harmony.engine import Engine
    from tests.fake_embedder import FakeEmbedder

    engine = Engine(tmp_path / "data")
    embedder = FakeEmbedder()
    embedder.load_error = "CUDA OOM"
    engine._embedder = embedder

    client = TestClient(create_app(tmp_path / "data", preload_on_serve=False, engine=engine))
    resp = client.post("/v1/search/text", json={"query": "jazz"})
    assert resp.status_code == 503
    assert resp.json()["code"] == "model_not_ready"


def test_index_rejects_path_outside_roots(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    music_dir = tmp_path / "music"
    other_dir = tmp_path / "other"
    music_dir.mkdir()
    other_dir.mkdir()
    monkeypatch.setenv("HARMONY_INDEX_PATHS", str(music_dir))

    client = TestClient(create_app(tmp_path / "data", preload_on_serve=False))
    resp = client.post("/v1/index", json={"paths": [str(other_dir)], "embed": False})
    assert resp.status_code == 400
    assert resp.json()["code"] == "path_not_allowed"


def test_v1_health_hides_load_error_details(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from harmony.engine import Engine
    from tests.fake_embedder import FakeEmbedder

    engine = Engine(tmp_path / "data")
    embedder = FakeEmbedder()
    embedder.load_error = "secret/path/to/weights failed"
    engine._embedder = embedder

    client = TestClient(create_app(tmp_path / "data", preload_on_serve=False, engine=engine))
    body = client.get("/v1/health").json()
    assert body["status"] == "error"
    assert "secret" not in body["message"]
    assert "server logs" in body["message"]


def test_list_tracks_rejects_invalid_status(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    client = TestClient(create_app(tmp_path / "data", preload_on_serve=False))
    resp = client.get("/v1/library/tracks", params={"status": "bogus"})
    assert resp.status_code == 422


def test_unhandled_exception_returns_internal_error(tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("fastapi")
    from harmony.engine import Engine

    engine = Engine(tmp_path / "data")

    def boom(*_args, **_kwargs):
        raise RuntimeError("sensitive traceback detail")

    monkeypatch.setattr(engine, "stats", boom)
    client = TestClient(
        create_app(tmp_path / "data", preload_on_serve=False, engine=engine),
        raise_server_exceptions=False,
    )
    resp = client.get("/v1/library/stats")
    assert resp.status_code == 500
    body = resp.json()
    assert body["code"] == "internal_error"
    assert "sensitive" not in body["error"]


def test_search_text_with_artist_filters(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    import numpy as np

    from harmony.config import Config
    from harmony.engine import Engine
    from harmony.models import track_id_from_content_hash, utcnow
    from harmony.storage.metadata import MetadataStore
    from harmony.storage.vectors import VectorStore
    from tests.fake_embedder import FakeEmbedder

    track_a = track_id_from_content_hash("api-hash-a")
    track_b = track_id_from_content_hash("api-hash-b")
    track_c = track_id_from_content_hash("api-hash-c")

    cfg = Config(data_dir=tmp_path / "data")
    store = MetadataStore(cfg)
    vectors = VectorStore(cfg)
    version = cfg.embedding_version()
    now = utcnow().isoformat()

    for track_id, artist, album, vector in (
        (track_a, "Radiohead", "OK Computer", [1.0, 0.0, 0.0]),
        (track_b, "Michael Jackson", "Thriller", [0.9, 0.1, 0.0]),
        (track_c, "Beck", "Odelay", [0.8, 0.2, 0.0]),
    ):
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
        vectors.save_track_vector(
            track_id,
            np.array(vector, dtype=np.float32),
            version,
        )
    store.conn.commit()

    engine = Engine(cfg.data_dir)
    embedder = FakeEmbedder(dimension=3)
    embedder.is_loaded = True
    engine._embedder = embedder
    engine._get_index_manager().rebuild()

    client = TestClient(create_app(cfg.data_dir, preload_on_serve=False, engine=engine))
    resp = client.post(
        "/v1/search/text",
        json={
            "query": "melancholic piano",
            "k": 10,
            "filters": {"artists": ["Radiohead", "Michael Jackson"]},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    artists = {item["metadata"]["artist"] for item in body["items"]}
    assert artists == {"Radiohead", "Michael Jackson"}
    assert body["query"]["filters"] == {
        "artists": ["Radiohead", "Michael Jackson"],
    }

    store.close()
    engine.close()
