"""Tests for the HTTP API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from harmony.api.app import create_app


def test_health_and_init(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    app = create_app(tmp_path / "data", preload_on_serve=False)
    client = TestClient(app)

    assert client.get("/health").json() == {"status": "ok"}
    init_resp = client.post("/v1/init")
    assert init_resp.status_code == 200
    assert "data_dir" in init_resp.json()


def test_ready_without_index(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    app = create_app(tmp_path / "data", preload_on_serve=False)
    client = TestClient(app)
    client.post("/v1/init")
    ready = client.get("/v1/ready").json()
    assert ready["ready"] is False
    assert ready["index_size"] == 0


def test_library_tracks_empty(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    app = create_app(tmp_path / "data", preload_on_serve=False)
    client = TestClient(app)
    client.post("/v1/init")
    resp = client.get("/v1/library/tracks")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] == 0
    assert body["items"] == []


def test_purge_requires_flags(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    app = create_app(tmp_path / "data", preload_on_serve=False)
    client = TestClient(app)
    client.post("/v1/init")
    resp = client.post("/v1/library/purge", json={})
    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"
