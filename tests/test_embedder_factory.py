"""Tests for embedder factory and backend registry."""

from __future__ import annotations

import pytest

from harmony.config import Config
from harmony.embedding.backends.muq_mulan import MuQMuLanEmbedder
from harmony.embedding.factory import (
    backend_dimension,
    create_embedder,
    list_backends,
)
from tests.fake_embedder import FakeEmbedder


def test_list_backends_includes_muq_mulan() -> None:
    assert "muq-mulan" in list_backends()


def test_backend_dimension_for_muq_mulan() -> None:
    assert backend_dimension("muq-mulan") == 512


def test_unknown_backend_raises() -> None:
    cfg = Config()
    cfg.embedding.model = "unknown-model"
    with pytest.raises(ValueError, match="Unknown embedding model 'unknown-model'"):
        create_embedder(cfg)


def test_unknown_backend_dimension_raises() -> None:
    with pytest.raises(ValueError, match="Unknown embedding model 'unknown-model'"):
        backend_dimension("unknown-model")


def test_create_embedder_returns_muq_backend() -> None:
    cfg = Config()
    embedder = create_embedder(cfg)
    assert isinstance(embedder, MuQMuLanEmbedder)
    assert embedder.dimension == 512
    assert cfg.embedding.dimension == 512


def test_effective_dimension_without_explicit_override() -> None:
    cfg = Config()
    assert cfg.embedding.effective_dimension() == 512


def test_fake_embedder_satisfies_protocol() -> None:
    embedder = FakeEmbedder()
    assert embedder.name == "fake"
    assert embedder.dimension == 4
    assert embedder.keep_alive_policy.mode == "forever"
    embedder.preload()
    assert embedder.is_loaded is True
    embedder.unload()
    assert embedder.is_loaded is False
