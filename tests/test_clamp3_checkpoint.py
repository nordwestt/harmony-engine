"""Tests for CLaMP3 checkpoint resolution."""

from __future__ import annotations

from pathlib import Path

from harmony.config import Config
from harmony.embedding.backends.clamp3 import Clamp3Embedder


def test_clamp3_ignores_muq_checkpoint(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path)
    cfg.embedding.model = "clamp3"
    cfg.embedding.checkpoint = "OpenMuQ/MuQ-MuLan-large"
    embedder = Clamp3Embedder(cfg)
    assert embedder._clamp3_repo_id() == "sander-wood/clamp3"
