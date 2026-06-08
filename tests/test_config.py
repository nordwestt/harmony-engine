"""Tests for configuration."""

from pathlib import Path

from harmony.config import Config
from harmony.engine import Engine


def test_config_defaults(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path)
    assert cfg.database.path == "harmony.db"
    assert cfg.audio.target_sample_rate == 24000
    assert "clamp3" in cfg.embedding_version()


def test_config_roundtrip(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path)
    cfg.filesystem.paths = ["/music"]
    cfg.save()

    loaded = Config.load(tmp_path)
    assert loaded.filesystem.paths == ["/music"]
    assert loaded.audio.chunk_seconds == cfg.audio.chunk_seconds


def test_index_paths_from_env(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("HARMONY_INDEX_PATHS", "/music,/other")
    loaded = Config.load(tmp_path)
    assert loaded.filesystem.paths == ["/music", "/other"]


def test_ensure_initialized_only_runs_once(tmp_path: Path) -> None:
    engine = Engine(tmp_path)
    assert engine.needs_init() is True
    assert engine.ensure_initialized() is True
    assert (tmp_path / "config.yaml").exists()
    assert engine.ensure_initialized() is False


def test_index_paths_env_overrides_config_yaml(tmp_path: Path, monkeypatch) -> None:
    cfg = Config(data_dir=tmp_path)
    cfg.filesystem.paths = ["/from-yaml"]
    cfg.save()

    monkeypatch.setenv("HARMONY_INDEX_PATHS", "/music")
    loaded = Config.load(tmp_path)
    assert loaded.filesystem.paths == ["/music"]
