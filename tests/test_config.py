"""Tests for configuration."""

from pathlib import Path

from harmony.config import Config


def test_config_defaults(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path)
    assert cfg.database.path == "harmony.db"
    assert cfg.audio.target_sample_rate == 24000
    assert "muq-mulan" in cfg.embedding_version()


def test_config_roundtrip(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path)
    cfg.filesystem.paths = ["/music"]
    cfg.save()

    loaded = Config.load(tmp_path)
    assert loaded.filesystem.paths == ["/music"]
    assert loaded.audio.chunk_seconds == 10
