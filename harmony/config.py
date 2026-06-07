"""Configuration loading and defaults."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATA_DIR = Path.home() / ".harmony"


@dataclass
class DatabaseConfig:
    path: str = "harmony.db"


@dataclass
class EmbeddingConfig:
    model: str = "muq-mulan"
    device: str = "auto"
    batch_size: int = 16


@dataclass
class AudioConfig:
    target_sample_rate: int = 24000
    mono: bool = True
    chunk_seconds: int = 10
    overlap_seconds: int = 2
    min_chunk_seconds: int = 1


@dataclass
class SyncConfig:
    missing_grace_days: int = 7
    watch_debounce_seconds: int = 5
    hash_chunk_size_mb: int = 4
    primary_path_policy: str = "scan"  # scan | shortest | first_seen


@dataclass
class IndexConfig:
    backend: str = "faiss"
    metric: str = "cosine"
    build_track_index: bool = True
    build_chunk_index: bool = True
    compact_threshold: float = 0.10


@dataclass
class RetrievalConfig:
    default_k: int = 50
    default_granularity: str = "track"
    default_aggregation: str = "max"


@dataclass
class LocalSourceConfig:
    paths: list[str] = field(default_factory=list)
    extensions: list[str] = field(
        default_factory=lambda: [".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wav", ".opus"]
    )
    follow_symlinks: bool = False


@dataclass
class SourcesConfig:
    local: LocalSourceConfig = field(default_factory=LocalSourceConfig)


@dataclass
class Config:
    data_dir: Path = field(default_factory=lambda: DEFAULT_DATA_DIR)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)
    index: IndexConfig = field(default_factory=IndexConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)

    @property
    def db_path(self) -> Path:
        return self.data_dir / self.database.path

    @property
    def embeddings_dir(self) -> Path:
        return self.data_dir / "embeddings"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    def embedding_version(self) -> str:
        """Composite key for the current embedding configuration."""
        return (
            f"{self.embedding.model}@0.1:"
            f"{self.audio.chunk_seconds}:"
            f"{self.audio.overlap_seconds}:"
            f"{self.audio.target_sample_rate}"
        )

    def ensure_data_dir(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embeddings_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.ensure_data_dir()
        path = self.data_dir / "config.yaml"
        data = _config_to_dict(self)
        path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")

    @classmethod
    def load(cls, data_dir: Path | str | None = None) -> Config:
        if data_dir is None:
            env = os.environ.get("HARMONY_DATA_DIR")
            data_dir = Path(env) if env else DEFAULT_DATA_DIR
        else:
            data_dir = Path(data_dir).expanduser()

        config_path = data_dir / "config.yaml"
        if not config_path.exists():
            cfg = cls(data_dir=data_dir)
            return cfg

        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, data_dir=data_dir)

    @classmethod
    def from_dict(cls, data: dict[str, Any], *, data_dir: Path) -> Config:
        return cls(
            data_dir=data_dir,
            database=_merge(DatabaseConfig, data.get("database")),
            embedding=_merge(EmbeddingConfig, data.get("embedding")),
            audio=_merge(AudioConfig, data.get("audio")),
            sync=_merge(SyncConfig, data.get("sync")),
            index=_merge(IndexConfig, data.get("index")),
            retrieval=_merge(RetrievalConfig, data.get("retrieval")),
            sources=_sources_from_dict(data.get("sources", {})),
        )


def _merge(cls: type, data: dict[str, Any] | None):
    if not data:
        return cls()
    return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def _sources_from_dict(data: dict[str, Any]) -> SourcesConfig:
    local_raw = data.get("local", {})
    return SourcesConfig(local=_merge(LocalSourceConfig, local_raw))


def _config_to_dict(config: Config) -> dict[str, Any]:
    data = asdict(config)
    data["data_dir"] = str(config.data_dir)
    return data
