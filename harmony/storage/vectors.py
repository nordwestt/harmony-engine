"""Embedding vector file storage."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from harmony.config import Config
from harmony.models import validate_track_id


class VectorStore:
    def __init__(self, config: Config) -> None:
        self.config = config

    def version_dir(self, embedding_version: str | None = None) -> Path:
        version = embedding_version or self.config.embedding_version()
        path = self.config.embeddings_dir / version
        path.mkdir(parents=True, exist_ok=True)
        return path

    def track_vector_path(self, track_id: str, embedding_version: str | None = None) -> Path:
        safe_id = validate_track_id(track_id)
        base = self.version_dir(embedding_version) / "tracks"
        path = base / f"{safe_id}.npy"
        resolved = path.resolve()
        if not resolved.is_relative_to(base.resolve()):
            raise ValueError(f"Invalid track vector path for track_id: {track_id}")
        return path

    def chunk_vectors_path(self, track_id: str, embedding_version: str | None = None) -> Path:
        safe_id = validate_track_id(track_id)
        base = self.version_dir(embedding_version) / "chunks"
        path = base / f"{safe_id}.npy"
        resolved = path.resolve()
        if not resolved.is_relative_to(base.resolve()):
            raise ValueError(f"Invalid chunk vector path for track_id: {track_id}")
        return path

    def save_track_vector(
        self,
        track_id: str,
        vector: np.ndarray,
        embedding_version: str | None = None,
    ) -> Path:
        path = self.track_vector_path(track_id, embedding_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vector.astype(np.float32))
        return path

    def load_track_vector(
        self,
        track_id: str,
        embedding_version: str | None = None,
    ) -> np.ndarray | None:
        path = self.track_vector_path(track_id, embedding_version)
        if not path.exists():
            return None
        return np.load(path)

    def save_chunk_vectors(
        self,
        track_id: str,
        vectors: np.ndarray,
        embedding_version: str | None = None,
    ) -> Path:
        path = self.chunk_vectors_path(track_id, embedding_version)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vectors.astype(np.float32))
        return path

    def load_chunk_vectors(
        self,
        track_id: str,
        embedding_version: str | None = None,
    ) -> np.ndarray | None:
        path = self.chunk_vectors_path(track_id, embedding_version)
        if not path.exists():
            return None
        return np.load(path)

    def delete_track_vectors(self, track_id: str) -> None:
        """Remove stored vectors for a track across all embedding versions."""
        if not self.config.embeddings_dir.exists():
            return
        for version_dir in self.config.embeddings_dir.iterdir():
            if not version_dir.is_dir():
                continue
            for subdir in ("tracks", "chunks"):
                path = version_dir / subdir / f"{track_id}.npy"
                path.unlink(missing_ok=True)
