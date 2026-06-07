"""Build and persist the track-level search index."""

from __future__ import annotations

import logging

import numpy as np

from harmony.config import Config
from harmony.index.brute import BruteForceIndex
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore

logger = logging.getLogger(__name__)


class TrackIndexManager:
    def __init__(
        self,
        config: Config,
        store: MetadataStore,
        vectors: VectorStore,
    ) -> None:
        self.config = config
        self.store = store
        self.vectors = vectors
        self.index = BruteForceIndex()
        self._loaded = False

    @property
    def index_path(self) -> str:
        version = self.config.embedding_version()
        path = self.config.indexes_dir / version / "track.brute"
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path)

    def ensure_loaded(self) -> BruteForceIndex:
        if not self._loaded:
            self.index.load(self.index_path)
            if self.index.size == 0:
                self.rebuild()
            self._loaded = True
        return self.index

    def rebuild(self) -> int:
        """Rebuild index from all embedded active tracks on disk."""
        tracks = self.store.list_embedded_tracks()
        ids: list[str] = []
        vectors: list[np.ndarray] = []
        version = self.config.embedding_version()

        for track in tracks:
            vector = self.vectors.load_track_vector(track.track_id, version)
            if vector is not None:
                ids.append(track.track_id)
                vectors.append(vector)

        self.index = BruteForceIndex()
        if ids:
            self.index.add(ids, np.stack(vectors))
        self.index.save(self.index_path)
        self._loaded = True
        logger.info("Track index rebuilt with %d vectors", len(ids))
        return len(ids)

    def upsert_track(self, track_id: str, vector: np.ndarray) -> None:
        self.ensure_loaded()
        self.index.add([track_id], vector)
        self.index.save(self.index_path)

    def remove_track(self, track_id: str) -> None:
        self.ensure_loaded()
        self.index.remove([track_id])
        self.index.save(self.index_path)
