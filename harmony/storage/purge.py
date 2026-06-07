"""Remove missing/removed tracks and orphan data from disk."""

from __future__ import annotations

import logging

from harmony.config import Config
from harmony.models import TrackStatus, utcnow
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore

logger = logging.getLogger(__name__)


class LibraryPurge:
    def __init__(
        self,
        config: Config,
        store: MetadataStore,
        vectors: VectorStore,
    ) -> None:
        self.config = config
        self.store = store
        self.vectors = vectors

    def prune_missing(self) -> list[str]:
        """Mark all missing tracks as removed and delete them. Returns purged track IDs."""
        track_ids = self.store.list_track_ids_by_status(TrackStatus.MISSING)
        if not track_ids:
            return []

        self.store.set_tracks_status(track_ids, TrackStatus.REMOVED)
        self.delete_tracks(track_ids)
        return track_ids

    def purge_removed(self) -> list[str]:
        """Delete all tracks already in removed status."""
        track_ids = self.store.list_track_ids_by_status(TrackStatus.REMOVED)
        if not track_ids:
            return []

        self.delete_tracks(track_ids)
        return track_ids

    def purge_orphans(self) -> int:
        """Delete embedding files on disk not referenced by any track in the DB."""
        known = set(self.store.list_all_track_ids())
        deleted = 0
        version_dir = self.vectors.version_dir()
        tracks_dir = version_dir / "tracks"
        if not tracks_dir.exists():
            return 0

        for path in tracks_dir.glob("*.npy"):
            track_id = path.stem
            if track_id not in known:
                path.unlink(missing_ok=True)
                deleted += 1
                logger.info("Deleted orphan vector %s", path)

        return deleted

    def delete_tracks(self, track_ids: list[str]) -> int:
        """Delete track rows and their vector files."""
        for track_id in track_ids:
            self.vectors.delete_track_vectors(track_id)

        return self.store.delete_tracks(track_ids)
