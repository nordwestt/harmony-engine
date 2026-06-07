"""Harmony Engine facade."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from harmony.config import Config
from harmony.embedding.base import Embedder
from harmony.errors import (
    DependencyMissingError,
    ModelNotReadyError,
    PathNotAllowedError,
)
from harmony.embedding.factory import create_embedder
from harmony.embedding.pipeline import TrackEmbeddingPipeline
from harmony.index.manager import TrackIndexManager
from harmony.models import SearchResult, SyncReport
from harmony.retrieval.filters import Filters
from harmony.retrieval.search import SearchEngine
from harmony.models import validate_track_id
from harmony.scanner.filesystem import FilesystemScanner, validate_scan_paths
from harmony.storage.metadata import MetadataStore
from harmony.storage.purge import LibraryPurge
from harmony.storage.sync import LibrarySync
from harmony.storage.vectors import VectorStore

logger = logging.getLogger(__name__)


class Engine:
    """Main entry point for Harmony Engine."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.config = Config.load(data_dir)
        self._store: MetadataStore | None = None
        self._sync: LibrarySync | None = None
        self._vectors: VectorStore | None = None
        self._embedder: Embedder | None = None
        self._pipeline: TrackEmbeddingPipeline | None = None
        self._index_manager: TrackIndexManager | None = None
        self._purge: LibraryPurge | None = None
        self._search: SearchEngine | None = None

    @property
    def store(self) -> MetadataStore:
        if self._store is None:
            self.config.ensure_data_dir()
            self._store = MetadataStore(self.config)
        return self._store

    @property
    def sync(self) -> LibrarySync:
        if self._sync is None:
            self._sync = LibrarySync(self.config, self.store)
        return self._sync

    @property
    def vectors(self) -> VectorStore:
        if self._vectors is None:
            self._vectors = VectorStore(self.config)
        return self._vectors

    def needs_init(self) -> bool:
        """True when the data directory has never been initialized."""
        return not (self.config.data_dir / "config.yaml").exists()

    def init(self) -> None:
        """Initialize data directory, config, and database schema."""
        self.config.ensure_data_dir()
        self.config.save()
        _ = self.store  # runs migrations

    def ensure_initialized(self) -> bool:
        """Initialize on first use. Returns True if setup ran."""
        if not self.needs_init():
            return False
        self.init()
        logger.info("Initialized Harmony at %s", self.config.data_dir)
        return True

    def index(
        self,
        *,
        paths: list[str | Path] | None = None,
        full_rescan: bool = False,  # noqa: ARG002 — reserved for future use
        embed: bool = True,
        prune: bool = False,
        reembed: bool = False,
        on_embed_progress: Callable[[int, int, str], None] | None = None,
    ) -> SyncReport:
        """Scan filesystem, reconcile metadata, embed new/changed tracks, rebuild index."""
        self.ensure_initialized()
        allowed_roots = self.config.filesystem.paths
        if not allowed_roots:
            if paths:
                bootstrap = [str(Path(p).expanduser().resolve()) for p in paths]
                self.config.filesystem.paths = bootstrap
                self.config.save()
                allowed_roots = bootstrap
            else:
                raise PathNotAllowedError(
                    "No scan roots configured. Set filesystem.paths in config.yaml "
                    "or HARMONY_INDEX_PATHS"
                )

        scan_paths = validate_scan_paths(paths, allowed_roots)
        scanner = FilesystemScanner(scan_paths, config=self.config.filesystem)
        report = self.sync.reconcile(scanner)

        if prune:
            purged = self._get_purge().prune_missing()
            report.purged = len(purged)

        embedded_ids: list[str] = []
        if embed:
            embedded, failed, embedded_ids = self._get_pipeline().embed_pending(
                reembed=reembed,
                on_progress=on_embed_progress,
            )
            report.embedded = embedded
            report.failed += failed

        if reembed or report.purged > 0:
            if embedded_ids or report.purged > 0:
                self._get_index_manager().rebuild()
                self._invalidate_search()
        elif embedded_ids:
            version = self.config.embedding_version()
            manager = self._get_index_manager()
            for track_id in embedded_ids:
                vector = self.vectors.load_track_vector(track_id, version)
                if vector is not None:
                    manager.upsert_track(track_id, vector)
            self._invalidate_search()
        else:
            self._get_index_manager().ensure_loaded()

        return report

    def purge(
        self,
        *,
        missing: bool = False,
        removed: bool = False,
        orphans: bool = False,
    ) -> dict[str, int]:
        """Remove missing/removed tracks and orphan files from the store."""
        purge_lib = self._get_purge()
        counts = {"missing": 0, "removed": 0, "orphans": 0}

        if missing:
            counts["missing"] = len(purge_lib.prune_missing())
        if removed:
            counts["removed"] = len(purge_lib.purge_removed())
        if orphans:
            counts["orphans"] = purge_lib.purge_orphans()

        if counts["missing"] or counts["removed"]:
            self._get_index_manager().rebuild()
            self._invalidate_search()

        return counts

    def list_tracks(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        status: str | None = None,
    ) -> tuple[list[dict[str, object]], int]:
        tracks, total = self.store.list_tracks(offset=offset, limit=limit, status=status)
        items = [
            {
                "track_id": t.track_id,
                "title": t.title,
                "artist": t.artist,
                "album": t.album,
                "status": t.status.value,
                "primary_path": t.primary_path,
                "duration_ms": t.duration_ms,
                "embedding_version": t.embedding_version,
                "indexed_at": t.indexed_at.isoformat() if t.indexed_at else None,
            }
            for t in tracks
        ]
        return items, total

    def get_track_detail(self, track_id: str) -> dict[str, object] | None:
        validate_track_id(track_id)
        track = self.store.get_track(track_id)
        if track is None:
            return None
        return {
            "track": {
                "track_id": track.track_id,
                "title": track.title,
                "artist": track.artist,
                "album": track.album,
                "status": track.status.value,
                "primary_path": track.primary_path,
                "duration_ms": track.duration_ms,
                "embedding_version": track.embedding_version,
                "indexed_at": track.indexed_at.isoformat() if track.indexed_at else None,
            },
            "locations": [
                {
                    "location_id": loc.location_id,
                    "path": loc.path,
                    "is_primary": loc.is_primary,
                    "first_seen_at": loc.first_seen_at.isoformat(),
                    "last_seen_at": loc.last_seen_at.isoformat(),
                }
                for loc in track.locations
            ],
        }

    def list_sync_history(self, *, limit: int = 10) -> list[dict[str, object]]:
        return self.store.list_sync_runs(limit=limit)

    def is_ready(self) -> dict[str, object]:
        model = self.model_status()
        index_size = self._get_index_manager().ensure_loaded().size
        embedded = self.store.count_embedded_tracks()
        loading = bool(model["loading"])
        loaded = bool(model["loaded"])
        message: str | None = None
        if loading:
            message = (
                "Hey, I'm just busy downloading the weights from Hugging Face - please wait!"
            )
        elif model["load_error"]:
            message = str(model["load_error"])
        return {
            "ready": loaded and index_size > 0,
            "model_loaded": loaded,
            "model_loading": loading,
            "message": message,
            "index_size": index_size,
            "tracks_embedded": embedded,
        }

    def stats(self) -> dict[str, int | str]:
        """Return library statistics."""
        by_status = self.store.count_tracks_by_status()
        return {
            "data_dir": str(self.config.data_dir),
            "embedding_version": self.config.embedding_version(),
            "tracks_active": by_status.get("active", 0),
            "tracks_embedded": self.store.count_embedded_tracks(),
            "tracks_missing": by_status.get("missing", 0),
            "tracks_removed": by_status.get("removed", 0),
            "tracks_failed": by_status.get("failed", 0),
            "tracks_total": sum(by_status.values()),
            "index_size": self._get_index_manager().ensure_loaded().size,
        }

    def _ensure_search_ready(self) -> None:
        model = self.model_status()
        if model["loading"]:
            raise ModelNotReadyError(
                "Embedding model is still loading. Check GET /v1/ready for status."
            )
        if model["load_error"]:
            raise ModelNotReadyError(
                "Embedding model failed to load. Check server logs for details."
            )

    def search_by_text(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> SearchResult:
        """Search library by natural language query."""
        self._ensure_search_ready()
        try:
            return self._get_search().search_by_text(query, k=k, filters=filters)
        except ImportError as e:
            raise DependencyMissingError(str(e)) from e

    def search_by_track(
        self,
        track_id: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> SearchResult:
        validate_track_id(track_id)
        self._ensure_search_ready()
        try:
            return self._get_search().search_by_track(track_id, k=k, filters=filters)
        except ImportError as e:
            raise DependencyMissingError(str(e)) from e

    def preload_model(self) -> None:
        """Load embedding model weights into memory."""
        self._get_embedder().preload()

    def preload_model_background(self) -> None:
        """Start loading embedding model weights in a background thread."""
        self._get_embedder().preload_background()

    def model_status(self) -> dict[str, str | bool | int | None]:
        embedder = self._get_embedder()
        policy = embedder.keep_alive_policy
        return {
            "model": self.config.embedding.model,
            "loaded": embedder.is_loaded,
            "loading": embedder.is_loading,
            "load_error": embedder.load_error,
            "device": embedder.device,
            "keep_alive": policy.label,
            "checkpoint": self.config.embedding.checkpoint,
            "dimension": self.config.embedding.effective_dimension(),
        }

    def close(self) -> None:
        if self._embedder is not None:
            self._embedder.unload()
        if self._store is not None:
            self._store.close()

    def _get_embedder(self) -> Embedder:
        if self._embedder is None:
            self._embedder = create_embedder(self.config)
        return self._embedder

    def _get_pipeline(self) -> TrackEmbeddingPipeline:
        if self._pipeline is None:
            self._pipeline = TrackEmbeddingPipeline(
                self.config,
                self.store,
                self.vectors,
                self._get_embedder(),
            )
        return self._pipeline

    def _get_index_manager(self) -> TrackIndexManager:
        if self._index_manager is None:
            self._index_manager = TrackIndexManager(self.config, self.store, self.vectors)
        return self._index_manager

    def _get_purge(self) -> LibraryPurge:
        if self._purge is None:
            self._purge = LibraryPurge(self.config, self.store, self.vectors)
        return self._purge

    def _invalidate_search(self) -> None:
        self._search = None

    def _get_search(self) -> SearchEngine:
        if self._search is None:
            self._search = SearchEngine(
                self.config,
                self.store,
                self._get_embedder(),
                lambda: self._get_index_manager().ensure_loaded(),
                self.vectors,
            )
        return self._search

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
