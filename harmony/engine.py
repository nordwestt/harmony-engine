"""Harmony Engine facade."""

from __future__ import annotations

from pathlib import Path

from harmony.config import Config
from harmony.embedding.muq_mulan import MuQMuLanEmbedder
from harmony.index.brute import BruteForceIndex
from harmony.models import SearchResult, SyncReport
from harmony.retrieval.filters import Filters
from harmony.retrieval.search import SearchEngine
from harmony.sources.filesystem import FilesystemScanner
from harmony.storage.metadata import MetadataStore
from harmony.storage.sync import LibrarySync
from harmony.storage.vectors import VectorStore


class Engine:
    """Main entry point for Harmony Engine."""

    def __init__(self, data_dir: Path | str | None = None) -> None:
        self.config = Config.load(data_dir)
        self._store: MetadataStore | None = None
        self._sync: LibrarySync | None = None
        self._vectors: VectorStore | None = None
        self._embedder: MuQMuLanEmbedder | None = None
        self._track_index: BruteForceIndex | None = None
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

    def init(self) -> None:
        """Initialize data directory, config, and database schema."""
        self.config.ensure_data_dir()
        self.config.save()
        _ = self.store  # runs migrations

    def index(
        self,
        *,
        paths: list[str | Path] | None = None,
        full_rescan: bool = False,  # noqa: ARG002 — reserved for future use
    ) -> SyncReport:
        """Scan filesystem paths and reconcile library metadata.

        Embedding is not yet wired; this pass discovers files and updates Turso.
        """
        scan_paths = paths or self.config.sources.local.paths
        if not scan_paths:
            raise ValueError("No paths provided. Pass paths= or configure sources.local.paths")

        scanner = FilesystemScanner(scan_paths, config=self.config.sources.local)
        return self.sync.reconcile(scanner)

    def stats(self) -> dict[str, int | str]:
        """Return library statistics."""
        by_status = self.store.count_tracks_by_status()
        return {
            "data_dir": str(self.config.data_dir),
            "embedding_version": self.config.embedding_version(),
            "tracks_active": by_status.get("active", 0),
            "tracks_missing": by_status.get("missing", 0),
            "tracks_removed": by_status.get("removed", 0),
            "tracks_failed": by_status.get("failed", 0),
            "tracks_total": sum(by_status.values()),
        }

    def search_by_text(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> SearchResult:
        """Search library by natural language query."""
        return self._get_search().search_by_text(query, k=k, filters=filters)

    def search_by_track(
        self,
        track_id: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> SearchResult:
        return self._get_search().search_by_track(track_id, k=k, filters=filters)

    def close(self) -> None:
        if self._store is not None:
            self._store.close()

    def _get_search(self) -> SearchEngine:
        if self._search is None:
            self._embedder = MuQMuLanEmbedder(self.config.embedding)
            self._track_index = BruteForceIndex()
            self._search = SearchEngine(
                self.config,
                self.store,
                self._embedder,
                self._track_index,
            )
        return self._search

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
