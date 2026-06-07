"""Search engine facade."""

from __future__ import annotations

import time
from collections.abc import Callable

from harmony.config import Config
from harmony.embedding.base import Embedder
from harmony.errors import IndexEmptyError
from harmony.index.base import IndexBackend
from harmony.models import QueryInfo, ScoredItem, SearchResult
from harmony.retrieval.filters import Filters
from harmony.storage.metadata import MetadataStore
from harmony.storage.vectors import VectorStore


class SearchEngine:
    def __init__(
        self,
        config: Config,
        store: MetadataStore,
        embedder: Embedder,
        index_provider: Callable[[], IndexBackend],
        vectors: VectorStore,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self._index_provider = index_provider
        self.vectors = vectors

    def _get_index(self) -> IndexBackend:
        return self._index_provider()

    def search_by_text(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> SearchResult:
        k = k or self.config.retrieval.default_k
        t0 = time.perf_counter()

        track_index = self._get_index()
        if track_index.size == 0:
            raise IndexEmptyError(
                "No embedded tracks in the index. Run: harmony index /path/to/music"
            )

        vector = self.embedder.embed_text(query)
        ids, scores = track_index.search(vector, k=k * 2)

        items = self._build_results(ids, scores, k=k, filters=filters)
        embedded = self.store.count_embedded_tracks()
        return SearchResult(
            items=items,
            query=QueryInfo(type="text", value=query, filters=filters.__dict__ if filters else None),
            total_indexed=embedded,
            took_ms=int((time.perf_counter() - t0) * 1000),
        )

    def search_by_track(
        self,
        track_id: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> SearchResult:
        k = k or self.config.retrieval.default_k
        t0 = time.perf_counter()

        track_index = self._get_index()
        if track_index.size == 0:
            raise IndexEmptyError(
                "No embedded tracks in the index. Run: harmony index /path/to/music"
            )

        vector = self.vectors.load_track_vector(track_id, self.config.embedding_version())
        if vector is None:
            raise ValueError(f"No embedding stored for track {track_id}")

        ids, scores = track_index.search(vector, k=k + 1)
        items = self._build_results(
            ids,
            scores,
            k=k,
            filters=filters,
            exclude_track_ids={track_id},
        )
        embedded = self.store.count_embedded_tracks()
        return SearchResult(
            items=items,
            query=QueryInfo(type="track", value=track_id, filters=filters.__dict__ if filters else None),
            total_indexed=embedded,
            took_ms=int((time.perf_counter() - t0) * 1000),
        )

    def _build_results(
        self,
        ids: list[str],
        scores: list[float],
        *,
        k: int,
        filters: Filters | None,
        exclude_track_ids: set[str] | None = None,
    ) -> list[ScoredItem]:
        exclude = set(exclude_track_ids or [])
        if filters:
            exclude.update(filters.exclude_track_ids)

        items: list[ScoredItem] = []
        for track_id, score in zip(ids, scores, strict=False):
            if track_id in exclude:
                continue
            track = self.store.get_track(track_id)
            if track is None or track.status.value == "removed":
                continue
            items.append(
                ScoredItem(
                    track_id=track_id,
                    score=score,
                    rank=len(items) + 1,
                    match_granularity="track",
                    metadata=track,
                )
            )
            if len(items) >= k:
                break
        return items
