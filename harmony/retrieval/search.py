"""Search engine facade."""

from __future__ import annotations

import time

from harmony.config import Config
from harmony.embedding.base import Embedder
from harmony.index.base import IndexBackend
from harmony.models import QueryInfo, ScoredItem, SearchResult
from harmony.retrieval.filters import Filters
from harmony.storage.metadata import MetadataStore


class SearchEngine:
    def __init__(
        self,
        config: Config,
        store: MetadataStore,
        embedder: Embedder,
        track_index: IndexBackend,
    ) -> None:
        self.config = config
        self.store = store
        self.embedder = embedder
        self.track_index = track_index

    def search_by_text(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> SearchResult:
        k = k or self.config.retrieval.default_k
        t0 = time.perf_counter()

        vector = self.embedder.embed_text(query)
        ids, scores = self.track_index.search(vector, k=k * 2)

        items: list[ScoredItem] = []
        for rank, (track_id, score) in enumerate(zip(ids, scores, strict=False)):
            if filters and track_id in filters.exclude_track_ids:
                continue
            track = self.store.get_track(track_id)
            if track is None or track.status.value == "removed":
                continue
            items.append(
                ScoredItem(
                    track_id=track_id,
                    score=score,
                    rank=rank + 1,
                    match_granularity="track",
                    metadata=track,
                )
            )
            if len(items) >= k:
                break

        active = self.store.count_tracks_by_status().get("active", 0)
        return SearchResult(
            items=items,
            query=QueryInfo(type="text", value=query, filters=filters.__dict__ if filters else None),
            total_indexed=active,
            took_ms=int((time.perf_counter() - t0) * 1000),
        )

    def search_by_track(
        self,
        track_id: str,
        *,
        k: int | None = None,
        filters: Filters | None = None,
    ) -> SearchResult:
        raise NotImplementedError("search_by_track requires stored embeddings")
