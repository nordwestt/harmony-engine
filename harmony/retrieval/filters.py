"""Metadata filters for pre-search narrowing."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Filters:
    artists: list[str] | None = None
    albums: list[str] | None = None
    genres: list[str] | None = None
    year_min: int | None = None
    year_max: int | None = None
    duration_min_ms: int | None = None
    duration_max_ms: int | None = None
    paths_glob: list[str] | None = None
    exclude_track_ids: list[str] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not any(
            [
                self.artists,
                self.albums,
                self.genres,
                self.year_min is not None,
                self.year_max is not None,
                self.duration_min_ms is not None,
                self.duration_max_ms is not None,
                self.paths_glob,
                self.exclude_track_ids,
            ]
        )
