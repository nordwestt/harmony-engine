"""Metadata filters for pre-search narrowing."""

from __future__ import annotations

from dataclasses import dataclass, field

from harmony.models import Track


def _norm(value: str) -> str:
    return value.strip().casefold()


def normalize_string_list(values: list[str] | None) -> list[str] | None:
    """Strip strings, drop empties, return None if nothing remains."""
    if not values:
        return None
    cleaned = [v.strip() for v in values if v.strip()]
    return cleaned or None


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

    def has_metadata_filters(self) -> bool:
        """True when artist/album/genre filters are set (post-filter on ANN results)."""
        return bool(self.artists or self.albums or self.genres)

    def matches(self, track: Track) -> bool:
        """Return True if track satisfies active metadata filters."""
        if self.artists:
            artist_values = {_norm(a) for a in self.artists}
            if _norm(track.artist) not in artist_values:
                return False
        if self.albums:
            album_values = {_norm(a) for a in self.albums}
            if _norm(track.album) not in album_values:
                return False
        return True
