"""Core data types for Harmony Engine."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

# Namespace for deterministic track IDs from content hashes.
TRACK_ID_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")


class TrackStatus(str, Enum):
    ACTIVE = "active"
    MISSING = "missing"
    REMOVED = "removed"
    FAILED = "failed"


class PathChangeReason(str, Enum):
    MOVED = "moved"
    RENAMED = "renamed"
    PRIMARY_CHANGED = "primary_changed"


def track_id_from_content_hash(content_hash: str) -> str:
    """Derive a stable track ID from a SHA-256 content hash."""
    return str(uuid.uuid5(TRACK_ID_NAMESPACE, content_hash))


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class TrackLocation:
    location_id: str
    track_id: str
    path: str
    is_primary: bool
    first_seen_at: datetime
    last_seen_at: datetime


@dataclass
class Track:
    track_id: str
    content_hash: str
    status: TrackStatus
    primary_path: str
    duration_ms: int
    title: str
    artist: str
    album: str
    embedding_version: str
    indexed_at: datetime | None = None
    last_seen_at: datetime | None = None
    album_artist: str | None = None
    year: int | None = None
    genre: str | None = None
    disc_number: int | None = None
    track_number: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    locations: list[TrackLocation] = field(default_factory=list)


@dataclass
class Chunk:
    chunk_id: str
    track_id: str
    index: int
    start_ms: int
    end_ms: int
    embedding_path: str | None = None


@dataclass
class ScannedFile:
    """A file discovered during a source scan."""

    path: str
    content_hash: str
    size_bytes: int | None = None
    mtime: float | None = None
    title: str | None = None
    artist: str | None = None
    album: str | None = None
    duration_ms: int | None = None


@dataclass
class SyncReport:
    added: int = 0
    updated_metadata: int = 0
    moved: int = 0
    duplicates_found: int = 0
    missing: int = 0
    removed: int = 0
    reembedded: int = 0
    failed: int = 0
    skipped: int = 0
    duration_ms: int = 0

    def total_changes(self) -> int:
        return (
            self.added
            + self.updated_metadata
            + self.moved
            + self.duplicates_found
            + self.missing
            + self.removed
            + self.reembedded
            + self.failed
        )


@dataclass
class BestChunkMatch:
    chunk_id: str
    start_ms: int
    end_ms: int
    chunk_score: float


@dataclass
class ScoredItem:
    track_id: str
    score: float
    rank: int
    match_granularity: str
    metadata: Track
    best_chunk: BestChunkMatch | None = None


@dataclass
class QueryInfo:
    type: str
    value: str
    filters: dict[str, Any] | None = None


@dataclass
class SearchResult:
    items: list[ScoredItem]
    query: QueryInfo
    total_indexed: int
    took_ms: int
