"""Harmony Engine — music library indexing and vector search."""

from harmony.engine import Engine
from harmony.models import (
    ScoredItem,
    SearchResult,
    SyncReport,
    Track,
    TrackLocation,
    TrackStatus,
)

__all__ = [
    "Engine",
    "ScoredItem",
    "SearchResult",
    "SyncReport",
    "Track",
    "TrackLocation",
    "TrackStatus",
]

__version__ = "0.1.0"
