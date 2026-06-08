"""Pydantic models for the Harmony HTTP API."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field, field_validator

TRACK_ID_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)

MAX_QUERY_LENGTH = 512
MAX_INDEX_PATHS = 32
MAX_PATH_STRING_LENGTH = 4096
MAX_FILTER_LIST = 32


class SearchFilters(BaseModel):
    artists: list[str] | None = Field(default=None, max_length=MAX_FILTER_LIST)
    albums: list[str] | None = Field(default=None, max_length=MAX_FILTER_LIST)

    @field_validator("artists", "albums")
    @classmethod
    def normalize_filter_lists(cls, values: list[str] | None) -> list[str] | None:
        if values is None:
            return None
        cleaned = [v.strip() for v in values if v.strip()]
        return cleaned or None


class ErrorResponse(BaseModel):
    error: str
    code: str


class IndexRequest(BaseModel):
    paths: list[str] | None = Field(
        default=None,
        max_length=MAX_INDEX_PATHS,
        description="Directories to scan; defaults to HARMONY_INDEX_PATHS or filesystem.paths",
    )
    full_rescan: bool = False
    embed: bool = True
    prune: bool = False
    reembed: bool = False
    async_: bool = Field(default=False, alias="async")

    @field_validator("paths")
    @classmethod
    def validate_path_strings(cls, paths: list[str] | None) -> list[str] | None:
        if paths is None:
            return None
        for path in paths:
            if len(path) > MAX_PATH_STRING_LENGTH:
                raise ValueError(
                    f"Path exceeds maximum length of {MAX_PATH_STRING_LENGTH} characters"
                )
        return paths


class IndexJobResponse(BaseModel):
    job_id: str
    status: str


class IndexJobStatus(BaseModel):
    job_id: str
    status: str
    phase: str | None = None
    embedded: int = 0
    total_pending: int = 0
    failed: int = 0
    error: str | None = None
    report: dict[str, Any] | None = None
    created_at: str | None = None
    updated_at: str | None = None
    started_at: str | None = None
    finished_at: str | None = None


class SyncReportResponse(BaseModel):
    added: int = 0
    moved: int = 0
    skipped: int = 0
    missing: int = 0
    removed: int = 0
    embedded: int = 0
    purged: int = 0
    failed: int = 0
    duration_ms: int = 0


class TextSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LENGTH)
    k: int = Field(default=50, ge=1, le=500)
    filters: SearchFilters | None = None

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        stripped = query.strip()
        if not stripped:
            raise ValueError("Query must not be empty or whitespace-only")
        return stripped


class TrackSearchRequest(BaseModel):
    track_id: str
    k: int = Field(default=50, ge=1, le=500)
    filters: SearchFilters | None = None

    @field_validator("track_id")
    @classmethod
    def validate_track_id_format(cls, track_id: str) -> str:
        if not TRACK_ID_PATTERN.match(track_id):
            raise ValueError(f"Invalid track ID format: {track_id}")
        return track_id.lower()


class PurgeRequest(BaseModel):
    missing: bool = False
    removed: bool = False
    orphans: bool = False


class InitResponse(BaseModel):
    data_dir: str
    status: str = "initialized"


class HealthResponse(BaseModel):
    status: str
    message: str


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
    model_loading: bool = False
    message: str | None = None
    index_size: int
    tracks_embedded: int


class TrackMetadata(BaseModel):
    track_id: str
    title: str
    artist: str
    album: str
    status: str
    primary_path: str
    duration_ms: int
    embedding_version: str | None = None
    indexed_at: str | None = None


class TrackDetailResponse(BaseModel):
    track: TrackMetadata
    locations: list[dict[str, Any]]


class TracksListResponse(BaseModel):
    items: list[TrackMetadata]
    total: int
    offset: int
    limit: int


class SearchItemResponse(BaseModel):
    track_id: str
    score: float
    rank: int
    match_granularity: str
    metadata: dict[str, Any]


class SearchResponse(BaseModel):
    items: list[SearchItemResponse]
    query: dict[str, Any]
    total_indexed: int
    took_ms: int
