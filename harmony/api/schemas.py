"""Pydantic models for the Harmony HTTP API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error: str
    code: str


class IndexRequest(BaseModel):
    paths: list[str] | None = None
    full_rescan: bool = False
    embed: bool = True
    prune: bool = False
    reembed: bool = False
    async_: bool = Field(default=False, alias="async")


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
    query: str
    k: int = Field(default=50, ge=1, le=500)


class TrackSearchRequest(BaseModel):
    track_id: str
    k: int = Field(default=50, ge=1, le=500)


class PurgeRequest(BaseModel):
    missing: bool = False
    removed: bool = False
    orphans: bool = False


class InitResponse(BaseModel):
    data_dir: str
    status: str = "initialized"


class ReadyResponse(BaseModel):
    ready: bool
    model_loaded: bool
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
