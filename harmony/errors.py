"""Domain exceptions for Harmony Engine."""

from __future__ import annotations


class HarmonyError(Exception):
    """Base class for domain errors with HTTP mapping."""

    http_status: int = 500
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class IndexEmptyError(HarmonyError):
    """No embedded tracks in the search index."""

    http_status = 503
    code = "index_not_ready"


class ModelNotReadyError(HarmonyError):
    """Embedding model is loading or failed to load."""

    http_status = 503
    code = "model_not_ready"


class DependencyMissingError(HarmonyError):
    """Required optional dependency is not installed."""

    http_status = 503
    code = "dependency_missing"


class PathNotAllowedError(HarmonyError):
    """Index path is outside configured scan roots."""

    http_status = 400
    code = "path_not_allowed"


class InvalidTrackIdError(HarmonyError):
    """Track ID is not a valid UUID."""

    http_status = 400
    code = "invalid_track_id"
