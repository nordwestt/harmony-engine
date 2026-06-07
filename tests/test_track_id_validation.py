"""Tests for track ID validation."""

from __future__ import annotations

import uuid

import pytest

from harmony.errors import InvalidTrackIdError
from harmony.models import TRACK_ID_NAMESPACE, track_id_from_content_hash, validate_track_id


def test_validate_track_id_accepts_uuid() -> None:
    track_id = track_id_from_content_hash("abc123")
    assert validate_track_id(track_id) == track_id.lower()


def test_validate_track_id_rejects_path_traversal() -> None:
    with pytest.raises(InvalidTrackIdError):
        validate_track_id("../../secrets")


def test_validate_track_id_rejects_garbage() -> None:
    with pytest.raises(InvalidTrackIdError):
        validate_track_id("not-a-uuid")


def test_validate_track_id_normalizes_case() -> None:
    raw = str(uuid.uuid5(TRACK_ID_NAMESPACE, "hash"))
    upper = raw.upper()
    assert validate_track_id(upper) == raw.lower()
