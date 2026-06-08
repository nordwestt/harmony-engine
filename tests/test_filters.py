"""Tests for metadata search filters."""

from __future__ import annotations

from harmony.models import Track, TrackStatus
from harmony.retrieval.filters import Filters, normalize_string_list


def _track(*, artist: str = "Radiohead", album: str = "OK Computer") -> Track:
    return Track(
        track_id="00000000-0000-0000-0000-000000000001",
        content_hash="abc",
        status=TrackStatus.ACTIVE,
        primary_path="/music/track.flac",
        duration_ms=1000,
        title="Track",
        artist=artist,
        album=album,
        embedding_version="v1",
    )


def test_normalize_string_list_strips_and_drops_empty() -> None:
    assert normalize_string_list([" Radiohead ", ""]) == ["Radiohead"]
    assert normalize_string_list(["  ", ""]) is None
    assert normalize_string_list(None) is None


def test_matches_artist_or_within_list() -> None:
    filters = Filters(artists=["Radiohead", "Michael Jackson"])
    assert filters.matches(_track(artist="Radiohead"))
    assert filters.matches(_track(artist="Michael Jackson", album="Thriller"))
    assert not filters.matches(_track(artist="Beck"))


def test_matches_album_or_within_list() -> None:
    filters = Filters(albums=["OK Computer", "Thriller"])
    assert filters.matches(_track(album="OK Computer"))
    assert filters.matches(_track(artist="Michael Jackson", album="Thriller"))
    assert not filters.matches(_track(album="Odelay"))


def test_matches_artist_and_album_combined() -> None:
    filters = Filters(artists=["Radiohead"], albums=["OK Computer"])
    assert filters.matches(_track(artist="Radiohead", album="OK Computer"))
    assert not filters.matches(_track(artist="Radiohead", album="Kid A"))
    assert not filters.matches(_track(artist="Beck", album="OK Computer"))


def test_matches_case_insensitive() -> None:
    filters = Filters(artists=["radiohead"], albums=["ok computer"])
    assert filters.matches(_track(artist="Radiohead", album="OK Computer"))


def test_matches_no_filters_always_true() -> None:
    filters = Filters()
    assert filters.matches(_track())
