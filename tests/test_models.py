"""Tests for core models."""

from harmony.models import SyncReport, track_id_from_content_hash


def test_track_id_is_stable() -> None:
    h = "abc123" * 8
    assert track_id_from_content_hash(h) == track_id_from_content_hash(h)
    assert track_id_from_content_hash(h) != track_id_from_content_hash(h + "x")


def test_sync_report_total_changes() -> None:
    report = SyncReport(added=2, moved=1, skipped=10)
    assert report.total_changes() == 3
