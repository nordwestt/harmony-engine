"""Tests for database layer."""

from pathlib import Path

from harmony.storage.db import connect, migrate


def test_migrate_creates_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "test.db"
    conn = connect(db_path)
    migrate(conn)

    tables = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    names = {row[0] for row in tables}
    assert "tracks" in names
    assert "track_locations" in names
    assert "sync_runs" in names
    conn.close()
