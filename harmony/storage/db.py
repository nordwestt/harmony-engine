"""Turso / SQLite database connection and migrations."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2
_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def connect(db_path: Path | str) -> Any:
    """Open a database connection.

    Uses pyturso when available, falls back to stdlib sqlite3.
    Returns a connection with sqlite3-compatible API.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import turso  # type: ignore[import-untyped]

        return turso.connect(str(db_path))
    except ImportError:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn


def migrate(conn: Any) -> None:
    """Apply schema migrations idempotently."""
    current = _get_schema_version(conn)
    if current is None:
        _apply_schema(conn)
        _set_schema_version(conn, SCHEMA_VERSION)
        conn.commit()
        return

    if current < SCHEMA_VERSION:
        # Future migrations go here.
        _set_schema_version(conn, SCHEMA_VERSION)
        conn.commit()


def _get_schema_version(conn: Any) -> int | None:
    try:
        row = conn.execute("SELECT version FROM schema_version LIMIT 1").fetchone()
    except Exception:
        return None
    if row is None:
        return None
    return int(row[0] if not hasattr(row, "keys") else row["version"])


def _set_schema_version(conn: Any, version: int) -> None:
    conn.execute("DELETE FROM schema_version")
    conn.execute("INSERT INTO schema_version (version) VALUES (?)", (version,))


def _apply_schema(conn: Any) -> None:
    sql = _SCHEMA_PATH.read_text(encoding="utf-8")
    conn.executescript(sql)
