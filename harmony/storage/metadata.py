"""Metadata CRUD against Turso."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from harmony.config import Config
from harmony.models import (
    ScannedFile,
    Track,
    TrackLocation,
    TrackStatus,
    track_id_from_content_hash,
    utcnow,
)
from harmony.storage.db import connect, migrate


def _iso(dt: datetime | None) -> str | None:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _parse_iso(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value)


class MetadataStore:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.conn = connect(config.db_path)
        migrate(self.conn)

    def close(self) -> None:
        self.conn.close()

    def count_tracks_by_status(self) -> dict[str, int]:
        rows = self.conn.execute(
            "SELECT status, COUNT(*) AS n FROM tracks GROUP BY status"
        ).fetchall()
        return {row[0]: int(row[1]) for row in rows}

    def get_track_by_content_hash(self, content_hash: str) -> Track | None:
        row = self.conn.execute(
            "SELECT * FROM tracks WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_track(row)

    def get_track_by_path(self, path: str) -> Track | None:
        row = self.conn.execute(
            """
            SELECT t.* FROM tracks t
            JOIN track_locations l ON l.track_id = t.track_id
            WHERE l.path = ?
            """,
            (path,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_track(row)

    def get_track(self, track_id: str) -> Track | None:
        row = self.conn.execute(
            "SELECT * FROM tracks WHERE track_id = ?", (track_id,)
        ).fetchone()
        if row is None:
            return None
        track = self._row_to_track(row)
        track.locations = self.get_locations(track_id)
        return track

    def get_locations(self, track_id: str) -> list[TrackLocation]:
        rows = self.conn.execute(
            "SELECT * FROM track_locations WHERE track_id = ? ORDER BY is_primary DESC",
            (track_id,),
        ).fetchall()
        return [self._row_to_location(r) for r in rows]

    def list_active_tracks(self) -> list[Track]:
        rows = self.conn.execute(
            "SELECT * FROM tracks WHERE status IN ('active', 'missing') ORDER BY artist, album, title"
        ).fetchall()
        return [self._row_to_track(r) for r in rows]

    def list_tracks_pending_embedding(self) -> list[Track]:
        version = self.config.embedding_version()
        rows = self.conn.execute(
            """
            SELECT * FROM tracks
            WHERE status IN ('active', 'failed')
              AND (indexed_at IS NULL OR embedding_version != ?)
            ORDER BY artist, album, title
            """,
            (version,),
        ).fetchall()
        return [self._row_to_track(r) for r in rows]

    def list_embedded_tracks(self) -> list[Track]:
        version = self.config.embedding_version()
        rows = self.conn.execute(
            """
            SELECT * FROM tracks
            WHERE status = 'active'
              AND indexed_at IS NOT NULL
              AND embedding_version = ?
            ORDER BY artist, album, title
            """,
            (version,),
        ).fetchall()
        return [self._row_to_track(r) for r in rows]

    def count_embedded_tracks(self) -> int:
        version = self.config.embedding_version()
        row = self.conn.execute(
            """
            SELECT COUNT(*) FROM tracks
            WHERE status = 'active'
              AND indexed_at IS NOT NULL
              AND embedding_version = ?
            """,
            (version,),
        ).fetchone()
        return int(row[0])

    def mark_track_embedded(
        self,
        track_id: str,
        *,
        duration_ms: int,
        version: str,
    ) -> None:
        now_iso = _iso(utcnow())
        self.conn.execute(
            """
            UPDATE tracks SET
                status = ?, indexed_at = ?, embedding_version = ?,
                duration_ms = ?, updated_at = ?
            WHERE track_id = ?
            """,
            (
                TrackStatus.ACTIVE.value,
                now_iso,
                version,
                duration_ms,
                now_iso,
                track_id,
            ),
        )
        self.conn.commit()

    def mark_track_failed(self, track_id: str, when: datetime) -> None:
        now_iso = _iso(when)
        self.conn.execute(
            "UPDATE tracks SET status = ?, updated_at = ? WHERE track_id = ?",
            (TrackStatus.FAILED.value, now_iso, track_id),
        )
        self.conn.commit()

    def upsert_scanned_file(self, scanned: ScannedFile, *, now: datetime | None = None) -> str:
        """Register or update a scanned file. Returns track_id."""
        now = now or utcnow()
        now_iso = _iso(now)
        assert now_iso is not None

        existing = self.get_track_by_content_hash(scanned.content_hash)
        track_id = existing.track_id if existing else track_id_from_content_hash(scanned.content_hash)
        title = scanned.title or Path(scanned.path).stem
        artist = scanned.artist or "Unknown Artist"
        album = scanned.album or "Unknown Album"
        duration_ms = scanned.duration_ms or 0
        embedding_version = self.config.embedding_version()

        if existing is None:
            self.conn.execute(
                """
                INSERT INTO tracks (
                    track_id, content_hash, status, primary_path,
                    duration_ms, title, artist, album, embedding_version,
                    indexed_at, last_seen_at, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
                """,
                (
                    track_id,
                    scanned.content_hash,
                    TrackStatus.ACTIVE.value,
                    scanned.path,
                    duration_ms,
                    title,
                    artist,
                    album,
                    embedding_version,
                    now_iso,
                    now_iso,
                    now_iso,
                ),
            )
        else:
            self.conn.execute(
                """
                UPDATE tracks SET
                    status = ?, primary_path = ?, last_seen_at = ?, updated_at = ?,
                    title = ?, artist = ?, album = ?, duration_ms = ?
                WHERE track_id = ?
                """,
                (
                    TrackStatus.ACTIVE.value,
                    scanned.path,
                    now_iso,
                    now_iso,
                    title,
                    artist,
                    album,
                    duration_ms,
                    track_id,
                ),
            )

        self._upsert_location(track_id, scanned.path, now=now)
        self.conn.commit()
        return track_id

    def list_track_ids_by_status(self, status: TrackStatus) -> list[str]:
        rows = self.conn.execute(
            "SELECT track_id FROM tracks WHERE status = ?",
            (status.value,),
        ).fetchall()
        return [row[0] for row in rows]

    def list_all_track_ids(self) -> list[str]:
        rows = self.conn.execute("SELECT track_id FROM tracks").fetchall()
        return [row[0] for row in rows]

    def set_tracks_status(self, track_ids: list[str], status: TrackStatus) -> None:
        if not track_ids:
            return
        now_iso = _iso(utcnow())
        for track_id in track_ids:
            self.conn.execute(
                "UPDATE tracks SET status = ?, updated_at = ? WHERE track_id = ?",
                (status.value, now_iso, track_id),
            )
        self.conn.commit()

    def delete_tracks(self, track_ids: list[str]) -> int:
        if not track_ids:
            return 0
        for track_id in track_ids:
            self.conn.execute("DELETE FROM chunks WHERE track_id = ?", (track_id,))
            self.conn.execute("DELETE FROM track_locations WHERE track_id = ?", (track_id,))
            self.conn.execute("DELETE FROM path_history WHERE track_id = ?", (track_id,))
            self.conn.execute("DELETE FROM embedding_jobs WHERE track_id = ?", (track_id,))
            self.conn.execute("DELETE FROM tracks WHERE track_id = ?", (track_id,))
        self.conn.commit()
        return len(track_ids)

    def mark_missing_not_in_scan(self, seen_paths: set[str]) -> int:
        """Mark active tracks whose primary path wasn't seen in the latest scan."""
        now_iso = _iso(utcnow())
        rows = self.conn.execute(
            "SELECT track_id, primary_path FROM tracks WHERE status = 'active'"
        ).fetchall()

        count = 0
        for row in rows:
            track_id = row[0]
            primary_path = row[1]
            if primary_path not in seen_paths:
                self.conn.execute(
                    "UPDATE tracks SET status = 'missing', updated_at = ? WHERE track_id = ?",
                    (now_iso, track_id),
                )
                count += 1
        self.conn.commit()
        return count

    def expire_missing_tracks(self) -> int:
        """Transition missing tracks past grace period to removed."""
        cutoff = utcnow() - timedelta(days=self.config.sync.missing_grace_days)
        cutoff_iso = _iso(cutoff)
        assert cutoff_iso is not None

        rows = self.conn.execute(
            """
            SELECT track_id FROM tracks
            WHERE status = 'missing' AND last_seen_at < ?
            """,
            (cutoff_iso,),
        ).fetchall()

        now_iso = _iso(utcnow())
        for row in rows:
            self.conn.execute(
                "UPDATE tracks SET status = 'removed', updated_at = ? WHERE track_id = ?",
                (now_iso, row[0]),
            )
        self.conn.commit()
        return len(rows)

    def record_sync_run(self, run_id: str, report: Any, started_at: datetime) -> None:
        from harmony.models import SyncReport

        assert isinstance(report, SyncReport)
        self.conn.execute(
            """
            INSERT INTO sync_runs (
                run_id, started_at, finished_at,
                added, updated_metadata, moved, duplicates_found,
                missing, removed, reembedded, failed, skipped, duration_ms
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                _iso(started_at),
                _iso(utcnow()),
                report.added,
                report.updated_metadata,
                report.moved,
                report.duplicates_found,
                report.missing,
                report.removed,
                report.reembedded,
                report.failed,
                report.skipped,
                report.duration_ms,
            ),
        )
        self.conn.commit()

    def _upsert_location(self, track_id: str, path: str, *, now: datetime) -> None:
        now_iso = _iso(now)
        assert now_iso is not None

        existing = self.conn.execute(
            "SELECT location_id, path FROM track_locations WHERE track_id = ? AND is_primary = 1",
            (track_id,),
        ).fetchone()

        if existing and existing[1] != path:
            self.conn.execute(
                """
                INSERT INTO path_history (track_id, old_path, new_path, changed_at, reason)
                VALUES (?, ?, ?, ?, ?)
                """,
                (track_id, existing[1], path, now_iso, "moved"),
            )

        row = self.conn.execute(
            "SELECT location_id FROM track_locations WHERE path = ?",
            (path,),
        ).fetchone()

        if row is None:
            location_id = str(uuid.uuid4())
            self.conn.execute(
                """
                INSERT INTO track_locations (
                    location_id, track_id, path, is_primary,
                    first_seen_at, last_seen_at
                ) VALUES (?, ?, ?, 1, ?, ?)
                """,
                (location_id, track_id, path, now_iso, now_iso),
            )
        else:
            self.conn.execute(
                """
                UPDATE track_locations SET
                    track_id = ?, last_seen_at = ?, is_primary = 1
                WHERE location_id = ?
                """,
                (track_id, now_iso, row[0]),
            )

        self.conn.execute(
            "UPDATE track_locations SET is_primary = 0 WHERE track_id = ? AND path != ?",
            (track_id, path),
        )

    def _row_to_track(self, row: Any) -> Track:
        keys = row.keys() if hasattr(row, "keys") else None
        get = (lambda k: row[k]) if keys else (lambda k: row[_TRACK_COLS.index(k)])

        extra_raw = get("extra_json")
        return Track(
            track_id=get("track_id"),
            content_hash=get("content_hash"),
            status=TrackStatus(get("status")),
            primary_path=get("primary_path"),
            duration_ms=int(get("duration_ms")),
            title=get("title"),
            artist=get("artist"),
            album=get("album"),
            album_artist=get("album_artist"),
            year=get("year"),
            genre=get("genre"),
            disc_number=get("disc_number"),
            track_number=get("track_number"),
            extra=json.loads(extra_raw or "{}"),
            indexed_at=_parse_iso(get("indexed_at")),
            last_seen_at=_parse_iso(get("last_seen_at")),
            embedding_version=get("embedding_version"),
        )

    def _row_to_location(self, row: Any) -> TrackLocation:
        keys = row.keys() if hasattr(row, "keys") else None
        get = (lambda k: row[k]) if keys else (lambda k: row[_LOCATION_COLS.index(k)])

        return TrackLocation(
            location_id=get("location_id"),
            track_id=get("track_id"),
            path=get("path"),
            is_primary=bool(get("is_primary")),
            first_seen_at=_parse_iso(get("first_seen_at")) or utcnow(),
            last_seen_at=_parse_iso(get("last_seen_at")) or utcnow(),
        )


_TRACK_COLS = [
    "track_id", "content_hash", "status", "primary_path",
    "duration_ms", "title", "artist", "album", "album_artist", "year", "genre",
    "disc_number", "track_number", "extra_json", "indexed_at", "last_seen_at",
    "embedding_version", "created_at", "updated_at",
]

_LOCATION_COLS = [
    "location_id", "track_id", "path", "is_primary",
    "first_seen_at", "last_seen_at",
]
