"""Library reconciliation: adds, moves, removes."""

from __future__ import annotations

import time
import uuid
from datetime import datetime
from pathlib import Path

from harmony.config import Config
from harmony.models import ScannedFile, SyncReport, Track, utcnow
from harmony.scanner.base import FilesystemScannerProtocol
from harmony.storage.metadata import MetadataStore


def _resolved_metadata(scanned: ScannedFile) -> tuple[str, str, str, int]:
    title = scanned.title or Path(scanned.path).stem
    artist = scanned.artist or "Unknown Artist"
    album = scanned.album or "Unknown Album"
    duration_ms = scanned.duration_ms or 0
    return title, artist, album, duration_ms


def _metadata_changed(existing: Track, scanned: ScannedFile) -> bool:
    title, artist, album, duration_ms = _resolved_metadata(scanned)
    return (
        existing.title != title
        or existing.artist != artist
        or existing.album != album
        or existing.album_artist != scanned.album_artist
        or existing.year != scanned.year
        or existing.genre != scanned.genre
        or existing.disc_number != scanned.disc_number
        or existing.track_number != scanned.track_number
        or existing.duration_ms != duration_ms
    )


class LibrarySync:
    def __init__(self, config: Config, store: MetadataStore) -> None:
        self.config = config
        self.store = store

    def reconcile(self, scanner: FilesystemScannerProtocol) -> SyncReport:
        """Compare filesystem scan against stored state and update metadata."""
        started = utcnow()
        t0 = time.perf_counter()
        report = SyncReport()

        scanned_files = list(scanner.scan())
        seen_paths: set[str] = set()

        for scanned in scanned_files:
            seen_paths.add(scanned.path)
            existing = self.store.get_track_by_content_hash(scanned.content_hash)

            if existing is None:
                by_path = self.store.get_track_by_path(scanned.path)
                if by_path and by_path.content_hash != scanned.content_hash:
                    # Same path, different content — handled as new track for now.
                    report.reembedded += 1
                else:
                    report.added += 1
            elif existing.primary_path != scanned.path:
                report.moved += 1
            elif _metadata_changed(existing, scanned):
                report.updated_metadata += 1
            else:
                report.skipped += 1

            self.store.upsert_scanned_file(scanned)

        report.missing = self.store.mark_missing_not_in_scan(seen_paths)
        report.removed = self.store.expire_missing_tracks()
        report.duration_ms = int((time.perf_counter() - t0) * 1000)

        self.store.record_sync_run(str(uuid.uuid4()), report, started_at=started)
        return report
