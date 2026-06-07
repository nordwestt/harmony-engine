"""Library reconciliation: adds, moves, removes."""

from __future__ import annotations

import time
import uuid
from datetime import datetime

from harmony.config import Config
from harmony.models import ScannedFile, SyncReport, utcnow
from harmony.scanner.base import FilesystemScannerProtocol
from harmony.storage.metadata import MetadataStore


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
            else:
                report.skipped += 1

            self.store.upsert_scanned_file(scanned)

        report.missing = self.store.mark_missing_not_in_scan(seen_paths)
        report.removed = self.store.expire_missing_tracks()
        report.duration_ms = int((time.perf_counter() - t0) * 1000)

        self.store.record_sync_run(str(uuid.uuid4()), report, started_at=started)
        return report
