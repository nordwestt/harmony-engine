"""Background index job runner with single-writer lock."""

from __future__ import annotations

import json
import logging
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from harmony.engine import Engine
from harmony.models import SyncReport, utcnow

logger = logging.getLogger(__name__)


@dataclass
class IndexJobState:
    job_id: str
    status: str = "pending"
    phase: str | None = None
    embedded: int = 0
    total_pending: int = 0
    failed: int = 0
    error: str | None = None
    report: dict[str, Any] | None = None
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: utcnow().isoformat())
    started_at: str | None = None
    finished_at: str | None = None


def _sync_report_to_dict(report: SyncReport) -> dict[str, Any]:
    return {
        "added": report.added,
        "moved": report.moved,
        "skipped": report.skipped,
        "missing": report.missing,
        "removed": report.removed,
        "embedded": report.embedded,
        "purged": report.purged,
        "failed": report.failed,
        "duration_ms": report.duration_ms,
    }


class IndexJobRunner:
    """Runs index jobs in background threads; one active job per data directory."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._lock = threading.Lock()
        self._active_job_id: str | None = None
        self._jobs: dict[str, IndexJobState] = {}

    def _is_job_active(self, job_id: str | None) -> bool:
        if job_id is None:
            return False
        job = self._jobs.get(job_id)
        return job is not None and job.status in ("pending", "running")

    @property
    def has_active_job(self) -> bool:
        with self._lock:
            return self._is_job_active(self._active_job_id)

    def start(
        self,
        *,
        paths: list[str | Path] | None = None,
        full_rescan: bool = False,
        embed: bool = True,
        prune: bool = False,
        reembed: bool = False,
    ) -> IndexJobState:
        with self._lock:
            if self._is_job_active(self._active_job_id):
                raise RuntimeError("index_job_running")

            job_id = str(uuid.uuid4())
            state = IndexJobState(job_id=job_id)
            self._jobs[job_id] = state
            self._active_job_id = job_id
            self._persist_job(state)

        thread = threading.Thread(
            target=self._run_job,
            args=(job_id, paths, full_rescan, embed, prune, reembed),
            daemon=True,
            name=f"harmony-index-{job_id[:8]}",
        )
        thread.start()
        return state

    def get(self, job_id: str) -> IndexJobState | None:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]
        return self._load_job(job_id)

    def _run_job(
        self,
        job_id: str,
        paths: list[str | Path] | None,
        full_rescan: bool,
        embed: bool,
        prune: bool,
        reembed: bool,
    ) -> None:
        state = self._jobs[job_id]
        state.status = "running"
        state.phase = "scan"
        state.started_at = utcnow().isoformat()
        state.updated_at = state.started_at
        self._persist_job(state)

        def on_progress(done: int, total: int, _label: str) -> None:
            state.phase = "embed"
            state.embedded = done
            state.total_pending = total
            state.updated_at = utcnow().isoformat()
            self._persist_job(state)

        try:
            report = self._engine.index(
                paths=paths,
                full_rescan=full_rescan,
                embed=embed,
                prune=prune,
                reembed=reembed,
                on_embed_progress=on_progress if embed else None,
            )
            state.status = "completed"
            state.phase = "done"
            state.embedded = report.embedded
            state.failed = report.failed
            state.report = _sync_report_to_dict(report)
        except Exception as e:
            logger.exception("Index job %s failed", job_id)
            state.status = "failed"
            state.error = str(e)
        finally:
            state.finished_at = utcnow().isoformat()
            state.updated_at = state.finished_at
            self._persist_job(state)
            with self._lock:
                if self._active_job_id == job_id:
                    self._active_job_id = None

    def _persist_job(self, state: IndexJobState) -> None:
        self._engine.store.upsert_index_job(
            job_id=state.job_id,
            status=state.status,
            phase=state.phase,
            embedded=state.embedded,
            total_pending=state.total_pending,
            failed=state.failed,
            error=state.error,
            report_json=json.dumps(state.report) if state.report else None,
            created_at=state.created_at,
            updated_at=state.updated_at,
            started_at=state.started_at,
            finished_at=state.finished_at,
        )

    def _load_job(self, job_id: str) -> IndexJobState | None:
        row = self._engine.store.get_index_job(job_id)
        if row is None:
            return None
        report = json.loads(row["report_json"]) if row.get("report_json") else None
        return IndexJobState(
            job_id=row["job_id"],
            status=row["status"],
            phase=row.get("phase"),
            embedded=int(row.get("embedded") or 0),
            total_pending=int(row.get("total_pending") or 0),
            failed=int(row.get("failed") or 0),
            error=row.get("error"),
            report=report,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
        )
