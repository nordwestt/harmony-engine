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

INTERRUPTED_MESSAGE = "Server restarted while job was in progress"


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
    params: dict[str, Any] | None = None
    embed_base: int = 0
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


def _job_params(
    *,
    paths: list[str | Path] | None,
    full_rescan: bool,
    embed: bool,
    prune: bool,
    reembed: bool,
) -> dict[str, Any]:
    return {
        "paths": [str(p) for p in paths] if paths else None,
        "full_rescan": full_rescan,
        "embed": embed,
        "prune": prune,
        "reembed": reembed,
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

    def _has_active_job_locked(self) -> bool:
        if self._is_job_active(self._active_job_id):
            return True
        return self._engine.store.has_active_index_job()

    @property
    def has_active_job(self) -> bool:
        with self._lock:
            return self._has_active_job_locked()

    def recover_stale_jobs(self) -> int:
        """Mark in-flight jobs as interrupted after a process restart."""
        stale = self._engine.store.list_index_jobs(status_in=("pending", "running"))
        if not stale:
            return 0

        now = utcnow().isoformat()
        for row in stale:
            job_id = row["job_id"]
            logger.warning("Marking stale index job %s as interrupted", job_id)
            state = self._state_from_row(row)
            state.status = "interrupted"
            state.error = INTERRUPTED_MESSAGE
            state.finished_at = now
            state.updated_at = now
            if not state.params:
                state.params = self._resolve_resume_params(state)
            self._jobs[job_id] = state
            self._persist_job(state)
            if self._active_job_id == job_id:
                self._active_job_id = None

        return len(stale)

    def maybe_resume_interrupted_job(self) -> IndexJobState | None:
        """Resume the latest interrupted embed job when tracks remain pending."""
        if not self._engine.config.jobs.resume_on_startup:
            return None
        if self.has_active_job:
            return None

        row = self._engine.store.get_latest_index_job(status="interrupted")
        if row is None:
            return None

        state = self._state_from_row(row)
        params = self._resolve_resume_params(state)
        if not params.get("embed"):
            return None

        try:
            state = self.resume(row["job_id"])
            logger.info("Auto-resuming interrupted index job %s", state.job_id)
            return state
        except (RuntimeError, ValueError) as e:
            logger.warning("Could not auto-resume job %s: %s", row["job_id"], e)
            return None

    def start(
        self,
        *,
        paths: list[str | Path] | None = None,
        full_rescan: bool = False,
        embed: bool = True,
        prune: bool = False,
        reembed: bool = False,
    ) -> IndexJobState:
        params = _job_params(
            paths=paths,
            full_rescan=full_rescan,
            embed=embed,
            prune=prune,
            reembed=reembed,
        )
        with self._lock:
            if self._has_active_job_locked():
                raise RuntimeError("index_job_running")

            job_id = str(uuid.uuid4())
            state = IndexJobState(job_id=job_id, params=params)
            self._jobs[job_id] = state
            self._active_job_id = job_id
            self._persist_job(state)

        self._spawn_worker(job_id, params)
        return state

    def resume(self, job_id: str) -> IndexJobState:
        """Continue an interrupted job under the same job_id."""
        with self._lock:
            if self._has_active_job_locked():
                raise RuntimeError("index_job_running")

            state = self._load_job(job_id)
            if state is None:
                raise ValueError(f"Job not found: {job_id}")
            if state.status != "interrupted":
                raise ValueError(f"Job cannot be resumed (status={state.status})")

            params = self._resolve_resume_params(state)
            if not params.get("embed"):
                raise ValueError(
                    "Job cannot be resumed (original job did not include embedding). "
                    "Start a new index with {\"embed\": true}."
                )

            state.embed_base = state.embedded
            state.status = "pending"
            state.error = None
            state.finished_at = None
            state.phase = None
            state.params = params
            state.updated_at = utcnow().isoformat()
            self._jobs[job_id] = state
            self._active_job_id = job_id
            self._persist_job(state)

        self._spawn_worker(job_id, params)
        return state

    def get(self, job_id: str) -> IndexJobState | None:
        with self._lock:
            if job_id in self._jobs:
                return self._jobs[job_id]
        return self._load_job(job_id)

    def _spawn_worker(self, job_id: str, params: dict[str, Any]) -> None:
        thread = threading.Thread(
            target=self._run_job,
            args=(
                job_id,
                params.get("paths"),
                params.get("full_rescan", False),
                params.get("embed", True),
                params.get("prune", False),
                params.get("reembed", False),
            ),
            daemon=True,
            name=f"harmony-index-{job_id[:8]}",
        )
        thread.start()

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
        state.started_at = state.started_at or utcnow().isoformat()
        state.updated_at = state.started_at
        self._persist_job(state)

        base = state.embed_base

        def on_progress(done: int, total: int, _label: str) -> None:
            state.phase = "embed"
            state.embedded = base + done
            state.total_pending = base + total
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
            if embed:
                state.embedded = base + report.embedded
                if state.total_pending < state.embedded:
                    state.total_pending = state.embedded
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
            params_json=json.dumps(state.params) if state.params else None,
            created_at=state.created_at,
            updated_at=state.updated_at,
            started_at=state.started_at,
            finished_at=state.finished_at,
        )

    def _resolve_resume_params(self, state: IndexJobState) -> dict[str, Any]:
        """Return stored job params, or reconstruct defaults for legacy jobs."""
        if state.params:
            return state.params

        logger.info(
            "Reconstructing missing params for interrupted job %s",
            state.job_id,
        )
        embed = (
            state.phase == "embed"
            or state.embedded > 0
            or state.total_pending > 0
            or self._engine.store.has_tracks_pending_embedding()
        )
        # Legacy async index jobs almost always included embedding.
        if state.phase in ("scan", None) and state.embedded == 0:
            embed = True

        return _job_params(
            paths=None,
            full_rescan=False,
            embed=embed,
            prune=False,
            reembed=False,
        )

    def _parse_params(self, raw: str | None) -> dict[str, Any]:
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Ignoring invalid params_json on index job: %r", raw)
            return {}
        return parsed if isinstance(parsed, dict) else {}

    def _state_from_row(self, row: dict[str, Any]) -> IndexJobState:
        report = None
        if row.get("report_json"):
            try:
                report = json.loads(row["report_json"])
            except json.JSONDecodeError:
                logger.warning(
                    "Ignoring invalid report_json on job %s",
                    row.get("job_id"),
                )
        return IndexJobState(
            job_id=row["job_id"],
            status=row["status"],
            phase=row.get("phase"),
            embedded=int(row.get("embedded") or 0),
            total_pending=int(row.get("total_pending") or 0),
            failed=int(row.get("failed") or 0),
            error=row.get("error"),
            report=report,
            params=self._parse_params(row.get("params_json")),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row.get("started_at"),
            finished_at=row.get("finished_at"),
        )

    def _load_job(self, job_id: str) -> IndexJobState | None:
        row = self._engine.store.get_index_job(job_id)
        if row is None:
            return None
        return self._state_from_row(row)
