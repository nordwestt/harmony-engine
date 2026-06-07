"""Tests for background index jobs."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import patch

from harmony.config import Config
from harmony.engine import Engine
from harmony.jobs.runner import IndexJobRunner
from harmony.models import SyncReport


def _wait_for_job(runner: IndexJobRunner, job_id: str, *, timeout: float = 5.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = runner.get(job_id)
        if state is not None and state.status in ("completed", "failed"):
            return
        time.sleep(0.02)
    state = runner.get(job_id)
    status = state.status if state else "missing"
    raise AssertionError(f"job {job_id} did not finish in {timeout}s (status={status})")


def test_index_job_runner_tracks_state(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "data")
    engine.init()
    runner = IndexJobRunner(engine)

    report = SyncReport(added=1, embedded=2, duration_ms=100)

    with patch.object(engine, "index", return_value=report):
        state = runner.start(paths=["/music"], embed=False)
        _wait_for_job(runner, state.job_id)
        state = runner.get(state.job_id)
        assert state is not None
        assert state.status == "completed"
        assert state.report is not None
        assert state.report["embedded"] == 2
    engine.close()


def test_index_job_single_writer(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "data")
    engine.init()
    runner = IndexJobRunner(engine)
    running = threading.Event()
    release = threading.Event()

    def slow_index(**_kwargs: object) -> SyncReport:
        running.set()
        assert release.wait(timeout=5)
        return SyncReport()

    with patch.object(engine, "index", side_effect=slow_index):
        state = runner.start(embed=False)
        assert running.wait(timeout=5)
        try:
            runner.start(embed=False)
            raised = False
        except RuntimeError as e:
            raised = True
            assert str(e) == "index_job_running"
        assert raised
        release.set()
        _wait_for_job(runner, state.job_id)
    engine.close()


def test_index_job_persisted(tmp_path: Path) -> None:
    cfg = Config(data_dir=tmp_path / "data")
    engine = Engine(cfg.data_dir)
    engine.init()
    runner = IndexJobRunner(engine)

    with patch.object(engine, "index", return_value=SyncReport(embedded=1)):
        state = runner.start(embed=False)
        _wait_for_job(runner, state.job_id)
        state = runner.get(state.job_id)
        assert state is not None

    row = engine.store.get_index_job(state.job_id)
    assert row is not None
    assert row["status"] == "completed"
    engine.close()
