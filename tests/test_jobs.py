"""Tests for background index jobs."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from harmony.config import Config
from harmony.engine import Engine
from harmony.jobs.runner import INTERRUPTED_MESSAGE, IndexJobRunner
from harmony.models import SyncReport, track_id_from_content_hash, utcnow

TRACK_ID = track_id_from_content_hash("job-test-track")


def _wait_for_job(
    runner: IndexJobRunner,
    job_id: str,
    *,
    timeout: float = 5.0,
    terminal: tuple[str, ...] = ("completed", "failed", "interrupted"),
) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        state = runner.get(job_id)
        if state is not None and state.status in terminal:
            return
        time.sleep(0.02)
    state = runner.get(job_id)
    status = state.status if state else "missing"
    raise AssertionError(f"job {job_id} did not finish in {timeout}s (status={status})")


def _insert_pending_track(engine: Engine, path: str) -> None:
    now = utcnow().isoformat()
    version = engine.config.embedding_version()
    engine.store.conn.execute(
        """
        INSERT INTO tracks (
            track_id, content_hash, status, primary_path,
            duration_ms, title, artist, album, embedding_version,
            indexed_at, last_seen_at, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, ?, ?, ?)
        """,
        (
            TRACK_ID,
            "hash-job-test",
            "active",
            path,
            0,
            "Tone",
            "Test",
            "Album",
            version,
            now,
            now,
            now,
        ),
    )
    engine.store.conn.commit()


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
    assert row["params_json"] is not None
    engine.close()


def test_recover_stale_jobs_with_v2_column_order(tmp_path: Path) -> None:
    """Migrated DBs append params_json last; reads must not confuse it with created_at."""
    import sqlite3

    from harmony.storage.db import connect, migrate

    db_path = tmp_path / "data" / "harmony.db"
    db_path.parent.mkdir(parents=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        CREATE TABLE schema_version (version INTEGER NOT NULL)
        """
    )
    conn.execute("INSERT INTO schema_version (version) VALUES (2)")
    conn.execute(
        """
        CREATE TABLE index_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            phase TEXT,
            embedded INTEGER NOT NULL DEFAULT 0,
            total_pending INTEGER NOT NULL DEFAULT 0,
            failed INTEGER NOT NULL DEFAULT 0,
            error TEXT,
            report_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            started_at TEXT,
            finished_at TEXT
        )
        """
    )
    conn.commit()
    conn.close()

    conn = connect(db_path)
    migrate(conn)
    now = "2024-06-07T12:00:00+00:00"
    params = json.dumps({"embed": True, "paths": ["/music"]})
    conn.execute(
        """
        INSERT INTO index_jobs (
            job_id, status, phase, embedded, total_pending, failed,
            error, report_json, created_at, updated_at, started_at,
            finished_at, params_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "v2-job",
            "running",
            "embed",
            1,
            5,
            0,
            None,
            None,
            now,
            now,
            now,
            None,
            params,
        ),
    )
    conn.commit()
    conn.close()

    engine = Engine(tmp_path / "data")
    runner = IndexJobRunner(engine)
    count = runner.recover_stale_jobs()
    assert count == 1
    row = engine.store.get_index_job("v2-job")
    assert row is not None
    assert row["status"] == "interrupted"
    assert json.loads(row["params_json"]) == {"embed": True, "paths": ["/music"]}
    engine.close()


def test_recover_stale_jobs_marks_interrupted(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "data")
    engine.init()
    runner = IndexJobRunner(engine)
    now = utcnow().isoformat()

    engine.store.upsert_index_job(
        job_id="stale-job",
        status="running",
        phase="embed",
        embedded=3,
        total_pending=10,
        failed=0,
        error=None,
        report_json=None,
        params_json=json.dumps({"embed": True, "paths": None}),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=None,
    )

    count = runner.recover_stale_jobs()
    assert count == 1

    row = engine.store.get_index_job("stale-job")
    assert row is not None
    assert row["status"] == "interrupted"
    assert row["error"] == INTERRUPTED_MESSAGE
    assert row["finished_at"] is not None
    engine.close()


def test_db_active_lock_blocks_new_runner(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "data")
    engine.init()
    now = utcnow().isoformat()

    engine.store.upsert_index_job(
        job_id="db-running",
        status="running",
        phase="embed",
        embedded=0,
        total_pending=5,
        failed=0,
        error=None,
        report_json=None,
        params_json=None,
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=None,
    )

    runner = IndexJobRunner(engine)
    with pytest.raises(RuntimeError, match="index_job_running"):
        runner.start(embed=False)
    engine.close()


def test_resume_progress_is_cumulative(tmp_path: Path) -> None:
    """Resumed jobs continue embedded/total_pending from pre-interrupt values."""
    engine = Engine(tmp_path / "data")
    engine.init()
    music = tmp_path / "music"
    music.mkdir()
    _insert_pending_track(engine, str(music / "song.flac"))

    now = utcnow().isoformat()
    params = {"paths": [str(music)], "embed": True, "full_rescan": False, "prune": False, "reembed": False}
    engine.store.upsert_index_job(
        job_id="cumulative",
        status="interrupted",
        phase="embed",
        embedded=12,
        total_pending=22,
        failed=0,
        error=INTERRUPTED_MESSAGE,
        report_json=None,
        params_json=json.dumps(params),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=now,
    )

    runner = IndexJobRunner(engine)
    snapshots: list[tuple[int, int]] = []

    def fake_index(**kwargs: object) -> SyncReport:
        cb = kwargs.get("on_embed_progress")
        if cb:
            cb(0, 10, "")
            state = runner.get("cumulative")
            assert state is not None
            snapshots.append((state.embedded, state.total_pending))
            cb(2, 10, "a")
            state = runner.get("cumulative")
            assert state is not None
            snapshots.append((state.embedded, state.total_pending))
        return SyncReport(embedded=2, failed=0)

    with patch.object(engine, "index", side_effect=fake_index):
        runner.resume("cumulative")
        _wait_for_job(runner, "cumulative")

    assert snapshots[0] == (12, 22)
    assert snapshots[1] == (14, 22)
    final = runner.get("cumulative")
    assert final is not None
    assert final.embedded == 14
    assert final.total_pending == 22
    engine.close()


def test_resume_legacy_interrupted_job_without_params(tmp_path: Path) -> None:
    """Jobs interrupted before params_json existed can still resume."""
    engine = Engine(tmp_path / "data")
    engine.init()
    music = tmp_path / "music"
    music.mkdir()
    engine.config.filesystem.paths = [str(music)]
    engine.config.save()
    _insert_pending_track(engine, str(music / "song.flac"))

    now = utcnow().isoformat()
    engine.store.upsert_index_job(
        job_id="legacy-interrupted",
        status="interrupted",
        phase="embed",
        embedded=5,
        total_pending=20,
        failed=0,
        error=INTERRUPTED_MESSAGE,
        report_json=None,
        params_json=None,
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=now,
    )

    runner = IndexJobRunner(engine)
    with patch.object(engine, "index", return_value=SyncReport(embedded=15)) as mock_index:
        state = runner.resume("legacy-interrupted")
        assert state.job_id == "legacy-interrupted"
        _wait_for_job(runner, "legacy-interrupted")
        final = runner.get("legacy-interrupted")
        assert final is not None
        assert final.status == "completed"
        assert final.params is not None
        assert final.params["embed"] is True
        mock_index.assert_called_once()

    row = engine.store.get_index_job("legacy-interrupted")
    assert row is not None
    assert row["params_json"] is not None
    engine.close()


def test_resume_interrupted_job(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "data")
    engine.init()
    music = tmp_path / "music"
    music.mkdir()
    _insert_pending_track(engine, str(music / "song.flac"))

    now = utcnow().isoformat()
    params = {"paths": [str(music)], "full_rescan": False, "embed": True, "prune": False, "reembed": False}
    engine.store.upsert_index_job(
        job_id="resume-me",
        status="interrupted",
        phase="embed",
        embedded=1,
        total_pending=3,
        failed=0,
        error=INTERRUPTED_MESSAGE,
        report_json=None,
        params_json=json.dumps(params),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=now,
    )

    runner = IndexJobRunner(engine)
    report = SyncReport(embedded=2, failed=0)

    with patch.object(engine, "index", return_value=report) as mock_index:
        state = runner.resume("resume-me")
        assert state.job_id == "resume-me"
        _wait_for_job(runner, "resume-me")
        final = runner.get("resume-me")
        assert final is not None
        assert final.status == "completed"
        mock_index.assert_called_once()
        call_kwargs = mock_index.call_args.kwargs
        assert call_kwargs["paths"] == [str(music)]
        assert call_kwargs["embed"] is True

    engine.close()


def test_maybe_resume_on_startup(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "data")
    engine.init()
    engine.config.jobs.resume_on_startup = True
    music = tmp_path / "music"
    music.mkdir()
    _insert_pending_track(engine, str(music / "song.flac"))

    now = utcnow().isoformat()
    params = {"paths": [str(music)], "embed": True, "prune": False, "reembed": False, "full_rescan": False}
    engine.store.upsert_index_job(
        job_id="auto-resume",
        status="interrupted",
        phase="embed",
        embedded=0,
        total_pending=1,
        failed=0,
        error=INTERRUPTED_MESSAGE,
        report_json=None,
        params_json=json.dumps(params),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=now,
    )

    runner = IndexJobRunner(engine)
    with patch.object(engine, "index", return_value=SyncReport(embedded=1)):
        resumed = runner.maybe_resume_interrupted_job()
        assert resumed is not None
        assert resumed.job_id == "auto-resume"
        _wait_for_job(runner, "auto-resume")

    engine.close()


def test_maybe_resume_skips_when_disabled(tmp_path: Path) -> None:
    engine = Engine(tmp_path / "data")
    engine.init()
    engine.config.jobs.resume_on_startup = False
    music = tmp_path / "music"
    music.mkdir()
    _insert_pending_track(engine, str(music / "song.flac"))

    now = utcnow().isoformat()
    engine.store.upsert_index_job(
        job_id="no-auto",
        status="interrupted",
        phase="embed",
        embedded=0,
        total_pending=1,
        failed=0,
        error=INTERRUPTED_MESSAGE,
        report_json=None,
        params_json=json.dumps({"embed": True}),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=now,
    )

    runner = IndexJobRunner(engine)
    assert runner.maybe_resume_interrupted_job() is None
    row = engine.store.get_index_job("no-auto")
    assert row is not None
    assert row["status"] == "interrupted"
    engine.close()


def test_startup_recovery_via_api(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from harmony.api.app import create_app

    data_dir = tmp_path / "data"
    engine = Engine(data_dir)
    engine.init()
    music = tmp_path / "music"
    music.mkdir()
    engine.config.filesystem.paths = [str(music)]
    engine.config.save()
    _insert_pending_track(engine, str(music / "song.flac"))

    now = utcnow().isoformat()
    params = {"paths": [str(music)], "embed": True, "prune": False, "reembed": False, "full_rescan": False}
    engine.store.upsert_index_job(
        job_id="startup-job",
        status="running",
        phase="embed",
        embedded=0,
        total_pending=1,
        failed=0,
        error=None,
        report_json=None,
        params_json=json.dumps(params),
        created_at=now,
        updated_at=now,
        started_at=now,
        finished_at=None,
    )
    engine.close()

    with patch.object(Engine, "index", return_value=SyncReport(embedded=1)) as mock_index:
        app = create_app(data_dir, preload_on_serve=False)
        with TestClient(app) as client:
            deadline = time.time() + 5.0
            status = ""
            while time.time() < deadline:
                row = client.get("/v1/index/jobs/startup-job").json()
                status = row["status"]
                if status == "completed":
                    break
                time.sleep(0.05)
            assert status == "completed"
        mock_index.assert_called()

    fresh = Engine(data_dir)
    job = fresh.store.get_index_job("startup-job")
    assert job is not None
    assert job["status"] == "completed"
    fresh.close()
