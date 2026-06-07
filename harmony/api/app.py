"""FastAPI application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from harmony.api.schemas import (
    ErrorResponse,
    HealthResponse,
    IndexJobResponse,
    IndexJobStatus,
    IndexRequest,
    InitResponse,
    PurgeRequest,
    ReadyResponse,
    SearchResponse,
    SyncReportResponse,
    TextSearchRequest,
    TrackDetailResponse,
    TrackSearchRequest,
    TracksListResponse,
)

from harmony.engine import Engine
from harmony.jobs.runner import IndexJobRunner
from harmony.models import SyncReport

MODEL_LOADING_MESSAGE = (
    "Hey, I'm just busy downloading the weights from Hugging Face - please wait!"
)


def _search_result_to_response(result: Any, query: dict[str, Any]) -> dict[str, Any]:
    return {
        "items": [
            {
                "track_id": i.track_id,
                "score": i.score,
                "rank": i.rank,
                "match_granularity": i.match_granularity,
                "metadata": {
                    "title": i.metadata.title,
                    "artist": i.metadata.artist,
                    "album": i.metadata.album,
                    "primary_path": i.metadata.primary_path,
                    "duration_ms": i.metadata.duration_ms,
                },
            }
            for i in result.items
        ],
        "query": query,
        "total_indexed": result.total_indexed,
        "took_ms": result.took_ms,
    }


def _sync_report_response(report: SyncReport) -> dict[str, Any]:
    return SyncReportResponse(
        added=report.added,
        moved=report.moved,
        skipped=report.skipped,
        missing=report.missing,
        removed=report.removed,
        embedded=report.embedded,
        purged=report.purged,
        failed=report.failed,
        duration_ms=report.duration_ms,
    ).model_dump()


def _job_status_response(state: Any) -> dict[str, Any]:
    return IndexJobStatus(
        job_id=state.job_id,
        status=state.status,
        phase=state.phase,
        embedded=state.embedded,
        total_pending=state.total_pending,
        failed=state.failed,
        error=state.error,
        report=state.report,
        created_at=state.created_at,
        updated_at=state.updated_at,
        started_at=state.started_at,
        finished_at=state.finished_at,
    ).model_dump()


def create_app(
    data_dir: Path | str | None = None,
    *,
    preload_on_serve: bool | None = None,
    engine: Engine | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Harmony Engine",
        version="0.1.0",
        description="Music library indexing and vector search API",
    )
    engine = engine or Engine(data_dir)
    jobs = IndexJobRunner(engine)

    @app.exception_handler(HTTPException)
    async def http_exception_handler(_request: Request, exc: HTTPException) -> JSONResponse:
        code = "http_error"
        if exc.status_code == 409:
            code = "index_job_running"
        elif exc.status_code == 404:
            code = "not_found"
        elif exc.status_code == 400:
            code = "bad_request"
        elif exc.status_code == 501:
            code = "not_implemented"
        detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(error=detail, code=code).model_dump(),
        )

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        _request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=ErrorResponse(error=str(exc), code="validation_error").model_dump(),
        )

    @app.on_event("startup")
    def _startup() -> None:
        engine.ensure_initialized()
        if preload_on_serve is not None:
            engine.config.embedding.preload_on_serve = preload_on_serve
        if engine.config.embedding.preload_on_serve:
            engine.preload_model_background()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        engine.close()

    @app.post("/v1/init", response_model=InitResponse)
    def init_library() -> dict[str, str]:
        """Idempotent setup — normally automatic on first server start."""
        engine.init()
        return InitResponse(data_dir=str(engine.config.data_dir)).model_dump()

    @app.get("/v1/ready", response_model=ReadyResponse)
    def ready() -> dict[str, Any]:
        return ReadyResponse(**engine.is_ready()).model_dump()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/health", response_model=HealthResponse)
    def v1_health() -> dict[str, str]:
        model = engine.model_status()
        if model["loading"]:
            return HealthResponse(
                status="starting",
                message=MODEL_LOADING_MESSAGE,
            ).model_dump()
        if model["load_error"]:
            return HealthResponse(
                status="error",
                message=str(model["load_error"]),
            ).model_dump()
        return HealthResponse(status="ok", message="Ready").model_dump()

    @app.get("/v1/library/stats")
    def library_stats() -> dict[str, Any]:
        stats = engine.stats()
        stats["model"] = engine.model_status()
        return stats

    @app.get("/v1/library/tracks", response_model=TracksListResponse)
    def list_tracks(
        offset: int = Query(default=0, ge=0),
        limit: int = Query(default=50, ge=1, le=500),
        status: str | None = Query(default=None),
    ) -> dict[str, Any]:
        items, total = engine.list_tracks(offset=offset, limit=limit, status=status)
        return TracksListResponse(
            items=items,  # type: ignore[arg-type]
            total=total,
            offset=offset,
            limit=limit,
        ).model_dump()

    @app.get("/v1/library/tracks/{track_id}", response_model=TrackDetailResponse)
    def get_track(track_id: str) -> dict[str, Any]:
        detail = engine.get_track_detail(track_id)
        if detail is None:
            raise HTTPException(status_code=404, detail=f"Track not found: {track_id}")
        return TrackDetailResponse(**detail).model_dump()  # type: ignore[arg-type]

    @app.get("/v1/library/sync")
    def sync_history(limit: int = Query(default=10, ge=1, le=100)) -> dict[str, Any]:
        return {"items": engine.list_sync_history(limit=limit)}

    @app.post("/v1/library/purge")
    def purge_library(req: PurgeRequest) -> dict[str, int]:
        if not req.missing and not req.removed and not req.orphans:
            raise HTTPException(
                status_code=400,
                detail="Specify missing, removed, and/or orphans",
            )
        return engine.purge(missing=req.missing, removed=req.removed, orphans=req.orphans)

    @app.get("/v1/model/status")
    def model_status() -> dict[str, Any]:
        return engine.model_status()

    @app.post("/v1/model/preload")
    def model_preload() -> dict[str, Any]:
        engine.preload_model()
        return engine.model_status()

    @app.post("/v1/model/unload")
    def model_unload() -> dict[str, Any]:
        if engine._embedder is not None:
            engine._embedder.unload()
        return engine.model_status()

    @app.post("/v1/index")
    def start_index(req: IndexRequest) -> dict[str, Any]:
        if req.async_:
            if jobs.has_active_job:
                raise HTTPException(
                    status_code=409,
                    detail="An index job is already running",
                )
            try:
                state = jobs.start(
                    paths=req.paths,
                    full_rescan=req.full_rescan,
                    embed=req.embed,
                    prune=req.prune,
                    reembed=req.reembed,
                )
            except RuntimeError as e:
                if str(e) == "index_job_running":
                    raise HTTPException(
                        status_code=409,
                        detail="An index job is already running",
                    ) from e
                raise
            return IndexJobResponse(job_id=state.job_id, status=state.status).model_dump()

        try:
            report: SyncReport = engine.index(
                paths=req.paths,
                full_rescan=req.full_rescan,
                embed=req.embed,
                prune=req.prune,
                reembed=req.reembed,
            )
        except (ValueError, NotImplementedError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        return _sync_report_response(report)

    @app.get("/v1/index/jobs/{job_id}", response_model=IndexJobStatus)
    def index_job_status(job_id: str) -> dict[str, Any]:
        state = jobs.get(job_id)
        if state is None:
            raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")
        return _job_status_response(state)

    @app.post("/v1/search/text", response_model=SearchResponse)
    def search_text(req: TextSearchRequest) -> dict[str, Any]:
        try:
            result = engine.search_by_text(req.query, k=req.k)
        except (NotImplementedError, RuntimeError) as e:
            raise HTTPException(status_code=501, detail=str(e)) from e

        return _search_result_to_response(
            result,
            {"type": "text", "value": req.query},
        )

    @app.post("/v1/search/track", response_model=SearchResponse)
    def search_track(req: TrackSearchRequest) -> dict[str, Any]:
        try:
            result = engine.search_by_track(req.track_id, k=req.k)
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        except (NotImplementedError, RuntimeError) as e:
            raise HTTPException(status_code=501, detail=str(e)) from e

        return _search_result_to_response(
            result,
            {"type": "track", "value": req.track_id},
        )

    return app
