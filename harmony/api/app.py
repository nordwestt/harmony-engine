"""FastAPI application."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from harmony.engine import Engine
from harmony.models import SyncReport


class IndexRequest(BaseModel):
    paths: list[str] | None = None
    full_rescan: bool = False
    prune: bool = False
    reembed: bool = False


class TextSearchRequest(BaseModel):
    query: str
    k: int = Field(default=50, ge=1, le=500)


def create_app(
    data_dir: Path | str | None = None,
    *,
    preload_on_serve: bool | None = None,
) -> FastAPI:
    app = FastAPI(
        title="Harmony Engine",
        version="0.1.0",
        description="Music library indexing and vector search API",
    )
    engine = Engine(data_dir)

    @app.on_event("startup")
    def _startup() -> None:
        if preload_on_serve is not None:
            engine.config.embedding.preload_on_serve = preload_on_serve
        if engine.config.embedding.preload_on_serve:
            engine.preload_model()

    @app.on_event("shutdown")
    def _shutdown() -> None:
        engine.close()

    @app.get("/v1/library/stats")
    def library_stats() -> dict[str, Any]:
        stats = engine.stats()
        stats["model"] = engine.model_status()
        return stats

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
        try:
            report: SyncReport = engine.index(
                paths=req.paths,
                full_rescan=req.full_rescan,
                prune=req.prune,
                reembed=req.reembed,
            )
        except (ValueError, NotImplementedError) as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

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

    @app.post("/v1/search/text")
    def search_text(req: TextSearchRequest) -> dict[str, Any]:
        try:
            result = engine.search_by_text(req.query, k=req.k)
        except (NotImplementedError, RuntimeError) as e:
            raise HTTPException(status_code=501, detail=str(e)) from e

        return {
            "items": [
                {
                    "track_id": i.track_id,
                    "score": i.score,
                    "rank": i.rank,
                    "metadata": {
                        "title": i.metadata.title,
                        "artist": i.metadata.artist,
                        "album": i.metadata.album,
                        "primary_path": i.metadata.primary_path,
                    },
                }
                for i in result.items
            ],
            "query": {"type": "text", "value": req.query},
            "total_indexed": result.total_indexed,
            "took_ms": result.took_ms,
        }

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app
