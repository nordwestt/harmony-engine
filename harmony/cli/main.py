"""Harmony CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from harmony import __version__
from harmony.client import (
    detect_api_url,
    index_library,
    library_stats,
    purge_library,
    search_text as api_search_text,
    sync_history as api_sync_history,
    wait_for_index_job,
)
from harmony.engine import Engine
from harmony.errors import PathNotAllowedError


def _resolve_api(local: bool) -> str | None:
    if local:
        return None
    return detect_api_url()


def _api_hint() -> None:
    click.echo(
        "Tip: start `harmony serve` for a shared engine (no restart after indexing).",
        err=True,
    )


@click.group()
@click.version_option(version=__version__, prog_name="harmony")
@click.option(
    "--data-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Harmony data directory (default: ~/.harmony)",
)
@click.pass_context
def cli(ctx: click.Context, data_dir: Path | None) -> None:
    """Harmony Engine — music library indexing and vector search."""
    ctx.ensure_object(dict)
    ctx.obj["data_dir"] = data_dir


@cli.command()
@click.option("--local", is_flag=True, help="Run in-process instead of via API")
@click.pass_context
def init(ctx: click.Context, local: bool) -> None:
    """Initialize data directory and database (optional — ``harmony serve`` does this automatically)."""
    api_url = _resolve_api(local)
    if api_url:
        from harmony.client import init_library

        result = init_library(api_url)
        click.echo(f"Initialized Harmony at {result['data_dir']}")
        return

    engine = Engine(ctx.obj["data_dir"])
    engine.init()
    click.echo(f"Initialized Harmony at {engine.config.data_dir}")
    engine.close()


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--full", is_flag=True, help="Force full rescan (reserved)")
@click.option("--watch", is_flag=True, help="Watch filesystem for changes (not yet implemented)")
@click.option("--no-embed", is_flag=True, help="Scan metadata only, skip embedding")
@click.option(
    "--prune",
    is_flag=True,
    help="Delete tracks no longer on disk (skips the missing-file grace period)",
)
@click.option(
    "--reembed",
    is_flag=True,
    help="Re-embed all tracks even if already indexed",
)
@click.option(
    "--async",
    "run_async",
    is_flag=True,
    help="Run indexing as a background API job (poll until complete)",
)
@click.option("--local", is_flag=True, help="Run in-process instead of via API")
@click.pass_context
def index(
    ctx: click.Context,
    paths: tuple[Path, ...],
    full: bool,
    watch: bool,
    no_embed: bool,
    prune: bool,
    reembed: bool,
    run_async: bool,
    local: bool,
) -> None:
    """Scan and reconcile a music library."""
    if watch:
        click.echo("Watch mode is not yet implemented.", err=True)
        sys.exit(1)

    api_url = _resolve_api(local)
    path_strs = [str(p) for p in paths] if paths else None

    if api_url:
        try:
            if not no_embed:
                click.echo("Scanning library via API…", err=True)
            payload = index_library(
                api_url,
                paths=path_strs,
                full_rescan=full,
                embed=not no_embed,
                prune=prune,
                reembed=reembed,
                async_=run_async,
            )
            if run_async:
                job_id = payload["job_id"]
                click.echo(f"Index job started: {job_id}", err=True)

                def on_progress(status: dict) -> None:
                    phase = status.get("phase") or status.get("status")
                    embedded = status.get("embedded", 0)
                    total = status.get("total_pending", 0)
                    if total:
                        click.echo(f"  {phase}: {embedded}/{total}", err=True)

                payload = wait_for_index_job(api_url, job_id, on_progress=on_progress)
                report = payload.get("report") or {}
            else:
                report = payload
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)

        click.echo(f"Added:      {report.get('added', 0)}")
        click.echo(f"Moved:      {report.get('moved', 0)}")
        click.echo(f"Skipped:    {report.get('skipped', 0)}")
        click.echo(f"Embedded:   {report.get('embedded', 0)}")
        click.echo(f"Purged:     {report.get('purged', 0)}")
        click.echo(f"Failed:     {report.get('failed', 0)}")
        click.echo(f"Missing:    {report.get('missing', 0)}")
        click.echo(f"Removed:    {report.get('removed', 0)}")
        click.echo(f"Duration:   {report.get('duration_ms', 0)}ms")
        return

    _api_hint()
    engine = Engine(ctx.obj["data_dir"])
    try:
        if not no_embed:
            click.echo("Scanning library…", err=True)
        report = engine.index(
            paths=path_strs,
            full_rescan=full,
            embed=not no_embed,
            prune=prune,
            reembed=reembed,
        )
    except (ValueError, NotImplementedError, ImportError, PathNotAllowedError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        engine.close()

    click.echo(f"Added:      {report.added}")
    click.echo(f"Moved:      {report.moved}")
    click.echo(f"Skipped:    {report.skipped}")
    click.echo(f"Embedded:   {report.embedded}")
    click.echo(f"Purged:     {report.purged}")
    click.echo(f"Failed:     {report.failed}")
    click.echo(f"Missing:    {report.missing}")
    click.echo(f"Removed:    {report.removed}")
    click.echo(f"Duration:   {report.duration_ms}ms")


@cli.command()
@click.option("--local", is_flag=True, help="Run in-process instead of via API")
@click.pass_context
def status(ctx: click.Context, local: bool) -> None:
    """Show library statistics."""
    api_url = _resolve_api(local)
    if api_url:
        try:
            stats = library_stats(api_url)
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        for key, value in stats.items():
            click.echo(f"{key}: {value}")
        return

    engine = Engine(ctx.obj["data_dir"])
    try:
        stats = engine.stats()
    finally:
        engine.close()

    for key, value in stats.items():
        click.echo(f"{key}: {value}")


@cli.command("sync-history")
@click.option("--limit", default=10, show_default=True)
@click.option("--local", is_flag=True, help="Run in-process instead of via API")
@click.pass_context
def sync_history(ctx: click.Context, limit: int, local: bool) -> None:
    """Show recent library sync reports."""
    api_url = _resolve_api(local)
    if api_url:
        try:
            payload = api_sync_history(api_url, limit=limit)
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        rows = payload.get("items", [])
        if not rows:
            click.echo("No sync runs recorded yet.")
            return
        for row in rows:
            click.echo(
                f"{row['started_at']}  "
                f"+{row['added']} moved={row['moved']} "
                f"missing={row['missing']} removed={row['removed']} "
                f"({row['duration_ms']}ms)"
            )
        return

    engine = Engine(ctx.obj["data_dir"])
    try:
        rows = engine.list_sync_history(limit=limit)
    finally:
        engine.close()

    if not rows:
        click.echo("No sync runs recorded yet.")
        return

    for row in rows:
        click.echo(
            f"{row['started_at']}  "
            f"+{row['added']} moved={row['moved']} "
            f"missing={row['missing']} removed={row['removed']} "
            f"({row['duration_ms']}ms)"
        )


@cli.group()
def bench() -> None:
    """Benchmark embedding performance."""


@bench.command("encode")
@click.argument("path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model",
    default=None,
    help="Embedding backend (e.g. muq-mulan, clap-music); overrides config",
)
@click.option(
    "--checkpoint",
    default=None,
    help="Model checkpoint; overrides config embedding.checkpoint",
)
@click.option("--json", "as_json", is_flag=True, help="Output JSON")
@click.pass_context
def bench_encode(
    ctx: click.Context,
    path: Path,
    model: str | None,
    checkpoint: str | None,
    as_json: bool,
) -> None:
    """Time encoding a single audio file (load, resample, chunk, embed)."""
    from harmony.config import Config
    from harmony.embedding.benchmark import benchmark_encode

    config = Config.load(ctx.obj["data_dir"])
    if model is not None:
        config.embedding.model = model
    if checkpoint is not None:
        config.embedding.checkpoint = checkpoint
    try:
        result = benchmark_encode(path, config)
    except (ValueError, ImportError, OSError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)

    if as_json:
        import dataclasses

        click.echo(json.dumps(dataclasses.asdict(result), indent=2))
        return

    click.echo(f"File:        {result.path}")
    click.echo(f"Duration:    {result.duration_s:.1f}s")
    click.echo(f"Sample rate: {result.source_sample_rate} → {result.target_sample_rate} Hz")
    click.echo(f"Chunks:      {result.chunks} ({result.batches} batches of {result.batch_size})")
    click.echo(f"Model:       {result.model} on {result.device}")
    click.echo(f"Vector dim:  {result.vector_dim}")
    click.echo(f"Load:        {result.load_ms:.0f} ms")
    if result.resample_ms > 0:
        click.echo(f"Resample:    {result.resample_ms:.0f} ms")
    click.echo(f"Chunk:       {result.chunk_ms:.0f} ms")
    if result.model_load_ms > 0:
        click.echo(f"Model load:  {result.model_load_ms:.0f} ms")
    click.echo(f"Embed:       {result.embed_ms:.0f} ms")
    click.echo(f"Total:       {result.total_ms:.0f} ms ({result.total_ms / 1000:.1f}s)")


@cli.group()
def search() -> None:
    """Search the indexed library."""


@search.command("text")
@click.argument("query")
@click.option("--k", default=50, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Output JSON")
@click.option(
    "--api",
    "api_url",
    default=None,
    help="Use a running harmony serve API (overrides auto-detect)",
)
@click.option("--local", is_flag=True, help="Run in-process instead of via API")
@click.pass_context
def search_text(
    ctx: click.Context,
    query: str,
    k: int,
    as_json: bool,
    api_url: str | None,
    local: bool,
) -> None:
    """Search by natural language query."""
    if local:
        api_url = None
    else:
        api_url = api_url or detect_api_url()

    if api_url:
        try:
            payload = api_search_text(api_url, query, k=k)
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        _print_search_payload(payload, as_json=as_json)
        return

    _api_hint()
    engine = Engine(ctx.obj["data_dir"])
    try:
        result = engine.search_by_text(query, k=k)
    except (NotImplementedError, RuntimeError, ImportError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        engine.close()

    if as_json:
        payload = {
            "items": [
                {
                    "track_id": i.track_id,
                    "score": i.score,
                    "rank": i.rank,
                    "title": i.metadata.title,
                    "artist": i.metadata.artist,
                    "path": i.metadata.primary_path,
                }
                for i in result.items
            ],
            "total_indexed": result.total_indexed,
            "took_ms": result.took_ms,
        }
        click.echo(json.dumps(payload, indent=2))
        return

    if not result.items:
        click.echo("No results.")
        return

    for item in result.items:
        click.echo(
            f"{item.score:6.3f}  {item.metadata.artist} — {item.metadata.title}"
        )


def _print_search_payload(payload: dict, *, as_json: bool) -> None:
    if as_json:
        click.echo(json.dumps(payload, indent=2))
        return

    items = payload.get("items", [])
    if not items:
        click.echo("No results.")
        return

    for item in items:
        meta = item.get("metadata", {})
        click.echo(
            f"{item['score']:6.3f}  {meta.get('artist', '')} — {meta.get('title', '')}"
        )


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.option(
    "--no-preload",
    is_flag=True,
    help="Do not load the embedding model until the first search/index request",
)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int, no_preload: bool) -> None:
    """Start the HTTP API server (keeps the model loaded between searches)."""
    try:
        import uvicorn
        from harmony.api.app import create_app
    except ImportError:
        click.echo(
            "API server requires optional dependencies. "
            "Install with: uv sync --extra api --extra embed --extra embed-muq",
            err=True,
        )
        sys.exit(1)

    engine = Engine(ctx.obj["data_dir"])
    policy = engine.model_status()["keep_alive"]
    engine.close()

    click.echo(f"Model keep-alive: {policy}", err=True)
    click.echo(
        f"API listening on http://{host}:{port}  "
        f"(CLI auto-detects this server; or set HARMONY_API_URL=http://{host}:{port})",
        err=True,
    )

    app = create_app(
        data_dir=ctx.obj["data_dir"],
        preload_on_serve=False if no_preload else None,
    )
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option(
    "--missing",
    is_flag=True,
    help="Delete tracks not found on disk (same as index --prune)",
)
@click.option("--removed", is_flag=True, help="Delete tracks already marked removed")
@click.option("--orphans", is_flag=True, help="Delete orphan vector files on disk")
@click.option("--local", is_flag=True, help="Run in-process instead of via API")
@click.pass_context
def purge(ctx: click.Context, missing: bool, removed: bool, orphans: bool, local: bool) -> None:
    """Purge missing/removed tracks and orphan data."""
    if not missing and not removed and not orphans:
        click.echo("Specify --missing, --removed, and/or --orphans", err=True)
        sys.exit(1)

    api_url = _resolve_api(local)
    if api_url:
        try:
            counts = purge_library(
                api_url,
                missing=missing,
                removed=removed,
                orphans=orphans,
            )
        except RuntimeError as e:
            click.echo(f"Error: {e}", err=True)
            sys.exit(1)
        if missing:
            click.echo(f"Purged missing: {counts.get('missing', 0)}")
        if removed:
            click.echo(f"Purged removed: {counts.get('removed', 0)}")
        if orphans:
            click.echo(f"Purged orphans: {counts.get('orphans', 0)}")
        return

    engine = Engine(ctx.obj["data_dir"])
    try:
        counts = engine.purge(missing=missing, removed=removed, orphans=orphans)
    finally:
        engine.close()

    if missing:
        click.echo(f"Purged missing: {counts['missing']}")
    if removed:
        click.echo(f"Purged removed: {counts['removed']}")
    if orphans:
        click.echo(f"Purged orphans: {counts['orphans']}")


if __name__ == "__main__":
    cli()
