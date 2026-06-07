"""Harmony CLI."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from harmony import __version__
from harmony.engine import Engine


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
@click.pass_context
def init(ctx: click.Context) -> None:
    """Initialize data directory and database."""
    engine = Engine(ctx.obj["data_dir"])
    engine.init()
    click.echo(f"Initialized Harmony at {engine.config.data_dir}")


@cli.command()
@click.argument("paths", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.option("--full", is_flag=True, help="Force full rescan (reserved)")
@click.option("--watch", is_flag=True, help="Watch filesystem for changes (not yet implemented)")
@click.option("--no-embed", is_flag=True, help="Scan metadata only, skip embedding")
@click.pass_context
def index(
    ctx: click.Context,
    paths: tuple[Path, ...],
    full: bool,
    watch: bool,
    no_embed: bool,
) -> None:
    """Scan and reconcile a music library."""
    if watch:
        click.echo("Watch mode is not yet implemented.", err=True)
        sys.exit(1)

    engine = Engine(ctx.obj["data_dir"])
    path_strs = [str(p) for p in paths] if paths else None

    try:
        report = engine.index(paths=path_strs, full_rescan=full, embed=not no_embed)
    except (ValueError, NotImplementedError) as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)
    finally:
        engine.close()

    click.echo(f"Added:      {report.added}")
    click.echo(f"Moved:      {report.moved}")
    click.echo(f"Skipped:    {report.skipped}")
    click.echo(f"Embedded:   {report.embedded}")
    click.echo(f"Failed:     {report.failed}")
    click.echo(f"Missing:    {report.missing}")
    click.echo(f"Removed:    {report.removed}")
    click.echo(f"Duration:   {report.duration_ms}ms")


@cli.command()
@click.pass_context
def status(ctx: click.Context) -> None:
    """Show library statistics."""
    engine = Engine(ctx.obj["data_dir"])
    try:
        stats = engine.stats()
    finally:
        engine.close()

    for key, value in stats.items():
        click.echo(f"{key}: {value}")


@cli.command("sync-history")
@click.option("--limit", default=10, show_default=True)
@click.pass_context
def sync_history(ctx: click.Context, limit: int) -> None:
    """Show recent library sync reports."""
    engine = Engine(ctx.obj["data_dir"])
    try:
        rows = engine.store.conn.execute(
            """
            SELECT * FROM sync_runs
            ORDER BY started_at DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        engine.close()

    if not rows:
        click.echo("No sync runs recorded yet.")
        return

    for row in rows:
        def col(name: str, idx: int) -> object:
            return row[name] if hasattr(row, "keys") else row[idx]

        click.echo(
            f"{col('started_at', 1)}  "
            f"+{col('added', 3)} moved={col('moved', 5)} "
            f"missing={col('missing', 7)} removed={col('removed', 8)} "
            f"({col('duration_ms', 12)}ms)"
        )


@cli.group()
def search() -> None:
    """Search the indexed library."""


@search.command("text")
@click.argument("query")
@click.option("--k", default=50, show_default=True)
@click.option("--json", "as_json", is_flag=True, help="Output JSON")
@click.pass_context
def search_text(ctx: click.Context, query: str, k: int, as_json: bool) -> None:
    """Search by natural language query."""
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


@cli.command()
@click.option("--host", default="127.0.0.1", show_default=True)
@click.option("--port", default=8000, show_default=True)
@click.pass_context
def serve(ctx: click.Context, host: str, port: int) -> None:
    """Start the HTTP API server."""
    try:
        import uvicorn
        from harmony.api.app import create_app
    except ImportError:
        click.echo(
            "API server requires optional dependencies. "
            "Install with: pip install harmony-engine[api]",
            err=True,
        )
        sys.exit(1)

    app = create_app(data_dir=ctx.obj["data_dir"])
    uvicorn.run(app, host=host, port=port)


@cli.command()
@click.option("--removed", is_flag=True, help="Purge removed tracks")
@click.option("--orphans", is_flag=True, help="Purge orphan vector files")
@click.pass_context
def purge(ctx: click.Context, removed: bool, orphans: bool) -> None:
    """Purge removed tracks and orphan data."""
    if not removed and not orphans:
        click.echo("Specify --removed and/or --orphans", err=True)
        sys.exit(1)
    click.echo("Purge is not yet implemented.", err=True)
    sys.exit(1)


if __name__ == "__main__":
    cli()
