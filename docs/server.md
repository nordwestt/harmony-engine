# Harmony Server

`harmony serve` is the primary runtime for Harmony Engine. One long-lived process owns the embedding model, search index, and database. All indexing and search go through its HTTP API.

For self-hosting with Docker, see [docker.md](docker.md).

## Quick start

```bash
# Terminal 1 — start the engine
uv sync --extra db --extra embed --extra api --group dev
uv run harmony serve

# Terminal 2 — CLI talks to the server automatically
uv run harmony index ~/music --prune
uv run harmony search text "melancholic piano" --k 10
uv run harmony status
```

The CLI auto-detects a server at `http://127.0.0.1:8000` (via `/health`). Set an explicit URL with:

```bash
export HARMONY_API_URL=http://127.0.0.1:8000
```

## Why server-first?

Running `harmony index` in a **separate process** does not update a running server. Each standalone CLI command also reloads the ~5 GB MuQ-MuLan model. With `harmony serve`:

- Index updates are visible to search **immediately** (no restart)
- The model stays loaded between requests
- Long index jobs can run in the background without blocking search

Use `--local` on any command to force in-process mode (bootstrap, CI, no server):

```bash
uv run harmony index ~/music --local
```

## API overview

Base path: `/v1`. Interactive docs: `http://127.0.0.1:8000/docs`

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Liveness probe (always `ok` once the server is listening) |
| `GET /v1/health` | Startup status; returns a friendly message while model weights download |
| `GET /v1/ready` | Model loaded and index has vectors |
| `POST /v1/init` | Initialize data directory |
| `POST /v1/index` | Scan, embed, update index |
| `GET /v1/index/jobs/{job_id}` | Background index job status |
| `POST /v1/search/text` | Text → similar tracks |
| `POST /v1/search/track` | Track → similar tracks |
| `GET /v1/library/stats` | Library statistics |
| `GET /v1/library/tracks` | Paginated track list |
| `GET /v1/library/tracks/{id}` | Single track + paths |
| `GET /v1/library/sync` | Sync run history |
| `POST /v1/library/purge` | Remove missing/removed/orphan data |

Errors return `{"error": "...", "code": "..."}`.

## Async indexing

Large libraries can block for minutes. Use async mode to return immediately and poll progress:

```bash
# API
curl -X POST http://127.0.0.1:8000/v1/index \
  -H 'Content-Type: application/json' \
  -d '{"paths": ["/home/you/music"], "async": true}'

# CLI
uv run harmony index ~/music --async
```

Only one index job runs per data directory at a time. A second request returns `409 Conflict`.

Search remains available during indexing. Each newly embedded track is searchable as soon as it is upserted into the index.

## Hot reload

The server invalidates its search cache after every index or purge. You do not need to restart `harmony serve` after adding music.

## Model keep-alive

See [model-cache.md](model-cache.md) for `embedding.keep_alive` settings (`immediate`, minutes, `forever`).
