# Harmony Server

`harmony serve` is the primary runtime for Harmony Engine. One long-lived process owns the embedding model, search index, and database. All indexing and search go through its HTTP API.

For self-hosting with Docker, see [docker.md](docker.md).

## Quick start

```bash
# Terminal 1 — start the engine
uv sync --extra db --extra embed --extra embed-clamp3 --extra api --group dev
uv run harmony serve

# Terminal 2 — CLI talks to the server automatically
# (no separate init step — the server creates ~/.harmony on first start)
uv run harmony index ~/music --prune
uv run harmony search text "melancholic piano" --k 10
uv run harmony status
```

The CLI auto-detects a server at `http://127.0.0.1:8000` (via `/health`). Set an explicit URL with:

```bash
export HARMONY_API_URL=http://127.0.0.1:8000
```

## Why server-first?

Running `harmony index` in a **separate process** does not update a running server. Each standalone CLI command also reloads the embedding model. With `harmony serve`:

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
| `POST /v1/init` | Optional idempotent setup (automatic on first `harmony serve`) |
| `POST /v1/index` | Scan, embed, update index |
| `GET /v1/index/jobs/{job_id}` | Background index job status |
| `POST /v1/index/jobs/{job_id}/resume` | Resume an interrupted embed job |
| `POST /v1/search/text` | Text → similar tracks |
| `POST /v1/search/track` | Track → similar tracks |
| `GET /v1/library/stats` | Library statistics |
| `GET /v1/library/tracks` | Paginated track list |
| `GET /v1/library/tracks/{id}` | Single track + paths |
| `GET /v1/library/sync` | Sync run history |
| `POST /v1/library/purge` | Remove missing/removed/orphan data |

Errors return `{"error": "...", "code": "..."}`.

Common error codes:

| Code | HTTP | Meaning |
|------|------|---------|
| `validation_error` | 422 | Invalid request body or query parameter |
| `path_not_allowed` | 400 | Index path outside configured scan roots |
| `invalid_track_id` | 400 | Track ID is not a valid UUID |
| `index_not_ready` | 503 | No embedded tracks yet — run an index job first |
| `model_not_ready` | 503 | Model is loading or failed to load |
| `internal_error` | 500 | Unexpected server failure (details in logs only) |

## Health and readiness

| Endpoint | Use |
|----------|-----|
| `GET /health` | **Liveness** — returns `ok` once the HTTP server is listening |
| `GET /v1/health` | Startup status while model weights download |
| `GET /v1/ready` | **Readiness** — model loaded and index has vectors (required for search) |

Search endpoints return `503` with `index_not_ready` or `model_not_ready` when the engine is not ready to serve queries. Poll `/v1/ready` before searching on a fresh install.

## Security (trusted network)

Harmony Engine has **no built-in authentication or rate limiting**. It is designed for self-hosting on a trusted network (localhost or a private LAN).

- `harmony serve` binds `127.0.0.1` by default.
- Docker binds `0.0.0.0` — do not expose port 8000 to the public internet without a reverse proxy (Caddy, nginx, etc.) that terminates TLS and enforces access control.
- Index paths are restricted to configured scan roots (`filesystem.paths` or `HARMONY_INDEX_PATHS`). Requests to scan directories outside those roots are rejected.
- Destructive endpoints (`POST /v1/library/purge`, `POST /v1/index`, model load/unload) are unauthenticated by design.

## Configuration

| Variable | Description |
|----------|-------------|
| `HARMONY_DATA_DIR` | Data directory (default `~/.harmony`, `/data` in Docker) |
| `HARMONY_INDEX_PATHS` | Comma-separated default scan roots when index `paths` are omitted |
| `HARMONY_API_URL` | API base URL for the CLI (default auto-detect `http://127.0.0.1:8000`) |

On first start, the server creates `config.yaml`, the SQLite database, and required folders. Set `HARMONY_INDEX_PATHS` (or `filesystem.paths` in `config.yaml`) so `POST /v1/index` with an empty body knows where your music lives.

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

### Job resume

Async jobs persist state to the database. If the server restarts while a job is `pending` or `running`, it is marked `interrupted` on startup. Embedding is checkpointed per track — unembedded tracks are picked up when the job resumes.

By default (`jobs.resume_on_startup: true` in `config.yaml`), the server automatically resumes the latest interrupted embed job when tracks remain pending. Disable with:

```yaml
jobs:
  resume_on_startup: false
```

Manual resume (reuses the same `job_id`):

```bash
curl -X POST http://127.0.0.1:8000/v1/index/jobs/{job_id}/resume
```

Jobs interrupted before params were persisted (older versions) reconstruct default settings (`embed: true`, configured scan paths) on resume.

Job statuses: `pending` → `running` → `completed` | `failed` | `interrupted` (resumable).

Progress fields on `GET /v1/index/jobs/{job_id}`:

- `embedded` — tracks successfully embedded **across the whole job**, including work done before an interrupt
- `total_pending` — total embed scope (`embedded` at start of embed phase + tracks remaining this run)
- On resume, counts continue from where they left off (e.g. 12/22 → 13/22) rather than resetting

Search remains available during indexing. Each newly embedded track is searchable as soon as it is upserted into the index.

## Hot reload

The server invalidates its search cache after every index or purge. You do not need to restart `harmony serve` after adding music.

## Model keep-alive

See [model-cache.md](model-cache.md) for `embedding.keep_alive` settings (`immediate`, minutes, `forever`).
