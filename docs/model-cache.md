# Model keep-alive

The default embedding model (CLaMP3 SAAS + MERT + XLM-R) is several GB of weights. Loading it takes tens of seconds. Harmony lets you control how long the embedding model stays in memory **within a running process**.

## Why each CLI search feels slow

`harmony search text "..."` starts a **new Python process** every time. That process loads the model, searches, then exits. Keep-alive settings in `config.yaml` do not help across separate CLI invocations.

For interactive searching, use **`harmony serve`** and point the CLI at the API.

## Recommended workflow

Terminal 1 — start the server (loads model once):

```bash
uv sync --extra api --extra embed --extra embed-clamp3
uv run harmony serve
```

Terminal 2 — fast searches (no model reload):

```bash
export HARMONY_API_URL=http://127.0.0.1:8000
uv run harmony search text "melancholic piano" --k 10
uv run harmony search text "dreamy ambient" --k 10
```

Or pass `--api` explicitly:

```bash
uv run harmony search text "dreamy ambient" --api http://127.0.0.1:8000
```

## Configuration

In `~/.harmony/config.yaml`:

```yaml
embedding:
  keep_alive: 5            # see options below (default: 5 minutes)
  preload_on_serve: true   # warm-load model when harmony serve starts
```

### `keep_alive` options

| Value | Behavior |
|-------|----------|
| `false`, `0`, `immediate`, `off` | Unload model after each embed/search operation |
| `30`, `30m`, `30min` | Keep loaded; unload after 30 minutes **since last use** |
| `forever`, `always`, `true` | Keep loaded until the process exits |

Default: **`5`** minutes since last use (good balance for Docker and interactive search). Use **`forever`** on a dedicated search server if you prefer never unloading.

### `preload_on_serve`

When `true` (default), `harmony serve` loads the model at startup so the first search is also fast. Use `harmony serve --no-preload` to defer loading until the first request.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/model/status` | Loaded state, device, keep-alive policy |
| `POST` | `/v1/model/preload` | Load model into memory |
| `POST` | `/v1/model/unload` | Release model from memory |

## One-shot CLI (index / single search)

For `harmony index` and standalone `harmony search` without `--api`, the process still loads and unloads per run. Use `keep_alive: immediate` on memory-constrained machines so RAM is freed as soon as the command finishes.

Indexing batches all chunk embeddings in a **session** so the model is not unloaded between chunks within the same track or index run.

## Examples

Keep model hot for 15 minutes between searches on a shared server:

```yaml
embedding:
  keep_alive: 15
```

Minimize RAM when running occasional one-off searches locally:

```yaml
embedding:
  keep_alive: immediate
```
