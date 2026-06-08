# Docker

Run Harmony as a long-lived API server with Docker Compose. Data and the Hugging Face model cache persist in a named volume; your music library is mounted read-only.

**Published images:** [ghcr.io/harmony-search/harmony-engine](https://github.com/harmony-search/harmony-engine/pkgs/container/harmony-engine)

| Tag | Description |
|-----|-------------|
| `latest` | CPU (works everywhere) |
| `cuda` | NVIDIA GPU (CUDA 12.1) |
| `v0.1.0` / `0.1.0` | CPU semver release |
| `v0.1.0-cuda` | GPU semver release |

## Prerequisites

- Docker and Docker Compose
- For GPU: [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)

## Quick start (CPU)

```bash
git clone https://github.com/harmony-search/harmony-engine.git
cd harmony-engine

export MUSIC_PATH=~/music   # host path to your library
docker compose up -d

curl http://localhost:8000/v1/health
```

Index and search. The compose file sets `HARMONY_INDEX_PATHS=/music`, so you can omit `paths`:

```bash
curl -X POST http://localhost:8000/v1/index \
  -H 'Content-Type: application/json' \
  -d '{}'

curl -X POST http://localhost:8000/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query": "melancholic piano", "k": 10}'
```

On first start the server auto-creates `/data/config.yaml`, the database, and index folders — no separate init step.

## GPU

```bash
export MUSIC_PATH=~/music
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Harmony auto-detects CUDA when `embedding.device` is `auto` (default).

## Volumes and layout

| Mount | Purpose |
|-------|---------|
| `$HARMONY_DATA_PATH` → `/data` | Database, embeddings, search index, config, model cache |
| `$MUSIC_PATH` → `/music` (ro) | Music library for indexing |

Defaults: `./harmony-data` and `./music` in the project directory. For a persistent home-directory store:

```bash
export HARMONY_DATA_PATH=$HOME/.harmony
export MUSIC_PATH=~/music
docker compose up -d
```

Inside `/data`:

```
/data/
├── config.yaml
├── harmony.db
├── huggingface/          # HF_HOME — model weights cached here
├── embeddings/{version}/tracks/
└── indexes/{version}/
```

On first start (`preload_on_serve: true` by default), the server downloads **CLaMP3** weights into `/data/models/clamp3/` and MERT/XLM-R checkpoints into `/data/huggingface`. Subsequent restarts reuse the cache. No model weights are baked into the image.

The model stays loaded for **5 minutes** after the last search/index (`keep_alive: 5`). If searches feel slow because weights reload every time, check `/data/config.yaml` — an older `keep_alive: immediate` overrides the default. Set `keep_alive: 5` or `forever` there.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HARMONY_DATA_PATH` | `./harmony-data` | Host path mounted at `/data` (DB, indexes, HF cache) |
| `MUSIC_PATH` | `./music` | Host path mounted at `/music` |
| `PORT` | `8000` | Host port mapped to the API |
| `HARMONY_IMAGE` | `ghcr.io/.../harmony-engine:latest` | Override image (e.g. pinned version) |
| `HARMONY_DATA_DIR` | `/data` | Set inside container; usually leave default |
| `HARMONY_INDEX_PATHS` | `/music` | Default scan roots for index jobs (comma-separated) |
| `HF_HOME` | `/data/huggingface` | Hugging Face cache directory |

## Upgrading

```bash
docker compose pull
docker compose up -d
```

Or pin a version:

```bash
export HARMONY_IMAGE=ghcr.io/harmony-search/harmony-engine:v0.1.0
docker compose up -d
```

## Host CLI against the container

Install Harmony locally (or use curl) and point the CLI at the container:

```bash
export HARMONY_API_URL=http://127.0.0.1:8000
uv run harmony index /music          # path must match container mount
uv run harmony search text "jazz" --k 10
```

Use the API for indexing when the CLI runs on the host but music is only visible inside the container:

```bash
curl -X POST http://localhost:8000/v1/index \
  -H 'Content-Type: application/json' \
  -d '{"paths": ["/music"], "prune": true}'
```

## Build locally

```bash
# CPU
docker build -t harmony-engine:local .
docker image ls harmony-engine:local

# GPU
docker build -f Dockerfile.gpu -t harmony-engine:cuda-local .
docker image ls harmony-engine:cuda-local
```

## Image size

Images bundle PyTorch and the CLaMP3 Python stack, but **not** model weights — those download on first server startup (or first index) into `$HARMONY_DATA_PATH` on the host.

| Image | Typical compressed pull | Notes |
|-------|-------------------------|-------|
| `latest` (CPU) | ~1.5–2 GB | CPU-only PyTorch wheel |
| `cuda` (GPU) | ~3–4 GB | CUDA runtime base + cu121 PyTorch |

The GPU image cannot shrink much further: the NVIDIA CUDA runtime and cu121 PyTorch are required for embedding on a GPU.

## Health checks

The default compose healthcheck uses `GET /health` (liveness — server is listening). For stricter readiness (model loaded and index populated), override the healthcheck:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/v1/ready"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 120s
```

Use a longer `start_period` on first run while model weights download.

## Security

The container listens on `0.0.0.0:8000` with **no authentication**. Treat it as a trusted-network service:

- Bind to localhost only if you do not need remote access: `ports: ["127.0.0.1:8000:8000"]`
- Or place a reverse proxy in front for TLS and access control
- Index paths are limited to `HARMONY_INDEX_PATHS` (default `/music`)

See [server.md](server.md) for error codes and readiness endpoints.

## API reference

See [server.md](server.md) for endpoints, async indexing, and health checks.
