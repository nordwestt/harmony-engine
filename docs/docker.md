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

curl http://localhost:8000/health
```

Index and search (paths are **inside the container**):

```bash
curl -X POST http://localhost:8000/v1/index \
  -H 'Content-Type: application/json' \
  -d '{"paths": ["/music"]}'

curl -X POST http://localhost:8000/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query": "melancholic piano", "k": 10}'
```

The entrypoint runs `harmony init` on first boot if `/data/config.yaml` is missing.

## GPU

```bash
export MUSIC_PATH=~/music
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

Harmony auto-detects CUDA when `embedding.device` is `auto` (default).

## Volumes and layout

| Mount | Purpose |
|-------|---------|
| `harmony-data` → `/data` | Database, embeddings, search index, config |
| `$MUSIC_PATH` → `/music` (ro) | Music library for indexing |

Inside `/data`:

```
/data/
├── config.yaml
├── harmony.db
├── huggingface/          # HF_HOME — model weights cached here
├── embeddings/{version}/tracks/
└── indexes/{version}/
```

First index run downloads **OpenMuQ/MuQ-MuLan-large** into `/data/huggingface`. Subsequent restarts reuse the cache.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MUSIC_PATH` | `./music` | Host path mounted at `/music` |
| `PORT` | `8000` | Host port mapped to the API |
| `HARMONY_IMAGE` | `ghcr.io/.../harmony-engine:latest` | Override image (e.g. pinned version) |
| `HARMONY_DATA_DIR` | `/data` | Set inside container; usually leave default |
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

Images bundle PyTorch and the MuQ-MuLan Python stack, but **not** the model weights (those download to `/data/huggingface` on first index).

| Image | Typical compressed pull | Notes |
|-------|-------------------------|-------|
| `latest` (CPU) | ~1.5–2 GB | CPU-only PyTorch wheel |
| `cuda` (GPU) | ~3–4 GB | CUDA runtime base + cu121 PyTorch |

The GPU image cannot shrink much further: the NVIDIA CUDA runtime and cu121 PyTorch are required for embedding on a GPU.

## API reference

See [server.md](server.md) for endpoints, async indexing, and health checks.
