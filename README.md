# Harmony Engine

Self-hosted music library indexer and vector search engine with pluggable embedding backends. The default embedder is [CLaMP3](https://github.com/sanderwood/clamp3) SAAS; [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) is available via `embedding.model: muq-mulan`.

Harmony turns a local music collection into searchable embeddings and exposes retrieval primitives (text → tracks, track → similar tracks, etc.) that higher-level apps — playlist generators, DJ tools, recommendation UIs — build on.

**Specification:** [engine-spec.md](engine-spec.md)

## Status

**Phase 0** — working today:

- Filesystem scan + content-hash identity + library sync
- Audio load, resample (24 kHz), chunk, embed (default: CLaMP3)
- Track vectors persisted to disk + brute-force cosine search
- `harmony search text "dreamy night drive"`

**Next:** FAISS, chunk-level index, metadata tag extraction (mutagen).

## Quick start

### Docker (recommended for self-hosting)

Published images: `ghcr.io/harmony-search/harmony-engine` — tags `latest` (CPU), `cuda` (GPU), `v0.1.0` / `v0.1.0-cuda` (semver).

```bash
git clone https://github.com/harmony-search/harmony-engine.git
cd harmony-engine

export MUSIC_PATH=~/music          # host path mounted read-only at /music
docker compose up -d

curl http://localhost:8000/v1/health   # ok once ready; "starting" while weights download

# Index (/music is the default scan path in Docker — paths optional)
curl -X POST http://localhost:8000/v1/index \
  -H 'Content-Type: application/json' \
  -d '{}'

curl -X POST http://localhost:8000/v1/search/text \
  -H 'Content-Type: application/json' \
  -d '{"query": "melancholic piano", "k": 10}'
```

GPU:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

See [docs/docker.md](docs/docker.md) for volumes, env vars, upgrades, and troubleshooting.

### From source (development)

Uses [uv](https://docs.astral.sh/uv/). Run the server first; the CLI auto-detects it and indexes/search without restart.

```bash
uv sync --extra db --extra embed --extra embed-clamp3 --extra api --group dev

# terminal 1 — the engine
uv run harmony serve

# terminal 2 — everything hits the API (data dir auto-initializes on first request)
uv run harmony index ~/Music          # scan + embed (downloads model on first run)
uv run harmony index ~/Music --prune  # also delete tracks removed from disk
uv run harmony status
uv run harmony search text "melancholic piano" --k 10
```

Set `HARMONY_API_URL` explicitly if the server is not on `127.0.0.1:8000`. Use `--local` on any command for in-process mode (CI, no server).

See [docs/server.md](docs/server.md) for the full API and async indexing. See [docs/model-cache.md](docs/model-cache.md) for `keep_alive` settings.

Metadata-only rescan (no GPU work):

```bash
uv run harmony index ~/Music --no-embed
```

## Data directory

Default: `~/.harmony` (or `/data` in Docker)

```
~/.harmony/
├── config.yaml
├── harmony.db
├── embeddings/{version}/tracks/{track_id}.npy
└── indexes/{version}/track.brute.{npy,json}
```

First embed run downloads **CLaMP3** weights (plus MERT and XLM-RoBERTa) into `~/.harmony/models/clamp3/` and `/data/huggingface` (Docker). GPU recommended; CPU works but is slower. Use `embedding.model: muq-mulan` for the MuQ backend.

## Project layout

```
harmony/
├── engine.py
├── scanner/       # filesystem discovery
├── audio/         # decode, resample, chunk
├── embedding/     # embedder backends + pipeline
├── index/         # brute-force ANN + manager
├── retrieval/     # search
└── cli/
```

## License

MIT (engine). CLaMP3 upstream code is MIT ([sanderwood/clamp3](https://github.com/sanderwood/clamp3)). MuQ-MuLan weights are [CC-BY-NC 4.0](https://huggingface.co/OpenMuQ/MuQ-MuLan-large) when that backend is used.
