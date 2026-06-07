# Harmony Engine

Self-hosted music library indexer and vector search engine powered by [MuQ-MuLan](https://huggingface.co/OpenMuQ/MuQ-MuLan-large).

Harmony turns a local music collection into searchable embeddings and exposes retrieval primitives (text → tracks, track → similar tracks, etc.) that higher-level apps — playlist generators, DJ tools, recommendation UIs — build on.

**Specification:** [engine-spec.md](engine-spec.md)

## Status

**Phase 0** — working today:

- Filesystem scan + content-hash identity + library sync
- Audio load, resample (24 kHz), chunk, MuQ-MuLan embed
- Track vectors persisted to disk + brute-force cosine search
- `harmony search text "dreamy night drive"`

**Next:** FAISS, chunk-level index, metadata tag extraction (mutagen).

## Quick start

Uses [uv](https://docs.astral.sh/uv/).

**Recommended:** run the server first; the CLI auto-detects it and indexes/search without restart.

```bash
uv sync --extra db --extra embed --extra api --group dev

# terminal 1 — the engine
uv run harmony serve

# terminal 2 — everything hits the API
uv run harmony init
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

Default: `~/.harmony`

```
~/.harmony/
├── config.yaml
├── harmony.db
├── embeddings/{version}/tracks/{track_id}.npy
└── indexes/{version}/track.brute.{npy,json}
```

First embed run downloads **OpenMuQ/MuQ-MuLan-large** from Hugging Face (~700M params). GPU recommended; CPU works but is slower.

## Project layout

```
harmony/
├── engine.py
├── scanner/       # filesystem discovery
├── audio/         # decode, resample, chunk
├── embedding/     # MuQ-MuLan + pipeline
├── index/         # brute-force ANN + manager
├── retrieval/     # search
└── cli/
```

## License

MIT (engine). MuQ-MuLan weights are [CC-BY-NC 4.0](https://huggingface.co/OpenMuQ/MuQ-MuLan-large).
