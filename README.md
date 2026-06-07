# Harmony Engine

Self-hosted music library indexer and vector search engine powered by [MuQ-MuLan](https://github.com/muqmulan/muq-mulan).

Harmony turns a local music collection into searchable embeddings and exposes retrieval primitives (text → tracks, track → similar tracks, etc.) that higher-level apps — playlist generators, DJ tools, recommendation UIs — build on.

**Specification:** [engine-spec.md](engine-spec.md)

## Status

**Phase 0 scaffold** — working today:

- Project structure, config, CLI
- Turso/SQLite metadata store with schema migrations
- Filesystem scan + content-hash identity
- Library sync (adds, moves, missing/removal grace period)
- Brute-force index stub

**Not yet implemented:** MuQ-MuLan embedding, FAISS, audio pipeline, search.

## Quick start

Uses [uv](https://docs.astral.sh/uv/).

```bash
uv sync --extra db --group dev

uv run harmony init          # creates ~/.harmony
uv run harmony index ~/Music # scan your library
uv run harmony status

uv run pytest
```

Or use the bootstrap script:

```bash
bash scripts/bootstrap.sh
```

## Data directory

By default Harmony uses `~/.harmony`:

```
~/.harmony/
├── config.yaml    # settings snapshot
├── harmony.db     # metadata (Turso / SQLite-compatible)
├── embeddings/    # vector files (once embedding is wired)
└── indexes/       # FAISS indexes
```

Override with `--data-dir` on CLI commands or `HARMONY_DATA_DIR` in the environment.

## Project layout

```
harmony/           # Python package
├── engine.py      # Engine facade
├── config.py      # Configuration
├── storage/       # Turso DB, vectors, sync
├── scanner/       # filesystem discovery
├── audio/         # decode, resample, chunk
├── embedding/     # MuQ-MuLan wrapper
├── index/         # FAISS / brute-force ANN
├── retrieval/     # search, filters, aggregation
├── api/           # FastAPI server
└── cli/           # `harmony` command
```

## License

MIT
