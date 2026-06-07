# Harmony Engine

Self-hosted music library indexer and vector search engine powered by [MuQ-MuLan](https://github.com/muqmulan/muq-mulan).

Harmony turns a local music collection into searchable embeddings and exposes retrieval primitives (text → tracks, track → similar tracks, etc.) that higher-level apps — playlist generators, DJ tools, recommendation UIs — build on.

**Specification:** [engine-spec.md](engine-spec.md)

## Status

**Phase 0 scaffold** — working today:

- Project structure, config, CLI
- Turso/SQLite metadata store with schema migrations
- Local filesystem scan + content-hash identity
- Library sync (adds, moves, missing/removal grace period)
- Brute-force index stub

**Not yet implemented:** MuQ-MuLan embedding, FAISS, audio pipeline, search.

## Quick start

```bash
# Option A: bootstrap script
bash scripts/bootstrap.sh

# Option B: manual
python -m venv .venv
source .venv/bin/activate
pip install -e ".[db,dev]"

harmony init
harmony index /path/to/music
harmony status
pytest
```

## Project layout

```
harmony/           # Python package
├── engine.py      # Engine facade
├── config.py      # Configuration
├── storage/       # Turso DB, vectors, sync
├── sources/       # filesystem scanner
├── audio/         # decode, resample, chunk
├── embedding/     # MuQ-MuLan wrapper
├── index/         # FAISS / brute-force ANN
├── retrieval/     # search, filters, aggregation
├── api/           # FastAPI server
└── cli/           # `harmony` command
```

## Self-hosting

Everything lives in a single `data_dir`:

- `harmony.db` — metadata (Turso / SQLite-compatible)
- `embeddings/` — vector files
- `indexes/` — FAISS indexes

No external database server required.

## License

MIT
