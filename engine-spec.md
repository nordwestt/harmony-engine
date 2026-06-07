# Harmony Engine — Specification

A library indexing and vector search engine powered by **MuQ-MuLan**. It turns a music collection into searchable embeddings and exposes retrieval primitives that higher-level applications (playlist generators, DJ tools, recommendation UIs, automations) can build on.

**In scope:** ingest, embed, persist, index, search.  
**Out of scope:** playlist sequencing, energy curves, graph walks, M3U export, pushing playlists to media servers.

**Deployment model:** fully self-hosted. A single `data_dir` on disk — no external database server, no cloud dependency. One process (CLI, library, or API server) owns the store.

---

## 1. Purpose

Harmony Engine answers one question well:

> *Given a music library, how do I find tracks (or moments within tracks) that match a text description, an audio seed, or another track — fast, persistently, and programmatically?*

Consumers of this engine should never need to touch MuQ-MuLan, FAISS, or audio preprocessing directly. They get stable IDs, metadata, scores, and embedding vectors when they need to go further.

---

## 2. Design principles

1. **Embeddings are the product.** Everything else exists to produce, store, and query vectors reliably.
2. **Chunks are first-class.** Track-level vectors are derived; chunk-level vectors are the source of truth for fine-grained search.
3. **Persist everything.** A rebuild should only re-embed what changed (file hash, model version, or chunking config).
4. **Model-agnostic interface.** MuQ-MuLan is the default embedder, but the engine talks in terms of `Embedder`, not a specific checkpoint.
5. **Retrieval, not curation.** Return ranked candidates with scores and metadata. Ordering 30 songs into a journey is someone else's job.
6. **Three surfaces, one core.** The same logic is exposed as a Python library, a CLI, and an HTTP API.
7. **Self-hosted by default.** No managed services required. Metadata, vectors, and indexes live in a portable `data_dir` you own.
8. **Content is identity, path is location.** Tracks are keyed by audio content, not filesystem path. Reorganizing folders must not trigger re-embedding.
9. **Libraries change continuously.** Adds, deletes, moves, tag edits, and duplicate files are normal operations — not exceptional rebuilds.

---

## 3. Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  Scanner    │ ──▶ │  Ingestion   │ ──▶ │  Embedding  │ ──▶ │   Storage    │
│ filesystem  │     │ decode/chunk │     │  MuQ-MuLan  │     │ turso + files │
└─────────────┘     └──────────────┘     └─────────────┘     └──────┬───────┘
                                                                    │
                                                                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│  CLI / API  │ ◀── │  Retrieval   │ ◀── │    Index    │ ◀── │   Vectors    │
│  / Library  │     │ search/rank  │     │ FAISS (ANN) │     │ chunk+track  │
└─────────────┘     └──────────────┘     └─────────────┘     └──────────────┘
```

### Layers

| Layer | Responsibility |
|-------|----------------|
| **Scanner** | Walk configured directories and discover audio files |
| **Ingestion** | Decode, mono mix (optional), resample to 24 kHz, chunk |
| **Embedding** | Audio → vector; text → vector; track pooling from chunks |
| **Storage** | [Turso](https://github.com/tursodatabase/turso) metadata DB, embedding files, index manifests |
| **Index** | ANN indexes at chunk and track granularity |
| **Retrieval** | Query parsing, search, aggregation, filtering, MMR |
| **Surfaces** | `harmony` CLI, REST API, Python `harmony` package |

### Self-hosting stack

Everything runs in-process on the user's machine:

| Component | Choice | Why |
|-----------|--------|-----|
| Metadata store | **Turso** (`pyturso`) | In-process, SQLite-compatible, Rust core — no separate DB daemon |
| Vector ANN | **FAISS** | Fast approximate search at 10k–100k+ scale; mmap-friendly |
| Embedding files | NumPy memmap / Parquet | Large sequential blobs; keep out of the SQL row store |
| API server | FastAPI (optional) | Thin wrapper; same `Engine` as CLI/library |

Turso holds structured, queryable state (tracks, paths, jobs, index manifests). FAISS holds the dense vector index. This split keeps metadata queries fast and vector search scalable without operating two services.

> **Note:** Turso has experimental in-DB vector support; vector *indexing* is on its roadmap. Harmony uses FAISS for ANN today. If Turso vector indexing matures, the `IndexBackend` interface allows swapping or combining backends without changing the public API.

### Filesystem-only

Harmony reads music directly from disk. Point it at a directory — `/music`, a NAS mount, an external drive — and it walks the tree, hashes files, and embeds them.

---

## 4. Data model

### 4.1 Track

```yaml
track_id: string          # UUIDv5 derived from content_hash — stable across path moves
content_hash: string      # SHA-256 of file bytes (primary identity)
status: enum              # active | missing | removed | failed
primary_path: string      # current canonical path (see §5.4 track_locations)
duration_ms: int
title: string
artist: string
album: string
album_artist: string | null
year: int | null
genre: string | null
disc_number: int | null
track_number: int | null
extra: object             # passthrough tags from file metadata (future: mutagen)
indexed_at: datetime
last_seen_at: datetime    # last time file appeared in a filesystem scan
embedding_version: string # see §4.4
```

### 4.1.1 Track location

Paths are mutable. A track may temporarily exist at multiple paths (duplicates, moves in progress).

```yaml
location_id: string
track_id: string
path: string              # absolute filesystem path
is_primary: bool          # one primary path per track
first_seen_at: datetime
last_seen_at: datetime
```

### 4.1.2 Path history

Optional audit trail for debugging reorganizations:

```yaml
track_id: string
old_path: string
new_path: string
changed_at: datetime
reason: "moved" | "renamed" | "primary_changed"
```

### 4.2 Chunk

```yaml
chunk_id: string          # "{track_id}:{index}"
track_id: string
index: int                # 0-based order in track
start_ms: int
end_ms: int
embedding_path: string    # relative path under data dir
```

### 4.3 Embedding record

```yaml
entity_type: "chunk" | "track" | "query"
entity_id: string
model: string             # e.g. "muq-mulan-large"
model_version: string
dimension: int
vector: float32[d]        # on disk; not always returned by API
created_at: datetime
```

### 4.4 Embedding version

A composite key that triggers re-embedding when any component changes:

```
{model_name}@{model_revision}:{chunk_seconds}s:{overlap_seconds}s:{sample_rate}
```

Example: `muq-mulan@1.0:10:2:24000`

### 4.5 Index manifest

```yaml
index_id: string
granularity: "chunk" | "track"
embedding_version: string
backend: "faiss" | "brute"
metric: "cosine" | "ip"
vector_count: int
built_at: datetime
path: string
```

---

## 5. Ingestion

### 5.1 Filesystem scanner

Harmony discovers music by walking configured directories:

```python
class FilesystemScanner(Protocol):
    def scan(self) -> Iterator[ScannedFile]: ...
    def resolve_path(self, path: str) -> Path: ...
```

| Config | Default | Notes |
|--------|---------|-------|
| `paths` | `[]` | Root directories to scan (required for `index`) |
| `extensions` | `.flac`, `.mp3`, … | Audio extensions to include |
| `follow_symlinks` | `false` | Whether `os.walk` follows symlinks |

The scanner hashes file contents and reads paths. It does not embed. Tag extraction (artist, album, title) via `mutagen` is planned but optional — filename-based fallbacks work for v1.

### 5.2 Audio pipeline

Default parameters (configurable):

| Parameter | Default | Notes |
|-----------|---------|-------|
| `target_sample_rate` | 24000 | MuQ-MuLan expectation |
| `mono` | true | Optional stereo preserve for future models |
| `chunk_seconds` | 10 | |
| `overlap_seconds` | 2 | |
| `min_chunk_seconds` | 1 | Skip tail shorter than this |

Pipeline:

```
resolve → decode → resample → chunk → queue for embedding
```

### 5.3 Library sync (incremental indexing)

Each `index` run is a **reconciliation** between what the filesystem scan finds and what Turso stores. The goal: embed the minimum, never lose searchability during transitions, and survive messy real-world libraries.

#### Reconciliation algorithm

```
scan_result = scanner.scan()         # (path, content_hash, metadata) per file
known       = db.get_active_tracks()

for each (path, hash, meta) in scan_result:
    match by content_hash first, then by path (for in-place edits)

apply transitions (see §5.4)
produce SyncReport
```

#### What triggers work

| Event | Detection | Action | Re-embed? | FAISS update? |
|-------|-----------|--------|-----------|---------------|
| **New file** | `content_hash` unknown | Create track, embed | Yes | Add vectors |
| **File edited in place** | Same path, `content_hash` changed | New `track_id`; old track → `removed` | Yes (new) | Remove old, add new |
| **File moved / renamed** | Known `content_hash`, new path | Update `track_locations`; log path history | No | No |
| **Duplicate copy** | Known `content_hash`, extra path | Add alias location; one embedding shared | No | No |
| **Unchanged file** | Same hash + current `embedding_version` | Touch `last_seen_at` only | No | No |
| **Tag/metadata change** | Same hash, different tags | Update metadata row | No | No |
| **File absent** | Known path not in scan | `status → missing` (see grace period) | No | No |
| **Confirmed deletion** | Missing past grace period | `status → removed`; tombstone in FAISS | No | Mark removed |
| **Model/chunk config change** | `embedding_version` mismatch | Queue re-embed | Yes | Rebuild affected |

Embedding is expensive; everything else is a cheap metadata update in Turso.

### 5.4 Change handling details

#### Moves and reorganization

Users frequently rename albums, reshuffle folders, or switch drive mount points. The engine must treat these as **location updates**, not new music:

1. Scan computes `content_hash` for every discovered file.
2. Hash matches an existing `track_id` → upsert `track_locations` row for the new path.
3. Set `is_primary` on the path seen in the current scan; demote stale paths.
4. Return `track_id` and embeddings unchanged. Search and `search_by_track` keep working.

#### Duplicates

Same audio at `/music/Artist/A.flac` and `/music/Backup/A.flac`:

- One `track_id`, one set of embeddings.
- Multiple `track_locations` rows.
- `primary_path` follows config (`first_seen`, `shortest_path`, or explicit).

#### Grace period for missing files

Files may vanish temporarily (external drive unmounted, NAS offline, sync in progress). Don't immediately purge embeddings.

| Phase | `status` | Searchable? | Duration |
|-------|----------|-------------|----------|
| Seen in last scan | `active` | Yes | — |
| Not seen once | `missing` | Yes | configurable, default 7 days |
| Missing past grace | `removed` | No (tombstoned) | until `purge` |

`missing` tracks remain in the FAISS index so search results don't flicker during brief outages. `removed` tracks are excluded from search; vectors stay on disk until explicit `purge`.

#### In-place content replacement

Same path, different audio (user overwrote the file):

- Old `track_id` (old hash) → `removed`.
- New `track_id` (new hash) → embed from scratch.
- Path history records the transition.

#### Sync report

Every reconciliation returns a summary (CLI, API, logs):

```yaml
SyncReport:
  added: int
  updated_metadata: int
  moved: int              # path changes, no re-embed
  duplicates_found: int
  missing: int            # entered grace period
  removed: int            # past grace period
  reembedded: int
  failed: int
  skipped: int            # unchanged
  duration_ms: int
```

### 5.5 Filesystem watch mode

`harmony index --watch`:

- Use `watchdog` (or platform notify API) with **debounce** (default 5s) to coalesce burst events (e.g. `mv album/ /new/album/`).
- Batch changes into a single reconciliation pass.
- Never embed on every individual notify — wait for quiescence.
- Search remains available during background reconciliation (see §13.2).

### 5.6 Index job states

```
pending → decoding → embedding → indexing → ready
                              ↘ failed (retryable, stores error)
```

Jobs are resumable. Embedding is the expensive step and must be checkpointed per track.

---

## 6. Embedding

### 6.1 Embedder interface

```python
class Embedder(Protocol):
    @property
    def name(self) -> str: ...
    @property
    def dimension(self) -> int: ...

    def embed_audio(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray: ...
    def embed_text(self, text: str) -> np.ndarray: ...
    def embed_text_batch(self, texts: list[str]) -> np.ndarray: ...
```

Default implementation: **MuQ-MuLan** wrapper with lazy model load and batching.

### 6.2 Track embedding derivation

Track vector is derived, never independently modeled:

```
track_embedding = mean(chunk_embeddings)
```

Alternatives exposed as config for advanced callers:

- `mean` (default)
- `max_norm` — chunk with highest L2 norm (proxy for "most energetic" section)
- `first` — first chunk only (useful for intros)

The engine always stores chunk embeddings. Track embeddings are cached for fast track-level index queries.

### 6.3 Text embedding

Text queries use the same joint embedding space as audio. This is the primary search mode:

```
"melancholic rainy evening piano" → vector → nearest chunks/tracks
```

Text embeddings are ephemeral unless explicitly cached (see §8.5).

---

## 7. Storage layout

```
{data_dir}/
├── config.yaml              # engine config snapshot
├── harmony.db               # Turso database (SQLite-compatible single file)
├── embeddings/
│   └── {embedding_version}/
│       ├── chunks/{track_id}.parquet   # or .npy per chunk batch
│       └── tracks.parquet              # denormalized track vectors
├── indexes/
│   └── {embedding_version}/
│       ├── chunk.faiss
│       ├── track.faiss
│       └── manifest.json
└── logs/
```

The entire `data_dir` is **portable**: copy, back up, or rsync it to another machine. No environment-specific connection strings.

### Turso (metadata store)

Harmony uses [Turso](https://github.com/tursodatabase/turso) via **`pyturso`** — an in-process, SQLite-compatible database written in Rust. No separate database server process.

```python
import turso

con = turso.connect(f"{data_dir}/harmony.db")
```

**Why Turso over plain SQLite**

| Property | Benefit for Harmony |
|----------|---------------------|
| In-process | Zero ops — ideal for self-hosted CLI and single-node API |
| SQLite-compatible | Standard SQL, single-file DB, familiar tooling (`turso` CLI) |
| Rust core | Fast metadata filtering (artist, year, path) during search pre-filter |
| `BEGIN CONCURRENT` | Better write throughput when reconciliation batches many path updates |
| Portable file format | `harmony.db` travels with `data_dir` |

**Division of labour**

| Store | Holds | Does not hold |
|-------|-------|---------------|
| Turso (`harmony.db`) | Tracks, paths, chunks, jobs, tombstones, sync state | Raw embedding vectors |
| Embedding files | `float32` vectors per chunk/track | — |
| FAISS indexes | ANN structure for fast similarity search | Metadata |

### Turso schema (minimum)

- `tracks` — metadata, `content_hash`, `status`, `embedding_version`
- `track_locations` — path ↔ `track_id` mapping (supports duplicates and moves)
- `path_history` — optional audit log for reorganization debugging
- `chunks` — chunk boundaries + embedding file pointers
- `embedding_jobs` — resumable job queue
- `indexes` — FAISS manifest rows
- `sync_runs` — history of `SyncReport` summaries
- `query_cache` — optional text → vector cache

Indexes on Turso tables:

```sql
CREATE INDEX idx_tracks_content_hash ON tracks(content_hash);
CREATE INDEX idx_tracks_status ON tracks(status);
CREATE INDEX idx_locations_path ON track_locations(path);
CREATE INDEX idx_locations_track ON track_locations(track_id);
CREATE INDEX idx_chunks_track ON chunks(track_id);
```

Embeddings on disk as **Parquet** (column: `vector` as fixed-size list) or **NumPy** memmap for simplicity in v1.

### FAISS incremental updates

When tracks are added or removed, the ANN index must stay consistent:

| Operation | FAISS action |
|-----------|--------------|
| New track embedded | `add()` vectors; persist index |
| Track `removed` | Mark ID in Turso `index_tombstones`; filter at query time (no full rebuild) |
| Bulk purge | `remove_ids()` or periodic compact rebuild (`harmony index --rebuild-index`) |
| Path-only move | No FAISS change |

Periodic compaction (weekly or when `removed` > 10% of index) reclaims tombstoned slots without re-embedding.

---

## 8. Retrieval API (core product surface)

All search operations return a common result shape:

```yaml
SearchResult:
  items: list[ScoredItem]
  query: QueryInfo          # what was searched, filters applied
  total_indexed: int        # corpus size for context
  took_ms: int

ScoredItem:
  track_id: string
  score: float              # higher = more similar (cosine sim or neg distance)
  rank: int
  match_granularity: "track" | "chunk"
  best_chunk:               # present when chunk-level search + track aggregation
    chunk_id: string
    start_ms: int
    end_ms: int
    chunk_score: float
  metadata: Track            # denormalized for convenience
```

---

### 8.1 Search modes (must-have)

#### `search_by_text`

The headline feature. Natural language → similar music.

```python
engine.search_by_text(
    query: str,
    *,
    k: int = 50,
    granularity: "track" | "chunk" = "track",
    filters: Filters | None = None,
    exclude_track_ids: list[str] | None = None,
)
```

#### `search_by_track`

Audio-audio similarity using a library track as seed.

```python
engine.search_by_track(
    track_id: str,
    *,
    k: int = 50,
    granularity: "track" | "chunk" = "track",
    use: "track_embedding" | "chunk_embedding" = "track_embedding",
    chunk_id: str | None = None,   # search from a specific moment
    filters: Filters | None = None,
    exclude_track_ids: list[str] | None = None,
)
```

When `use="chunk_embedding"` and `chunk_id` is set, find music similar to *that moment*.

#### `search_by_audio`

Ad-hoc audio without indexing the file. Useful for "hum this" or uploaded clip prototypes.

```python
engine.search_by_audio(
    audio: Path | bytes | np.ndarray,
    sample_rate: int | None = None,
    *,
    k: int = 50,
    granularity: "track" | "chunk" = "track",
    filters: Filters | None = None,
)
```

Embeds on the fly; does not add to the library index unless caller runs `index` separately.

#### `search_by_vector`

Escape hatch for applications that compose their own vectors (e.g. blend multiple text prompts).

```python
engine.search_by_vector(
    vector: np.ndarray,
    *,
    k: int = 50,
    granularity: "track" | "chunk" = "track",
    filters: Filters | None = None,
)
```

#### `similar_chunks`

Returns chunk-level hits without aggregating to tracks. For "find the drop that sounds like X" use cases.

```python
engine.similar_chunks(
    query: str | np.ndarray,
    query_type: "text" | "vector",
    *,
    k: int = 100,
    filters: Filters | None = None,
)
```

---

### 8.2 Query composition (must-have)

#### `search_by_blend`

Weighted combination of multiple text and/or track seeds. This is retrieval-level blending, not playlist logic.

```python
engine.search_by_blend(
    terms: list[BlendTerm],
    *,
    k: int = 50,
    granularity: "track" | "chunk" = "track",
    normalize_weights: bool = True,
    filters: Filters | None = None,
)

BlendTerm:
  type: "text" | "track" | "vector"
  value: str | np.ndarray
  weight: float
```

Implementation: embed each term → L2-normalize → weighted average → L2-normalize → search.

---

### 8.3 Chunk → track aggregation

When `granularity="track"` but the chunk index is queried (recommended default for quality):

| Strategy | Behavior |
|----------|----------|
| `max` (default) | Track score = best matching chunk score |
| `mean_top3` | Mean of top 3 chunk scores per track |
| `sum_topk` | Sum of top-k chunk scores (rewards multi-match tracks) |

Expose as `aggregation: "max" | "mean_top3" | "sum_topk"`.

---

### 8.4 Filtering (must-have)

Pre-filter (metadata SQL) or post-filter (on results). Pre-filter is preferred at scale.

```python
Filters:
  artists: list[str] | None          # exact or fuzzy, configurable
  albums: list[str] | None
  genres: list[str] | None
  year_min: int | None
  year_max: int | None
  duration_min_ms: int | None
  duration_max_ms: int | None
  paths_glob: list[str] | None        # filter by path pattern
```

---

### 8.5 Diversity in retrieval (should-have)

Lightweight deduplication at search time — not playlist sequencing.

```python
SearchOptions:
  mmr: bool = False                  # Maximal Marginal Relevance
  mmr_lambda: float = 0.5
  max_per_artist: int | None = None  # cap artist frequency in top-k
```

This helps downstream apps get varied candidate pools without baking in playlist rules.

---

### 8.6 Direct embedding access (should-have)

For apps that want custom ranking or their own ANN experiments:

```python
engine.embed_text(text: str) -> np.ndarray
engine.embed_text_batch(texts: list[str]) -> np.ndarray
engine.get_track_embedding(track_id: str) -> np.ndarray
engine.get_chunk_embeddings(track_id: str) -> list[tuple[Chunk, np.ndarray]]
engine.get_tracks_batch(track_ids: list[str]) -> list[Track]
```

---

### 8.7 Text query cache (nice-to-have)

```python
engine.cache_query(text: str) -> str   # returns query_id
engine.search_by_query_id(query_id: str, **kwargs) -> SearchResult
```

Avoids re-tokenizing hot prompts in UI loops.

---

## 9. HTTP API

Base path: `/v1`. JSON in/out. OpenAPI spec generated from route definitions.

### 9.1 Indexing

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/index` | Start sync/reconcile job `{ paths?, full_rescan? }` |
| `GET` | `/index/{job_id}` | Job status + progress + latest `SyncReport` |
| `DELETE` | `/index/{job_id}` | Cancel running job |
| `GET` | `/library/stats` | Track count by status, chunk count, index health, embedding version |
| `GET` | `/library/tracks` | Paginated track list + metadata |
| `GET` | `/library/tracks/{track_id}` | Single track + chunk map + all known paths |
| `GET` | `/library/sync` | History of sync runs and reports |
| `POST` | `/library/purge` | Remove `removed` tracks and orphan vectors from disk |

### 9.2 Search

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search/text` | Text query |
| `POST` | `/search/track` | Seed track query |
| `POST` | `/search/audio` | Multipart audio upload |
| `POST` | `/search/blend` | Weighted multi-term query |
| `POST` | `/search/vector` | Raw vector (base64 float32) |
| `POST` | `/search/chunks` | Chunk-level results |

### 9.3 Embeddings

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/embed/text` | Text → vector (no search) |
| `GET` | `/embed/track/{track_id}` | Stored track vector |
| `GET` | `/embed/track/{track_id}/chunks` | All chunk vectors for a track |

### 9.4 Example

```http
POST /v1/search/text
Content-Type: application/json

{
  "query": "dreamy ambient night drive",
  "k": 50,
  "granularity": "track",
  "aggregation": "max",
  "filters": { "year_min": 2010 },
  "options": { "max_per_artist": 3 }
}
```

```json
{
  "items": [
    {
      "track_id": "a1b2c3",
      "score": 0.82,
      "rank": 1,
      "match_granularity": "track",
      "best_chunk": {
        "chunk_id": "a1b2c3:4",
        "start_ms": 32000,
        "end_ms": 42000,
        "chunk_score": 0.85
      },
      "metadata": {
        "title": "Night Owl",
        "artist": "Example Artist",
        "album": "Nocturne",
        "duration_ms": 245000
      }
    }
  ],
  "query": { "type": "text", "value": "dreamy ambient night drive" },
  "total_indexed": 12450,
  "took_ms": 38
}
```

---

## 10. CLI

Binary name: `harmony`

```
harmony init [--data-dir PATH]
harmony index [paths...] [--full] [--watch]
harmony status                              # library stats + last sync report
harmony sync-history [--limit 10]           # past reconciliation reports
harmony search text <query> [--k 50] [--json]
harmony search track <track_id> [--chunk <chunk_id>] [--k 50]
harmony search blend --text "ambient:0.6" --track <id>:0.4
harmony serve [--host] [--port] [--data-dir]
harmony purge [--removed] [--orphans]
```

`harmony index` prints a `SyncReport` on completion (added, moved, reembedded, etc.).  
`harmony search` prints human-readable tables by default; `--json` for scripting.

---

## 11. Python library

```python
from harmony import Engine

engine = Engine(data_dir="~/.harmony")

# Index or re-sync (adds, moves, removes — moves don't re-embed)
report = engine.index(paths=["/music"])
print(f"Moved {report.moved}, added {report.added}, re-embedded {report.reembedded}")

# Search
results = engine.search_by_text("melancholic piano", k=30)

for item in results.items:
    print(item.score, item.metadata.artist, item.metadata.title)
    if item.best_chunk:
        print(f"  best moment: {item.best_chunk.start_ms}ms")
```

The library is the source of truth. CLI and HTTP are thin wrappers.

---

## 12. Configuration

```yaml
data_dir: ~/.harmony

database:
  path: harmony.db      # Turso/SQLite file inside data_dir
  # no host, port, or credentials — fully local

embedding:
  model: muq-mulan
  device: auto          # cuda | cpu | auto
  batch_size: 16
  keep_alive: forever   # immediate | minutes (30) | forever
  preload_on_serve: true

audio:
  target_sample_rate: 24000
  mono: true
  chunk_seconds: 10
  overlap_seconds: 2

sync:
  missing_grace_days: 7       # before a missing file becomes removed
  watch_debounce_seconds: 5   # coalesce filesystem events
  hash_chunk_size_mb: 4       # streaming SHA-256 for large FLACs
  primary_path_policy: scan   # scan | shortest | first_seen

index:
  backend: faiss
  metric: cosine
  build_track_index: true
  build_chunk_index: true
  compact_threshold: 0.10     # rebuild FAISS when >10% tombstoned

retrieval:
  default_k: 50
  default_granularity: track
  default_aggregation: max

filesystem:
  paths: []                 # e.g. ["/music", "/mnt/nas/audio"]
  extensions: [".flac", ".mp3", ".m4a", ".aac", ".ogg", ".wav", ".opus"]
  follow_symlinks: false
```

---

## 13. Operational concerns

### 13.1 Self-hosting requirements

| Requirement | Detail |
|-------------|--------|
| Disk | `data_dir` size ≈ embeddings (~1–4 KB/track vector) + FAISS index + `harmony.db` (small) |
| RAM | FAISS index mmap'd; GPU RAM for embedding only |
| Processes | One Python process; Turso in-process (no DB daemon) |
| Network | None required after model weights are cached |
| Backup | Copy `data_dir`; `harmony.db` is a single consistent SQLite-format file |

A minimal deployment is `pip install harmony` → `harmony init` → `harmony index /music` → `harmony serve`.

### 13.2 Performance targets (informative)

| Corpus | Initial embed | Incremental sync (moves only) | Text search p95 |
|--------|---------------|-------------------------------|-----------------|
| 10k tracks | < 2h (GPU) | < 30s (hash + Turso writes) | < 100ms |
| 100k tracks | background job | < 5min (hash-bound) | < 200ms |

Hashing dominates move-only syncs; embedding dominates adds. Brute-force cosine is acceptable for MVP (<2k tracks) before FAISS kicks in.

### 13.3 Concurrency

- **Indexing:** single-writer per `data_dir` (one embed job at a time). Turso `BEGIN CONCURRENT` allows batched path/metadata writes during reconciliation without blocking readers.
- **Search during sync:** always available. Newly embedded tracks appear after their FAISS `add()` completes; removed tracks are filtered via tombstones immediately.
- **FAISS:** mmap'd index; concurrent read queries from API threads.
- **API:** embed and search can overlap if GPU memory allows.

### 13.4 Failure handling

- Per-track failures do not abort the job; recorded in `embedding_jobs` with error message. `status = failed` with retry on next sync.
- Unreadable file mid-scan → skip, log, continue.
- Corrupt FAISS index → detect via manifest checksum; `harmony index --rebuild-index` rebuilds ANN from stored vectors without re-embedding.
- Corrupt `harmony.db` → restore from backup; embeddings and FAISS files are independent and reusable.

### 13.5 Versioning & migrations

- `embedding_version` change → background re-embed job for affected tracks.
- Schema migrations via Turso SQL + `schema_version` table (standard SQLite migration pattern).
- `content_hash` algorithm change → one-time background re-hash (no re-embed unless hash differs).
- Breaking API changes bump `/v2`.

---

## 14. Explicit non-goals

The engine **does not**:

- Generate ordered playlists or sequences
- Apply energy curves, BPM matching, or harmonic mixing rules
- Cluster candidates into sub-vibes for journey planning
- Export M3U/PLS (downstream apps use `track_id` + metadata)
- Media server API integrations
- Push playlists to external services
- Train or fine-tune MuQ-MuLan
- Stream audio to clients (only paths/IDs)

---

## 15. MVP → v1 roadmap

### Phase 0 — MVP (prove the loop)

- [ ] Filesystem scanner
- [ ] Audio load, resample, chunk
- [ ] MuQ-MuLan embed (track-level only)
- [ ] Brute-force cosine search
- [ ] Turso (`pyturso`) metadata + numpy embedding files
- [ ] Content-hash identity + path moves without re-embed
- [ ] CLI: `index`, `search text`, `status`
- [ ] Python library with `search_by_text`, `search_by_track`

### Phase 1 — Production retrieval

- [ ] Chunk-level embeddings + storage
- [ ] FAISS chunk + track indexes + tombstone filtering
- [ ] Chunk → track aggregation (`max`)
- [ ] `search_by_blend`, `search_by_audio`
- [ ] Filters (artist, year, duration) via Turso pre-filter
- [ ] Full library sync: adds, removes, moves, duplicates, `SyncReport`
- [ ] Missing-file grace period
- [ ] HTTP API + OpenAPI

### Phase 2 — Scale & polish

- [ ] `similar_chunks` mode
- [ ] Tag extraction via mutagen (artist, album from file tags)
- [ ] MMR + `max_per_artist`
- [ ] Filesystem watch mode with debounce
- [ ] `search_by_vector` + batch text embed
- [ ] FAISS compact rebuild without re-embed
- [ ] `path_history` audit log

---

## 16. What downstream projects get for free

A playlist generator, radio mode, or "vibe DJ" app built on Harmony Engine can:

1. Call `search_by_text` / `search_by_blend` to get a **candidate pool** with scores.
2. Pull **chunk timestamps** to align transitions or crossfades.
3. Use `search_by_track` for "more like this" without re-implementing embeddings.
4. Fetch raw vectors for custom sequencing (graph walk, energy curves) in its own repo.
5. Apply its own rules on top of `ScoredItem` lists — the engine never needs to know.
6. Rely on **stable `track_id`s** across library reorganizations — playlist apps store IDs, not paths.
7. Run `engine.index()` on a schedule or watch; handle `SyncReport` to show "12 new tracks indexed" in their UI.

---

## 17. Open questions

| Question | Lean |
|----------|------|
| Track ID derivation | `UUIDv5(namespace, content_hash)` — deterministic, survives moves |
| Hash without full read | Stream SHA-256; optional size+mtime fast-path to skip re-hash of unchanged files |
| Parquet vs memmap npy for v1 | npy per track (simpler); migrate to Parquet at 50k+ chunks |
| GPU requirement | CPU works; document GPU as recommended for indexing |
| Include path in search results? | Return `primary_path`; include all `track_locations` on detail endpoint |
| Turso beta status | Acceptable for self-hosted use with backup guidance; fall back to `sqlite3` stdlib if `pyturso` unavailable |
| FAISS vs Turso vectors | FAISS for ANN now; revisit when Turso vector indexing ships |
| License for model weights | Document MuQ-MuLan terms in README |

---

## 18. Suggested project layout

```
harmony/
├── pyproject.toml
├── README.md
├── harmony/
│   ├── __init__.py          # Engine facade
│   ├── config.py
│   ├── scanner/
│   │   ├── base.py
│   │   └── filesystem.py    # directory walk + content hashing
│   ├── audio/
│   │   ├── loader.py
│   │   ├── resample.py
│   │   └── chunking.py
│   ├── embedding/
│   │   ├── base.py
│   │   └── muq_mulan.py
│   ├── storage/
│   │   ├── db.py            # Turso connection + migrations
│   │   ├── metadata.py
│   │   ├── vectors.py
│   │   └── sync.py          # reconciliation + SyncReport
│   ├── index/
│   │   ├── base.py
│   │   └── faiss_index.py
│   ├── retrieval/
│   │   ├── search.py
│   │   ├── aggregate.py
│   │   └── filters.py
│   ├── api/
│   │   └── app.py
│   └── cli/
│       └── main.py
└── tests/
```

No `playlist/` module. That lives in a separate repository that depends on `harmony` as a library or talks to its HTTP API.
