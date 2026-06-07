# Harmony Engine — Specification

A library indexing and vector search engine powered by **MuQ-MuLan**. It turns a music collection into searchable embeddings and exposes retrieval primitives that higher-level applications (playlist generators, DJ tools, recommendation UIs, automations) can build on.

**In scope:** ingest, embed, persist, index, search.  
**Out of scope:** playlist sequencing, energy curves, graph walks, M3U export, pushing playlists to media servers.

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

---

## 3. Architecture

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐     ┌──────────────┐
│   Sources   │ ──▶ │  Ingestion   │ ──▶ │  Embedding  │ ──▶ │   Storage    │
│ local/API   │     │ decode/chunk │     │  MuQ-MuLan  │     │ sqlite/parquet│
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
| **Sources** | Discover tracks from filesystem globs, Jellyfin, Subsonic |
| **Ingestion** | Decode, mono mix (optional), resample to 24 kHz, chunk |
| **Embedding** | Audio → vector; text → vector; track pooling from chunks |
| **Storage** | Metadata DB, embedding files, index manifests |
| **Index** | ANN indexes at chunk and track granularity |
| **Retrieval** | Query parsing, search, aggregation, filtering, MMR |
| **Surfaces** | `harmony` CLI, REST API, Python `harmony` package |

---

## 4. Data model

### 4.1 Track

```yaml
track_id: string          # stable UUID, survives path moves if keyed by content hash
source_id: string         # upstream ID (Jellyfin GUID, Subsonic id, or "local")
path: string              # absolute path or virtual URI
content_hash: string      # SHA-256 of file bytes (or size+mtime fallback)
duration_ms: int
title: string
artist: string
album: string
album_artist: string | null
year: int | null
genre: string | null
disc_number: int | null
track_number: int | null
extra: object             # passthrough tags from source adapter
indexed_at: datetime
embedding_version: string # see §4.4
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

### 5.1 Source adapters

Each adapter implements `SourceAdapter`:

```python
class SourceAdapter(Protocol):
    def scan(self) -> Iterator[TrackRef]: ...
    def resolve_audio(self, track_id: str) -> Path | bytes: ...
    def fetch_metadata(self, track_id: str) -> TrackMetadata: ...
```

**v1 adapters:**

| Adapter | Config |
|---------|--------|
| `local` | Root paths, glob patterns, follow symlinks |
| `jellyfin` | Base URL, API key, user ID, music libraries |
| `subsonic` | Base URL, username, password/token |

Adapters only discover and fetch. They do not embed.

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

### 5.3 Incremental indexing

On each `index` run:

1. Scan source for current track set.
2. Compare `content_hash` + `embedding_version` against stored records.
3. **New** → full pipeline.
4. **Changed** → re-embed; invalidate old chunks.
5. **Unchanged** → skip.
6. **Missing from source** → mark `status = removed` (soft delete; vectors kept until `purge`).

Optional filesystem watcher mode (`index --watch`) for local sources.

### 5.4 Index job states

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
├── metadata.db              # SQLite: tracks, chunks, jobs, indexes
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

### SQLite tables (minimum)

- `tracks` — metadata + indexing state
- `chunks` — chunk boundaries + embedding file pointers
- `embedding_jobs` — resumable job queue
- `indexes` — manifest rows
- `query_cache` — optional text → vector cache

Embeddings on disk as **Parquet** (column: `vector` as fixed-size list) or **NumPy** memmap for simplicity in v1.

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
  source_ids: list[str] | None       # limit to one adapter
  paths_glob: list[str] | None       # local path patterns
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
| `POST` | `/index` | Start indexing job `{ source, paths?, full_rescan? }` |
| `GET` | `/index/{job_id}` | Job status + progress |
| `DELETE` | `/index/{job_id}` | Cancel running job |
| `GET` | `/library/stats` | Track count, chunk count, index health, embedding version |
| `GET` | `/library/tracks` | Paginated track list + metadata |
| `GET` | `/library/tracks/{track_id}` | Single track + chunk map |
| `POST` | `/library/purge` | Remove soft-deleted tracks and orphan vectors |

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
harmony index <source> [paths...] [--full] [--watch]
harmony status
harmony search text <query> [--k 50] [--json]
harmony search track <track_id> [--chunk <chunk_id>] [--k 50]
harmony search blend --text "ambient:0.6" --track <id>:0.4
harmony serve [--host] [--port] [--data-dir]
harmony purge [--removed] [--orphans]
```

`harmony search` prints human-readable tables by default; `--json` for scripting.

---

## 11. Python library

```python
from harmony import Engine

engine = Engine(data_dir="~/.harmony")

# One-time setup
engine.index(source="local", paths=["/music"])

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

embedding:
  model: muq-mulan
  device: auto          # cuda | cpu | auto
  batch_size: 16

audio:
  target_sample_rate: 24000
  mono: true
  chunk_seconds: 10
  overlap_seconds: 2

index:
  backend: faiss
  metric: cosine
  build_track_index: true
  build_chunk_index: true

retrieval:
  default_k: 50
  default_granularity: track
  default_aggregation: max

sources:
  local:
    paths: []
  jellyfin:
    url: null
    api_key: null
```

---

## 13. Operational concerns

### 13.1 Performance targets (informative)

| Corpus | Index build | Text search p95 |
|--------|-------------|-----------------|
| 10k tracks | < 2h initial embed (GPU) | < 100ms |
| 100k tracks | incremental only on changes | < 200ms |

Brute-force cosine is acceptable for MVP (<2k tracks) before FAISS kicks in.

### 13.2 Concurrency

- Indexing: single-writer (one embed job at a time per data dir).
- Search: multi-reader (read-only index mmap / FAISS concurrent search).
- API: embed and search can run concurrently if GPU memory allows batching.

### 13.3 Failure handling

- Per-track failures do not abort the job; recorded in `embedding_jobs` with error message.
- Corrupt index → detect via manifest checksum; offer `harmony index --rebuild-index` without re-embedding.

### 13.4 Versioning & migrations

- `embedding_version` change → background re-embed job.
- Schema migrations via SQLite pragmas + version table.
- Breaking API changes bump `/v2`.

---

## 14. Explicit non-goals

The engine **does not**:

- Generate ordered playlists or sequences
- Apply energy curves, BPM matching, or harmonic mixing rules
- Cluster candidates into sub-vibes for journey planning
- Export M3U/PLS (downstream apps use `track_id` + metadata)
- Push playlists to Jellyfin/Subsonic (adapter is read-only in v1)
- Train or fine-tune MuQ-MuLan
- Stream audio to clients (only paths/IDs)

---

## 15. MVP → v1 roadmap

### Phase 0 — MVP (prove the loop)

- [ ] Local source adapter
- [ ] Audio load, resample, chunk
- [ ] MuQ-MuLan embed (track-level only)
- [ ] Brute-force cosine search
- [ ] SQLite metadata + numpy embedding files
- [ ] CLI: `index`, `search text`, `status`
- [ ] Python library with `search_by_text`, `search_by_track`

### Phase 1 — Production retrieval

- [ ] Chunk-level embeddings + storage
- [ ] FAISS chunk + track indexes
- [ ] Chunk → track aggregation (`max`)
- [ ] `search_by_blend`, `search_by_audio`
- [ ] Filters (artist, year, duration)
- [ ] Incremental indexing by content hash
- [ ] HTTP API + OpenAPI

### Phase 2 — Scale & integrations

- [ ] Jellyfin + Subsonic source adapters
- [ ] `similar_chunks` mode
- [ ] MMR + `max_per_artist`
- [ ] Filesystem watch mode
- [ ] `search_by_vector` + batch text embed
- [ ] Index rebuild without re-embed

---

## 16. What downstream projects get for free

A playlist generator, radio mode, or "vibe DJ" app built on Harmony Engine can:

1. Call `search_by_text` / `search_by_blend` to get a **candidate pool** with scores.
2. Pull **chunk timestamps** to align transitions or crossfades.
3. Use `search_by_track` for "more like this" without re-implementing embeddings.
4. Fetch raw vectors for custom sequencing (graph walk, energy curves) in its own repo.
5. Apply its own rules on top of `ScoredItem` lists — the engine never needs to know.

---

## 17. Open questions

| Question | Lean |
|----------|------|
| Track ID stability | Prefer content hash; fall back to path + source_id |
| Parquet vs memmap npy for v1 | npy per track (simpler); migrate to Parquet at 50k+ chunks |
| GPU requirement | CPU works; document GPU as recommended for indexing |
| Include path in search results? | Yes for local; URI for API sources |
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
│   ├── sources/
│   │   ├── local.py
│   │   ├── jellyfin.py
│   │   └── subsonic.py
│   ├── audio/
│   │   ├── loader.py
│   │   ├── resample.py
│   │   └── chunking.py
│   ├── embedding/
│   │   ├── base.py
│   │   └── muq_mulan.py
│   ├── storage/
│   │   ├── metadata.py
│   │   └── vectors.py
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
