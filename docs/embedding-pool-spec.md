# Harmony Embedding Pool — Specification

A centralized, community-maintained cache of MuQ-MuLan track embeddings. Harmony clients look up vectors before running local GPU work, and optionally contribute vectors after embedding — dramatically reducing index time for large libraries.

**In scope:** canonical song identity, pool lookup/upload protocol, server-side consensus and outlier rejection, client integration in the embed pipeline.  
**Out of scope:** chunk-level vectors (v1), P2P transport, hosting/ops runbooks, MusicBrainz resolver implementation details (referenced, not specified here).

**Deployment model:** optional. The engine remains fully self-hosted; the pool is an accelerator operated on a central VPS. Clients work offline when the pool is unreachable or disabled.

---

## 1. Purpose

Embedding a large library with MuQ-MuLan is slow — hours to days on CPU, and still costly on GPU for collections of tens of thousands of tracks. Many users share overlapping catalog (same releases, same scene rips, mainstream albums). Pre-computed embeddings keyed by song identity can skip redundant work.

The pool answers:

> *Before I embed this track locally, does the community already have a trustworthy vector I can import?*

And reciprocally:

> *After I embed locally, can I contribute that vector to help others?*

Vectors are **semantic consensus approximations** — good for text search and rough similarity across a library. They are not a byte-for-byte substitute for embedding the user's exact file when maximum per-track fidelity is required (see §9).

---

## 2. Design principles

1. **Opt-in.** Pool use and contribution are disabled by default; users enable explicitly.
2. **Local engine stays authoritative.** Imported vectors land in the same `VectorStore` as locally computed ones. The user's `content_hash` / `track_id` is unchanged.
3. **Version-strict.** Every request carries `embedding_version`. Vectors are only interchangeable when it matches exactly (model, chunking, sample rate). Unsupported or retired versions are rejected; deprecated versions are lookup-only.
4. **Trust at the source.** Consensus, outlier rejection, and confidence scoring run on the server before vectors are served to clients.
5. **Layered identity.** Exact file hash first; canonical song ID second — never rely on fuzzy metadata alone.
6. **Fail open locally.** Pool miss or low confidence → fall back to local embed. Pool outage must never block indexing.
7. **Privacy-aware.** Lookups reveal song identity to the pool operator; contribution is explicit.

---

## 3. Architecture

```
┌──────────────────┐         ┌─────────────────────────────┐
│  Harmony Engine  │         │  Embedding Pool (VPS)         │
│                  │         │                               │
│  Scanner         │         │  Identity resolver            │
│       ↓          │  GET    │  Consensus store              │
│  Embed pipeline ─┼────────▶│  Outlier rejection            │
│       ↓          │  POST   │  Confidence scoring           │
│  VectorStore     │◀────────┤                               │
└──────────────────┘         └─────────────────────────────┘
```

### Lookup path (client)

1. On first use, `GET /v1/pool/versions` — confirm local `embedding_version` is supported (see §5.3).
2. Scan discovers track; metadata and `content_hash` are known.
3. Resolve **canonical song ID** (see §4).
4. `GET /v1/pool/embeddings` with identity + `embedding_version`.
5. If hit and `confidence ≥ threshold` → import vector into `VectorStore`, mark track embedded, skip GPU.
6. On miss or low confidence → local `TrackEmbeddingPipeline.embed_track`.
7. If contribution enabled and version is **active** → `POST /v1/pool/embeddings` with vector + metadata.

### Integration point

The hook lives in `TrackEmbeddingPipeline.embed_track`, before audio load and model inference:

```python
def embed_track(self, track: Track) -> np.ndarray:
    if self.pool is not None:
        cached = self.pool.try_fetch(track)
        if cached is not None:
            self._persist_imported(track, cached)
            return cached.vector
    # ... existing load → chunk → embed path ...
```

---

## 4. Canonical song identity

Harmony locally keys tracks by `content_hash` (SHA-256 of file bytes). The pool keys by **canonical song ID** so different rips of the same recording can share a vector.

Lookups proceed in priority order:

| Tier | Key | Confidence | Notes |
|------|-----|------------|-------|
| **1 — Exact** | `content_hash` | Highest | Byte-identical file. No averaging needed; store and serve the exact vector. |
| **2 — Recording** | MusicBrainz `recording_mbid` | High | Resolved once at ingest via MusicBrainz search API. Stable across encodings. |
| **3 — Normalized metadata** | `canonical_id` derived from normalized `(artist, title, duration_ms, album?)` | Medium | Fallback when MBID unavailable. Requires tag extraction (mutagen). |

### Tier 1: content hash

If the pool has an entry keyed by `content_hash` + `embedding_version`, serve it directly. No consensus blending — the vector is deterministic for identical inputs.

### Tier 2: MusicBrainz recording MBID

At ingest (or first pool interaction), resolve metadata to a recording MBID:

```
artist + title + album + duration → MusicBrainz search → recording_mbid
```

Store `recording_mbid` on the local track record. Pool entries keyed by MBID hold the **consensus aggregate** across contributors.

### Tier 3: normalized metadata fallback

When MBID resolution fails, derive a stable ID:

```
canonical_id = sha256(normalize(artist) + "|" + normalize(title) + "|" + duration_bucket)
```

**Normalization rules (v1):**

- Lowercase, strip leading/trailing whitespace
- Remove punctuation except spaces
- Strip leading "The "
- Collapse `feat.`, `ft.`, `featuring` to a canonical separator
- `duration_bucket` = `duration_ms` rounded to nearest 1000 ms

**Disambiguation filters** (reject candidate matches):

- Duration must agree within ±3000 ms
- Fuzzy title + artist score (e.g. RapidFuzz token-set ratio) must be ≥ 90

### Local vs pool identity

| Field | Scope | Used for |
|-------|-------|----------|
| `track_id` / `content_hash` | Local engine | File identity, index, search-by-track on user's audio |
| `canonical_song_id` | Pool | Shared cache lookup and contribution |
| `recording_mbid` | Both (stored locally) | Preferred canonical song ID |

A single local track always keeps its `content_hash`-derived `track_id`. The imported vector is associated with that track in `VectorStore` regardless of which pool tier supplied it.

---

## 5. Embedding version (compatibility key)

Every pool request includes an **`embedding_version`** — the primary compatibility gate between client and server. Clients only download vectors for the version they are running; the server only accepts uploads tagged with a version it currently supports. Cross-version mixing is never allowed.

### 5.1 Format

Derived from Harmony's local `Config.embedding_version()`:

```
{model}@{revision}:{chunk_seconds}:{overlap_seconds}:{sample_rate}
```

Example: `muq-mulan@0.1:10:2:24000`

Any change to model checkpoint, model revision, chunk length, overlap, or target sample rate produces a **new** version string and therefore a separate pool namespace.

### 5.2 Server version policy

The pool operator maintains a version manifest:

| Status | Meaning |
|--------|---------|
| **active** | Lookups and uploads accepted |
| **deprecated** | Lookups accepted; uploads rejected (read-only sunset) |
| **retired** | Lookups and uploads rejected; data retained but not served |

When Harmony ships a new default model (e.g. `muq-mulan@0.2:10:2:24000`), the operator adds it as **active** and may deprecate the previous version. Old entries remain in storage but are invisible to clients on the new version.

### 5.3 Client version negotiation

On pool connect (first lookup or explicit prefetch), the client:

1. Computes `embedding_version` from local config.
2. Calls `GET /v1/pool/versions` (or reads `supported_embedding_versions` from health).
3. If local version is **not** in the server's active or deprecated list → disable pool for this run; log a warning; embed locally only.
4. If local version is **deprecated** → allow lookup, disable contribution (even if `contribute: true`).
5. If local version is **active** → full lookup + optional contribution.

The client **never** sends a different `embedding_version` than its local config produces. The client **never** imports a vector whose `embedding_version` in the response does not exactly match local config (defensive check).

### 5.4 Dimension coupling

`embedding_version` implicitly defines vector dimension (512 for MuQ-MuLan-large). If a future model uses a different dimension, the version string changes and the API returns `dimension` explicitly in lookup responses. Uploads with mismatched dimension are rejected.

---

## 6. Server data model

### 6.1 Pool entry (consensus)

```yaml
canonical_song_id: string       # recording_mbid or normalized hash
identity_tier: enum             # exact | recording | metadata
embedding_version: string
aggregate_vector: float32[512]  # L2-normalized
upload_count: int
confidence: float               # 0.0–1.0, derived from agreement stats
duration_ms: int                # median across uploads
title: string                   # display / debug
artist: string
album: string | null
created_at: datetime
updated_at: datetime
stats:
  mean_similarity: float        # mean pairwise sim of accepted uploads to aggregate
  min_similarity: float         # sim of most recent accepted upload
  rejected_count: int           # outlier rejections
```

### 6.2 Exact-file entry

```yaml
content_hash: string            # SHA-256 hex
embedding_version: string
vector: float32[512]            # exact, not averaged
uploaded_at: datetime
```

Exact entries take precedence over consensus entries for the same lookup.

### 6.3 Quarantine record (outliers)

```yaml
canonical_song_id: string
embedding_version: string
vector: float32[512]
rejected_at: datetime
similarity_to_aggregate: float
reason: string                  # "below_reject_threshold" | "duration_mismatch" | ...
uploader_key_hash: string | null
```

Quarantined vectors are not served. Operators may inspect for abuse or mis-tagged uploads.

---

## 7. Consensus and trust algorithm

All trust logic runs **server-side** on upload. Clients receive only accepted aggregates (or exact vectors) plus a `confidence` score.

### 7.1 Upload handling

```
on upload(canonical_song_id, vector, metadata, embedding_version):

    if embedding_version is retired or not in manifest:
        return rejected_version(version=embedding_version)
    if embedding_version is deprecated:
        return rejected_version(version=embedding_version, reason="deprecated")
    assert vector.shape == (512,) and is finite
    vector = L2_normalize(vector)

    # Tier 1 fast path
    if metadata.content_hash is present:
        store exact entry at content_hash key
        # also feed into consensus below if a canonical_song_id is resolved

    entry = get_or_create_consensus(canonical_song_id, embedding_version)

    if entry.upload_count == 0:
        entry.aggregate_vector = vector
        entry.upload_count = 1
        entry.confidence = 0.3    # single contributor — low confidence
        return accepted

    sim = cosine_similarity(entry.aggregate_vector, vector)

    if sim < REJECT_THRESHOLD:      # default: 0.85
        quarantine(vector, sim)
        entry.rejected_count += 1
        return rejected

    if sim < WARN_THRESHOLD:        # default: 0.95
        # Suspicious but not fatal — heavy downweight
        w_new = 1 / (entry.upload_count + 10)
    else:
        # Strong agreement — standard incremental update
        w_new = 1 / (entry.upload_count + 1)

    entry.aggregate_vector = L2_normalize(
        (1 - w_new) * entry.aggregate_vector + w_new * vector
    )
    entry.upload_count += 1
    recompute_confidence(entry)
    return accepted
```

### 7.2 Confidence score

Expose to clients so they can decide whether to trust an import or re-embed locally.

```
confidence = min(1.0,
    0.3 + 0.7 * mean_similarity * (1 - exp(-upload_count / 5))
)
```

| `upload_count` | `mean_similarity` | Approximate `confidence` |
|----------------|-------------------|--------------------------|
| 1 | — | 0.3 |
| 3 | 0.97 | ~0.85 |
| 10 | 0.96 | ~0.95 |
| 1 | 0.70 | 0.3 (would have been rejected) |

**Client import threshold (default):** `confidence ≥ 0.7`. Below threshold, the client treats the response as a miss and embeds locally. Configurable.

### 7.3 Constants (server config)

| Constant | Default | Description |
|----------|---------|-------------|
| `REJECT_THRESHOLD` | 0.85 | Below this cosine sim → quarantine, no blend |
| `WARN_THRESHOLD` | 0.95 | Between reject and warn → blend with heavy downweight |
| `DURATION_TOLERANCE_MS` | 3000 | Max duration difference for metadata-tier identity |

---

## 8. HTTP API

Base URL configured by client (e.g. `https://pool.harmony.example`). All payloads JSON. Vectors transmitted as base64-encoded `float32` little-endian blobs (512 × 4 = 2048 bytes) or as a JSON float array (simpler, larger on wire).

**URL versioning** (`/v1/`) refers to the pool HTTP API schema. **`embedding_version`** refers to the embedding model/config compatibility — orthogonal concerns. A `/v2/` API could still serve the same `embedding_version` strings.

### 8.0 Versions (discovery)

```
GET /v1/pool/versions
```

No parameters. Returns the server's embedding version manifest. Clients should call this once per session before bulk index operations.

**Response 200:**

```json
{
  "api_version": "v1",
  "embedding_versions": [
    {
      "embedding_version": "muq-mulan@0.2:10:2:24000",
      "status": "active",
      "dimension": 512,
      "entry_count": 85000
    },
    {
      "embedding_version": "muq-mulan@0.1:10:2:24000",
      "status": "deprecated",
      "dimension": 512,
      "entry_count": 142000,
      "deprecated_at": "2026-06-01T00:00:00Z",
      "retires_at": "2026-12-01T00:00:00Z"
    }
  ]
}
```

### 8.1 Lookup

```
GET /v1/pool/embeddings
```

**Query parameters (provide as many identity fields as available):**

| Param | Required | Description |
|-------|----------|-------------|
| `embedding_version` | yes | Must match client config |
| `content_hash` | no | Tier 1 exact lookup |
| `recording_mbid` | no | Tier 2 |
| `artist` | no | Tier 3 fallback |
| `title` | no | Tier 3 fallback |
| `album` | no | Tier 3 fallback |
| `duration_ms` | no | Tier 3 disambiguation |

**Resolution order:** `content_hash` → `recording_mbid` → normalized metadata match.

**Response 200:**

```json
{
  "found": true,
  "tier": "recording",
  "canonical_song_id": "a1b2c3d4-...",
  "embedding_version": "muq-mulan@0.1:10:2:24000",
  "vector": "<base64>",
  "dimension": 512,
  "confidence": 0.91,
  "upload_count": 7,
  "duration_ms": 245000
}
```

**Response 404 (not found):**

```json
{
  "found": false,
  "embedding_version": "muq-mulan@0.2:10:2:24000"
}
```

**Response 400 (unsupported version):**

```json
{
  "error": "unsupported_embedding_version",
  "embedding_version": "muq-mulan@0.1:10:2:24000",
  "message": "This embedding version is retired. See GET /v1/pool/versions.",
  "supported_versions": ["muq-mulan@0.2:10:2:24000"]
}
```

Returned when `embedding_version` is retired or unknown. Client should disable pool and embed locally.

### 8.2 Contribute

```
POST /v1/pool/embeddings
```

**Request body:**

```json
{
  "embedding_version": "muq-mulan@0.1:10:2:24000",
  "vector": "<base64>",
  "content_hash": "abc123...",
  "recording_mbid": "a1b2c3d4-...",
  "artist": "Radiohead",
  "title": "Everything In Its Right Place",
  "album": "Kid A",
  "duration_ms": 245000
}
```

**Response 200 (accepted):**

```json
{
  "accepted": true,
  "canonical_song_id": "a1b2c3d4-...",
  "upload_count": 8,
  "confidence": 0.93,
  "blended": true
}
```

**Response 200 (rejected outlier):**

```json
{
  "accepted": false,
  "reason": "below_reject_threshold",
  "similarity": 0.72
}
```

**Response 400 (version rejected):**

```json
{
  "accepted": false,
  "error": "unsupported_embedding_version",
  "embedding_version": "muq-mulan@0.1:10:2:24000",
  "reason": "deprecated",
  "message": "Uploads are disabled for deprecated versions.",
  "supported_versions": ["muq-mulan@0.2:10:2:24000"]
}
```

Returned when the version is unknown, retired, or deprecated. Prevents clients on old models from polluting or creating entries in a namespace the server no longer maintains.

### 8.3 Health

```
GET /v1/pool/health
```

```json
{
  "status": "ok",
  "api_version": "v1",
  "supported_embedding_versions": ["muq-mulan@0.2:10:2:24000"],
  "deprecated_embedding_versions": ["muq-mulan@0.1:10:2:24000"],
  "entry_count": 227000
}
```

Lightweight check. For full per-version metadata (counts, retirement dates), use `GET /v1/pool/versions`.

### 8.4 Rate limits

| Endpoint | Default limit |
|----------|---------------|
| `GET` lookup | 600/min per client key |
| `POST` contribute | 60/min per client key |

Unauthenticated access may be allowed for lookups; contribution requires an API key (`Authorization: Bearer <key>`). Exact auth policy is an ops decision.

---

## 9. Limitations and honest tradeoffs

### What the pool is good for

- Skipping GPU work during initial index of large libraries
- Text → track search (primary Harmony retrieval mode)
- Approximate similarity across a catalog where exact per-file fidelity is unnecessary

### What the pool is not

- **Not a replacement for local embed when precision matters.** Track → similar seeded from the user's exact audio may differ from search results based on a consensus vector. Clients may offer `harmony index --reembed-pool` to refresh imported tracks locally.
- **Not stable across different musical versions.** Live vs studio, remixes, and covers may map to the same metadata identity; outlier rejection mitigates but does not eliminate this.
- **Not useful without tag extraction.** Tier 3 identity requires artist/title from tags (mutagen). Filename-only scanning yields poor hit rates.

### MuQ-MuLan semantic layer

MuQ-MuLan embeddings capture musical semantics (mood, instrumentation, energy) in a joint text–audio space. Consensus vectors approximate that semantic neighborhood across similar rips. They are not a single Platonic vector per work — they are the **centroid of agreeing contributors** in embedding space.

---

## 10. Client configuration

In `~/.harmony/config.yaml`:

```yaml
pool:
  enabled: false                              # opt-in
  url: "https://pool.harmony.example"
  api_key: null                               # required for contribution
  contribute: false                           # opt-in upload after local embed
  min_confidence: 0.7                         # minimum to import
  timeout_seconds: 10
  on_unreachable: "embed"                     # embed | skip (skip = mark pending)
  on_version_mismatch: "embed"                # embed | disable (if local version unsupported)
  check_versions_on_start: true               # prefetch GET /v1/pool/versions
```

`embedding_version` is not configurable here — it is always derived from the local `embedding` and `audio` config. The pool must support that version or the client falls back per `on_version_mismatch`.

Environment overrides:

| Variable | Description |
|----------|-------------|
| `HARMONY_POOL_URL` | Pool base URL |
| `HARMONY_POOL_API_KEY` | Contribution API key |
| `HARMONY_POOL_ENABLED` | `true` / `false` |

CLI flags (proposed):

```bash
uv run harmony index ~/music --pool              # enable lookup for this run
uv run harmony index ~/music --pool --contribute # also upload after local embed
uv run harmony index ~/music --reembed-pool      # replace pool-imported vectors locally
```

---

## 11. Privacy and abuse

### Privacy

- Lookups reveal `content_hash` and/or metadata to the pool operator.
- Consider batching lookups in a future revision to reduce per-track leakage.
- No audio bytes are ever uploaded — only precomputed vectors and metadata.

### Abuse mitigation

- Rate limits per API key and IP
- Outlier rejection limits impact of poisoned vectors
- Contribution requires API key (accountability)
- Quarantine log for manual review
- Optional: require `content_hash` on contribution so exact-tier entries can be audited

---

## 12. Implementation phases

### Phase 1 — Server MVP

- [ ] VPS service with SQLite or Postgres + blob storage for vectors
- [ ] Tier 1 (`content_hash`) and Tier 3 (normalized metadata) identity
- [ ] Consensus algorithm with reject/warn thresholds
- [ ] Embedding version manifest (active / deprecated / retired)
- [ ] `GET /v1/pool/versions`, lookup, contribute, health endpoints
- [ ] Version mismatch errors on lookup and upload
- [ ] Basic rate limiting

### Phase 2 — Client integration

- [ ] `PoolClient` in `harmony/pool/` (lookup, contribute, config)
- [ ] Hook in `TrackEmbeddingPipeline.embed_track`
- [ ] `config.yaml` pool section + env overrides
- [ ] `--pool` / `--contribute` CLI flags
- [ ] Index stats: `pool_hits`, `pool_misses`, `pool_rejected` in status output

### Phase 3 — Identity improvements

- [ ] Mutagen tag extraction (prerequisite for tier 3 quality)
- [ ] MusicBrainz recording MBID resolver (tier 2)
- [ ] Store `recording_mbid` and `canonical_song_id` in local metadata DB

### Phase 4 — Polish

- [ ] `harmony index --reembed-pool`
- [ ] Batch lookup API (`POST /v1/pool/embeddings/lookup-batch`)
- [ ] Pool status in `harmony status` (hit rate, contribution count)
- [ ] Chunk-level vectors (separate namespace, larger payloads)

---

## 13. Open questions

1. **Chunk vectors.** Track-level vectors are ~2 KB; chunk vectors are ~10–40 KB per track. Worth pooling in a later version for chunk-level search, or track-level only forever?
2. **Auth model.** Open lookups + keyed contribution, or keyed for both?
3. **NC license.** MuQ-MuLan is CC-BY-NC 4.0. Does operating a public embedding pool require legal review of derived-work redistribution?
4. **Re-embed policy.** Should the client background-re-embed low-confidence imports, or leave them until the user opts in?
5. **Operator.** Project-hosted pool vs self-hosted pool URL (already supported via config) — document third-party mirrors?

---

## 14. Related documents

- [engine-spec.md](../engine-spec.md) — core engine, embedding version, storage layout
- [server.md](server.md) — Harmony API server
- [model-cache.md](model-cache.md) — local model keep-alive (orthogonal to pool)
