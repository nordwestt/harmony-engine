
# 🧠 System overview: “Vibe → Playlist Engine”

Think in 5 layers:

[ Ingestion ] → [ Embeddings ] → [ Index ] → [ Retrieval ] → [ Playlist Engine ] → API

---

# 1) 🎧 Ingestion layer (library → normalized audio)

### Input

- Local files (FLAC, MP3, AAC, etc.)
- Jellyfin / Subsonic APIs

### Processing

- Decode → mono (optional) → **resample to 24kHz**
- Chunk long tracks (important!)

audio = load(file)  
audio_24k = resample(audio, target_sr=24000)  
chunks = split(audio_24k, chunk_size=10s, overlap=2s)

### Why chunking matters

- Music embeddings are **time-local**
- A song can have multiple “vibes” (intro ≠ drop ≠ outro)

👉 Store:

- track_id
- chunk_id
- waveform (or temp path)

---

# 2) 🧬 Embedding layer (core intelligence)

### Model choice

- **MuQ-MuLan** (ideal for your use case)

You’ll compute:

### A. Audio embeddings

chunk_embedding = model.encode_audio(chunk)  
track_embedding = mean_pool(chunks)

### B. Text embeddings

vibe_embedding = model.encode_text("dreamy ambient night drive")

---

# 3) 📦 Vector storage / index

Use a proper ANN index:

- FAISS (best default)
- Qdrant (if you want API-first)
- Weaviate (more infra-heavy)

### What to store

Two levels:

### Chunk-level index

- finer semantic resolution
- better retrieval

### Track-level index

- faster + simpler ranking

👉 Recommended:

- retrieve on chunks
- aggregate to tracks

---

# 4) 🔎 Retrieval layer (vibe → candidates)

Naive version:

candidates = top_k_similar(vibe_embedding, chunk_index, k=500)

Then:

- group by track
- score = max or avg of chunk scores

---

## ⚠️ Improve this (important)

### Hybrid scoring

score =   
  0.7 * sim(vibe, track_embedding) +  
  0.3 * sim(seed_tracks, track_embedding)

### Add diversity penalty

Avoid 10 songs from same artist:

score -= λ * artist_overlap_penalty

---

# 5) 🎛️ Playlist engine (this is the secret sauce)

Most people get this wrong.

You don’t want:

> “top 30 closest songs”

You want:

> “coherent journey with variation”

---

## Step A: Candidate pool

- top 200–500 tracks

## Step B: Clustering

Cluster candidates:

- k-means (k ≈ 5–10)
- or HDBSCAN

This gives:

- sub-vibes inside the main vibe

---

## Step C: Sequencing

### Method 1: Greedy path (simple + good)

playlist = [best_track]  
  
while len < N:  
    next = argmax(  
        α * sim(vibe, track) +  
        β * sim(previous_track, track) -  
        γ * repetition_penalty  
    )

---

### Method 2: Graph walk (better)

- Nodes = tracks
- Edge weight = similarity
- Constrain:
    - smooth transitions
    - gradual drift

---

### Method 3: Energy curve (underrated 🔥)

Estimate energy:

- RMS / spectral features
- or embedding norm proxy

Shape:

- start low → build → peak → resolve

👉 This alone makes playlists feel _intentional_

---

# 6) 🌐 API layer

Expose a simple service:

### Endpoints

POST /embed/library  
POST /vibe_playlist  
POST /similar_tracks

### Example

POST /vibe_playlist  
{  
  "prompt": "melancholic rainy evening piano",  
  "length": 25,  
  "seed_tracks": ["track_123"],  
  "diversity": 0.3  
}

---

# 7) 🔌 Integration layer (Jellyfin/Subsonic)

For Jellyfin:

- Pull library via API
- Map track IDs ↔ file paths
- Push playlists back via API

Same idea for Subsonic

---

# ⚙️ Optional upgrades (worth it)

## 1. Metadata fusion

Combine:

- embeddings
- genre / year / BPM

final_score =   
  0.6 * embedding +  
  0.2 * genre_match +  
  0.2 * tempo_match

---

## 2. Caching

- Precompute embeddings once
- Store in disk (Parquet / SQLite)

---

## 3. Incremental updates

- Watch filesystem
- Embed new tracks only

---

## 4. Multi-vibe blending

"prompt": ["ambient", "nostalgic", "lofi"],  
"weights": [0.5, 0.3, 0.2]

---

# 🧭 Minimal viable stack

If I had to keep it tight:

- Model: MuQ-MuLan
- Index: FAISS
- Backend: FastAPI
- Storage: SQLite + Parquet
- Integration: Jellyfin API


Structure example
```

music-vibe-engine/
├── README.md
├── pyproject.toml
├── .env.example

├── vibe/                     # core package
│   ├── __init__.py
│   │
│   ├── config.py            # global config (paths, model choice, params)
│   │
│   ├── audio/
│   │   ├── loader.py        # load + decode audio files
│   │   ├── resample.py      # resample to 24kHz
│   │   ├── chunking.py      # split into chunks
│   │
│   ├── embedding/
│   │   ├── model.py         # MuQ-MuLan wrapper
│   │   ├── audio_embedder.py
│   │   ├── text_embedder.py
│   │   ├── pooling.py       # mean / attention pooling
│   │
│   ├── index/
│   │   ├── faiss_index.py   # build/load/search index
│   │   ├── schema.py        # track/chunk metadata
│   │
│   ├── retrieval/
│   │   ├── search.py        # vibe → candidates
│   │   ├── scoring.py       # hybrid scoring logic
│   │
│   ├── playlist/
│   │   ├── generator.py     # main playlist logic
│   │   ├── sequencing.py    # ordering logic
│   │   ├── diversity.py     # anti-repetition rules
│   │
│   ├── storage/
│   │   ├── embeddings.py    # save/load embeddings (parquet/sqlite)
│   │   ├── metadata.py      # track metadata store
│   │
│   └── utils/
│       ├── logging.py
│       ├── progress.py
│
├── cli/
│   ├── main.py              # entrypoint
│   ├── embed.py             # `embed` command
│   ├── playlist.py          # `playlist` command
│
├── data/
│   ├── embeddings/          # cached embeddings
│   ├── index/               # FAISS index
│   ├── metadata.db
│
├── scripts/
│   ├── bootstrap.sh
│
└── tests/
    ├── test_embedding.py
    ├── test_playlist.py

```

## Notes

# 🧠 Core design principles (important for your agent)

### 1. Separate _audio → embedding_ from _embedding → playlist_

Don’t mix them. You’ll want to swap models later.

---

### 2. Treat chunks as first-class citizens

Even if CLI v1 only outputs track-level playlists.

---

### 3. Everything should be reloadable from disk

No recomputing embeddings unless necessary.

---

# ⚙️ CLI interface (simple but powerful)

### 1) Embed a library

python -m cli.main embed /path/to/music

What it does:

- scans files
- resamples + chunks
- computes embeddings
- saves:
    - embeddings → `data/embeddings/`
    - FAISS index → `data/index/`
    - metadata → `metadata.db`

---

### 2) Generate playlist

python -m cli.main playlist \  
  --prompt "melancholic rainy night piano" \  
  --length 25

Optional flags:

--seed track1.mp3 track2.mp3  
--diversity 0.3  
--output playlist.m3u

---

# 🧩 Key module sketches

## 🎧 audio/chunking.py

def chunk_audio(waveform, sr, chunk_seconds=10, overlap=2):  
    step = chunk_seconds - overlap  
    chunks = []  
    for i in range(0, len(waveform), int(step * sr)):  
        chunk = waveform[i:i + int(chunk_seconds * sr)]  
        if len(chunk) < sr:  # skip tiny tail  
            continue  
        chunks.append(chunk)  
    return chunks

---

## 🧬 embedding/audio_embedder.py

class AudioEmbedder:  
    def __init__(self, model):  
        self.model = model  
  
    def embed_track(self, chunks):  
        chunk_embeds = [self.model.encode_audio(c) for c in chunks]  
        return np.mean(chunk_embeds, axis=0)

---

## 🔎 retrieval/search.py

def retrieve_candidates(vibe_embedding, index, k=300):  
    distances, ids = index.search(vibe_embedding, k)  
    return ids

---

## 🎛️ playlist/generator.py

def generate_playlist(candidates, embeddings, length=25):  
    playlist = [candidates[0]]  
  
    while len(playlist) < length:  
        best = None  
        best_score = -1  
  
        for track in candidates:  
            if track in playlist:  
                continue  
  
            score = (  
                0.7 * sim_to_vibe(track) +  
                0.3 * sim_to_last(track, playlist[-1]) -  
                repetition_penalty(track, playlist)  
            )  
  
            if score > best_score:  
                best = track  
                best_score = score  
  
        playlist.append(best)  
  
    return playlist

---

# 💾 Storage format (keep it simple)

### Metadata (SQLite)

tracks(  
  id TEXT PRIMARY KEY,  
  path TEXT,  
  artist TEXT,  
  album TEXT  
)  
  
chunks(  
  id TEXT,  
  track_id TEXT,  
  embedding_path TEXT  
)

---

### Embeddings

- Store as `.npy` or Parquet
- One file per track or per chunk batch

## 🚀 Step-by-step build plan (for your agent)

### Phase 1 (MVP)

- [ ]  load audio
- [ ]  resample to 24kHz
- [ ]  chunk
- [ ]  embed (track-level only)
- [ ]  brute-force cosine search (no FAISS yet)
- [ ]  generate playlist

👉 This already works and is testable

---

### Phase 2

- [ ]  add FAISS
- [ ]  persist embeddings
- [ ]  speed up retrieval

---

### Phase 3

- [ ]  chunk-level indexing
- [ ]  diversity penalties
- [ ]  better sequencing

---

### Phase 4

- [ ]  API layer
- [ ]  Jellyfin integration