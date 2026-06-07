-- Harmony Engine schema (SQLite / Turso compatible)

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS tracks (
    track_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL UNIQUE,
    status TEXT NOT NULL DEFAULT 'active',
    primary_path TEXT NOT NULL,
    duration_ms INTEGER NOT NULL DEFAULT 0,
    title TEXT NOT NULL DEFAULT '',
    artist TEXT NOT NULL DEFAULT '',
    album TEXT NOT NULL DEFAULT '',
    album_artist TEXT,
    year INTEGER,
    genre TEXT,
    disc_number INTEGER,
    track_number INTEGER,
    extra_json TEXT NOT NULL DEFAULT '{}',
    indexed_at TEXT,
    last_seen_at TEXT,
    embedding_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS track_locations (
    location_id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL REFERENCES tracks(track_id),
    path TEXT NOT NULL UNIQUE,
    is_primary INTEGER NOT NULL DEFAULT 0,
    first_seen_at TEXT NOT NULL,
    last_seen_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS path_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    track_id TEXT NOT NULL REFERENCES tracks(track_id),
    old_path TEXT,
    new_path TEXT,
    changed_at TEXT NOT NULL,
    reason TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL REFERENCES tracks(track_id),
    chunk_index INTEGER NOT NULL,
    start_ms INTEGER NOT NULL,
    end_ms INTEGER NOT NULL,
    embedding_path TEXT,
    UNIQUE(track_id, chunk_index)
);

CREATE TABLE IF NOT EXISTS embedding_jobs (
    job_id TEXT PRIMARY KEY,
    track_id TEXT NOT NULL,
    status TEXT NOT NULL,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS indexes (
    index_id TEXT PRIMARY KEY,
    granularity TEXT NOT NULL,
    embedding_version TEXT NOT NULL,
    backend TEXT NOT NULL,
    metric TEXT NOT NULL,
    vector_count INTEGER NOT NULL DEFAULT 0,
    built_at TEXT NOT NULL,
    path TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_tombstones (
    index_id TEXT NOT NULL,
    entity_id TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    tombstoned_at TEXT NOT NULL,
    PRIMARY KEY (index_id, entity_id, entity_type)
);

CREATE TABLE IF NOT EXISTS sync_runs (
    run_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    added INTEGER NOT NULL DEFAULT 0,
    updated_metadata INTEGER NOT NULL DEFAULT 0,
    moved INTEGER NOT NULL DEFAULT 0,
    duplicates_found INTEGER NOT NULL DEFAULT 0,
    missing INTEGER NOT NULL DEFAULT 0,
    removed INTEGER NOT NULL DEFAULT 0,
    reembedded INTEGER NOT NULL DEFAULT 0,
    failed INTEGER NOT NULL DEFAULT 0,
    skipped INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS query_cache (
    query_id TEXT PRIMARY KEY,
    text TEXT NOT NULL UNIQUE,
    vector_path TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tracks_content_hash ON tracks(content_hash);
CREATE INDEX IF NOT EXISTS idx_tracks_status ON tracks(status);
CREATE INDEX IF NOT EXISTS idx_locations_path ON track_locations(path);
CREATE INDEX IF NOT EXISTS idx_locations_track ON track_locations(track_id);
CREATE INDEX IF NOT EXISTS idx_chunks_track ON chunks(track_id);
