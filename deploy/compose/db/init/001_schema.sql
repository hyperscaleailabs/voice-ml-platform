-- Schema for the pgvector-backed RAG store and telemetry sink.
-- Mirrors the dataclasses in src/vmp/types.py (Document, Chunk) and the
-- GraphStore node/edge model. Runs once on first container start.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          TEXT PRIMARY KEY,
    path        TEXT NOT NULL,
    text        TEXT NOT NULL,
    sha         TEXT NOT NULL,
    meta        JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS documents_sha_idx ON documents (sha);

-- Embedding dimension is a deployment choice. 384 matches small
-- sentence-transformers models and the default hashing embedder; change the
-- dimension here and in configs/rag.toml together.
CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents (id) ON DELETE CASCADE,
    start_line  INTEGER NOT NULL,
    end_line    INTEGER NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector(384),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chunks_document_idx ON chunks (document_id);
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Graph RAG: entities and relations extracted from chunks.
CREATE TABLE IF NOT EXISTS nodes (
    id          TEXT PRIMARY KEY,
    kind        TEXT NOT NULL,
    name        TEXT NOT NULL,
    props       JSONB NOT NULL DEFAULT '{}'::jsonb,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS nodes_kind_name_idx ON nodes (kind, name);

CREATE TABLE IF NOT EXISTS edges (
    id          BIGSERIAL PRIMARY KEY,
    src         TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    dst         TEXT NOT NULL REFERENCES nodes (id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    chunk_id    TEXT REFERENCES chunks (id) ON DELETE SET NULL,
    props       JSONB NOT NULL DEFAULT '{}'::jsonb,
    UNIQUE (src, dst, relation)
);
CREATE INDEX IF NOT EXISTS edges_src_idx ON edges (src);
CREATE INDEX IF NOT EXISTS edges_dst_idx ON edges (dst);

-- Stage traces. One row per JSONL trace line:
-- keys ts, session, turn, event, span, seq, ms, payload.
CREATE TABLE IF NOT EXISTS telemetry_events (
    id          BIGSERIAL PRIMARY KEY,
    ts          DOUBLE PRECISION NOT NULL,
    session     TEXT NOT NULL,
    turn        INTEGER NOT NULL,
    event       TEXT NOT NULL,          -- "<stage>.start" | "<stage>.end"
    span        TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    ms          DOUBLE PRECISION,
    payload     JSONB NOT NULL DEFAULT '{}'::jsonb,
    received_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS telemetry_session_turn_idx ON telemetry_events (session, turn, seq);
CREATE INDEX IF NOT EXISTS telemetry_event_ts_idx ON telemetry_events (event, ts);
