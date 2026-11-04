-- FilingCheck schema. Postgres 16 + pgvector.
-- The vector dimension (384) must match config EMBEDDING_DIM and the default
-- EMBEDDING_MODEL (BAAI/bge-small-en-v1.5, run locally). Changing the model to one
-- with a different dimension requires changing this file and re-ingesting.
-- Owned by contracts-architect. Safe to run more than once.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id           BIGSERIAL PRIMARY KEY,
    chunk_id     TEXT        NOT NULL,
    company      TEXT        NOT NULL,
    ticker       TEXT        NOT NULL,
    cik          TEXT        NOT NULL CHECK (cik ~ '^[0-9]{10}$'),
    fiscal_year  INTEGER     NOT NULL,
    item         TEXT        NOT NULL CHECK (item IN ('1A', '1C', '7')),
    chunk_index  INTEGER     NOT NULL CHECK (chunk_index >= 0),
    text         TEXT        NOT NULL,
    token_count  INTEGER     NOT NULL CHECK (token_count >= 1),
    filing_url   TEXT        NOT NULL,
    embedding    vector(384)  NOT NULL,
    tsv          tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    CONSTRAINT chunks_chunk_id_key UNIQUE (chunk_id),
    CONSTRAINT chunks_position_key UNIQUE (ticker, fiscal_year, item, chunk_index)
);

-- Full-text search (BM25-style ranking via ts_rank_cd).
CREATE INDEX IF NOT EXISTS chunks_tsv_gin ON chunks USING GIN (tsv);

-- Approximate nearest neighbour search on cosine distance.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Hard filters applied before ranking.
CREATE INDEX IF NOT EXISTS chunks_ticker_year ON chunks (ticker, fiscal_year);
