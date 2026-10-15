"""Database helpers: connect to Postgres and create the tickets table."""
import os

import psycopg
from dotenv import load_dotenv
from pgvector.psycopg import register_vector

load_dotenv()

EMBEDDING_DIM = 384  # size of the all-MiniLM-L6-v2 embeddings

CREATE_TABLE = f"""
CREATE TABLE IF NOT EXISTS tickets (
    ticket_id   TEXT PRIMARY KEY,
    text        TEXT NOT NULL,
    answer      TEXT NOT NULL,
    category    TEXT NOT NULL,
    queue       TEXT,
    priority    TEXT,
    type        TEXT,
    tags        TEXT[] DEFAULT '{{}}',
    source      TEXT NOT NULL DEFAULT 'dataset' CHECK (source IN ('dataset', 'learned')),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    embedding   vector({EMBEDDING_DIM})
);
"""

# HNSW is simpler than IVFFlat: IVFFlat must be "trained" on data that is already in the
# table (and re-built as the data grows), while HNSW works from an empty table and keeps
# itself up to date as rows are added - which the agent will do later when it "learns".
CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS tickets_embedding_idx
ON tickets USING hnsw (embedding vector_cosine_ops);
"""


def get_conn() -> psycopg.Connection:
    """Open a connection using DATABASE_URL from .env (vector type is registered)."""
    conn = psycopg.connect(os.environ["DATABASE_URL"], autocommit=True)
    conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    register_vector(conn)
    return conn


def init_db() -> None:
    """Create the table and index if they do not exist yet. Safe to run many times."""
    with get_conn() as conn:
        conn.execute(CREATE_TABLE)
        conn.execute(CREATE_INDEX)


if __name__ == "__main__":
    init_db()
    print("tickets table and HNSW index ready")
