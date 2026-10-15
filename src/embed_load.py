"""Embed the knowledge base and load it into Postgres. Safe to re-run.

Run from the project root:  python src/embed_load.py
"""
import json
import time
from pathlib import Path

import pandas as pd
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from db import EMBEDDING_DIM, get_conn, init_db

ROOT = Path(__file__).resolve().parent.parent
KB_CSV = ROOT / "data" / "processed" / "knowledge_base.csv"
EVAL_CSV = ROOT / "data" / "processed" / "eval_set.csv"
MODEL_NAME = "all-MiniLM-L6-v2"
BATCH_SIZE = 64

INSERT = """
INSERT INTO tickets (ticket_id, text, answer, category, queue, priority, type, tags, source, embedding)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'dataset', %s)
ON CONFLICT (ticket_id) DO NOTHING
"""


def main():
    start = time.time()
    kb = pd.read_csv(KB_CSV)
    eval_ids = set(pd.read_csv(EVAL_CSV)["ticket_id"])
    # Safety check: eval tickets must never reach the database
    assert not eval_ids & set(kb["ticket_id"]), "Eval tickets found in the knowledge base!"

    init_db()
    with get_conn() as conn:
        in_db = {r[0] for r in conn.execute("SELECT ticket_id FROM tickets")}
        if in_db & eval_ids:
            raise SystemExit("Eval tickets are in the database - clean it before continuing.")

        # Rows already loaded: refresh their answer text if the cleaned CSV changed
        existing = kb[kb["ticket_id"].isin(in_db)]
        with conn.cursor() as cur:
            cur.executemany(
                "UPDATE tickets SET answer = %s WHERE ticket_id = %s AND answer IS DISTINCT FROM %s",
                [(a, i, a) for i, a in zip(existing["ticket_id"], existing["answer"])],
            )

        todo = kb[~kb["ticket_id"].isin(in_db)]
        print(f"Knowledge base: {len(kb)} rows, already in database: {len(kb) - len(todo)}, to load: {len(todo)}")

        if len(todo):
            model = SentenceTransformer(MODEL_NAME)
            assert model.get_sentence_embedding_dimension() == EMBEDDING_DIM
            for i in tqdm(range(0, len(todo), BATCH_SIZE), desc="Embedding + loading", unit="batch"):
                batch = todo.iloc[i : i + BATCH_SIZE]
                vectors = model.encode(batch["text"].tolist(), normalize_embeddings=True)
                rows = [
                    (r.ticket_id, r.text, r.answer, r.category, r.queue, r.priority, r.type,
                     json.loads(r.tags), vec)
                    for r, vec in zip(batch.itertuples(index=False), vectors)
                ]
                with conn.cursor() as cur:
                    cur.executemany(INSERT, rows)

        total = conn.execute("SELECT count(*) FROM tickets").fetchone()[0]
    print(f"Loaded {len(todo)} new rows. Table now has {total} rows. Took {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
