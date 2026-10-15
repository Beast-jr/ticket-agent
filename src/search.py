"""Find the past tickets most similar to a new one."""
from sentence_transformers import SentenceTransformer

from db import get_conn

MODEL_NAME = "all-MiniLM-L6-v2"
_model = None  # loaded on first use, then reused


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def search_similar(text: str, k: int = 5) -> list[dict]:
    """Return the k most similar tickets. similarity is 0-1 (higher = more similar)."""
    vec = _get_model().encode(text, normalize_embeddings=True)
    sql = """
        SELECT ticket_id, text, answer, category, priority, 1 - (embedding <=> %s) AS similarity
        FROM tickets
        ORDER BY embedding <=> %s
        LIMIT %s
    """
    with get_conn() as conn:
        rows = conn.execute(sql, (vec, vec, k)).fetchall()
    return [
        {
            "ticket_id": r[0],
            "text": r[1],
            "answer": r[2],
            "category": r[3],
            "priority": r[4],
            "similarity": max(0.0, min(1.0, float(r[5]))),
        }
        for r in rows
    ]
