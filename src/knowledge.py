"""Add new tickets to the knowledge base while the agent runs ("learning")."""
import csv
import re
from pathlib import Path

import search
from db import get_conn

LEARNED_PREFIX = "L-"  # learned tickets get ids like L-T12345, so they never clash with dataset ids

_CITATION = re.compile(r"\s*\[T\d+\]")


def strip_citations(reply: str) -> str:
    """Remove [T01234] source tags: they point at tickets that mean nothing outside this one run."""
    return _CITATION.sub("", reply).strip()


def add_learned_ticket(ticket_id: str, text: str, answer: str, category: str,
                       priority: str | None = None) -> bool:
    """Embed a ticket and insert it with source = 'learned'.

    Returns True if a row was added, False if that id was already there (so running it twice,
    for example after a restart, never creates duplicates)."""
    learned_id = f"{LEARNED_PREFIX}{ticket_id}"
    vec = search.embed_text(text)  # looked up on the module so tests can replace it
    with get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO tickets (ticket_id, text, answer, category, priority, tags, source, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, 'learned', %s)
            ON CONFLICT (ticket_id) DO NOTHING
            """,
            (learned_id, text, answer, category, priority, ["learned"], vec),
        )
        return cur.rowcount == 1


_eval_ids = None


def is_eval_ticket(ticket_id: str) -> bool:
    """True if this id belongs to the held-out eval set. Eval tickets must never enter the database."""
    global _eval_ids
    if _eval_ids is None:
        path = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval_set.csv"
        _eval_ids = set()
        if path.exists():
            with path.open(encoding="utf-8", newline="") as f:
                _eval_ids = {row["ticket_id"] for row in csv.DictReader(f)}
    return ticket_id in _eval_ids
