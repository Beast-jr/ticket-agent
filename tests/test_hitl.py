"""Tests for human review and learning (Day 8).

The LLM and the embedding model are mocked (no Groq call, no model download). A REAL local Postgres
is used, but a separate database called `tickets_test`, so the real knowledge base is never touched."""
import hashlib
import re

import numpy as np
import psycopg
import pytest
from psycopg.conninfo import conninfo_to_dict, make_conninfo

import agent.nodes as nodes
import search
from agent.checkpoint import get_checkpointer
from agent.graph import (get_saved_state, is_waiting_for_human, list_paused, resume_ticket, run_ticket)
from db import get_conn, init_db
from helpers import CLASSIFY, TICKET, FakeLLM, draft_json, similar_tickets, verdict

TEST_DB = "tickets_test"
NEAR_TICKET = "Concerns about hospital security systems: suspected data breach, we are examining the access logs."
HUMAN_REPLY = "Thank you for reporting this. Please send us the access logs and we will investigate."


# --- fakes and fixtures ------------------------------------------------------

def fake_embed(text):
    """A cheap stand-in for the embedding model: hash each word into one of 384 slots.
    Texts that share most words get almost the same vector, so 'near-identical' texts match."""
    vec = np.zeros(384)
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % 384] += 1.0
    norm = np.linalg.norm(vec)
    return vec / norm if norm else vec


def low_match_tickets():
    """Search results that do not resemble the ticket at all: confidence ends below LOW -> escalate."""
    return [{**t, "similarity": 0.20, "category": "Billing and Payments"} for t in similar_tickets()]


@pytest.fixture(scope="session")
def test_url():
    """Create the tickets_test database if needed and point DATABASE_URL at it for the whole session."""
    import os
    base = os.environ["DATABASE_URL"]
    admin = make_conninfo(base, dbname="postgres")
    with psycopg.connect(admin, autocommit=True) as conn:
        if not conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)).fetchone():
            conn.execute(f'CREATE DATABASE "{TEST_DB}"')
    url = make_conninfo(base, dbname=TEST_DB)
    mp = pytest.MonkeyPatch()
    mp.setenv("DATABASE_URL", url)
    yield url
    mp.undo()


@pytest.fixture
def saver(test_url, monkeypatch):
    """A Postgres checkpointer on the test database, with every table emptied before the test."""
    import os
    assert conninfo_to_dict(os.environ["DATABASE_URL"])["dbname"] == TEST_DB, "refusing to touch a real database"
    init_db()
    s = get_checkpointer(test_url)
    s.conn.execute("TRUNCATE tickets, checkpoints, checkpoint_blobs, checkpoint_writes")
    monkeypatch.setattr(search, "embed_text", fake_embed)
    yield s
    s.conn.close()


def install_llm(monkeypatch, script, similar=None):
    llm = FakeLLM(script)
    monkeypatch.setattr(nodes, "call_llm", llm)
    monkeypatch.setattr(nodes, "search_similar", lambda query, k=5: similar if similar is not None else low_match_tickets())
    return llm


def approve(reply=HUMAN_REPLY, action="approve", by="Dana"):
    return {"action": action, "reply": reply, "decided_by": by}


def learned_rows():
    with get_conn() as conn:
        return conn.execute("SELECT ticket_id, answer, source FROM tickets WHERE source = 'learned'").fetchall()


# --- a) an escalated ticket pauses and its state is saved ---------------------

def test_escalated_ticket_pauses_and_state_is_saved(saver, monkeypatch):
    llm = install_llm(monkeypatch, {"classify": [CLASSIFY]})
    result = run_ticket(TICKET, "TEST-A", checkpointer=saver, learn=True)

    assert "__interrupt__" in result  # the run stopped and is waiting
    assert is_waiting_for_human("TEST-A", saver)
    saved = get_saved_state("TEST-A", saver).values
    assert saved["status"] == "escalated"
    assert "PROBLEM:" in saved["escalation_summary"]
    assert saved["category"] == "Technical & IT Support"
    assert [p["ticket_id"] for p in list_paused(saver)] == ["TEST-A"]
    assert llm.steps == ["classify"]  # a template summary: the only LLM call was classify


# --- b) restart simulation: new graph, new connection, same thread_id ---------

def test_resume_after_restart(saver, test_url, monkeypatch):
    llm = install_llm(monkeypatch, {"classify": [CLASSIFY]})
    run_ticket(TICKET, "TEST-B", checkpointer=saver)
    saver.conn.close()  # "the program was closed"

    saver2 = get_checkpointer(test_url)  # "the program was started again": new connection, new graph
    try:
        assert is_waiting_for_human("TEST-B", saver2)
        final = resume_ticket("TEST-B", approve(), saver2)
    finally:
        saver2.conn.close()

    assert final["status"] == "replied_by_human"
    assert final["final_reply"] == HUMAN_REPLY
    assert llm.steps == ["classify"]  # resuming did NOT redo classify or the escalation summary


# --- c) approve -> replied_by_human, and the learn node inserted it -----------

def test_approve_replies_and_learns(saver, monkeypatch):
    install_llm(monkeypatch, {"classify": [CLASSIFY]})
    run_ticket(TICKET, "TEST-C", checkpointer=saver, learn=True)
    final = resume_ticket("TEST-C", approve(), saver)

    assert final["status"] == "replied_by_human"
    assert final["human_decision"]["decided_by"] == "Dana"
    assert final["human_decision"]["action"] == "approve"
    assert final["human_decision"]["decided_at"]  # a timestamp was stored
    assert final["learned"] is True and final["learned_ticket_id"] == "L-TEST-C"
    assert learned_rows() == [("L-TEST-C", HUMAN_REPLY, "learned")]


def test_edit_also_replies_and_records_the_action(saver, monkeypatch):
    install_llm(monkeypatch, {"classify": [CLASSIFY]})
    run_ticket(TICKET, "TEST-C2", checkpointer=saver, learn=True)
    final = resume_ticket("TEST-C2", approve("Edited reply text.", action="edit", by="Sam"), saver)
    assert final["status"] == "replied_by_human"
    assert final["human_decision"]["action"] == "edit"
    assert final["final_reply"] == "Edited reply text."


# --- d) reject -> closed, nothing learned -------------------------------------

def test_reject_closes_and_learns_nothing(saver, monkeypatch):
    install_llm(monkeypatch, {"classify": [CLASSIFY]})
    run_ticket(TICKET, "TEST-D", checkpointer=saver, learn=True)
    final = resume_ticket("TEST-D", {"action": "reject", "decided_by": "Dana", "note": "spam"}, saver)

    assert final["status"] == "closed"
    assert final["final_reply"] == ""
    assert final["human_decision"]["action"] == "reject"
    assert not final.get("learned")
    assert learned_rows() == []
    assert not is_waiting_for_human("TEST-D", saver)  # it is finished, no longer in the queue


# --- e) after learning, a near-identical new ticket finds the learned one -----

def test_learned_ticket_is_found_by_a_near_identical_new_ticket(saver, monkeypatch):
    install_llm(monkeypatch, {"classify": [CLASSIFY]})
    run_ticket(TICKET, "TEST-E", checkpointer=saver, learn=True)
    resume_ticket("TEST-E", approve(), saver)

    found = search.search_similar(NEAR_TICKET, k=3)  # the real search function on the test database
    assert found[0]["ticket_id"] == "L-TEST-E"
    assert found[0]["source"] == "learned"
    assert found[0]["answer"] == HUMAN_REPLY
    assert found[0]["similarity"] > 0.8


# --- learning rules -----------------------------------------------------------

def test_a_self_checked_reply_is_learned_without_citation_tags(saver, monkeypatch):
    llm = install_llm(monkeypatch,
                      {"classify": [CLASSIFY], "draft": [draft_json("Please send the access logs [T00001].")],
                       "check": [verdict("supported")]},
                      similar=similar_tickets())
    final = run_ticket(TICKET, "TEST-F", learn=True)  # no checkpointer needed: nothing to wait for
    assert final["status"] == "replied"
    assert final["learned"] is True
    assert learned_rows() == [("L-TEST-F", "Please send the access logs.", "learned")]  # [T00001] removed


def test_nothing_is_learned_unless_learning_is_switched_on(saver, monkeypatch):
    install_llm(monkeypatch, {"classify": [CLASSIFY], "draft": [draft_json()], "check": [verdict("supported")]},
                similar=similar_tickets())
    final = run_ticket(TICKET, "TEST-G")  # learn defaults to False
    assert final["status"] == "replied"
    assert not final.get("learned")
    assert learned_rows() == []


def test_eval_set_tickets_are_never_learned(saver, monkeypatch):
    import csv
    from pathlib import Path
    eval_path = Path(__file__).resolve().parent.parent / "data" / "processed" / "eval_set.csv"
    eval_id = next(csv.DictReader(eval_path.open(encoding="utf-8")))["ticket_id"]
    install_llm(monkeypatch, {"classify": [CLASSIFY], "draft": [draft_json()], "check": [verdict("supported")]},
                similar=similar_tickets())
    final = run_ticket(TICKET, eval_id, learn=True)
    assert final["status"] == "replied"
    assert not final.get("learned")
    assert learned_rows() == []
    assert any("eval-set ticket" in line for line in final["steps_log"])


# --- decisions are validated ---------------------------------------------------

def test_a_bad_decision_is_refused_and_the_ticket_keeps_waiting(saver, monkeypatch):
    install_llm(monkeypatch, {"classify": [CLASSIFY]})
    run_ticket(TICKET, "TEST-H", checkpointer=saver, learn=True)

    for bad in [{"action": "maybe", "decided_by": "Dana"},
                {"action": "approve", "reply": "", "decided_by": "Dana"},
                {"action": "approve", "reply": "ok"},          # nobody signed it
                "approve"]:
        with pytest.raises(ValueError):
            resume_ticket("TEST-H", bad, saver)
    assert is_waiting_for_human("TEST-H", saver)  # still paused, nothing was damaged

    assert resume_ticket("TEST-H", approve(), saver)["status"] == "replied_by_human"


def test_resuming_a_ticket_that_is_not_waiting_is_refused(saver, monkeypatch):
    install_llm(monkeypatch, {"classify": [CLASSIFY]})
    run_ticket(TICKET, "TEST-I", checkpointer=saver)
    resume_ticket("TEST-I", approve(), saver)
    with pytest.raises(ValueError, match="not waiting"):
        resume_ticket("TEST-I", approve(), saver)  # a second decision on a finished ticket
