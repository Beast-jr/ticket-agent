"""Run all 100 eval tickets + 10 off-topic tickets through the graph and count outcomes.

To save LLM tokens, the draft step is replaced by a stub here: drafting never changes which
route a ticket takes, and in the logs the real draft step is always exactly 1 LLM call, so the
call count adds 1 for every ticket that reaches draft. The Day 7 self-check is stubbed out too
(these are the Day 6 numbers, which were defined before the check existed).

Progress is saved after every ticket to docs/routing_day6.csv. If the LLM rate limit is hit the
script stops cleanly; run it again later and it continues where it stopped.
Run from the project root:  python scripts/evaluate_routing.py
"""
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd
from groq import RateLimitError

import agent.graph as graph_module
from agent.graph import path_taken, run_ticket
from off_topic import OFF_TOPIC

LOG_FILE = ROOT / "logs" / "llm_calls.jsonl"
OUT_FILE = ROOT / "docs" / "routing_day6.csv"
FIELDS = ["group", "ticket_id", "outcome", "path", "attempts", "first_confidence", "last_confidence",
          "llm_calls", "escalation_paragraph"]


def draft_stub(state):
    """Stand-in for the draft node (no LLM call). Counted as 1 LLM call below."""
    log = list(state.get("steps_log", []))
    log.append("draft: (skipped in evaluation)")
    return {"status": "drafted", "attempts": state.get("attempts", 0) + 1, "steps_log": log}


def check_stub(state):
    """The self-check (Day 7) is not part of the Day 6 numbers: pass every stub draft straight through."""
    return {"check_result": "supported"}


# build_graph() looks these names up when the graph is first built
graph_module.draft = draft_stub
graph_module.check = check_stub


def llm_calls() -> int:
    return len(LOG_FILE.read_text(encoding="utf-8").splitlines()) if LOG_FILE.exists() else 0


def outcome(path: list[str], status: str) -> str:
    if status == "escalated":
        return "escalated"
    if "rewrite_query" in path and "reconcile_category" in path:
        return "drafted after reconcile + rewrite"
    if "reconcile_category" in path:
        return "drafted after reconcile"
    if "rewrite_query" in path:
        return "drafted after rewrite"
    return "drafted directly"


def report(df: pd.DataFrame) -> None:
    pd.set_option("display.width", 200)
    print("\nOutcome counts:")
    print(pd.crosstab(df["outcome"], df["group"], margins=True))
    print("\nAverage LLM calls per ticket (draft counted as 1):")
    print(df.groupby("group")["llm_calls"].mean().round(2).to_string())
    print("overall:", round(df["llm_calls"].mean(), 2))
    print("\nAverage LLM calls by outcome:")
    print(df.groupby("outcome")["llm_calls"].mean().round(2).to_string())
    esc = df[(df.group == "in_domain") & (df.outcome == "escalated")]
    print(f"\nEscalated in-domain tickets: {len(esc)}")
    if len(esc):
        print(esc[["ticket_id", "path", "first_confidence", "last_confidence"]].to_string(index=False))


eval_df = pd.read_csv(ROOT / "data" / "processed" / "eval_set.csv")
tickets = [("in_domain", t.ticket_id, t.text) for t in eval_df.itertuples()]
tickets += [("off_topic", f"OFF{i:02d}", text) for i, text in enumerate(OFF_TOPIC, 1)]

done = set(pd.read_csv(OUT_FILE)["ticket_id"]) if OUT_FILE.exists() else set()
if not OUT_FILE.exists():
    with OUT_FILE.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, FIELDS).writeheader()
print(f"{len(done)} of {len(tickets)} tickets already done")

try:
    for group, tid, text in tickets:
        if tid in done:
            continue
        before = llm_calls()
        s = run_ticket(text, tid)
        path = path_taken(s)
        row = {
            "group": group, "ticket_id": tid, "outcome": outcome(path, s["status"]),
            "path": " > ".join(path), "attempts": s["attempts"],
            "first_confidence": s["confidence_history"][0], "last_confidence": s["confidence_history"][-1],
            "llm_calls": llm_calls() - before + (1 if s["status"] == "drafted" else 0),
            "escalation_paragraph": (("template" if s["top_score"] < 0.55 else "llm")
                                     if s["status"] == "escalated" else ""),
        }
        with OUT_FILE.open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, FIELDS).writerow(row)
        done.add(tid)
        if len(done) % 10 == 0:
            print(f"  {len(done)}/{len(tickets)}")
except RateLimitError as e:
    print(f"\nSTOPPED by the LLM rate limit after {len(done)} tickets. Run this script again later.\n{str(e)[:300]}")
    sys.exit(1)

report(pd.read_csv(OUT_FILE))
