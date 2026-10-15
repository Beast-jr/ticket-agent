"""Four deliberate test cases for the router and its loops.

Run from the project root:  python scripts/try_cases.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from agent.graph import path_taken, run_ticket

LOG_FILE = ROOT / "logs" / "llm_calls.jsonl"
eval_df = pd.read_csv(ROOT / "data" / "processed" / "eval_set.csv").set_index("ticket_id")
cal = pd.read_csv(ROOT / "docs" / "calibration.csv")
cal["conf"] = 0.8 * cal.top1 + 0.2 * cal.agreement

# b) a label-conflict ticket: high top-1 score, agreement 0, confidence in the middle band
conflict = cal[(cal.set == "in_domain") & (cal.top1 >= 0.85) & (cal.agreement == 0) & (cal.conf < 0.75)]
conflict_id = conflict.sort_values("top1").iloc[0].ticket_id

CASES = [
    ("a) easy in-domain ticket", "T15090", eval_df.loc["T15090"].text),
    (f"b) label conflict ({conflict_id})", conflict_id, eval_df.loc[conflict_id].text),
    ("c) vague ticket", "VAGUE", "it's broken again since yesterday, same as last time, please fix asap"),
    ("d) nonsense", "NONSENSE", "qwerty zxcvb lorem asdf 12345 !!!"),
]


def llm_calls() -> int:
    return len(LOG_FILE.read_text(encoding="utf-8").splitlines()) if LOG_FILE.exists() else 0


for title, tid, text in CASES:
    before = llm_calls()
    s = run_ticket(text, tid)
    print("=" * 90)
    print(title)
    print("ticket:", " ".join(text.split())[:160])
    print("path:  ", " -> ".join(path_taken(s)))
    print("attempts:", s["attempts"], "  LLM calls:", llm_calls() - before)
    hist = s["confidence_history"]
    print("confidence:", " -> ".join(f"{c:.2f}" for c in hist), f"  (first {hist[0]:.2f}, last {hist[-1]:.2f})")
    print("category:", s["category"], "  status:", s["status"])
    if s["status"] == "escalated":
        print("\nESCALATION SUMMARY:\n" + s["escalation_summary"])
    else:
        print("\nREPLY:\n" + s.get("draft_reply", "(none)"))
    print("\nsteps_log:")
    for line in s["steps_log"]:
        print("  -", line)
