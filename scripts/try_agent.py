"""Run 5 eval tickets end to end (classify -> retrieve -> draft) and show everything.

Run from the project root:  python scripts/try_agent.py
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from agent.graph import run_ticket

LOG_FILE = ROOT / "logs" / "llm_calls.jsonl"
N = 5


def log_line_count() -> int:
    return len(LOG_FILE.read_text(encoding="utf-8").splitlines()) if LOG_FILE.exists() else 0


eval_df = pd.read_csv(ROOT / "data" / "processed" / "eval_set.csv").sample(N, random_state=11)

for _, t in eval_df.iterrows():
    before = log_line_count()
    start = time.time()
    s = run_ticket(t.text, t.ticket_id)
    seconds = time.time() - start
    calls = log_line_count() - before

    print("=" * 90)
    print(f"TICKET {t.ticket_id}: {t.text[:200]!r}")
    print(f"category: {s['category']} (real: {t.category})   priority: {s['priority']} (real: {t.priority})")
    print("top 3 similar:", ", ".join(f"{x['ticket_id']} ({x['similarity']:.2f}, {x['category']})"
                                      for x in s["similar_tickets"][:3]))
    print(f"\nREPLY:\n{s['draft_reply']}")
    print(f"\nREAL ANSWER (from dataset):\n{t.answer}")
    print("\nANALYSIS CARD:")
    print(json.dumps(s.get("analysis", {}), indent=2))
    print("used_ticket_ids:", s.get("used_ticket_ids"))
    print("\nsteps_log:")
    for line in s["steps_log"]:
        print("  -", line)
    print(f"\nstatus: {s['status']}   time: {seconds:.1f}s   LLM calls: {calls}")
