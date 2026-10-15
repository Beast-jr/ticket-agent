"""Run 10 eval tickets + 3 off-topic tickets through the graph and show which route each takes.

Run from the project root:  python scripts/try_router.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from agent.graph import run_ticket
from off_topic import OFF_TOPIC

STATUS_TO_ROUTE = {"replied": "draft", "drafted": "draft", "draft_failed": "draft",
                   "check_failed": "draft", "escalated": "escalate"}

tickets = [(t.ticket_id, t.text) for t in
           pd.read_csv(ROOT / "data" / "processed" / "eval_set.csv").sample(10, random_state=5).itertuples()]
tickets += [(f"OFF{i:02d}", OFF_TOPIC[i - 1]) for i in (1, 4, 5)]  # cooking, Tesla, "help"

routes = Counter()
print(f"{'ticket':<8}{'category':<32}{'top1':>6}{'agree':>7}{'conf':>6}  {'route':<14}{'status':<14}text")
for tid, text in tickets:
    s = run_ticket(text, tid)
    route = STATUS_TO_ROUTE.get(s["status"], s["status"])
    routes[route] += 1
    print(f"{tid:<8}{s['category']:<32}{s['top_score']:>6.2f}{s['agreement']:>7.1f}{s['confidence']:>6.2f}"
          f"  {route:<14}{s['status']:<14}{text[:100]!r}".replace("\\n", " "))

print("\nTickets per route:")
for route in ("draft", "rewrite_query", "escalate"):
    print(f"  {route:<14}{routes[route]}")
