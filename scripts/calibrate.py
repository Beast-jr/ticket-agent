"""Measure similarity scores for in-domain vs off-topic tickets, to pick routing thresholds.

Runs classify + retrieve only (no draft). Saves docs/calibration.csv.
Run from the project root:  python scripts/calibrate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import pandas as pd

from agent.nodes import classify, retrieve
from off_topic import OFF_TOPIC


def measure(text: str) -> dict:
    state = {"text": text, "steps_log": []}
    state.update(classify(state))
    state.update(retrieve(state))
    sims = state["similar_tickets"]
    scores = [t["similarity"] for t in sims]
    agreement = sum(t["category"] == state["category"] for t in sims) / len(sims)
    return {
        "category": state["category"],
        "top1": scores[0],
        "avg5": sum(scores) / len(scores),
        "agreement": agreement,
    }


rows = []
eval_df = pd.read_csv(ROOT / "data" / "processed" / "eval_set.csv")
for i, t in enumerate(eval_df.itertuples(), 1):
    rows.append({"set": "in_domain", "ticket_id": t.ticket_id, "text": t.text[:80].replace("\n", " "),
                 "real_category": t.category, **measure(t.text)})
    if i % 20 == 0:
        print(f"  in-domain {i}/100")
for i, text in enumerate(OFF_TOPIC, 1):
    rows.append({"set": "off_topic", "ticket_id": f"OFF{i:02d}", "text": text[:80], "real_category": "",
                 **measure(text)})

df = pd.DataFrame(rows)
df.round(4).to_csv(ROOT / "docs" / "calibration.csv", index=False)

pd.set_option("display.width", 200)
for col in ["top1", "avg5", "agreement"]:
    print(f"\n{col}  (min / 25% / median / 75% / max)")
    print(df.groupby("set")[col].describe()[["min", "25%", "50%", "75%", "max"]].round(3))

print("\nOff-topic tickets in detail:")
print(df[df.set == "off_topic"][["ticket_id", "category", "top1", "avg5", "agreement", "text"]].round(3).to_string(index=False))
