"""Run 20 eval tickets through the classify graph and report accuracy.

Run from the project root:  python scripts/try_classify.py
"""
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from agent.graph import run_ticket

N = 20
eval_df = pd.read_csv(ROOT / "data" / "processed" / "eval_set.csv").sample(N, random_state=42)

cat_ok = pri_ok = 0
cat_confusions = Counter()
for _, t in eval_df.iterrows():
    s = run_ticket(t.text, t.ticket_id)
    c_hit, p_hit = s["category"] == t.category, s["priority"] == t.priority
    cat_ok += c_hit
    pri_ok += p_hit
    if not c_hit:
        cat_confusions[(t.category, s["category"])] += 1
    print(f"{t.ticket_id}  category {'OK ' if c_hit else 'XX '} real={t.category!r} pred={s['category']!r}")
    print(f"{'':8}priority {'OK ' if p_hit else 'XX '} real={t.priority} pred={s['priority']}   | {t.text[:70]!r}".replace("\\n", " "))

print("\n" + "=" * 70)
print(f"Category accuracy: {cat_ok}/{N} = {cat_ok / N:.0%}")
print(f"Priority accuracy: {pri_ok}/{N} = {pri_ok / N:.0%}")
if cat_confusions:
    print("\nMost common category mix-ups (real -> predicted):")
    for (real, pred), n in cat_confusions.most_common():
        print(f"  {n}x  {real}  ->  {pred}")
