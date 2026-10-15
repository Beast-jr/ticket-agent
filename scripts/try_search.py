"""Quick test: run 3 eval tickets through search_similar and print the matches.

Run from the project root:  python scripts/try_search.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from search import search_similar

eval_df = pd.read_csv(ROOT / "data" / "processed" / "eval_set.csv")

for _, t in eval_df.sample(3, random_state=1).iterrows():
    print("=" * 90)
    print(f"QUERY {t.ticket_id}  (real category: {t.category})")
    print(t.text[:300].replace("\n", " "))
    print("-" * 90)
    for m in search_similar(t.text, k=5):
        flag = "OK " if m["category"] == t.category else "   "
        print(f"{flag}{m['similarity']:.2f}  [{m['category']}]  {m['text'][:100]!r}".replace("\\n", " "))
    print()
