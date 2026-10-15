"""Day 2: clean the raw ticket CSV and split it into a knowledge base and an eval set.

Run from the project root:  python src/prepare_data.py
"""
import json
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "tickets.csv"
OUT_DIR = ROOT / "data" / "processed"

# --- Settings ---------------------------------------------------------------

# Original queue -> final category (see docs/DATA_NOTES.md)
QUEUE_TO_CATEGORY = {
    "Technical Support": "Technical & IT Support",
    "IT Support": "Technical & IT Support",
    "Product Support": "Product Support",
    "Customer Service": "Customer Service",
    "General Inquiry": "Customer Service",
    "Billing and Payments": "Billing and Payments",
    "Returns and Exchanges": "Returns and Exchanges",
    "Service Outages and Maintenance": "Service Outages and Maintenance",
    "Sales and Pre-Sales": "Sales and Pre-Sales",
    "Human Resources": "Human Resources",
}

# Placeholder tag -> neutral replacement. Order matters: "Dear <name>" is handled first.
PLACEHOLDER_REPLACEMENTS = {
    "<name>": "there",
    "<tel_num>": "our support line",
    "<email>": "our support email",
    "<acc_num>": "your account",
    "<website_url>": "our website",
    "<link>": "our website",
    "<amount>": "the amount",
    "<br>": " ",  # a leftover line-break tag, just a space
}
FALLBACK_REPLACEMENT = "[redacted]"  # any tag we did not plan for
PLACEHOLDER_PATTERN = re.compile(r"<[a-z_]+>", re.IGNORECASE)

MIN_WORDS = 6  # drop bodies/answers with 5 words or fewer
EVAL_SIZE = 100
SEED = 42
TAG_COLUMNS = [f"tag_{i}" for i in range(1, 9)]


# --- Helpers ----------------------------------------------------------------

def word_count(s: pd.Series) -> pd.Series:
    return s.fillna("").str.split().str.len()


def clean_placeholders(text: str) -> str:
    if not isinstance(text, str):
        return text
    text = re.sub(r"\b(Dear|Hello|Hi|Hey)\s+<name>", r"\1 there", text, flags=re.IGNORECASE)
    text = re.sub(r"\bDear there\b", "Dear customer", text)
    # A tag at the start of a line ("<name>, thank you...") becomes a greeting
    text = re.sub(r"(^|\n)[ \t]*<name>(?=[\s,:])", r"\1Hello", text, flags=re.IGNORECASE)

    def swap(match):
        return PLACEHOLDER_REPLACEMENTS.get(match.group(0).lower(), FALLBACK_REPLACEMENT)

    text = PLACEHOLDER_PATTERN.sub(swap, text)
    return re.sub(r"[ \t]{2,}", " ", text).strip()  # tidy double spaces


def make_text(subject, body) -> str:
    body = str(body).strip()
    if isinstance(subject, str) and subject.strip():
        return f"{subject.strip()}\n\n{body}"
    return body


# --- Main -------------------------------------------------------------------

def main():
    df = pd.read_csv(RAW_CSV)
    # Stable id: based on the row position in the raw file, so it never changes between runs
    df["ticket_id"] = [f"T{i + 1:05d}" for i in range(len(df))]
    print(f"Raw rows:                      {len(df):>6}")

    df = df[df["language"] == "en"]
    print(f"English only:                  {len(df):>6}")

    # List every placeholder type found (before cleaning)
    found = pd.Series(
        re.findall(PLACEHOLDER_PATTERN, " ".join(df["body"].fillna("") + " " + df["answer"].fillna("")))
    ).value_counts()
    print("\nPlaceholders found -> how handled:")
    for tag, count in found.items():
        print(f"  {tag:<14}{count:>6}x  -> {PLACEHOLDER_REPLACEMENTS.get(tag.lower(), FALLBACK_REPLACEMENT)!r}")

    df = df[(word_count(df["body"]) >= MIN_WORDS) & (word_count(df["answer"]) >= MIN_WORDS)]
    print(f"\nBody and answer > 5 words:     {len(df):>6}")

    df = df.drop_duplicates(subset=["body", "answer"])
    print(f"After dropping duplicates:     {len(df):>6}")

    df = df.copy()
    df["category"] = df["queue"].map(QUEUE_TO_CATEGORY)
    assert df["category"].notna().all(), "A queue is missing from QUEUE_TO_CATEGORY"

    df["answer"] = df["answer"].map(clean_placeholders)
    df["text"] = [make_text(s, clean_placeholders(b)) for s, b in zip(df["subject"], df["body"])]
    df["tags"] = df[TAG_COLUMNS].apply(lambda r: [t for t in r if isinstance(t, str) and t.strip()], axis=1)

    left = PLACEHOLDER_PATTERN.findall(" ".join(df["text"] + " " + df["answer"]))
    print(f"Placeholders left in text/answer: {len(left)}")

    keep = ["ticket_id", "text", "answer", "category", "queue", "priority", "type", "tags"]
    df = df[keep]
    # tags are saved as JSON text inside the CSV (a CSV cannot hold a list)
    df["tags"] = df["tags"].map(json.dumps)

    kb, ev = train_test_split(df, test_size=EVAL_SIZE, stratify=df["category"], random_state=SEED)
    kb, ev = kb.sort_values("ticket_id"), ev.sort_values("ticket_id")
    assert not set(kb["ticket_id"]) & set(ev["ticket_id"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    kb.to_csv(OUT_DIR / "knowledge_base.csv", index=False)
    ev.to_csv(OUT_DIR / "eval_set.csv", index=False)

    print(f"\nKnowledge base: {len(kb)} rows   Eval set: {len(ev)} rows")
    print("\nCategory counts (knowledge base / eval):")
    print(pd.DataFrame({"knowledge_base": kb["category"].value_counts(), "eval": ev["category"].value_counts()}))


if __name__ == "__main__":
    main()
