"""The steps (nodes) of the agent. A node takes the state and returns what it wants to change."""
import json

from agent.state import TicketState
from llm import call_llm
from prepare_data import PLACEHOLDER_PATTERN, QUEUE_TO_CATEGORY, clean_placeholders
from search import search_similar

CATEGORIES = list(dict.fromkeys(QUEUE_TO_CATEGORY.values()))  # unique, in order
PRIORITIES = ["low", "medium", "high"]

# Short hints help the model, because several categories overlap a lot
CATEGORY_HINTS = {
    "Technical & IT Support": "something is broken or not working: login, VPN, network, software errors, security",
    "Product Support": "questions or problems about how a product or feature works",
    "Customer Service": "general questions, feedback, suggestions, account or service requests",
    "Billing and Payments": "invoices, charges, payments, refunds, subscription billing",
    "Returns and Exchanges": "returning, exchanging or replacing a purchased item",
    "Service Outages and Maintenance": "service is down, outage reports, planned maintenance",
    "Sales and Pre-Sales": "questions before buying: pricing, demos, product offers",
    "Human Resources": "HR topics: employees, staff, internal company matters",
}

SYSTEM = "You are a support-ticket triage assistant. You reply only with JSON."


def _build_prompt(ticket_text: str, strict: bool) -> str:
    category_lines = "\n".join(f"- {c}: {CATEGORY_HINTS[c]}" for c in CATEGORIES)
    prompt = (
        "Classify this support ticket.\n\n"
        f"Allowed categories (pick exactly one):\n{category_lines}\n\n"
        f"Allowed priorities: {', '.join(PRIORITIES)}\n\n"
        f"TICKET:\n{ticket_text}\n\n"
        'Reply with JSON only, in this shape: {"category": "...", "priority": "...", "reason": "one short sentence"}'
    )
    if strict:
        prompt += (
            "\n\nIMPORTANT: your last answer was invalid. "
            "\"category\" must be copied EXACTLY from the allowed categories list above, "
            f"and \"priority\" must be exactly one of: {', '.join(PRIORITIES)}. "
            "Return valid JSON and nothing else."
        )
    return prompt


def _parse(raw: str) -> tuple[str, str, str]:
    """Return (category, priority, reason) or raise ValueError if anything is wrong."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"broken JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    by_lower_cat = {c.lower(): c for c in CATEGORIES}
    category = by_lower_cat.get(str(data.get("category", "")).strip().lower())
    priority = str(data.get("priority", "")).strip().lower()
    if category is None:
        raise ValueError(f"category not allowed: {data.get('category')!r}")
    if priority not in PRIORITIES:
        raise ValueError(f"priority not allowed: {data.get('priority')!r}")
    return category, priority, str(data.get("reason", "")).strip()


def classify(state: TicketState) -> dict:
    """Pick a category and priority for the ticket. Falls back to Unknown / medium."""
    log = list(state.get("steps_log", []))
    problem = ""
    for strict in (False, True):  # first normal prompt, then one stricter retry
        raw = call_llm(
            _build_prompt(state["text"], strict), system=SYSTEM, json_mode=True,
            step="classify_strict" if strict else "classify",
        )
        try:
            category, priority, reason = _parse(raw)
            note = " (after a stricter retry)" if strict else ""
            log.append(f"classify: {category} / {priority}{note} - {reason}")
            return {"category": category, "priority": priority, "classify_reason": reason,
                    "status": "classified", "steps_log": log}
        except ValueError as e:
            problem = str(e)

    log.append(f"classify: LLM answer invalid twice ({problem}); fell back to Unknown / medium")
    return {"category": "Unknown", "priority": "medium", "classify_reason": f"fallback: {problem}",
            "status": "classified", "steps_log": log}


# --- retrieve ---------------------------------------------------------------

def retrieve(state: TicketState) -> dict:
    """Find the 5 most similar past tickets in the database."""
    query = state.get("search_query") or state["text"]
    found = search_similar(query, k=5)
    top = f"{found[0]['similarity']:.2f}" if found else "n/a"
    log = list(state.get("steps_log", []))
    log.append(f"retrieve: found {len(found)} similar tickets, top score {top}")
    return {"similar_tickets": found, "search_query": query, "steps_log": log}


# --- draft ------------------------------------------------------------------

SEVERITIES = ["low", "medium", "high"]
DRAFT_SYSTEM = (
    "You are a careful customer-support agent. You write replies using only the "
    "evidence you are given. You reply only with JSON."
)


def _build_draft_prompt(state: TicketState, strict: bool) -> str:
    past = "\n\n".join(
        f"[{t['ticket_id']}] (category: {t['category']})\n"
        f"PAST TICKET: {t['text']}\nPAST ANSWER: {t['answer']}"
        for t in state.get("similar_tickets", [])
    ) or "(no similar past tickets were found)"
    prompt = (
        f"NEW TICKET (category: {state.get('category')}, priority: {state.get('priority')}):\n"
        f"{state['text']}\n\n"
        f"SIMILAR PAST TICKETS AND HOW THEY WERE ANSWERED:\n{past}\n\n"
        "Write a reply to the new ticket and a short analysis card.\n"
        "RULES:\n"
        "- Base the reply ONLY on the past answers above. Reuse their advice and wording where it fits.\n"
        "- Do NOT invent product names, prices, phone numbers, email addresses, links, deadlines or policies "
        "that do not appear in the past tickets or the new ticket.\n"
        "- If the past tickets do not really fit the new ticket, say so politely in the reply and ask the "
        "customer for more details instead of guessing.\n"
        "- Never write placeholder tags such as <name> or <tel_num>.\n"
        "- used_ticket_ids: only the ids (like T01234) of past tickets you actually relied on; use [] if none.\n\n"
        'Reply with JSON only, in this shape:\n'
        '{"reply": "...", "used_ticket_ids": ["T..."], '
        '"analysis": {"summary": "...", "root_cause": "...", "suggested_fix": "...", "severity": "low|medium|high"}}'
    )
    if strict:
        prompt += (
            "\n\nIMPORTANT: your last answer was invalid. Return valid JSON in exactly that shape, "
            "with a non-empty reply, a list for used_ticket_ids, all four analysis fields as text, "
            f"and severity exactly one of: {', '.join(SEVERITIES)}."
        )
    return prompt


def _parse_draft(raw: str, allowed_ids: set[str]) -> tuple[str, list[str], dict, list[str]]:
    """Return (reply, used_ids, analysis, dropped_ids) or raise ValueError."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"broken JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")

    reply = data.get("reply")
    if not isinstance(reply, str) or not reply.strip():
        raise ValueError("reply is empty")

    used = data.get("used_ticket_ids")
    if not isinstance(used, list):
        raise ValueError("used_ticket_ids is not a list")
    used = [str(u) for u in used]
    kept = [u for u in used if u in allowed_ids]
    dropped = [u for u in used if u not in allowed_ids]

    raw_analysis = data.get("analysis")
    if not isinstance(raw_analysis, dict):
        raise ValueError("analysis is missing")
    analysis = {}
    for key in ("summary", "root_cause", "suggested_fix", "severity"):
        value = raw_analysis.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"analysis.{key} is missing")
        analysis[key] = value.strip()
    analysis["severity"] = analysis["severity"].lower()
    if analysis["severity"] not in SEVERITIES:
        raise ValueError(f"severity not allowed: {raw_analysis.get('severity')!r}")

    return reply.strip(), kept, analysis, dropped


def draft(state: TicketState) -> dict:
    """Write a reply and an analysis card from the new ticket + similar past tickets."""
    log = list(state.get("steps_log", []))
    attempts = state.get("attempts", 0) + 1
    allowed_ids = {t["ticket_id"] for t in state.get("similar_tickets", [])}
    problem = ""
    for strict in (False, True):  # one normal try, one stricter retry
        raw = call_llm(
            _build_draft_prompt(state, strict), system=DRAFT_SYSTEM, json_mode=True,
            step="draft_strict" if strict else "draft",
        )
        try:
            reply, used, analysis, dropped = _parse_draft(raw, allowed_ids)
        except ValueError as e:
            problem = str(e)
            continue

        if dropped:
            log.append(f"draft: dropped used_ticket_ids not in the search results: {dropped}")
        if PLACEHOLDER_PATTERN.search(reply):
            reply = clean_placeholders(reply)
            log.append("draft: removed a raw placeholder tag from the reply")
        note = " (after a stricter retry)" if strict else ""
        log.append(f"draft: reply written{note}, used {len(used)} past tickets, severity {analysis['severity']}")
        return {"draft_reply": reply, "analysis": analysis, "attempts": attempts,
                "status": "drafted", "steps_log": log, "used_ticket_ids": used}

    log.append(f"draft: LLM answer invalid twice ({problem}); status draft_failed")
    return {"draft_reply": "", "attempts": attempts, "status": "draft_failed", "steps_log": log,
            "used_ticket_ids": []}
