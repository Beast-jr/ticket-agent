"""The steps (nodes) of the agent. A node takes the state and returns what it wants to change."""
import json

from agent.state import TicketState
from llm import call_llm
from prepare_data import QUEUE_TO_CATEGORY

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
