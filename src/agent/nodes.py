"""The steps (nodes) of the agent. A node takes the state and returns what it wants to change."""
import json

from agent.state import TicketState
from llm import call_llm
from prepare_data import PLACEHOLDER_PATTERN, QUEUE_TO_CATEGORY, clean_placeholders
from search import search_similar

# --- Routing settings (calibrated on Day 5, see docs/ROUTING.md and docs/calibration.csv) ---
# confidence = W_TOP1 * (best similarity score) + W_AGREEMENT * (category agreement)
#
# Why this formula:
# - The best similarity score is the strongest signal. On the calibration run, off-topic
#   tickets scored at most 0.45 while real tickets scored 0.63-1.00.
# - Category agreement (how many of the 5 similar tickets have the same category as the
#   classified one) adds a second opinion, but this dataset's labels are noisy: even good
#   in-domain tickets have a median agreement of only 0.4. So it gets the smaller weight.
# - A plain weighted average is easy to explain and easy to re-tune.
W_TOP1 = 0.8
W_AGREEMENT = 0.2
HIGH = 0.75  # confidence >= HIGH            -> draft a reply
LOW = 0.45   # confidence < LOW              -> escalate to a human
#              LOW <= confidence < HIGH      -> the "middle band", see choose_route() below

# Middle-band settings (Day 6). Why the middle band is low tells us how to fix it:
# - TOP1_STRONG: in the calibration data (docs/calibration.csv) the 9 middle-band tickets with a
#   top score of 0.85 or more ALL had category agreement 0.0. The search worked, the labels
#   disagree -> reconcile the category. The 8 tickets below 0.85 are weak matches -> rewrite the query.
# - AGREEMENT_LOW: "low agreement" means fewer than half of the similar tickets share the category.
# - MAX_ATTEMPTS: never loop forever. After this many reconcile/rewrite steps, escalate.
# - OFF_TOPIC_TOP1: below this best-match score nothing in our history resembles the ticket
#   (off-topic tickets peaked at 0.45, real ones never went under 0.63), so escalation uses a
#   fixed template instead of an LLM call.
TOP1_STRONG = 0.85
AGREEMENT_LOW = 0.5
MAX_ATTEMPTS = 2
OFF_TOPIC_TOP1 = 0.55

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

# --- confidence + routing ---------------------------------------------------

def compute_confidence(similar: list[dict], category: str) -> tuple[float, float, float]:
    """Return (top_score, agreement, confidence) for a category and its similar tickets."""
    if not similar:
        return 0.0, 0.0, 0.0
    top_score = similar[0]["similarity"]
    agreement = sum(t["category"] == category for t in similar) / len(similar)
    confidence = round(W_TOP1 * top_score + W_AGREEMENT * agreement, 4)
    return round(top_score, 4), agreement, confidence


def choose_route(confidence: float, top_score: float, agreement: float,
                 attempts: int, reconciled: bool) -> str:
    """Decide the next step: draft, reconcile_category, rewrite_query or escalate."""
    if confidence >= HIGH:
        return "draft"
    if confidence < LOW:
        return "escalate"
    # Middle band from here on
    if attempts >= MAX_ATTEMPTS:
        return "escalate"  # we already tried enough times
    if top_score >= TOP1_STRONG and agreement < AGREEMENT_LOW and not reconciled:
        return "reconcile_category"  # good matches, but their labels disagree with ours
    return "rewrite_query"  # weak matches (or reconcile already tried): search again


def _route_for(state: TicketState) -> str:
    return choose_route(state["confidence"], state["top_score"], state["agreement"],
                        state.get("attempts", 0), state.get("reconciled", False))


def score_confidence(state: TicketState) -> dict:
    """How sure are we that the past tickets fit this one? A number from 0 to 1."""
    top_score, agreement, confidence = compute_confidence(
        state.get("similar_tickets", []), state.get("category"))
    history = list(state.get("confidence_history", [])) + [confidence]
    new = {**state, "confidence": confidence, "top_score": top_score, "agreement": agreement}
    log = list(state.get("steps_log", []))
    log.append(
        f"score_confidence: confidence {confidence:.2f} (top score {top_score:.2f}, "
        f"category agreement {agreement:.1f}) -> route: {_route_for(new)}"
    )
    return {"confidence": confidence, "top_score": top_score, "agreement": agreement,
            "confidence_history": history, "steps_log": log}


def router(state: TicketState) -> str:
    """Conditional edge: returns the NAME of the next node. It does not change the state."""
    return _route_for(state)


# --- reconcile_category -----------------------------------------------------

def _build_reconcile_prompt(state: TicketState) -> str:
    category_lines = "\n".join(f"- {c}: {CATEGORY_HINTS[c]}" for c in CATEGORIES)
    similar = "\n".join(
        f"- [{t['ticket_id']}] category: {t['category']} | {t['text'][:200]!r}"
        for t in state["similar_tickets"]
    )
    return (
        f"NEW TICKET:\n{state['text']}\n\n"
        f"Current category: {state['category']}\n\n"
        f"SIMILAR PAST TICKETS AND THE CATEGORY THEY WERE FILED UNDER:\n{similar}\n\n"
        f"Allowed categories:\n{category_lines}\n\n"
        "The past tickets are about the same topic as the new one, but your current category does not "
        "match how they were filed. The categories in this company's data overlap a lot, so the way "
        "similar tickets are filed is the best guide to how this kind of ticket is normally handled. "
        "Choose the single best category for the NEW ticket: follow the category most of the similar "
        "tickets share, and keep your current category only if the new ticket clearly does not belong "
        "there.\n"
        'Reply with JSON only: {"category": "...", "reason": "one short sentence"}'
    )


def _parse_category(raw: str) -> tuple[str, str]:
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise ValueError(f"broken JSON: {e}")
    if not isinstance(data, dict):
        raise ValueError("JSON is not an object")
    by_lower = {c.lower(): c for c in CATEGORIES}
    category = by_lower.get(str(data.get("category", "")).strip().lower())
    if category is None:
        raise ValueError(f"category not allowed: {data.get('category')!r}")
    return category, str(data.get("reason", "")).strip()


def reconcile_category(state: TicketState) -> dict:
    """Good matches but conflicting labels: ask the LLM for the best final category."""
    log = list(state.get("steps_log", []))
    old = state["category"]
    attempts = state.get("attempts", 0) + 1
    category, reason, problem = old, "", ""
    for strict in (False, True):
        prompt = _build_reconcile_prompt(state)
        if strict:
            prompt += "\n\nIMPORTANT: copy the category EXACTLY from the allowed list. Return valid JSON only."
        raw = call_llm(prompt, system=SYSTEM, json_mode=True,
                       step="reconcile_strict" if strict else "reconcile")
        try:
            category, reason = _parse_category(raw)
            problem = ""
            break
        except ValueError as e:
            problem = str(e)

    top_score, agreement, confidence = compute_confidence(state["similar_tickets"], category)
    history = list(state.get("confidence_history", [])) + [confidence]
    if problem:
        log.append(f"reconcile_category: LLM answer invalid twice ({problem}); kept {old!r}")
    else:
        verb = "kept" if category == old else f"changed {old!r} to"
        log.append(f"reconcile_category: {verb} {category!r} - {reason} "
                   f"(agreement {state['agreement']:.1f} -> {agreement:.1f}, "
                   f"confidence {state['confidence']:.2f} -> {confidence:.2f})")
    return {"category": category, "top_score": top_score, "agreement": agreement,
            "confidence": confidence, "confidence_history": history, "reconciled": True,
            "attempts": attempts, "steps_log": log}


# --- rewrite_query ----------------------------------------------------------

def rewrite_query(state: TicketState) -> dict:
    """Weak matches: rewrite the ticket into a short, clear search query."""
    log = list(state.get("steps_log", []))
    old_query = state.get("search_query") or state["text"]
    prompt = (
        "Rewrite this support ticket as a short search query for finding similar past tickets. "
        "Keep the main problem, the product or system involved, and the symptoms. "
        "Drop greetings, politeness and filler. Maximum 25 words. Do not invent details that are "
        "not in the ticket.\n\n"
        f"TICKET:\n{state['text']}\n\n"
        'Reply with JSON only: {"query": "..."}'
    )
    raw = call_llm(prompt, system=SYSTEM, json_mode=True, step="rewrite_query")
    try:
        new_query = str(json.loads(raw).get("query", "")).strip()
    except (json.JSONDecodeError, AttributeError, TypeError):
        new_query = ""
    note = ""
    if not new_query:
        new_query, note = old_query, " (LLM gave no usable query, kept the old one)"
    attempts = state.get("attempts", 0) + 1
    if " ".join(new_query.lower().split()) == " ".join(old_query.lower().split()):
        # A rewrite that changes nothing would repeat the same search: use up the retries instead
        attempts = MAX_ATTEMPTS
        note = " (same query as before, so no point retrying again)"
    log.append(f"rewrite_query: {old_query[:70]!r} -> {new_query[:70]!r}{note}")
    return {"search_query": new_query, "attempts": attempts, "steps_log": log}


# --- escalate ---------------------------------------------------------------

def _escalation_reason(state: TicketState) -> str:
    conf, top = state["confidence"], state["top_score"]
    if conf < LOW:
        return (f"confidence {conf:.2f} is below the escalation limit of {LOW}; "
                f"the best past match scores only {top:.2f}")
    return (f"still not confident after {state.get('attempts', 0)} retry steps "
            f"(confidence {conf:.2f}, a reply needs {HIGH}; best match score {top:.2f})")


def _problem_paragraph(state: TicketState) -> str:
    """One paragraph about the problem. Fixed template when nothing resembles the ticket."""
    top = state["top_score"]
    if top < OFF_TOPIC_TOP1:
        snippet = " ".join(state["text"].split())[:200]
        return (f"This ticket does not resemble anything in the support history (best past match "
                f"scores only {top:.2f}). It may be off-topic, too vague, or unreadable. "
                f"Original text: \"{snippet}\"")
    prompt = (
        "Write ONE short paragraph (2-4 sentences) for a human support agent who is taking over this "
        "ticket. Say what the customer's problem is and why an automatic reply was not safe to send. "
        "Use only the facts below. Do not invent details.\n\n"
        f"TICKET:\n{state['text']}\n\n"
        f"Category: {state.get('category')}, priority: {state.get('priority')}\n"
        f"Why it was escalated: {_escalation_reason(state)}\n"
        "Reply with the paragraph only."
    )
    return call_llm(prompt, step="escalate_summary").strip()


def escalate(state: TicketState) -> dict:
    """Hand the ticket to a human, with a summary that saves them time."""
    log = list(state.get("steps_log", []))
    tried = "\n".join(f"  - {line}" for line in log)
    closest = "\n".join(
        f"  - {t['ticket_id']} (score {t['similarity']:.2f}, {t['category']}): "
        f"{' '.join(t['text'].split())[:80]}"
        for t in state.get("similar_tickets", [])[:3]
    ) or "  (none)"
    summary = (
        f"PROBLEM: {_problem_paragraph(state)}\n\n"
        f"WHY ESCALATED: {_escalation_reason(state)}\n\n"
        f"WHAT WAS TRIED:\n{tried}\n\n"
        f"CLOSEST PAST TICKETS:\n{closest}"
    )
    log.append("escalate: ticket handed to a human with a summary "
               f"({'template' if state['top_score'] < OFF_TOPIC_TOP1 else 'LLM'} paragraph)")
    return {"status": "escalated", "escalation_summary": summary, "steps_log": log}
