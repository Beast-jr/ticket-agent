"""The shared state: one notebook that every step of the agent reads from and writes to."""
from typing import TypedDict


class Analysis(TypedDict, total=False):
    summary: str
    root_cause: str
    suggested_fix: str
    severity: str


class TicketState(TypedDict, total=False):
    ticket_id: str
    text: str                      # subject + body of the incoming ticket
    category: str                  # set by classify
    priority: str                  # low / medium / high, set by classify
    classify_reason: str           # one-line explanation from the LLM
    similar_tickets: list[dict]    # past tickets found by search (Day 4+)
    draft_reply: str               # the reply the agent writes (Day 4+)
    used_ticket_ids: list[str]     # which similar tickets the reply was based on
    analysis: Analysis             # summary, root cause, suggested fix, severity
    confidence: float              # how sure the agent is (0-1)
    top_score: float               # best similarity score from the search
    agreement: float               # share of similar tickets with the classified category
    attempts: int                  # how many retry steps (reconcile / rewrite) and drafts so far
    search_query: str              # what we searched for (can be rewritten on retry)
    check_result: str              # supported / not_supported (set by check)
    check_reason: str              # why the self-check said that
    check_attempts: int            # how many times the self-check has run
    redrafts: int                  # how many times the reply was rewritten after a failed check (max 1)
    status: str                    # new / classified / drafted / replied / escalated
    escalation_summary: str        # hand-over note for a human (set by escalate)
    reconciled: bool               # True once reconcile_category has run (it only runs once)
    confidence_history: list[float]  # confidence after each scoring, oldest first
    steps_log: list[str]           # plain-English diary of what happened
