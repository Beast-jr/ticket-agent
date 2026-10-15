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
    attempts: int                  # how many times we tried to draft a reply
    search_query: str              # what we searched for (can be rewritten on retry)
    check_result: str              # outcome of the self-check step
    status: str                    # e.g. new / classified / done
    steps_log: list[str]           # plain-English diary of what happened
