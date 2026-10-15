"""Wire the nodes together into a LangGraph.

Day 5:
START -> classify -> retrieve -> score_confidence -> router -+-> draft ----------+-> END
                                                              +-> rewrite_query --+
                                                              +-> escalate -------+
"""
from langgraph.graph import END, START, StateGraph

from agent.nodes import (classify, draft, escalate, retrieve, rewrite_query, router,
                         score_confidence)
from agent.state import TicketState


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("score_confidence", score_confidence)
    graph.add_node("draft", draft)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("escalate", escalate)

    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "score_confidence")
    # Conditional edge: router(state) returns a name, and the dict maps that name to a node
    graph.add_conditional_edges(
        "score_confidence",
        router,
        {"draft": "draft", "rewrite_query": "rewrite_query", "escalate": "escalate"},
    )
    graph.add_edge("draft", END)
    graph.add_edge("rewrite_query", END)
    graph.add_edge("escalate", END)
    return graph.compile()


_app = None  # built once, reused


def run_ticket(text: str, ticket_id: str = "") -> TicketState:
    """Run one ticket through the graph and return the final state."""
    global _app
    if _app is None:
        _app = build_graph()
    start_state: TicketState = {
        "ticket_id": ticket_id,
        "text": text,
        "attempts": 0,
        "status": "new",
        "steps_log": [],
    }
    return _app.invoke(start_state)
