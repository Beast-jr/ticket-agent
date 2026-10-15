"""Wire the nodes together into a LangGraph. Day 4: START -> classify -> retrieve -> draft -> END."""
from langgraph.graph import END, START, StateGraph

from agent.nodes import classify, draft, retrieve
from agent.state import TicketState


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify)
    graph.add_node("retrieve", retrieve)
    graph.add_node("draft", draft)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", "retrieve")
    graph.add_edge("retrieve", "draft")
    graph.add_edge("draft", END)
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
