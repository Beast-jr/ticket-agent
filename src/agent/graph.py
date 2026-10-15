"""Wire the nodes together into a LangGraph. Day 3: START -> classify -> END."""
from langgraph.graph import END, START, StateGraph

from agent.nodes import classify
from agent.state import TicketState


def build_graph():
    graph = StateGraph(TicketState)
    graph.add_node("classify", classify)
    graph.add_edge(START, "classify")
    graph.add_edge("classify", END)
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
