"""Shared test helpers: a fake LLM and fake tickets. No Groq call is ever made through these."""
import json

TICKET = "Concerns about hospital security systems. Suspected data breach, examining access logs."
CLASSIFY = json.dumps({"category": "Technical & IT Support", "priority": "high", "reason": "breach"})
SUMMARY_PARAGRAPH = "The customer reports a suspected data breach. A human should review the reply."


def similar_tickets():
    return [
        {"ticket_id": f"T0000{i}", "text": f"Past ticket {i} about a data breach", "answer": f"Past answer {i}",
         "category": "Technical & IT Support", "priority": "high", "similarity": 0.95 - i * 0.01}
        for i in range(1, 6)
    ]


def draft_json(reply="Please send the access logs [T00001].", cited=("T00001",)):
    return json.dumps({
        "reply": reply, "used_ticket_ids": list(cited),
        "analysis": {"summary": "s", "root_cause": "r", "suggested_fix": "f", "severity": "high"},
    })


def verdict(v, reason="because"):
    return json.dumps({"verdict": v, "reason": reason})


class FakeLLM:
    """Replaces call_llm. `script` maps a step name to the list of answers to give, in order.
    Any call that is not in the script fails the test, so 'no LLM call was made' is checkable."""

    def __init__(self, script):
        self.script = {k: list(v) for k, v in script.items()}
        self.calls = []  # (step, prompt)

    def __call__(self, prompt, system="", json_mode=False, step=""):
        self.calls.append((step, prompt))
        if not self.script.get(step):
            raise AssertionError(f"unexpected LLM call for step {step!r}")
        return self.script[step].pop(0)

    @property
    def steps(self):
        return [s for s, _ in self.calls]
