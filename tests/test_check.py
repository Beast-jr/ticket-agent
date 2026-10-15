"""Tests for the Day 7 self-check. The LLM and the database search are both mocked:
no Groq call and no Postgres connection happens in these tests."""
import pytest

import agent.graph as graph_module
import agent.nodes as nodes
from agent.graph import path_taken, run_ticket

from helpers import (CLASSIFY, SUMMARY_PARAGRAPH, TICKET, FakeLLM, draft_json, similar_tickets,
                     verdict)


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeLLM and a fake search. Usage: llm = fake({...script...})"""
    def install(script):
        llm = FakeLLM(script)
        monkeypatch.setattr(nodes, "call_llm", llm)
        monkeypatch.setattr(nodes, "search_similar", lambda query, k=5: similar_tickets())
        monkeypatch.setattr(graph_module, "_app", None)  # rebuild the graph with the patched nodes
        return llm
    return install


def state_with_reply(reply, cited):
    return {"text": TICKET, "draft_reply": reply, "used_ticket_ids": list(cited),
            "similar_tickets": similar_tickets(), "steps_log": []}


# a) good reply -> supported -> replied
def test_good_reply_is_supported_and_replied(fake):
    llm = fake({"classify": [CLASSIFY], "draft": [draft_json()], "check": [verdict("supported")]})
    s = run_ticket(TICKET, "T1")
    assert s["status"] == "replied"
    assert s["check_result"] == "supported"
    assert s["redrafts"] == 0
    assert llm.steps == ["classify", "draft", "check"]
    assert path_taken(s) == ["classify", "retrieve", "score_confidence", "draft", "check"]


# b) citing an id that is not in the results -> rule check fails, no LLM call
def test_rule_check_fails_for_unknown_id_without_calling_llm(fake):
    llm = fake({})  # empty script: ANY llm call raises
    result = nodes.check(state_with_reply("Send the logs [T99999].", ["T99999"]))
    assert result["check_result"] == "not_supported"
    assert "not in the search results" in result["check_reason"]
    assert llm.calls == []


def test_rule_check_fails_when_nothing_is_cited_without_calling_llm(fake):
    llm = fake({})
    result = nodes.check(state_with_reply("We will look into it.", []))
    assert result["check_result"] == "not_supported"
    assert "cites no past ticket" in result["check_reason"]
    assert llm.calls == []


def test_inline_citation_tags_count_even_if_the_list_is_empty(fake):
    fake({})
    reply, cited, _ = nodes._parse_draft(draft_json("Send logs [T00002].", cited=()))
    assert cited == ["T00002"]
    reply, cited, _ = nodes._parse_draft(draft_json("Send logs [T99999].", cited=("T00001",)))
    assert cited == ["T00001", "T99999"]  # the invented id is kept so the rule check can catch it


def test_invented_ids_in_the_graph_escalate_without_any_check_llm_call(fake):
    bad = draft_json("Send the logs [T99999].", cited=("T99999",))
    llm = fake({"classify": [CLASSIFY], "draft": [bad], "redraft": [bad],
                "escalate_summary": [SUMMARY_PARAGRAPH]})
    s = run_ticket(TICKET, "T2")
    assert s["status"] == "escalated"
    assert not any(step.startswith("check") for step in llm.steps)  # the LLM never checked anything
    assert "T99999" in s["escalation_summary"]


# c) LLM says not_supported, redraft passes -> replied after 1 redraft
def test_redraft_after_failed_check_then_replied(fake):
    reason = "the reply promises a 24 hour fix that the tickets never mention"
    llm = fake({
        "classify": [CLASSIFY],
        "draft": [draft_json("We will fix it within 24 hours [T00001].")],
        "check": [verdict("not_supported", reason), verdict("supported")],
        "redraft": [draft_json("Please send the access logs [T00001].")],
    })
    s = run_ticket(TICKET, "T3")
    assert s["status"] == "replied"
    assert s["redrafts"] == 1
    assert s["check_attempts"] == 2
    assert s["draft_reply"] == "Please send the access logs [T00001]."
    assert llm.steps == ["classify", "draft", "check", "redraft", "check"]
    redraft_prompt = dict(llm.calls[3:4])["redraft"]
    assert reason in redraft_prompt  # the check's reason was passed into the redraft prompt
    assert "24 hours" in redraft_prompt  # ...together with the rejected reply
    assert path_taken(s) == ["classify", "retrieve", "score_confidence", "draft", "check", "redraft", "check"]


# d) not_supported twice -> escalated, reason in the summary
def test_two_failed_checks_escalate_with_the_reason_in_the_summary(fake):
    llm = fake({
        "classify": [CLASSIFY],
        "draft": [draft_json("Reply one [T00001].")],
        "check": [verdict("not_supported", "first problem"), verdict("not_supported", "invented a refund policy")],
        "redraft": [draft_json("Reply two [T00001].")],
        "escalate_summary": [SUMMARY_PARAGRAPH],
    })
    s = run_ticket(TICKET, "T4")
    assert s["status"] == "escalated"
    summary = s["escalation_summary"]
    why = summary.split("WHY ESCALATED:")[1].split("WHAT WAS TRIED:")[0]
    assert "invented a refund policy" in why  # the check reason is in the "why escalated" part
    assert "LAST REJECTED DRAFT" in summary and "Reply two" in summary
    assert s["draft_reply"] == ""  # a rejected reply is never left where a customer could see it


# e) never more than one redraft
def test_never_more_than_one_redraft(fake):
    llm = fake({
        "classify": [CLASSIFY],
        "draft": [draft_json()],
        "check": [verdict("not_supported", "bad")] * 5,  # offers more checks than should ever happen
        "redraft": [draft_json()] * 5,                   # ...and more redrafts
        "escalate_summary": [SUMMARY_PARAGRAPH],
    })
    s = run_ticket(TICKET, "T5")
    assert s["redrafts"] == 1
    assert llm.steps.count("redraft") == 1
    assert llm.steps.count("draft") + llm.steps.count("redraft") == 2  # one draft + one redraft
    assert llm.steps.count("check") == 2
    assert path_taken(s).count("redraft") == 1


def test_after_check_router_allows_exactly_one_redraft():
    assert nodes.after_check({"check_result": "supported"}) == "done"
    assert nodes.after_check({"check_result": "not_supported", "redrafts": 0}) == "redraft"
    assert nodes.after_check({"check_result": "not_supported", "redrafts": 1}) == "escalate"
    assert nodes.after_check({"check_result": "not_supported", "redrafts": 2}) == "escalate"


def test_checker_giving_garbage_twice_counts_as_not_supported(fake):
    llm = fake({"check": ["not json"], "check_strict": ['{"verdict": "maybe"}']})
    result = nodes.check(state_with_reply("Send the logs [T00001].", ["T00001"]))
    assert result["check_result"] == "not_supported"
    assert "invalid answer twice" in result["check_reason"]
    assert llm.steps == ["check", "check_strict"]


def test_check_prompt_shows_only_the_cited_tickets(fake):
    llm = fake({"check": [verdict("supported")]})
    nodes.check(state_with_reply("Send logs [T00002].", ["T00002"]))
    prompt = llm.calls[0][1]
    assert "Past answer 2" in prompt
    assert "Past answer 1" not in prompt and "Past answer 3" not in prompt
