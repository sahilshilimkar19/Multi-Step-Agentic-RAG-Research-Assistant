"""Tests for grader_node."""
from __future__ import annotations

from agentic_rag.nodes.grader import grader_node
from agentic_rag.state import GradedDocument


def _state(**overrides):
    s = {
        "original_query": "q",
        "research_plan": [],
        "search_queries": [],
        "raw_documents": [],
        "graded_documents": [],
        "iteration_count": 1,
        "max_iterations": 4,
        "needs_more_research": True,
        "final_report": "",
        "next_action": "search",
        "messages": [],
    }
    s.update(overrides)
    return s


def test_grades_only_new_docs(patch_get_llm):
    existing = [
        GradedDocument(
            content="old", url="https://old", source="tavily",
            sub_question="q", relevance_score=0.9, is_grounded=True, rationale="r",
        )
    ]
    raw = [
        {"url": "https://old", "content": "old", "source": "tavily", "sub_question": "q"},
        {"url": "https://new", "content": "new content", "source": "tavily", "sub_question": "q"},
    ]
    patch_get_llm.queue(
        {"relevance_score": 0.8, "is_grounded": True, "rationale": "ok"}
    )
    out = grader_node(_state(graded_documents=existing, raw_documents=raw))
    # Only one call: the brand-new doc.
    assert len(patch_get_llm.calls) == 1
    assert len(out["graded_documents"]) == 2  # existing + new
    new_urls = {g.url for g in out["graded_documents"]}
    assert new_urls == {"https://old", "https://new"}


def test_returns_empty_delta_when_nothing_to_grade(patch_get_llm):
    existing = [
        GradedDocument(
            content="x", url="https://a", source="tavily",
            sub_question="q", relevance_score=0.5, is_grounded=True, rationale="r",
        )
    ]
    raw = [{"url": "https://a", "content": "x", "source": "tavily", "sub_question": "q"}]
    out = grader_node(_state(graded_documents=existing, raw_documents=raw))
    assert out == {}
    assert patch_get_llm.calls == []


def test_bad_json_grade_defaults_to_zero(patch_get_llm):
    raw = [{"url": "https://a", "content": "x", "source": "tavily", "sub_question": "q"}]
    patch_get_llm.queue("not json")
    out = grader_node(_state(raw_documents=raw))
    assert len(out["graded_documents"]) == 1
    g = out["graded_documents"][0]
    assert g.relevance_score == 0.0
    assert g.is_grounded is False


def test_clamps_relevance_score(patch_get_llm):
    raw = [{"url": "https://a", "content": "x", "source": "tavily", "sub_question": "q"}]
    patch_get_llm.queue({"relevance_score": 99.0, "is_grounded": True, "rationale": "r"})
    out = grader_node(_state(raw_documents=raw))
    assert out["graded_documents"][0].relevance_score == 1.0
