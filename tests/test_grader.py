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
    # One batched grader call covers the single new doc.
    patch_get_llm.queue(
        {"grades": [{"index": 0, "relevance_score": 0.8, "is_grounded": True, "rationale": "ok"}]}
    )
    out = grader_node(_state(graded_documents=existing, raw_documents=raw))
    # One LLM call: the batched grader -- the existing doc is filtered out before grading.
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
    patch_get_llm.queue(
        {"grades": [{"index": 0, "relevance_score": 99.0, "is_grounded": True, "rationale": "r"}]}
    )
    out = grader_node(_state(raw_documents=raw))
    assert out["graded_documents"][0].relevance_score == 1.0


def test_grader_batched_path(patch_get_llm):
    """Happy path: one batched LLM call covers the single new doc."""
    raw = [{"url": "https://a", "content": "x", "source": "tavily", "sub_question": "q"}]
    patch_get_llm.queue(
        {"grades": [{"index": 0, "relevance_score": 0.85, "is_grounded": True, "rationale": "r"}]}
    )
    out = grader_node(_state(raw_documents=raw))
    assert out["graded_documents"][0].relevance_score == 0.85
    assert len(patch_get_llm.calls) == 1


def test_grader_falls_back_to_per_doc_when_batch_raises(patch_get_llm):
    """If the batched call raises, fall back to per-doc structured grading."""
    raw = [{"url": "https://a", "content": "x", "source": "tavily", "sub_question": "q"}]
    patch_get_llm.queue(
        ValueError("batch schema mismatch"),
        {"relevance_score": 0.7, "is_grounded": True, "rationale": "fb"},
    )
    out = grader_node(_state(raw_documents=raw))
    assert out["graded_documents"][0].relevance_score == 0.7
    assert out["graded_documents"][0].rationale == "fb"
    # Two calls: failed batch + per-doc structured.
    assert len(patch_get_llm.calls) == 2


def test_grader_batches_five_docs_in_one_call(patch_get_llm):
    """Five docs go in one batched LLM call (not five individual calls)."""
    raw = [
        {"url": f"https://d{i}", "content": f"c{i}", "source": "tavily", "sub_question": "q"}
        for i in range(5)
    ]
    patch_get_llm.queue(
        {
            "grades": [
                {"index": i, "relevance_score": 0.8, "is_grounded": True, "rationale": "ok"}
                for i in range(5)
            ]
        }
    )
    out = grader_node(_state(raw_documents=raw))
    assert len(out["graded_documents"]) == 5
    assert len(patch_get_llm.calls) == 1
