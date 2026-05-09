"""Tests for planner_node + helpers."""
from __future__ import annotations

from agentic_rag.nodes.planner import _safe_json, _uncovered_plan_items, planner_node
from agentic_rag.state import GradedDocument


def _base_state(**overrides):
    state = {
        "original_query": "How does Mamba compare to Transformers on long-context tasks?",
        "research_plan": [],
        "search_queries": [],
        "raw_documents": [],
        "graded_documents": [],
        "iteration_count": 0,
        "max_iterations": 4,
        "needs_more_research": True,
        "final_report": "",
        "next_action": "search",
        "messages": [],
    }
    state.update(overrides)
    return state


# ---- _safe_json ----------------------------------------------------------

def test_safe_json_strips_fences():
    raw = '```json\n{"a": 1}\n```'
    assert _safe_json(raw, default={"a": 0}) == {"a": 1}


def test_safe_json_returns_default_on_garbage():
    assert _safe_json("not json at all", default={"fallback": True}) == {"fallback": True}


def test_safe_json_plain_json():
    assert _safe_json('{"x": 2}', default={}) == {"x": 2}


# ---- _uncovered_plan_items ----------------------------------------------

def test_uncovered_when_no_relevant_docs():
    plan = ["Compare Mamba scaling", "Discuss attention complexity"]
    assert _uncovered_plan_items(plan, []) == plan


def test_covered_when_keyword_matches():
    plan = ["Compare Mamba scaling on long sequences"]
    docs = [
        GradedDocument(
            content="Mamba scales linearly on long sequences.",
            url="u",
            source="tavily",
            sub_question="q",
            relevance_score=0.9,
            is_grounded=True,
            rationale="r",
        )
    ]
    assert _uncovered_plan_items(plan, docs) == []


# ---- planner_node first visit -------------------------------------------

def test_first_visit_builds_plan_and_queries(patch_get_llm):
    patch_get_llm.queue(
        {
            "plan": ["Sub-Q 1", "Sub-Q 2", "Sub-Q 3"],
            "initial_queries": ["q1", "q2", "q3"],
        }
    )
    out = planner_node(_base_state())
    assert out["research_plan"] == ["Sub-Q 1", "Sub-Q 2", "Sub-Q 3"]
    assert out["search_queries"] == ["q1", "q2", "q3"]
    assert out["next_action"] == "search"


def test_first_visit_falls_back_when_llm_returns_garbage(patch_get_llm):
    patch_get_llm.queue("not valid json")
    out = planner_node(_base_state())
    assert out["research_plan"] == [_base_state()["original_query"]]
    assert out["search_queries"] == [_base_state()["original_query"]]
    assert out["next_action"] == "search"


# ---- planner_node hard termination ---------------------------------------

def test_max_iterations_short_circuits_without_llm_call(patch_get_llm):
    state = _base_state(
        research_plan=["Sub-Q 1"], iteration_count=4, max_iterations=4
    )
    out = planner_node(state)
    assert out == {"next_action": "synthesize"}
    assert patch_get_llm.calls == []


def test_sufficiency_short_circuits_without_llm_call(patch_get_llm, monkeypatch):
    # Need 5 relevant docs (default min_relevant_docs) all of which cover the plan keyword.
    docs = [
        GradedDocument(
            content="content with mamba keyword present in the body",
            url=f"u{i}",
            source="tavily",
            sub_question="q",
            relevance_score=0.9,
            is_grounded=True,
            rationale="r",
        )
        for i in range(5)
    ]
    state = _base_state(
        research_plan=["Mamba long-context comparison"],
        graded_documents=docs,
        iteration_count=2,
    )
    out = planner_node(state)
    assert out == {"next_action": "synthesize"}
    assert patch_get_llm.calls == []


# ---- planner_node router branch -----------------------------------------

def test_router_branch_uses_llm_when_uncovered(patch_get_llm):
    patch_get_llm.queue(
        {
            "next_action": "search",
            "rationale": "still gaps",
            "queries": ["fresh-q-1", "fresh-q-2"],
        }
    )
    state = _base_state(
        research_plan=["Discuss cosmic background radiation"],
        graded_documents=[],
        iteration_count=1,
    )
    out = planner_node(state)
    assert out["next_action"] == "search"
    assert out["search_queries"] == ["fresh-q-1", "fresh-q-2"]
    assert len(patch_get_llm.calls) == 1


def test_router_falls_back_on_bad_json(patch_get_llm):
    patch_get_llm.queue("oops not json")
    state = _base_state(
        research_plan=["Discuss cosmic background radiation"],
        graded_documents=[],
        iteration_count=1,
    )
    out = planner_node(state)
    assert out["next_action"] == "search"
    # Falls back to uncovered plan items, capped at 3.
    assert "cosmic" in " ".join(out["search_queries"]).lower()
