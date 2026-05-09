"""Tests for HITL interrupt_before=['synthesizer']."""
from __future__ import annotations

from agentic_rag.graph import build_graph


def _initial_state(max_iter: int = 2) -> dict:
    return {
        "original_query": "test",
        "research_plan": [],
        "search_queries": [],
        "raw_documents": [],
        "graded_documents": [],
        "iteration_count": 0,
        "max_iterations": max_iter,
        "needs_more_research": True,
        "final_report": "",
        "next_action": "search",
        "messages": [],
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
    }


def test_review_pauses_before_synthesizer(patch_get_llm, patch_web_search, mem_saver):
    # Sequence: planner first-visit -> grade x2 -> router synthesize -> (PAUSE) -> synth.
    patch_get_llm.queue(
        {"plan": ["Sub-Q 1"], "initial_queries": ["q1"]},
        {
            "grades": [
                {"index": 0, "relevance_score": 0.9, "is_grounded": True, "rationale": "r"},
                {"index": 1, "relevance_score": 0.9, "is_grounded": True, "rationale": "r"},
            ]
        },
        {"next_action": "synthesize", "rationale": "enough", "queries": []},
        "# Report\n\nDone.",
    )

    graph = build_graph(checkpointer=mem_saver, interrupt_before=["synthesizer"])
    config = {"configurable": {"thread_id": "t-hitl"}}

    # First stream call should pause before synthesizer.
    events = list(graph.stream(_initial_state(max_iter=2), config=config, stream_mode="updates"))
    # No final_report yet because we paused.
    state = graph.get_state(config)
    assert state.values.get("final_report", "") == ""
    assert "synthesizer" in (state.next or ())
    assert events  # at least one event before pause

    # Resume with None -> synth runs.
    list(graph.stream(None, config=config, stream_mode="updates"))
    final = graph.get_state(config)
    assert "Done" in final.values["final_report"]


def test_no_review_runs_to_completion(patch_get_llm, patch_web_search, mem_saver):
    """interrupt_before defaults to None -> no pause."""
    patch_get_llm.queue(
        {"plan": ["S"], "initial_queries": ["q"]},
        {
            "grades": [
                {"index": 0, "relevance_score": 0.9, "is_grounded": True, "rationale": "r"},
                {"index": 1, "relevance_score": 0.9, "is_grounded": True, "rationale": "r"},
            ]
        },
        {"next_action": "synthesize", "rationale": "ok", "queries": []},
        "# Report\n\nFinished.",
    )

    graph = build_graph(checkpointer=mem_saver)
    config = {"configurable": {"thread_id": "t-no-review"}}
    final_state = graph.invoke(_initial_state(max_iter=2), config=config)
    assert "Finished" in final_state["final_report"]
