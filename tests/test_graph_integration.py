"""End-to-end integration test: full loop with stubbed LLM + canned search."""
from __future__ import annotations

from agentic_rag.graph import build_graph


def _initial_state(query: str = "test query", max_iterations: int = 2) -> dict:
    return {
        "original_query": query,
        "research_plan": [],
        "search_queries": [],
        "raw_documents": [],
        "graded_documents": [],
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "needs_more_research": True,
        "final_report": "",
        "next_action": "search",
        "messages": [],
    }


def test_full_loop_terminates_within_cap(patch_get_llm, patch_web_search, mem_saver):
    # Sequence of LLM responses for the run:
    #   1) planner first-visit  -> plan + initial_queries
    #   2) two grader calls (one per fake doc, fan-out)
    #   3) planner router -> synthesize (we want to terminate after one loop)
    #   4) synthesizer -> markdown report
    patch_get_llm.queue(
        {"plan": ["Sub-Q 1"], "initial_queries": ["q1"]},
        {"relevance_score": 0.9, "is_grounded": True, "rationale": "r"},
        {"relevance_score": 0.9, "is_grounded": True, "rationale": "r"},
        {"next_action": "synthesize", "rationale": "enough", "queries": []},
        "# Report\n\n## Executive Summary\nDone.",
    )

    graph = build_graph(checkpointer=mem_saver)
    config = {"configurable": {"thread_id": "t-int-1"}}
    final_state = graph.invoke(_initial_state(max_iterations=2), config=config)

    assert final_state["final_report"]
    assert "Done" in final_state["final_report"]
    assert final_state["iteration_count"] <= 2


def test_loop_respects_hard_cap_when_router_keeps_searching(
    patch_get_llm, patch_web_search, mem_saver
):
    # Router keeps saying 'search' -- the planner must still terminate at iteration_count >= max_iterations.
    # max_iterations=2 -> two search rounds, then forced synthesize.
    queue = [
        {"plan": ["Sub-Q 1"], "initial_queries": ["q1"]},  # planner first visit
        # iter 1 grading (2 fake docs)
        {"relevance_score": 0.5, "is_grounded": True, "rationale": "r"},
        {"relevance_score": 0.5, "is_grounded": True, "rationale": "r"},
        # router round 1 -> search again
        {"next_action": "search", "rationale": "more", "queries": ["q2"]},
        # iter 2 grading: only NEW docs would be graded, but our fake search returns same URLs
        # so nothing new -> no grader LLM calls. Then planner short-circuits at iteration_count >= max_iterations.
        # synthesizer
        "# Forced\n\n## Executive Summary\nCapped.",
    ]
    patch_get_llm.queue(*queue)

    graph = build_graph(checkpointer=mem_saver)
    config = {"configurable": {"thread_id": "t-int-2"}}
    final_state = graph.invoke(_initial_state(max_iterations=2), config=config)

    assert final_state["iteration_count"] == 2
    assert "Capped" in final_state["final_report"]
