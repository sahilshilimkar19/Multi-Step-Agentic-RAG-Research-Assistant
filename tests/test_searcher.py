"""Tests for searcher_node."""
from __future__ import annotations

from agentic_rag.nodes.searcher import searcher_node


def _state(**overrides):
    s = {
        "original_query": "q",
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
    s.update(overrides)
    return s


def test_empty_queries_still_increments_iteration():
    out = searcher_node(_state(search_queries=[], iteration_count=0))
    assert out == {"iteration_count": 1}


def test_aggregates_and_increments(patch_web_search):
    out = searcher_node(_state(search_queries=["q1", "q2"], iteration_count=1))
    # Two queries x 2 fake results = 4, but URLs collide -> dedupe to 2.
    assert len(out["raw_documents"]) == 2
    assert out["iteration_count"] == 2
    for doc in out["raw_documents"]:
        assert doc["sub_question"] in {"q1", "q2"}


def test_dedupes_against_existing_raw_documents(patch_web_search, fake_search_results):
    existing = [{"url": fake_search_results[0]["url"]}]
    out = searcher_node(_state(search_queries=["q"], raw_documents=existing))
    new_urls = {d["url"] for d in out["raw_documents"]}
    # The pre-existing doc stays, only the second fake result is added fresh.
    assert fake_search_results[0]["url"] in new_urls
    assert fake_search_results[1]["url"] in new_urls
    assert len(out["raw_documents"]) == 2


def test_one_failed_query_does_not_kill_iteration(monkeypatch):
    import agentic_rag.nodes.searcher as searcher_mod

    class _FlakyTool:
        calls = 0

        @classmethod
        def invoke(cls, payload, **_):
            cls.calls += 1
            if payload["query"] == "boom":
                raise RuntimeError("network down")
            return [
                {
                    "title": "ok",
                    "url": f"https://ok/{payload['query']}",
                    "content": "ok",
                    "source": "tavily",
                }
            ]

    monkeypatch.setattr(searcher_mod, "web_search", _FlakyTool)
    out = searcher_node(_state(search_queries=["boom", "good"]))
    assert len(out["raw_documents"]) == 1
    assert out["iteration_count"] == 1
