"""End-to-end test for the FastAPI server (PR-1 thin slice).

This is also the canonical regression test for the asyncio-run-inside-thread
risk: the searcher and grader nodes call `asyncio.run(...)` internally, and
this test exercises them through `graph.astream` (LangGraph's async path) to
prove that path works without a "cannot be called from a running event loop"
error.
"""

import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from langgraph.checkpoint.memory import MemorySaver

from agentic_rag.api import runs as runs_module


@pytest.fixture
def mem_server_saver(monkeypatch):
    """Replace the AsyncSqliteSaver factory with one that yields an in-memory saver."""
    saver = MemorySaver()

    @asynccontextmanager
    async def _factory():
        yield saver

    monkeypatch.setattr(runs_module, "saver_factory", _factory)
    runs_module._clear_pending_for_test()
    return saver


@pytest.fixture
def app(mem_server_saver, patch_get_llm, patch_web_search):
    # Programmed responses for one full loop:
    #   1) planner first-visit -> plan + initial_queries
    #   2) batched grader covering both fake search docs
    #   3) router -> synthesize
    #   4) synthesizer -> markdown report
    patch_get_llm.queue(
        {"plan": ["Sub-Q 1"], "initial_queries": ["q1"]},
        {
            "grades": [
                {"index": 0, "relevance_score": 0.9, "is_grounded": True, "rationale": "r"},
                {"index": 1, "relevance_score": 0.9, "is_grounded": True, "rationale": "r"},
            ]
        },
        {"next_action": "synthesize", "rationale": "enough", "queries": []},
        "# Report\n\n## Executive Summary\nDone.",
    )
    from agentic_rag.server import app as fastapi_app

    return fastapi_app


@pytest.mark.asyncio
async def test_full_run_streams_and_completes(app):
    """POST /api/runs then GET /stream -> at least one node event then 'end'."""
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/runs", json={"query": "Test query", "max_iterations": 2})
        assert r.status_code == 201, r.text
        thread_id = r.json()["thread_id"]
        assert thread_id

        node_events: list[dict] = []
        end_seen = False
        async with client.stream("GET", f"/api/runs/{thread_id}/stream") as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers["content-type"]
            current_event = None
            async for raw in resp.aiter_lines():
                if raw.startswith("event:"):
                    current_event = raw.split(":", 1)[1].strip()
                elif raw.startswith("data:") and current_event:
                    payload = raw.split(":", 1)[1].strip()
                    if current_event == "node":
                        node_events.append(json.loads(payload))
                    elif current_event == "end":
                        end_seen = True
                        break
                    elif current_event == "error":
                        pytest.fail(f"stream emitted error: {payload}")

        assert node_events, "expected at least one node event"
        assert end_seen, "expected an 'end' event"
        node_names = {e["node"] for e in node_events}
        assert "planner" in node_names
        assert "synthesizer" in node_names


@pytest.mark.asyncio
async def test_stream_unknown_thread_emits_error(app):
    transport = httpx.ASGITransport(app=app)
    async with (
        httpx.AsyncClient(transport=transport, base_url="http://test") as client,
        client.stream("GET", "/api/runs/does-not-exist/stream") as resp,
    ):
        assert resp.status_code == 200
        current_event = None
        saw_error = False
        async for raw in resp.aiter_lines():
            if raw.startswith("event:"):
                current_event = raw.split(":", 1)[1].strip()
            elif raw.startswith("data:") and current_event == "error":
                saw_error = True
                break
        assert saw_error


@pytest.mark.asyncio
async def test_health(app):
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/api/health")
        assert r.status_code == 200
        assert r.json() == {"status": "ok"}


def test_event_loop_not_required():
    """Sanity check that pytest-asyncio is wired."""
    assert asyncio.get_event_loop_policy() is not None
