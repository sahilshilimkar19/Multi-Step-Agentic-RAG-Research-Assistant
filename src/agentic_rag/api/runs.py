"""Routes for starting and streaming research runs.

PR-1 scope: only POST /api/runs and GET /api/runs/{thread_id}/stream.
Seeded initial states live in an in-memory dict keyed by thread_id; this is
replaced by a SQLite `run_meta` sidecar in PR-5 when HITL/PDFs land.
"""

import json
import logging
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import APIRouter
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agentic_rag.api._serialize import serialize_update
from agentic_rag.config import get_settings
from agentic_rag.graph import build_graph

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["runs"])

# PR-1: in-memory seeded states. Replaced by SQLite sidecar in PR-5.
_pending_inputs: dict[str, dict[str, Any]] = {}


class StartRunRequest(BaseModel):
    query: str
    max_iterations: int | None = Field(default=None, ge=1, le=20)


class StartRunResponse(BaseModel):
    thread_id: str


@asynccontextmanager
async def _default_saver() -> AsyncIterator[Any]:
    """Yield an AsyncSqliteSaver bound to the configured DB path.

    Tests monkeypatch the module-level `saver_factory` to inject a
    MemorySaver-backed context manager instead.
    """
    settings = get_settings()
    async with AsyncSqliteSaver.from_conn_string(settings.checkpoint_db) as cp:
        yield cp


saver_factory = _default_saver


def _build_initial_state(query: str, max_iterations: int) -> dict[str, Any]:
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
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
    }


@router.post("/runs", response_model=StartRunResponse, status_code=201)
async def start_run(req: StartRunRequest) -> StartRunResponse:
    """Allocate a thread_id and stash the seeded state for /stream to pick up."""
    settings = get_settings()
    thread_id = str(uuid.uuid4())
    max_iter = req.max_iterations or settings.max_iterations
    _pending_inputs[thread_id] = _build_initial_state(req.query, max_iter)
    logger.info("Started run thread_id=%s query=%r max_iter=%d", thread_id, req.query, max_iter)
    return StartRunResponse(thread_id=thread_id)


@router.get("/runs/{thread_id}/stream")
async def stream_run(thread_id: str) -> EventSourceResponse:
    """Stream node updates as Server-Sent Events.

    On first connect for a thread, consumes the pending seeded input.
    On reconnect (checkpoint exists), passes None so LangGraph resumes.
    """

    async def event_gen() -> AsyncIterator[dict[str, str]]:
        async with saver_factory() as cp:
            graph = build_graph(checkpointer=cp)
            config = {"configurable": {"thread_id": thread_id}}
            existing = await graph.aget_state(config)
            if existing.values:
                input_: Any = None
            else:
                input_ = _pending_inputs.pop(thread_id, None)
                if input_ is None:
                    yield {
                        "event": "error",
                        "data": json.dumps({"message": f"No pending run for thread {thread_id}"}),
                    }
                    return

            try:
                async for event in graph.astream(input_, config=config, stream_mode="updates"):
                    for node_name, update in event.items():
                        yield {
                            "event": "node",
                            "data": json.dumps(
                                {"node": node_name, "update": serialize_update(update)}
                            ),
                        }
            except Exception as e:
                logger.exception("stream_run failed for thread_id=%s", thread_id)
                yield {"event": "error", "data": json.dumps({"message": str(e)})}
                return

            state = await graph.aget_state(config)
            if state.next:
                yield {
                    "event": "paused",
                    "data": json.dumps({"reason": "review", "next": list(state.next)}),
                }
            else:
                yield {"event": "end", "data": "{}"}

    return EventSourceResponse(event_gen())


# Convenience for tests that need to query whether a thread is still pending.
def _pending_for_test(thread_id: str) -> dict[str, Any] | None:
    return _pending_inputs.get(thread_id)


def _clear_pending_for_test() -> None:
    _pending_inputs.clear()


def _get_thread_id_for_test_lookup() -> dict[str, dict[str, Any]]:
    """Tests inspect/seed _pending_inputs through this getter to avoid touching internals."""
    return _pending_inputs
