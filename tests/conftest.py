"""Shared fixtures: fake LLM, fake search, MemorySaver."""
from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from pydantic import BaseModel


class _StructuredRunnable:
    """Stub returned by FakeLLM.with_structured_output(schema)."""

    def __init__(self, parent: FakeLLM, schema: type[BaseModel]) -> None:
        self._parent = parent
        self._schema = schema

    def invoke(self, messages, **_: Any):  # noqa: ANN001
        self._parent.calls.append(messages)
        if not self._parent._queue:
            # Cannot construct an arbitrary schema without data; raise so the
            # caller falls back to the unstructured path.
            raise ValueError(f"FakeLLM has no queued response for {self._schema.__name__}")
        nxt = self._parent._queue.popleft()
        if isinstance(nxt, BaseException):
            raise nxt
        if isinstance(nxt, self._schema):
            return nxt
        if isinstance(nxt, dict):
            return self._schema(**nxt)
        # Strings/other -> trigger fallback path in the node under test.
        raise ValueError(
            f"FakeLLM cannot coerce {type(nxt).__name__} to {self._schema.__name__}"
        )


class FakeLLM:
    """LLM stub with a programmable response queue.

    Each .invoke() pops the next response. Strings -> AIMessage(text).
    Dicts -> AIMessage(json) so _safe_json can parse them.
    BaseException instances are raised to simulate provider errors.

    `.with_structured_output(schema)` returns a runnable whose .invoke()
    pops the queue and returns a Pydantic instance (constructed from a dict
    or passed through if already an instance). Strings/non-schema values
    raise so node code can exercise its fallback path.
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._queue: deque = deque(responses or [])
        self.calls: list[list] = []

    def queue(self, *responses: Any) -> FakeLLM:
        for r in responses:
            self._queue.append(r)
        return self

    def invoke(self, messages, **_: Any) -> AIMessage:
        self.calls.append(messages)
        if not self._queue:
            return AIMessage(content="{}")
        nxt = self._queue.popleft()
        if isinstance(nxt, BaseException):
            raise nxt
        if isinstance(nxt, dict):
            return AIMessage(content=json.dumps(nxt))
        if isinstance(nxt, AIMessage):
            return nxt
        return AIMessage(content=str(nxt))

    def with_structured_output(self, schema: type[BaseModel]) -> _StructuredRunnable:
        return _StructuredRunnable(self, schema)


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def patch_get_llm(monkeypatch, fake_llm: FakeLLM):
    """Replace get_llm everywhere it has been bound."""

    def _factory(*_a, **_kw):
        return fake_llm

    import agentic_rag.chat as chat_mod
    import agentic_rag.nodes.grader as grader_mod
    import agentic_rag.nodes.planner as planner_mod
    import agentic_rag.nodes.synthesizer as synth_mod

    monkeypatch.setattr(planner_mod, "get_llm", _factory)
    monkeypatch.setattr(grader_mod, "get_llm", _factory)
    monkeypatch.setattr(synth_mod, "get_llm", _factory)
    monkeypatch.setattr(chat_mod, "get_llm", _factory)
    return fake_llm


@pytest.fixture
def fake_search_results() -> list[dict]:
    return [
        {
            "title": "Mamba paper",
            "url": "https://example.com/mamba",
            "content": "Mamba is a state-space model with linear scaling on long-context tasks. Transformers use quadratic attention.",
            "source": "tavily",
        },
        {
            "title": "Transformer comparison",
            "url": "https://example.com/transformer",
            "content": "Transformers dominate language tasks but suffer on very long sequences due to attention cost.",
            "source": "tavily",
        },
    ]


@pytest.fixture
def patch_web_search(monkeypatch, fake_search_results: list[dict]):
    """Make searcher_node return canned results regardless of query."""
    import agentic_rag.nodes.searcher as searcher_mod

    class _FakeTool:
        @staticmethod
        def invoke(payload, **_):
            return list(fake_search_results)

    monkeypatch.setattr(searcher_mod, "web_search", _FakeTool)
    return fake_search_results


@pytest.fixture
def mem_saver() -> MemorySaver:
    return MemorySaver()
