"""Shared fixtures: fake LLM, fake search, MemorySaver."""
from __future__ import annotations

import json
from collections import deque
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver


class FakeLLM:
    """LLM stub with a programmable response queue.

    Each .invoke() pops the next response. Strings are wrapped in AIMessage.
    Dicts are JSON-serialized into AIMessage.content (so _safe_json can parse them).
    """

    def __init__(self, responses: list[Any] | None = None) -> None:
        self._queue: deque = deque(responses or [])
        self.calls: list[list] = []  # records the messages each invoke received

    def queue(self, *responses: Any) -> FakeLLM:
        for r in responses:
            self._queue.append(r)
        return self

    def invoke(self, messages, **_: Any) -> AIMessage:
        self.calls.append(messages)
        if not self._queue:
            return AIMessage(content="{}")
        nxt = self._queue.popleft()
        if isinstance(nxt, dict):
            return AIMessage(content=json.dumps(nxt))
        if isinstance(nxt, AIMessage):
            return nxt
        return AIMessage(content=str(nxt))


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def patch_get_llm(monkeypatch, fake_llm: FakeLLM):
    """Replace get_llm everywhere it has been bound."""

    def _factory(*_a, **_kw):
        return fake_llm

    import agentic_rag.nodes.grader as grader_mod
    import agentic_rag.nodes.planner as planner_mod
    import agentic_rag.nodes.synthesizer as synth_mod

    monkeypatch.setattr(planner_mod, "get_llm", _factory)
    monkeypatch.setattr(grader_mod, "get_llm", _factory)
    monkeypatch.setattr(synth_mod, "get_llm", _factory)
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
