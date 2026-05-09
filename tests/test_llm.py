"""Tests for the LLM retry wrapper."""
from __future__ import annotations

import pytest

from agentic_rag import llm as llm_mod
from agentic_rag.llm import _RetryingChat, _RetryingRunnable


class _FakeTransient(Exception):
    """Stand-in transient error injected via monkeypatch on _TRANSIENT."""


class _FakeNonTransient(Exception):
    pass


@pytest.fixture
def patch_transient(monkeypatch):
    """Replace _TRANSIENT with a tuple containing only our test exception."""
    monkeypatch.setattr(llm_mod, "_TRANSIENT", (_FakeTransient,))


def test_retries_on_transient_then_succeeds(patch_transient):
    calls = {"n": 0}

    class _Inner:
        def invoke(self, *_a, **_kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _FakeTransient("rate limited")
            return "ok"

    chat = _RetryingRunnable(_Inner())
    assert chat.invoke([]) == "ok"
    assert calls["n"] == 3


def test_does_not_retry_non_transient(patch_transient):
    calls = {"n": 0}

    class _Inner:
        def invoke(self, *_a, **_kw):
            calls["n"] += 1
            raise _FakeNonTransient("auth failure")

    chat = _RetryingRunnable(_Inner())
    with pytest.raises(_FakeNonTransient):
        chat.invoke([])
    assert calls["n"] == 1


def test_gives_up_after_three_attempts(patch_transient):
    calls = {"n": 0}

    class _Inner:
        def invoke(self, *_a, **_kw):
            calls["n"] += 1
            raise _FakeTransient("always failing")

    chat = _RetryingRunnable(_Inner())
    with pytest.raises(_FakeTransient):
        chat.invoke([])
    assert calls["n"] == 3


def test_with_structured_output_wraps_inner_result(patch_transient):
    """_RetryingChat.with_structured_output(...) returns a _RetryingRunnable."""
    structured_calls = {"n": 0}

    class _StructuredInner:
        def invoke(self, *_a, **_kw):
            structured_calls["n"] += 1
            if structured_calls["n"] < 2:
                raise _FakeTransient("hiccup")
            return {"ok": True}

    structured = _StructuredInner()

    class _Inner:
        def with_structured_output(self, _schema):
            return structured

        def invoke(self, *_a, **_kw):
            return "raw"

    chat = _RetryingChat(_Inner())
    runnable = chat.with_structured_output(object)
    assert isinstance(runnable, _RetryingRunnable)
    assert runnable.invoke([]) == {"ok": True}
    assert structured_calls["n"] == 2


def test_passthrough_attribute_access(patch_transient):
    class _Inner:
        special = "marker"

        def invoke(self, *_a, **_kw):
            return None

    chat = _RetryingRunnable(_Inner())
    assert chat.special == "marker"
