"""Tests for the follow-up chat command."""
from __future__ import annotations

from agentic_rag import chat as chat_mod
from agentic_rag.cache import ThreadCache
from agentic_rag.state import GradedDocument


class _FakeCache(ThreadCache):
    """Bypasses chromadb -- returns a canned doc list from .search."""

    def __init__(self, docs: list[GradedDocument]) -> None:
        self.thread_id = "fake"
        self._collection = object()  # any non-None marks .available True
        self._docs = docs

    def search(self, query: str, k: int = 30) -> list[GradedDocument]:  # type: ignore[override]
        return self._docs[:k]


def _gd(content: str, url: str) -> GradedDocument:
    return GradedDocument(
        content=content,
        url=url,
        source="tavily",
        sub_question="q",
        relevance_score=0.9,
        is_grounded=True,
        rationale="r",
    )


def test_chat_uses_cache_only(patch_get_llm):
    """Chat answers from cache only -- the chat module never imports web_search."""
    import agentic_rag.chat as chat_module
    import agentic_rag.tools.search as search_mod

    # Smoke check: chat.py must not depend on the web_search tool, by design.
    assert "web_search" not in dir(chat_module)
    # And the search module is fine to ignore here -- we just want to assert imports.
    assert search_mod is not None  # noqa: F841

    cache = _FakeCache([
        _gd("Mamba scales linearly on long sequences.", "https://m"),
    ])
    patch_get_llm.queue("Mamba is linear-time on long context [1].")

    text, docs = chat_mod.answer(
        thread_id="t", original_query="Mamba vs Transformers", question="How does Mamba scale?",
        cache=cache,
    )
    assert "[1]" in text
    assert len(docs) == 1
    assert docs[0].url == "https://m"


def test_chat_explicit_when_no_evidence(patch_get_llm):
    cache = _FakeCache([])
    patch_get_llm.queue("I do not have evidence to answer this.")
    text, docs = chat_mod.answer(
        thread_id="t", original_query="X", question="What is X?", cache=cache,
    )
    assert "evidence" in text.lower()
    assert docs == []
    # The LLM still gets called once, with the placeholder text in the prompt.
    last_user = patch_get_llm.calls[-1][-1].content
    assert "No cached evidence found." in last_user
