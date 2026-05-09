"""Tests for ThreadCache.

Uses Chroma's in-memory ephemeral client when available (filesystem
isolation via tmp_path). If chromadb is not installed, the no-op path
is exercised instead.
"""
from __future__ import annotations

import pytest

from agentic_rag.cache import ThreadCache, cache_dir
from agentic_rag.state import GradedDocument


def _gd(content: str, url: str, score: float = 0.9) -> GradedDocument:
    return GradedDocument(
        content=content,
        url=url,
        source="tavily",
        sub_question="q",
        relevance_score=score,
        is_grounded=True,
        rationale="r",
    )


def test_cache_dir_derives_under_checkpoint_parent():
    assert cache_dir("./checkpoints.sqlite").as_posix() == "runs/chroma"
    assert cache_dir("/var/data/checkpoints.sqlite").as_posix() == "/var/data/runs/chroma"


def test_cache_roundtrip(tmp_path):
    pytest.importorskip("chromadb")
    cache = ThreadCache("t-roundtrip", tmp_path)
    assert cache.available
    cache.add([
        _gd("Mamba scales linearly on long sequences.", "https://m"),
        _gd("Cooking pasta requires boiling water.", "https://p"),
    ])
    hits = cache.search("state space models long context", k=2)
    assert len(hits) >= 1
    urls = {h.url for h in hits}
    # The pasta doc should rank lower than the mamba doc; the top-1 must be mamba.
    assert hits[0].url == "https://m", urls


def test_cache_handles_no_query(tmp_path):
    pytest.importorskip("chromadb")
    cache = ThreadCache("t-empty-query", tmp_path)
    assert cache.search("", k=5) == []


def test_cache_no_op_when_unavailable(monkeypatch):
    """If the constructor fails, add/search should silently no-op."""
    # Force the constructor to fail by pointing at a nonsensical path mode.
    cache = ThreadCache.__new__(ThreadCache)
    cache.thread_id = "x"
    cache._collection = None
    cache.add([_gd("c", "u")])
    assert cache.search("q") == []
    assert cache.available is False
