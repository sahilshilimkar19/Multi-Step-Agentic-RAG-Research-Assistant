"""Tests for synthesizer_node."""
from __future__ import annotations

from agentic_rag.nodes.synthesizer import synthesizer_node
from agentic_rag.state import GradedDocument


def _state(**overrides):
    s = {
        "original_query": "How does Mamba compare to Transformers?",
        "research_plan": ["Sub-Q 1", "Sub-Q 2"],
        "search_queries": [],
        "raw_documents": [],
        "graded_documents": [],
        "iteration_count": 2,
        "max_iterations": 4,
        "needs_more_research": False,
        "final_report": "",
        "next_action": "synthesize",
        "messages": [],
    }
    s.update(overrides)
    return s


def test_synthesizer_assembles_prompt_with_relevant_docs(patch_get_llm):
    docs = [
        GradedDocument(
            content="Mamba scales linearly.", url="https://m", source="tavily",
            sub_question="q", relevance_score=0.9, is_grounded=True, rationale="r",
        ),
        # Filtered out: low relevance
        GradedDocument(
            content="irrelevant", url="https://i", source="tavily",
            sub_question="q", relevance_score=0.1, is_grounded=True, rationale="r",
        ),
        # Filtered out: not grounded
        GradedDocument(
            content="speculation", url="https://s", source="tavily",
            sub_question="q", relevance_score=0.9, is_grounded=False, rationale="r",
        ),
    ]
    patch_get_llm.queue("# Title\n\n## Executive Summary\nDone [1].")
    out = synthesizer_node(_state(graded_documents=docs))
    assert out["next_action"] == "end"
    assert "Done" in out["final_report"]
    # The prompt sent to the LLM must contain only the one passing doc.
    last_messages = patch_get_llm.calls[-1]
    user_content = last_messages[-1].content
    assert "https://m" in user_content
    assert "https://i" not in user_content
    assert "https://s" not in user_content


def test_synthesizer_handles_zero_relevant_docs(patch_get_llm):
    patch_get_llm.queue("# Empty\n\n## Limitations & Open Questions\nNo evidence.")
    out = synthesizer_node(_state(graded_documents=[]))
    user_content = patch_get_llm.calls[-1][-1].content
    assert "No relevant documents collected." in user_content
    assert out["final_report"]


def test_synthesizer_truncates_per_doc(patch_get_llm):
    long_content = "x" * 5000
    docs = [
        GradedDocument(
            content=long_content, url="https://m", source="tavily",
            sub_question="q", relevance_score=0.9, is_grounded=True, rationale="r",
        )
    ]
    patch_get_llm.queue("ok")
    synthesizer_node(_state(graded_documents=docs))
    user_content = patch_get_llm.calls[-1][-1].content
    # Truncation cap is 2000.
    assert "x" * 2001 not in user_content
    assert "x" * 2000 in user_content
