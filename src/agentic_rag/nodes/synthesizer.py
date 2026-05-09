"""Synthesizer node: produces the final markdown report from graded evidence."""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic_rag.config import get_settings
from agentic_rag.nodes.planner import get_llm
from agentic_rag.prompts import SYNTHESIZER_SYSTEM, SYNTHESIZER_USER
from agentic_rag.state import ResearchState

logger = logging.getLogger(__name__)

_MAX_DOCS_IN_CONTEXT = 30
_MAX_CHARS_PER_DOC = 2000


def synthesizer_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    llm = get_llm(settings.synthesizer_model, temperature=0.2)

    relevant = [
        d
        for d in (state.get("graded_documents") or [])
        if d.relevance_score >= settings.relevance_threshold and d.is_grounded
    ]
    relevant.sort(key=lambda d: d.relevance_score, reverse=True)
    relevant = relevant[:_MAX_DOCS_IN_CONTEXT]

    docs_text = "\n\n".join(
        f"[{i + 1}] URL: {d.url}\n"
        f"Source: {d.source} | Relevance: {d.relevance_score:.2f}\n"
        f"Content: {d.content[:_MAX_CHARS_PER_DOC]}"
        for i, d in enumerate(relevant)
    )

    plan = state.get("research_plan") or [state["original_query"]]
    logger.info("Synthesizer: writing report from %d docs", len(relevant))
    resp = llm.invoke(
        [
            SystemMessage(content=SYNTHESIZER_SYSTEM),
            HumanMessage(
                content=SYNTHESIZER_USER.format(
                    query=state["original_query"],
                    plan="\n".join(f"- {p}" for p in plan),
                    documents=docs_text or "No relevant documents collected.",
                )
            ),
        ]
    )
    report = resp.content if isinstance(resp.content, str) else str(resp.content)
    return {
        "final_report": report,
        "next_action": "end",
        "messages": [AIMessage(content="Report drafted.")],
    }
