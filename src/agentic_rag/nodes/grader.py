"""Grader node: LLM-as-judge over raw_documents, producing GradedDocument entries."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.config import get_settings
from agentic_rag.llm import get_llm
from agentic_rag.nodes.planner import _safe_json
from agentic_rag.prompts import GRADER_SYSTEM, GRADER_USER
from agentic_rag.state import GradedDocument, GraderOutput, ResearchState

logger = logging.getLogger(__name__)

_MAX_GRADER_DOC_CHARS = 4000


def _grade_one(llm, doc: dict, original_query: str) -> GradedDocument | None:
    messages = [
        SystemMessage(content=GRADER_SYSTEM),
        HumanMessage(
            content=GRADER_USER.format(
                query=original_query,
                sub_question=doc.get("sub_question", original_query),
                url=doc.get("url", ""),
                content=(doc.get("content", "") or "")[:_MAX_GRADER_DOC_CHARS],
            )
        ),
    ]
    score: float
    is_grounded: bool
    rationale: str
    try:
        try:
            structured_llm = llm.with_structured_output(GraderOutput)
            output: GraderOutput = structured_llm.invoke(messages)
            score = max(0.0, min(1.0, float(output.relevance_score)))
            is_grounded = bool(output.is_grounded)
            rationale = output.rationale
        except Exception as struct_err:
            logger.warning(
                "Grader structured output failed for %s (%s); falling back to JSON",
                doc.get("url"),
                struct_err,
            )
            resp = llm.invoke(messages)
            text = resp.content if isinstance(resp.content, str) else str(resp.content)
            data = _safe_json(
                text,
                default={
                    "relevance_score": 0.0,
                    "is_grounded": False,
                    "rationale": "grader parse failed",
                },
            )
            score = max(0.0, min(1.0, float(data.get("relevance_score", 0.0))))
            is_grounded = bool(data.get("is_grounded", False))
            rationale = data.get("rationale", "")
        return GradedDocument(
            content=doc.get("content", "") or "",
            url=doc.get("url", "") or "",
            source=doc.get("source", "unknown"),
            sub_question=doc.get("sub_question", original_query),
            relevance_score=score,
            is_grounded=is_grounded,
            rationale=rationale,
        )
    except Exception as e:
        logger.error("Grading failed for %s: %s", doc.get("url"), e)
        return None


def grader_node(state: ResearchState) -> dict[str, Any]:
    settings = get_settings()
    llm = get_llm(settings.grader_model)

    graded = state.get("graded_documents") or []
    already_graded = {d.url for d in graded}
    to_grade = [d for d in (state.get("raw_documents") or []) if d.get("url") not in already_graded]

    if not to_grade:
        logger.info("Grader: nothing new to grade")
        return {}

    logger.info("Grader: grading %d new docs", len(to_grade))

    new_grades: list[GradedDocument] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [ex.submit(_grade_one, llm, d, state["original_query"]) for d in to_grade]
        for fut in as_completed(futures):
            res = fut.result()
            if res is not None:
                new_grades.append(res)

    relevant = [
        g
        for g in new_grades
        if g.relevance_score >= settings.relevance_threshold and g.is_grounded
    ]
    logger.info("Grader: %d/%d passed threshold", len(relevant), len(new_grades))

    return {"graded_documents": graded + new_grades}
