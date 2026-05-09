"""Grader node: LLM-as-judge over raw_documents, producing GradedDocument entries."""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agentic_rag.cache import ThreadCache, cache_dir
from agentic_rag.config import get_settings
from agentic_rag.llm import UsageCollector, get_llm
from agentic_rag.nodes.planner import _safe_json
from agentic_rag.prompts import GRADER_SYSTEM, GRADER_USER
from agentic_rag.state import GradedDocument, GraderOutput, ResearchState

logger = logging.getLogger(__name__)

_MAX_GRADER_DOC_CHARS = 4000


def _grade_one(
    llm, doc: dict, original_query: str, usage: UsageCollector | None = None
) -> GradedDocument | None:
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
    cb_config = {"callbacks": [usage]} if usage is not None else {}
    score: float
    is_grounded: bool
    rationale: str
    try:
        try:
            structured_llm = llm.with_structured_output(GraderOutput)
            output: GraderOutput = structured_llm.invoke(messages, config=cb_config)
            score = max(0.0, min(1.0, float(output.relevance_score)))
            is_grounded = bool(output.is_grounded)
            rationale = output.rationale
        except Exception as struct_err:
            logger.warning(
                "Grader structured output failed for %s (%s); falling back to JSON",
                doc.get("url"),
                struct_err,
            )
            resp = llm.invoke(messages, config=cb_config)
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


def grader_node(
    state: ResearchState, config: RunnableConfig | None = None
) -> dict[str, Any]:
    """Score each NEW raw document for relevance + groundedness.

    Skips docs that are already in `graded_documents` (URL match), so resumes
    after a partial run do not re-grade. Grades up to 8 docs in parallel via
    ThreadPoolExecutor; per-doc failures return None and are dropped.
    Returns the appended graded_documents list and accumulated token usage.
    Writes graded docs through to the per-thread Chroma cache when the
    LangGraph config carries a `thread_id`.
    """
    iteration = state.get("iteration_count", 0)
    thread_id = ((config or {}).get("configurable") or {}).get("thread_id")
    with structlog.contextvars.bound_contextvars(node="grader", iteration=iteration):
        return _grader_body(state, thread_id)


def _grader_body(state: ResearchState, thread_id: str | None) -> dict[str, Any]:
    settings = get_settings()
    llm = get_llm(settings.grader_model)
    usage = UsageCollector()  # accumulated across all per-doc calls

    graded = state.get("graded_documents") or []
    already_graded = {d.url for d in graded}
    to_grade = [d for d in (state.get("raw_documents") or []) if d.get("url") not in already_graded]

    if not to_grade:
        logger.info("Grader: nothing new to grade")
        return {}

    logger.info("Grader: grading %d new docs", len(to_grade))

    new_grades: list[GradedDocument] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = [
            ex.submit(_grade_one, llm, d, state["original_query"], usage) for d in to_grade
        ]
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

    if thread_id and new_grades:
        cache = ThreadCache(thread_id, cache_dir(settings.checkpoint_db))
        cache.add(new_grades)

    return {
        "graded_documents": graded + new_grades,
        "token_usage": usage.as_dict(),
    }
