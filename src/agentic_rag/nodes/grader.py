"""Grader node: LLM-as-judge over raw_documents, producing GradedDocument entries."""

import asyncio
import logging
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agentic_rag.cache import ThreadCache, cache_dir
from agentic_rag.config import get_settings
from agentic_rag.llm import UsageCollector, get_llm
from agentic_rag.nodes.planner import _safe_json
from agentic_rag.prompts import (
    GRADER_BATCH_SYSTEM,
    GRADER_BATCH_USER,
    GRADER_SYSTEM,
    GRADER_USER,
)
from agentic_rag.state import (
    GradedDocument,
    GraderBatchOutput,
    GraderOutput,
    ResearchState,
)

logger = logging.getLogger(__name__)

_MAX_GRADER_DOC_CHARS = 4000
_BATCH_SIZE = 5
_GRADER_CONCURRENCY = 8


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


def _grade_batch(
    llm, docs: list[dict], original_query: str, usage: UsageCollector | None = None
) -> list[GradedDocument] | None:
    """Grade up to _BATCH_SIZE docs in one LLM call. Returns None on failure (caller falls back)."""
    if not docs:
        return []
    numbered = "\n\n".join(
        f"--- Document {i} ---\n"
        f"Sub-question: {d.get('sub_question', original_query)}\n"
        f"URL: {d.get('url', '')}\n"
        f"Content: {(d.get('content', '') or '')[:_MAX_GRADER_DOC_CHARS]}"
        for i, d in enumerate(docs)
    )
    messages = [
        SystemMessage(content=GRADER_BATCH_SYSTEM),
        HumanMessage(content=GRADER_BATCH_USER.format(query=original_query, documents=numbered)),
    ]
    cb_config = {"callbacks": [usage]} if usage is not None else {}
    try:
        structured_llm = llm.with_structured_output(GraderBatchOutput)
        output: GraderBatchOutput = structured_llm.invoke(messages, config=cb_config)
    except Exception as e:
        logger.warning("Batched grader failed (%s); falling back to per-doc", e)
        return None

    by_index = {item.index: item for item in output.grades}
    if len(by_index) != len(docs):
        logger.warning(
            "Batched grader returned %d items for %d docs; falling back",
            len(by_index),
            len(docs),
        )
        return None

    out: list[GradedDocument] = []
    for i, d in enumerate(docs):
        item = by_index.get(i)
        if item is None:
            return None
        score = max(0.0, min(1.0, float(item.relevance_score)))
        out.append(
            GradedDocument(
                content=d.get("content", "") or "",
                url=d.get("url", "") or "",
                source=d.get("source", "unknown"),
                sub_question=d.get("sub_question", original_query),
                relevance_score=score,
                is_grounded=bool(item.is_grounded),
                rationale=item.rationale,
            )
        )
    return out


async def _gather_batches(
    llm, docs: list[dict], original_query: str, usage: UsageCollector
) -> tuple[list[GradedDocument], list[dict]]:
    """Run all batched grader calls concurrently. Returns (graded, fallback_chunks)."""
    sem = asyncio.Semaphore(_GRADER_CONCURRENCY)
    chunks = [docs[i : i + _BATCH_SIZE] for i in range(0, len(docs), _BATCH_SIZE)]

    async def _one(chunk: list[dict]) -> tuple[list[GradedDocument] | None, list[dict]]:
        async with sem:
            result = await asyncio.to_thread(_grade_batch, llm, chunk, original_query, usage)
        return result, chunk

    results = await asyncio.gather(*(_one(c) for c in chunks))
    graded: list[GradedDocument] = []
    fallback: list[dict] = []
    for batch_result, chunk in results:
        if batch_result is None:
            fallback.extend(chunk)
        else:
            graded.extend(batch_result)
    return graded, fallback


async def _gather_per_doc(
    llm, docs: list[dict], original_query: str, usage: UsageCollector
) -> list[GradedDocument]:
    """Run per-doc grading concurrently for any docs that fell out of the batched path."""
    sem = asyncio.Semaphore(_GRADER_CONCURRENCY)

    async def _one(d: dict) -> GradedDocument | None:
        async with sem:
            return await asyncio.to_thread(_grade_one, llm, d, original_query, usage)

    results = await asyncio.gather(*(_one(d) for d in docs))
    return [r for r in results if r is not None]


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

    logger.info("Grader: grading %d new docs (batches of %d)", len(to_grade), _BATCH_SIZE)

    new_grades, fallback = asyncio.run(
        _gather_batches(llm, to_grade, state["original_query"], usage)
    )

    if fallback:
        logger.info("Grader: %d docs in per-doc fallback", len(fallback))
        per_doc = asyncio.run(_gather_per_doc(llm, fallback, state["original_query"], usage))
        new_grades.extend(per_doc)

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
