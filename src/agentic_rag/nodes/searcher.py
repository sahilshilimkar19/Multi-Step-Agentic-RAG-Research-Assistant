"""Searcher node: runs queries in parallel, dedupes by URL, increments iteration_count."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import structlog

from agentic_rag.state import ResearchState
from agentic_rag.tools.search import web_search

logger = logging.getLogger(__name__)

_SEARCH_CONCURRENCY = 5
_SEARCH_TIMEOUT_S = 30.0


def searcher_node(state: ResearchState) -> dict[str, Any]:
    """Run every query in `search_queries` in parallel; dedupe by URL.

    Uses ThreadPoolExecutor (max 5 workers) with a 30s per-query timeout.
    Failed queries are logged but do not abort the iteration. Increments
    `iteration_count` exactly once per call -- canonical place for the
    counter that drives the planner's hard-cap termination.
    """
    iteration = state.get("iteration_count", 0)
    with structlog.contextvars.bound_contextvars(node="searcher", iteration=iteration):
        return _searcher_body(state, iteration)


async def _run_one_query(query: str, sem: asyncio.Semaphore) -> list[dict]:
    """Run web_search.invoke off the event loop, bounded by the semaphore."""
    async with sem:
        try:
            results = await asyncio.wait_for(
                asyncio.to_thread(web_search.invoke, {"query": query, "max_results": 5}),
                timeout=_SEARCH_TIMEOUT_S,
            )
        except Exception as e:
            logger.error("Search failed for %r: %s", query, e)
            return []
    for r in results:
        r["sub_question"] = query
    return results


async def _gather_searches(queries: list[str]) -> list[dict]:
    sem = asyncio.Semaphore(_SEARCH_CONCURRENCY)
    nested = await asyncio.gather(*(_run_one_query(q, sem) for q in queries))
    out: list[dict] = []
    for batch in nested:
        out.extend(batch)
    return out


def _searcher_body(state: ResearchState, iteration: int) -> dict[str, Any]:
    queries = state.get("search_queries") or []

    if not queries:
        logger.warning("Searcher invoked with empty query list")
        return {"iteration_count": iteration + 1}

    logger.info("Searcher: running %d queries (iter %d -> %d)", len(queries), iteration, iteration + 1)
    new_docs = asyncio.run(_gather_searches(queries))

    seen = {d.get("url") for d in state.get("raw_documents", []) if d.get("url")}
    fresh: list[dict] = []
    for d in new_docs:
        url = d.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        fresh.append(d)
    logger.info("Searcher: %d fresh docs (%d duplicates filtered)", len(fresh), len(new_docs) - len(fresh))

    return {
        "raw_documents": (state.get("raw_documents") or []) + fresh,
        "iteration_count": iteration + 1,
    }
