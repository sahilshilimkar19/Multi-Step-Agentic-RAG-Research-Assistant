"""Searcher node: runs queries in parallel, dedupes by URL, increments iteration_count."""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from agentic_rag.state import ResearchState
from agentic_rag.tools.search import web_search

logger = logging.getLogger(__name__)


def searcher_node(state: ResearchState) -> dict[str, Any]:
    queries = state.get("search_queries") or []
    iteration = state.get("iteration_count", 0)

    if not queries:
        logger.warning("Searcher invoked with empty query list")
        return {"iteration_count": iteration + 1}

    logger.info("Searcher: running %d queries (iter %d -> %d)", len(queries), iteration, iteration + 1)

    new_docs: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(len(queries), 5)) as ex:
        future_to_query = {
            ex.submit(web_search.invoke, {"query": q, "max_results": 5}): q for q in queries
        }
        for fut in as_completed(future_to_query):
            q = future_to_query[fut]
            try:
                results = fut.result(timeout=30)
                for r in results:
                    r["sub_question"] = q
                new_docs.extend(results)
            except Exception as e:
                logger.error("Search failed for %r: %s", q, e)

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
