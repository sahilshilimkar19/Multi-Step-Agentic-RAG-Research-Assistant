"""Web search tool: Tavily primary, DuckDuckGo fallback."""
from __future__ import annotations

import logging
from typing import List

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    source: str


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def _tavily_search(query: str, max_results: int) -> List[dict]:
    tavily = TavilySearchResults(max_results=max_results)
    return tavily.invoke(query)


def _ddg_search(query: str, max_results: int) -> List[dict]:
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        return list(ddgs.text(query, max_results=max_results))


@tool
def web_search(query: str, max_results: int = 5) -> List[dict]:
    """Search the web. Tries Tavily first; falls back to DuckDuckGo on failure."""
    try:
        results = _tavily_search(query, max_results)
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("url", ""),
                content=r.get("content", ""),
                source="tavily",
            ).model_dump()
            for r in results
        ]
    except Exception as e:
        logger.warning("Tavily failed (%s); falling back to DuckDuckGo", e)
        try:
            results = _ddg_search(query, max_results)
        except Exception as e2:
            logger.error("DuckDuckGo also failed (%s); returning empty list", e2)
            return []
        return [
            SearchResult(
                title=r.get("title", ""),
                url=r.get("href", ""),
                content=r.get("body", ""),
                source="ddg",
            ).model_dump()
            for r in results
        ]
