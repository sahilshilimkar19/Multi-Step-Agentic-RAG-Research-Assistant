"""PDF loading: agent-callable @tool plus CLI bulk-corpus helper."""
from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def _load_one(path: str | Path, max_pages: int | None = None) -> list[dict]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF not found: {p}")

    loader = PyPDFLoader(str(p))
    docs = loader.load()
    if max_pages:
        docs = docs[:max_pages]

    abs_path = p.resolve()
    return [
        {
            "title": f"{p.name} (page {d.metadata.get('page', '?')})",
            "url": f"file://{abs_path}#page={d.metadata.get('page', 0)}",
            "content": d.page_content,
            "source": "pdf",
        }
        for d in docs
    ]


@tool
def load_pdf(path: str, max_pages: int | None = None) -> list[dict]:
    """Load a PDF from disk. Returns one dict per page with content and metadata."""
    return _load_one(path, max_pages)


def load_corpus(paths: Sequence[str | Path], original_query: str) -> list[dict]:
    """Pre-load multiple PDFs for the CLI; tags each page with the original query as sub_question."""
    docs: list[dict] = []
    for p in paths:
        try:
            for page in _load_one(p):
                page["sub_question"] = original_query
                docs.append(page)
        except Exception as e:
            logger.error("Failed to load PDF %s: %s", p, e)
    return docs
