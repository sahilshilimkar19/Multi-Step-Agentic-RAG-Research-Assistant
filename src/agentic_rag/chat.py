"""Follow-up chat over a thread's cached evidence (no web search).

Loads `final_report` from the SqliteSaver checkpoint and the per-thread
Chroma cache. Each user turn retrieves top-k similar docs from the
cache and asks the LLM to answer using ONLY those docs (with [n]
citations). If the cache is unavailable (chromadb missing, never
populated) or the thread has no checkpoint, the chat raises early so
the CLI can show a friendly error.
"""

import logging

from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.cache import ThreadCache, cache_dir
from agentic_rag.config import get_settings
from agentic_rag.llm import UsageCollector, get_llm
from agentic_rag.prompts import CHAT_SYSTEM, CHAT_USER
from agentic_rag.state import GradedDocument

logger = logging.getLogger(__name__)

_TOP_K = 8
_MAX_CHARS_PER_DOC = 2000


def _format_evidence(docs: list[GradedDocument]) -> str:
    if not docs:
        return "No cached evidence found."
    return "\n\n".join(
        f"[{i + 1}] URL: {d.url}\n"
        f"Source: {d.source} | Relevance: {d.relevance_score:.2f}\n"
        f"Content: {d.content[:_MAX_CHARS_PER_DOC]}"
        for i, d in enumerate(docs)
    )


def answer(
    thread_id: str,
    original_query: str,
    question: str,
    *,
    cache: ThreadCache | None = None,
) -> tuple[str, list[GradedDocument]]:
    """Answer a follow-up question using cached evidence only.

    Returns (markdown_answer, retrieved_docs). Caller injects a cache
    instance for tests; production callers omit it.
    """
    settings = get_settings()
    if cache is None:
        cache = ThreadCache(thread_id, cache_dir(settings.checkpoint_db))

    docs = cache.search(question, k=_TOP_K) if cache.available else []
    llm = get_llm(settings.synthesizer_model, temperature=0.2)
    usage = UsageCollector()
    resp = llm.invoke(
        [
            SystemMessage(content=CHAT_SYSTEM),
            HumanMessage(
                content=CHAT_USER.format(
                    query=original_query,
                    documents=_format_evidence(docs),
                    question=question,
                )
            ),
        ],
        config={"callbacks": [usage]},
    )
    text = resp.content if isinstance(resp.content, str) else str(resp.content)
    return text, docs
