"""Persistent vector cache per research thread, backed by ChromaDB.

`ThreadCache(thread_id, persist_dir)` opens (or creates) a Chroma
collection at `<persist_dir>/<thread_id>/`. The grader writes graded
documents through after each grading round; the synthesizer reads the
top-k most relevant docs by similarity to the original query (a more
semantically grounded selection than sorting state by relevance_score
alone). The chat command (B5) reuses the same cache to answer
follow-up questions from cached evidence only.

If chromadb fails to load (corrupt DB, missing native deps, etc), the
cache transparently no-ops: `available` is False, `add` is a noop, and
`search` returns []. Callers fall back to whatever is in state.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agentic_rag.state import GradedDocument

logger = logging.getLogger(__name__)


class ThreadCache:
    """Vector cache for one research thread."""

    def __init__(self, thread_id: str, persist_dir: Path) -> None:
        self.thread_id = thread_id
        self._collection: Any = None
        try:
            import chromadb  # local import: heavy dep

            persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(persist_dir / thread_id))
            self._collection = client.get_or_create_collection(name="docs")
        except Exception as e:
            logger.warning("Chroma cache unavailable for thread %s (%s)", thread_id, e)

    @property
    def available(self) -> bool:
        """True iff chromadb loaded and the collection is ready."""
        return self._collection is not None

    def add(self, docs: list[GradedDocument]) -> None:
        """Upsert docs into the collection, keyed by URL."""
        if not self.available or not docs:
            return
        try:
            self._collection.upsert(
                documents=[d.content for d in docs],
                metadatas=[
                    {
                        "url": d.url,
                        "source": d.source,
                        "sub_question": d.sub_question,
                        "relevance_score": float(d.relevance_score),
                        "is_grounded": bool(d.is_grounded),
                        "rationale": d.rationale,
                    }
                    for d in docs
                ],
                ids=[d.url or f"_anon_{i}" for i, d in enumerate(docs)],
            )
        except Exception as e:
            logger.error("Chroma upsert failed for thread %s: %s", self.thread_id, e)

    def search(self, query: str, k: int = 30) -> list[GradedDocument]:
        """Return the top-k docs by similarity to `query`."""
        if not self.available or not query:
            return []
        try:
            results = self._collection.query(query_texts=[query], n_results=k)
        except Exception as e:
            logger.error("Chroma query failed for thread %s: %s", self.thread_id, e)
            return []
        docs_field = results.get("documents") or [[]]
        metas_field = results.get("metadatas") or [[]]
        documents = docs_field[0] if docs_field else []
        metadatas = metas_field[0] if metas_field else []
        out: list[GradedDocument] = []
        for content, meta in zip(documents, metadatas, strict=False):
            meta = meta or {}
            out.append(
                GradedDocument(
                    content=content or "",
                    url=str(meta.get("url", "")),
                    source=str(meta.get("source", "unknown")),
                    sub_question=str(meta.get("sub_question", "")),
                    relevance_score=float(meta.get("relevance_score", 0.0) or 0.0),
                    is_grounded=bool(meta.get("is_grounded", True)),
                    rationale=str(meta.get("rationale", "")),
                )
            )
        return out


def cache_dir(checkpoint_db: str) -> Path:
    """Derive the Chroma persist dir from the checkpoint DB path.

    `./checkpoints.sqlite` -> `./runs/chroma`.
    """
    return Path(checkpoint_db).parent / "runs" / "chroma"
