"""FastAPI entrypoint. Run via `uvicorn agentic_rag.server:app --reload`."""

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from agentic_rag.api import runs as runs_module
from agentic_rag.config import get_settings
from agentic_rag.logging_config import configure_logging


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    yield


app = FastAPI(title="Agentic RAG", version="0.1.0", lifespan=lifespan)

# Localhost dev fallback path: allow the browser to hit FastAPI directly when
# Next.js dev rewrites buffer SSE. Gated so it never silently leaks in prod.
if os.getenv("AGENTIC_RAG_DEV") == "1":
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000"],
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )


@app.get("/api/health")
async def health() -> dict[str, str]:
    """Simple liveness probe."""
    return {"status": "ok"}


app.include_router(runs_module.router)
