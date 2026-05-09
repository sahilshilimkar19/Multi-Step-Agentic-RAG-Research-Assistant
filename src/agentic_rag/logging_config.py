"""structlog setup with stdlib bridging.

`configure_logging(level, json_output)` configures structlog and routes
existing `logging.getLogger(__name__).info(...)` calls through the same
processors, so the codebase keeps using stdlib loggers while we get JSON
output and contextvar-based field binding.

Per-run binding is done via `structlog.contextvars`:
- The CLI binds `thread_id` once before the graph stream starts.
- Each node wraps its body in `with bound_contextvars(node=..., iteration=...):`
  so log lines inside the node carry those fields automatically.
"""
from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(level: str = "INFO", json_output: bool = False) -> None:
    """Set up structlog + bridge stdlib logging through the same processors."""
    level_int = getattr(logging, level.upper(), logging.INFO)

    timestamper = structlog.processors.TimeStamper(fmt="iso")
    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if json_output:
        renderer: Any = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=False)

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(level_int),
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=shared_processors,
            processors=[
                structlog.stdlib.ProcessorFormatter.remove_processors_meta,
                renderer,
            ],
        )
    )
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level_int)
