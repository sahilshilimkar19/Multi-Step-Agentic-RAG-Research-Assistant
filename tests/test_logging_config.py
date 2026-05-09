"""Smoke tests for the logging configuration."""
from __future__ import annotations

import json
import logging

import structlog

from agentic_rag.logging_config import configure_logging


def test_json_output_produces_parseable_lines(capsys):
    configure_logging(level="INFO", json_output=True)
    logging.getLogger("test").info("hello %s", "world")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["event"].startswith("hello")
    assert record["level"] == "info"
    assert "timestamp" in record


def test_contextvar_thread_id_appears_in_log(capsys):
    configure_logging(level="INFO", json_output=True)
    structlog.contextvars.clear_contextvars()
    with structlog.contextvars.bound_contextvars(thread_id="abc-1", node="planner"):
        logging.getLogger("test").info("running")
    captured = capsys.readouterr()
    line = captured.err.strip().splitlines()[-1]
    record = json.loads(line)
    assert record["thread_id"] == "abc-1"
    assert record["node"] == "planner"
