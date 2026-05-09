"""JSON-safe serialization of LangGraph state deltas for SSE wire format.

Node updates contain LangChain `BaseMessage` objects (AIMessage / HumanMessage
/ SystemMessage / ToolMessage / ...) and Pydantic models like `GradedDocument`.
Plain `json.dumps` chokes on both. This module provides a single helper that
walks the delta and replaces non-JSON types with primitive equivalents.
"""

from typing import Any

from langchain_core.messages import BaseMessage
from pydantic import BaseModel


def _serialize_value(value: Any) -> Any:
    if isinstance(value, BaseMessage):
        return {"type": value.type, "content": value.content}
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_serialize_value(v) for v in value]
    if isinstance(value, tuple):
        return [_serialize_value(v) for v in value]
    return value


def serialize_update(update: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-serializable copy of a node's state-delta dict."""
    return {k: _serialize_value(v) for k, v in update.items()}
