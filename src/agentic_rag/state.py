"""Graph state schema + Pydantic models for structured per-doc data."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


class GradedDocument(BaseModel):
    """A retrieved document after the grader has scored it."""

    content: str
    url: str
    source: str  # "tavily" | "ddg" | "pdf"
    sub_question: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    is_grounded: bool
    rationale: str


class ResearchState(TypedDict):
    # Inputs (set once)
    original_query: str
    max_iterations: int

    # Plan (set by planner on first visit)
    research_plan: list[str]

    # Working memory (mutated each iteration)
    search_queries: list[str]
    raw_documents: list[dict]
    graded_documents: list[GradedDocument]
    iteration_count: int

    # Routing
    next_action: Literal["search", "synthesize", "end"]
    needs_more_research: bool

    # Output
    final_report: str

    # LangChain message history (streamable to UIs)
    messages: Annotated[list[BaseMessage], add_messages]
