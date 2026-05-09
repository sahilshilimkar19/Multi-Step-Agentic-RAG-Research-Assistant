"""Graph state schema + Pydantic models for structured per-doc data."""
from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field


def add_usage(left: dict | None, right: dict | None) -> dict:
    """Reducer that sums two {input_tokens, output_tokens} dicts."""
    a = left or {}
    b = right or {}
    return {
        "input_tokens": int(a.get("input_tokens", 0)) + int(b.get("input_tokens", 0)),
        "output_tokens": int(a.get("output_tokens", 0)) + int(b.get("output_tokens", 0)),
    }


class GradedDocument(BaseModel):
    """A retrieved document after the grader has scored it."""

    content: str
    url: str
    source: str  # "tavily" | "ddg" | "pdf"
    sub_question: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    is_grounded: bool
    rationale: str


class PlannerOutput(BaseModel):
    """Structured response from the planner LLM (first-visit decomposition)."""

    plan: list[str]
    initial_queries: list[str]


class RouterOutput(BaseModel):
    """Structured response from the router LLM (subsequent-visit decision)."""

    next_action: Literal["search", "synthesize"]
    rationale: str
    queries: list[str] = Field(default_factory=list)


class GraderOutput(BaseModel):
    """Structured response from the grader LLM (per-doc judgment).

    `relevance_score` is unbounded here so a slightly out-of-range LLM score
    does not crash the structured-output path; the grader clamps to [0, 1].
    """

    relevance_score: float
    is_grounded: bool
    rationale: str


class GraderBatchItem(BaseModel):
    """One verdict in a batched grader response."""

    index: int
    relevance_score: float
    is_grounded: bool
    rationale: str


class GraderBatchOutput(BaseModel):
    """Structured response from the batched grader LLM (multiple docs in one call)."""

    grades: list[GraderBatchItem]


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

    # Cumulative token usage across all node LLM calls
    token_usage: Annotated[dict, add_usage]
