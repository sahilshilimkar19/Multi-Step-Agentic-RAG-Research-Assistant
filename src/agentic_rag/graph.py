"""Compose the research graph. Caller owns the checkpointer's lifecycle."""
from __future__ import annotations

from collections.abc import Sequence

from langgraph.graph import END, START, StateGraph

from agentic_rag.nodes.grader import grader_node
from agentic_rag.nodes.planner import planner_node
from agentic_rag.nodes.searcher import searcher_node
from agentic_rag.nodes.synthesizer import synthesizer_node
from agentic_rag.state import ResearchState


def _route_from_planner(state: ResearchState) -> str:
    return state.get("next_action", "search")


def build_graph(checkpointer, *, interrupt_before: Sequence[str] | None = None):
    """Compose the research StateGraph and compile it with the given checkpointer.

    The caller owns the checkpointer's lifecycle (typically inside a
    `with SqliteSaver.from_conn_string(...) as cp:` block). Returns a
    `CompiledStateGraph` ready for `.stream()` or `.invoke()`.

    `interrupt_before` is forwarded to LangGraph's compile() to support
    human-in-the-loop pauses (e.g. `["synthesizer"]` to review evidence
    before drafting the final report).
    """
    g = StateGraph(ResearchState)

    g.add_node("planner", planner_node)
    g.add_node("searcher", searcher_node)
    g.add_node("grader", grader_node)
    g.add_node("synthesizer", synthesizer_node)

    g.add_edge(START, "planner")
    g.add_conditional_edges(
        "planner",
        _route_from_planner,
        {"search": "searcher", "synthesize": "synthesizer", "end": END},
    )
    g.add_edge("searcher", "grader")
    g.add_edge("grader", "planner")
    g.add_edge("synthesizer", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=list(interrupt_before or []))
