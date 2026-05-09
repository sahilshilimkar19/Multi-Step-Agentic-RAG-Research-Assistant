"""Planner / router node + the LLM factory used by every node."""
from __future__ import annotations

import json
import logging
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agentic_rag.config import get_settings
from agentic_rag.llm import UsageCollector, get_llm
from agentic_rag.prompts import (
    PLANNER_SYSTEM,
    PLANNER_USER,
    ROUTER_SYSTEM,
    ROUTER_USER,
)
from agentic_rag.state import GradedDocument, PlannerOutput, ResearchState, RouterOutput

logger = logging.getLogger(__name__)


def _safe_json(text: str, default: dict) -> dict:
    """Parse JSON defensively. Strips fenced code blocks; returns default on failure."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned.removesuffix("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Non-JSON LLM response; using default. First 200 chars: %r", text[:200])
        return default


def _uncovered_plan_items(plan: list[str], relevant_docs: list[GradedDocument]) -> list[str]:
    """A plan item is 'covered' if any relevant doc mentions one of its longer content words."""
    uncovered: list[str] = []
    for item in plan:
        keywords = [w.lower().strip(".,?!:;()[]\"'") for w in item.split() if len(w) > 4]
        if not keywords:
            continue
        covered = any(any(k in d.content.lower() for k in keywords) for d in relevant_docs)
        if not covered:
            uncovered.append(item)
    return uncovered


def planner_node(state: ResearchState) -> dict[str, Any]:
    iteration = state.get("iteration_count", 0)
    with structlog.contextvars.bound_contextvars(node="planner", iteration=iteration):
        return _planner_body(state, iteration)


def _planner_body(state: ResearchState, iteration: int) -> dict[str, Any]:
    settings = get_settings()
    llm = get_llm(settings.planner_model)
    usage = UsageCollector()
    cb_config = {"callbacks": [usage]}

    plan = state.get("research_plan") or []

    # First visit: build the plan and seed initial queries.
    if iteration == 0 and not plan:
        logger.info("Planner: initial planning for query=%r", state["original_query"])
        messages = [
            SystemMessage(content=PLANNER_SYSTEM),
            HumanMessage(content=PLANNER_USER.format(query=state["original_query"])),
        ]
        try:
            structured_llm = llm.with_structured_output(PlannerOutput)
            output: PlannerOutput = structured_llm.invoke(messages, config=cb_config)
            new_plan = output.plan or [state["original_query"]]
            new_queries = output.initial_queries or [state["original_query"]]
        except Exception as e:
            logger.warning("Planner structured output failed (%s); falling back to JSON", e)
            resp = llm.invoke(messages, config=cb_config)
            data = _safe_json(
                resp.content if isinstance(resp.content, str) else str(resp.content),
                default={
                    "plan": [state["original_query"]],
                    "initial_queries": [state["original_query"]],
                },
            )
            new_plan = data.get("plan") or [state["original_query"]]
            new_queries = data.get("initial_queries") or [state["original_query"]]
        return {
            "research_plan": new_plan,
            "search_queries": new_queries,
            "next_action": "search",
            "messages": [AIMessage(content=f"Plan: {new_plan}")],
            "token_usage": usage.as_dict(),
        }

    # Subsequent visits: route.
    relevant = [
        d
        for d in (state.get("graded_documents") or [])
        if d.relevance_score >= settings.relevance_threshold and d.is_grounded
    ]
    uncovered = _uncovered_plan_items(plan, relevant)

    # Hard termination -- max iterations.
    if iteration >= state["max_iterations"]:
        logger.info("Planner: hit max_iterations (%d), forcing synthesize", state["max_iterations"])
        return {"next_action": "synthesize"}

    # Sufficiency termination.
    if len(relevant) >= settings.min_relevant_docs and not uncovered:
        logger.info(
            "Planner: %d relevant docs, all plan items covered -> synthesize", len(relevant)
        )
        return {"next_action": "synthesize"}

    # Otherwise ask the router LLM.
    router_messages = [
        SystemMessage(
            content=ROUTER_SYSTEM.format(
                plan=plan,
                iteration=iteration,
                max_iterations=state["max_iterations"],
                relevant_count=len(relevant),
                uncovered=uncovered,
            )
        ),
        HumanMessage(content=ROUTER_USER),
    ]
    fallback_queries = uncovered[:3] or [state["original_query"]]
    try:
        structured_llm = llm.with_structured_output(RouterOutput)
        router_output: RouterOutput = structured_llm.invoke(router_messages, config=cb_config)
        next_action = router_output.next_action
        queries = router_output.queries or fallback_queries
        rationale = router_output.rationale
    except Exception as e:
        logger.warning("Router structured output failed (%s); falling back to JSON", e)
        resp = llm.invoke(router_messages, config=cb_config)
        data = _safe_json(
            resp.content if isinstance(resp.content, str) else str(resp.content),
            default={
                "next_action": "search",
                "rationale": "fallback: continue searching",
                "queries": fallback_queries,
            },
        )
        next_action = data.get("next_action", "search")
        if next_action not in ("search", "synthesize"):
            next_action = "search"
        queries = data.get("queries") or fallback_queries
        rationale = data.get("rationale", "")
    return {
        "next_action": next_action,
        "search_queries": queries,
        "messages": [AIMessage(content=f"Router: {rationale}")],
        "token_usage": usage.as_dict(),
    }
