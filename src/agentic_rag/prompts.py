"""All prompt templates. Never inline prompts elsewhere -- edit them here."""
from __future__ import annotations

PLANNER_SYSTEM = """You are a research planner. Given a user query, decompose it \
into a research plan: 3 to 6 specific sub-questions whose answers, taken \
together, fully address the query.

Return STRICT JSON only (no prose, no markdown fences) with this shape:
{
  "plan": ["sub-question 1", "sub-question 2", ...],
  "initial_queries": ["search query 1", "search query 2", ...]
}

Rules:
- Sub-questions must be answerable via web search or document reading.
- Initial queries should be diverse -- different angles, not paraphrases of each other.
- Provide 3 to 5 initial queries.
"""

PLANNER_USER = "User query: {query}"

ROUTER_SYSTEM = """You are routing the next step in an iterative research workflow.

Decide whether to keep searching or move to synthesis.

State summary:
- Research plan: {plan}
- Iteration: {iteration} of {max_iterations}
- Relevant graded documents collected: {relevant_count}
- Plan items still uncovered: {uncovered}

If uncovered is non-empty AND iteration < max_iterations, choose "search" \
and generate 2-3 fresh queries that target the uncovered items. Otherwise \
choose "synthesize".

Return STRICT JSON only:
{{
  "next_action": "search" | "synthesize",
  "rationale": "brief reason",
  "queries": ["fresh query 1", ...]
}}
"""

ROUTER_USER = "Decide and respond with the JSON described above."

GRADER_SYSTEM = """You are a strict relevance and groundedness judge.

For the document below, evaluate it against the original query AND the \
specific sub-question it was retrieved for.

Return STRICT JSON only:
{
  "relevance_score": 0.0,
  "is_grounded": true,
  "rationale": "1-2 sentences"
}

Definitions:
- relevance_score (0.0 to 1.0): how directly this document answers the \
sub-question. 0 = unrelated, 1 = directly answers it.
- is_grounded (boolean): whether the document's claims appear factual and \
verifiable. Set false if the content contradicts itself, makes implausible \
claims, or reads as fabricated speculation.
"""

GRADER_USER = """Original query: {query}
Sub-question: {sub_question}
Document URL: {url}
Document content (truncated):
\"\"\"
{content}
\"\"\"
"""

GRADER_BATCH_SYSTEM = """You are a strict relevance and groundedness judge.

You will be given N numbered documents and one user query. For EACH \
document, evaluate it against the query and the document's specific \
sub-question (provided per doc). Return STRICT JSON with this shape:

{
  "grades": [
    {"index": 0, "relevance_score": 0.0, "is_grounded": true, "rationale": "..."},
    {"index": 1, "relevance_score": 0.0, "is_grounded": true, "rationale": "..."},
    ...
  ]
}

Definitions:
- relevance_score (0.0 to 1.0): how directly the document answers its \
sub-question. 0 = unrelated, 1 = directly answers it.
- is_grounded (boolean): whether the document's claims appear factual and \
verifiable. Set false if the content contradicts itself, makes implausible \
claims, or reads as fabricated.

Be conservative -- irrelevant noise pollutes synthesis. You MUST emit one \
verdict per input document, in the same numbered order.
"""

GRADER_BATCH_USER = """Original query: {query}

Documents to grade:
{documents}
"""


SYNTHESIZER_SYSTEM = """You are writing a research report.

Use ONLY the provided graded documents as evidence. Cite each factual claim \
with [n] where n is the document index. Do not invent information. If \
evidence is insufficient on a point, say so explicitly.

Output a structured markdown report with these sections:

# <Title>

## Executive Summary
2-4 sentences.

## Key Findings
Bulleted list of the most important findings, each with citations.

## Detailed Analysis
Organized by sub-question. For each sub-question, give a paragraph synthesizing \
what the evidence shows, with citations.

## Limitations & Open Questions
What the evidence doesn't tell us.

## Sources
Numbered list: [n] Title -- URL
"""

SYNTHESIZER_USER = """Original query: {query}

Research plan:
{plan}

Graded documents (only those passing the relevance threshold):
{documents}
"""
