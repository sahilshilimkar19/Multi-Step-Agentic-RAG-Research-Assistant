# Multi-Step Agentic RAG Research Assistant

A production-grade research agent built on **LangGraph**. Instead of one-shot
RAG, it runs a self-correcting research loop — plan, search, grade, decide
to loop or synthesize — until evidence is sufficient or a hard iteration cap
is hit, then writes a structured markdown report with citations.

---

## Why this beats single-shot RAG

| Property              | Single-shot RAG   | This system                                          |
|-----------------------|-------------------|------------------------------------------------------|
| Query rewriting       | One static query  | Plan-driven, multi-query, refined per iteration      |
| Hallucination control | None              | LLM-as-judge grades each doc for groundedness        |
| Coverage              | Whatever 1 search returns | Loops until plan items are covered or cap hit |
| Resumability          | None              | `SqliteSaver` checkpoint per `thread_id`             |
| Observability         | Final answer only | Streamed per-node updates                            |

---

## Architecture

```
                  ┌─────────────┐
                  │   START     │
                  └──────┬──────┘
                         ▼
                  ┌─────────────┐
          ┌──────▶│   planner   │── route on next_action ──┐
          │       │  (router)   │                          │
          │       └──────┬──────┘                          │ synthesize
          │              │ search                          ▼
          │              ▼                          ┌─────────────┐
          │       ┌─────────────┐                   │ synthesizer │
          │       │  searcher   │                   └──────┬──────┘
          │       └──────┬──────┘                          ▼
          │              ▼                              ┌─────┐
          │       ┌─────────────┐                       │ END │
          └───────│   grader    │                       └─────┘
                  └─────────────┘
```

- **Planner** is the only router. On the first visit it decomposes the query
  into a research plan plus initial search queries; on subsequent visits it
  inspects graded evidence and decides to keep searching (with fresh queries)
  or move to synthesis.
- **Searcher** runs every query in parallel (Tavily primary, DuckDuckGo
  fallback), dedupes by URL, and increments `iteration_count`.
- **Grader** is an LLM-as-judge: scores each new doc 0–1 for relevance and
  flags `is_grounded`. Only un-graded docs are scored, so resumes are cheap.
- **Synthesizer** writes a markdown report with `[n]` citations, restricted
  to evidence that passed the relevance + grounded filter.

### Termination — three layers of defense

1. **Hard cap**: `iteration_count >= max_iterations` short-circuits to
   synthesize without an LLM call.
2. **Sufficiency**: `len(relevant_docs) >= min_relevant_docs` and all plan
   items covered short-circuits to synthesize without an LLM call.
3. **Voluntary**: the router LLM may return `next_action="synthesize"` early
   when evidence already covers every sub-question.

---

## Stack

- Python 3.11+
- LangGraph (`StateGraph` + `SqliteSaver` checkpointer)
- LangChain (LLM wrappers, document loaders, `@tool` decorator)
- LLMs: OpenAI (`gpt-4o`, `gpt-4o-mini`) and Anthropic (`claude-sonnet-4-5`)
  — swappable via env
- Web search: Tavily (primary) with DuckDuckGo fallback
- PDF: PyPDF / `PyPDFLoader`
- CLI: Typer + Rich (streaming output)
- Config: `pydantic-settings`
- Retries: `tenacity`

---

## Install

Requires Python 3.11+.

```bash
git clone https://github.com/sahilshilimkar19/Multi-Step-Agentic-RAG-Research-Assistant.git
cd Multi-Step-Agentic-RAG-Research-Assistant
python -m pip install -e ".[dev]"
cp .env.example .env
# fill in OPENAI_API_KEY / ANTHROPIC_API_KEY / TAVILY_API_KEY
```

---

## Usage

```bash
# Fresh run (web search only)
agentic-rag research "How does Mamba compare to Transformers on long-context tasks?"

# Pre-load PDFs into the corpus before planning
agentic-rag research "Summarize this paper's claims vs prior work" \
  --pdf data/pdfs/mamba.pdf --pdf data/pdfs/related.pdf

# Override the iteration cap
agentic-rag research "What are the trade-offs of state-space models?" --max-iterations 6

# Resume an interrupted run
agentic-rag resume <thread-id>

# List checkpointed runs
agentic-rag list-runs
```

The CLI streams every node update to the terminal via `graph.stream(stream_mode="updates")`,
showing the plan, current search queries, graded-doc count, and iteration
number as they happen. After END, the final markdown report is rendered
inline.

---

## Configuration

All knobs live in `.env` (see `.env.example`):

| Variable                | Purpose                                           | Default                |
|-------------------------|---------------------------------------------------|------------------------|
| `LLM_PROVIDER`          | `anthropic` or `openai`                           | `anthropic`            |
| `PLANNER_MODEL`         | Model used by the planner/router                  | `claude-sonnet-4-5`    |
| `GRADER_MODEL`          | Model used by the grader (called many times)      | `gpt-4o-mini`          |
| `SYNTHESIZER_MODEL`     | Model used to write the final report              | `claude-sonnet-4-5`    |
| `MAX_ITERATIONS`        | Hard cap on planner→searcher→grader loops         | `4`                    |
| `MIN_RELEVANT_DOCS`     | Sufficiency threshold for early termination       | `5`                    |
| `RELEVANCE_THRESHOLD`   | Minimum grader score for a doc to count           | `0.6`                  |
| `CHECKPOINT_DB`         | Path to the SQLite checkpoint DB                  | `./checkpoints.sqlite` |

The default tiering uses Claude Sonnet for planning + synthesis (quality
matters) and `gpt-4o-mini` for the grader (called once per retrieved doc —
the cost-sensitive role). Set `LLM_PROVIDER=openai` and adjust the model
fields to swap providers.

---

## Project layout

```
src/agentic_rag/
├── config.py             pydantic-settings: env-driven Settings + get_settings()
├── state.py              ResearchState TypedDict + GradedDocument
├── prompts.py            All prompt templates (never inline elsewhere)
├── tools/
│   ├── search.py         @tool web_search (Tavily → DDG fallback, tenacity retry)
│   └── pdf_loader.py     @tool load_pdf + load_corpus(paths) for CLI
├── nodes/
│   ├── planner.py        Planner/router + LLM factory + _safe_json
│   ├── searcher.py       Parallel queries, URL dedupe, iteration_count++
│   ├── grader.py         Parallel LLM-as-judge; only grades NEW docs
│   └── synthesizer.py    Final markdown report (top 30 docs, 2k chars each)
├── graph.py              StateGraph wiring; caller owns the checkpointer
└── cli.py                Typer: research / resume / list-runs
```

---

## Tests

```bash
pytest tests/ -v
```

Coverage:

- **Unit** — per-node tests with a `FakeLLM` (programmable response queue)
  and a stub `web_search`. Verifies short-circuit termination, JSON-parse
  fallbacks, score clamping, dedupe, and grade-only-new behaviour.
- **Integration** — `test_graph_integration.py` runs the full graph against
  an in-memory `MemorySaver` and asserts the loop terminates within
  `max_iterations` (regression test for infinite loops).

---

## Persistence and resumability

Every node transition is checkpointed via `SqliteSaver`. A run is identified
by its `thread_id`; passing `None` as the input to `graph.stream(...)` with
the same thread resumes from the last checkpoint. The CLI prints the
`thread_id` at the start of every run so it can be reused with
`agentic-rag resume <thread-id>`.

Multi-instance deployments should swap `SqliteSaver` for `PostgresSaver` —
the rest of the code is unchanged.

---

## Risks and mitigations

| Risk                              | Mitigation                                                    |
|-----------------------------------|---------------------------------------------------------------|
| Infinite loop                     | Triple defense: hard cap + sufficiency + voluntary terminate  |
| LLM rate limits                   | `tenacity` retry on Tavily; cheap grader (`gpt-4o-mini`)      |
| Token blowup                      | Grader truncates docs to 4k chars; synthesizer caps 30×2k     |
| Tavily + DDG both dead            | Returns `[]`; planner sees no progress and routes to synthesize |
| Non-JSON LLM output               | `_safe_json` strips fences and falls back at every parse site |
| Cost runaway                      | Anthropic-primary tiering, `gpt-4o-mini` grader, doc/char caps |
| Hallucination at synthesis        | Synthesizer prompt forbids invention; `[n]` citation required  |
