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
- **Searcher** runs every query concurrently via `asyncio.gather` with a
  `Semaphore(5)` (Tavily primary, DuckDuckGo fallback), dedupes by URL, and
  increments `iteration_count`.
- **Grader** is an LLM-as-judge: scores each new doc 0–1 for relevance and
  flags `is_grounded`. Grades **5 docs per LLM call** (batched structured
  output) with a per-doc fallback if the batch fails. Only un-graded docs
  are scored, and graded docs are written through to a per-thread Chroma
  vector cache so resumes are cheap.
- **Synthesizer** writes a markdown report with `[n]` citations, restricted
  to evidence that passed the relevance + grounded filter. When the Chroma
  cache is populated it pulls the top-k docs by similarity to the original
  query (a more semantically grounded selection than score-sorted state).

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
- LangGraph (`StateGraph` + `SqliteSaver` checkpointer; `interrupt_before`
  for HITL pauses)
- LangChain (LLM wrappers, document loaders, `@tool`,
  `.with_structured_output()` for Pydantic-validated planner/grader I/O)
- LLMs: OpenAI (`gpt-4o`, `gpt-4o-mini`) and Anthropic (`claude-sonnet-4-5`)
  — swappable via env, every call retried via `tenacity` on transient errors
- Web search: `langchain-tavily` (primary) with DuckDuckGo fallback
- Vector cache: `chromadb` PersistentClient, one collection per `thread_id`
- PDF: PyPDF / `PyPDFLoader`
- CLI: Typer + Rich (streaming + Markdown rendering + summary table)
- Config: `pydantic-settings`
- Logging: `structlog` (JSON or pretty); `thread_id` + `iteration` bound via
  contextvars
- CI: GitHub Actions running ruff + mypy + pytest on Python 3.11 and 3.12

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

# Pause before synthesis to inspect the graded evidence
agentic-rag research "<query>" --review

# Resume an interrupted run
agentic-rag resume <thread-id>

# Follow-up Q&A over a prior run's cached evidence (no fresh web search)
agentic-rag chat <thread-id>

# List checkpointed runs
agentic-rag list-runs
```

The CLI streams every node update to the terminal via `graph.stream(stream_mode="updates")`,
showing the plan, current search queries, graded-doc count, and iteration
number as they happen. After END, a Rich summary table reports
iterations, raw/graded counts, and tokens in/out; the final markdown report
is rendered inline and written to `runs/<thread_id>.md`.

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
| `LOG_LEVEL`             | structlog level (`DEBUG`/`INFO`/...)              | `INFO`                 |
| `LOG_JSON`              | Emit JSON log lines (otherwise human-friendly)    | `false`                |

The default tiering uses Claude Sonnet for planning + synthesis (quality
matters) and `gpt-4o-mini` for the grader (called once per retrieved doc —
the cost-sensitive role). Set `LLM_PROVIDER=openai` and adjust the model
fields to swap providers.

---

## Project layout

```
src/agentic_rag/
├── config.py             pydantic-settings: env-driven Settings + get_settings()
├── state.py              ResearchState + Pydantic schemas (Planner/Router/Grader/Batch)
├── prompts.py            All prompt templates (planner/router/grader/synth/chat)
├── llm.py                LLM factory: ChatAnthropic/ChatOpenAI + tenacity retry
│                          + UsageCollector callback for token tracking
├── logging_config.py     structlog setup; bridges stdlib logging
├── cache.py              ThreadCache: per-thread Chroma vector cache
├── chat.py               answer(): follow-up Q&A over cached evidence (no web)
├── tools/
│   ├── search.py         @tool web_search (langchain-tavily → DDG fallback)
│   └── pdf_loader.py     @tool load_pdf + load_corpus(paths) for CLI
├── nodes/
│   ├── planner.py        Planner/router; structured output + _safe_json fallback
│   ├── searcher.py       asyncio.gather queries (Semaphore=5), URL dedupe
│   ├── grader.py         Batched LLM-as-judge (5 docs/call) + per-doc fallback;
│   │                      writes through to Chroma cache
│   └── synthesizer.py    Final markdown report; reads top-k from cache
├── graph.py              StateGraph wiring; supports interrupt_before for HITL
└── cli.py                Typer: research / resume / chat / list-runs
```

---

## Tests

```bash
pytest tests/ -v          # 49 tests
ruff check src tests      # lint
mypy src                  # type-check
```

GitHub Actions runs all three on every push and PR, against Python 3.11
and 3.12.

Coverage:

- **Unit** — per-node and per-module tests with a `FakeLLM` (programmable
  response queue, supports `with_structured_output()` schema construction)
  and a stub `web_search`. Verifies short-circuit termination, structured
  output + `_safe_json` fallback paths, batched grader behaviour, score
  clamping, dedupe, retry-on-transient, JSON log output, Chroma cache
  roundtrip, and chat answering from cache only.
- **Integration** — `test_graph_integration.py` runs the full graph against
  an in-memory `MemorySaver` and asserts the loop terminates within
  `max_iterations` (regression test for infinite loops).
- **HITL** — `test_hitl.py` verifies `interrupt_before=["synthesizer"]`
  pauses correctly and resume completes the run.

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

## Web UI (preview)

A minimal Next.js 14 + FastAPI front end ships under `web/` and
`src/agentic_rag/server.py`. The CLI keeps working unchanged; the web UI is
purely additive and currently localhost-only. PR-1 is the thin slice:
`POST /api/runs` + an SSE stream — start a run, watch streamed
planner/searcher/grader updates, see the final markdown report. List-runs,
resume, chat, HITL `--review`, and PDF upload land in subsequent PRs.

### Run it (two terminals)

```bash
# Terminal A: FastAPI (Python core)
uvicorn agentic_rag.server:app --reload --port 8000

# Terminal B: Next.js dev server
cd web && npm install        # first time only
npm run dev                  # starts on :3000
```

Open <http://localhost:3000>, type a query, click Start. The browser
navigates to `/runs/<thread-id>` and streams node-by-node updates.

### SSE fallback

Next.js dev rewrites occasionally buffer Server-Sent Events. If you only see
the final report and never the intermediate panels, set:

```bash
# in web/.env.local (or your shell)
NEXT_PUBLIC_API_BASE=http://localhost:8000
```

Restart `npm run dev` and start the FastAPI server with
`AGENTIC_RAG_DEV=1` so it serves CORS for `http://localhost:3000`. The
browser will then connect to FastAPI directly, bypassing Next's proxy.

---

## Risks and mitigations

| Risk                              | Mitigation                                                    |
|-----------------------------------|---------------------------------------------------------------|
| Infinite loop                     | Triple defense: hard cap + sufficiency + voluntary terminate  |
| LLM rate limits                   | `tenacity` retry on every LLM `invoke`/`ainvoke` (3 attempts, 1→8s backoff) on `RateLimitError`/`APITimeoutError`/`APIConnectionError`; cheap batched grader on `gpt-4o-mini` |
| Token blowup                      | Grader truncates docs to 4k chars; synthesizer caps 30×2k; per-run token counts shown in summary |
| Tavily + DDG both dead            | Returns `[]`; planner sees no progress and routes to synthesize |
| Non-JSON LLM output               | `.with_structured_output(Schema)` first; on Pydantic / parser failure, `_safe_json` falls back at every parse site |
| Cost runaway                      | Anthropic-primary tiering, `gpt-4o-mini` batched grader (5 docs/call), doc/char caps, token tracking |
| Hallucination at synthesis        | Synthesizer prompt forbids invention; `[n]` citation required; chat command refuses to answer outside cached evidence |
