"""Typer CLI: research / resume / list-runs."""
from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path

import structlog
import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table

from agentic_rag.chat import answer as chat_answer
from agentic_rag.config import get_settings
from agentic_rag.graph import build_graph
from agentic_rag.logging_config import configure_logging
from agentic_rag.tools.pdf_loader import load_corpus

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _setup_logging(level: str, json_output: bool) -> None:
    configure_logging(level=level, json_output=json_output)


_RUNS_DIR = Path("runs")


def _render_summary(state_values: dict) -> Table:
    """Build a Rich table summarizing the run."""
    iterations = state_values.get("iteration_count", 0)
    raw_count = len(state_values.get("raw_documents") or [])
    graded = state_values.get("graded_documents") or []
    graded_count = len(graded)
    relevant_count = sum(
        1 for g in graded if getattr(g, "relevance_score", 0.0) >= 0.6 and getattr(g, "is_grounded", False)
    )
    usage = state_values.get("token_usage") or {}
    in_tok = int(usage.get("input_tokens", 0))
    out_tok = int(usage.get("output_tokens", 0))

    table = Table(title="Run Summary", show_header=False, title_style="bold green")
    table.add_column("metric", style="bold")
    table.add_column("value")
    table.add_row("Iterations", str(iterations))
    table.add_row("Raw documents", str(raw_count))
    table.add_row("Graded documents", f"{graded_count} ({relevant_count} relevant)")
    table.add_row("Tokens", f"{in_tok:,} in / {out_tok:,} out")
    return table


def _save_report(thread_id: str, report: str) -> Path:
    """Write the final report to runs/<thread_id>.md and return the path."""
    _RUNS_DIR.mkdir(exist_ok=True)
    path = _RUNS_DIR / f"{thread_id}.md"
    path.write_text(report, encoding="utf-8")
    return path


def _render_update(node_name: str, update: dict) -> None:
    console.rule(f"[bold cyan]{node_name}[/]")
    if update.get("messages"):
        for m in update["messages"]:
            content = getattr(m, "content", str(m))
            console.print(f"[dim]{content}[/]")
    if update.get("research_plan"):
        console.print("[bold]Plan:[/]")
        for p in update["research_plan"]:
            console.print(f"  - {p}")
    if update.get("search_queries"):
        console.print(f"[bold]Queries:[/] {update['search_queries']}")
    if update.get("graded_documents") is not None:
        console.print(f"[bold]Graded so far:[/] {len(update['graded_documents'])} docs")
    if update.get("iteration_count") is not None:
        console.print(f"[bold]Iteration:[/] {update['iteration_count']}")
    if update.get("next_action"):
        console.print(f"[bold]Next action:[/] {update['next_action']}")


@app.command()
def research(
    query: str = typer.Argument(..., help="Research question"),
    thread_id: str | None = typer.Option(None, "--thread-id", help="Reuse an existing thread"),
    max_iterations: int | None = typer.Option(
        None, "--max-iterations", help="Override MAX_ITERATIONS"
    ),
    pdf: list[Path] | None = typer.Option(
        None, "--pdf", help="Pre-load PDF(s) into the corpus before planning"
    ),
    review: bool = typer.Option(
        False, "--review", help="Pause before synthesis to review the graded evidence"
    ),
) -> None:
    """Run a new research task."""
    settings = get_settings()
    _setup_logging(settings.log_level, settings.log_json)

    tid = thread_id or str(uuid.uuid4())
    config = {"configurable": {"thread_id": tid}}

    seed_docs: list[dict] = []
    if pdf:
        console.print(f"[bold]Loading {len(pdf)} PDF(s) into corpus...[/]")
        seed_docs = load_corpus(pdf, original_query=query)
        console.print(f"[dim]Seeded {len(seed_docs)} pages.[/]")

    initial_state = {
        "original_query": query,
        "research_plan": [],
        "search_queries": [],
        "raw_documents": seed_docs,
        "graded_documents": [],
        "iteration_count": 0,
        "max_iterations": max_iterations or settings.max_iterations,
        "needs_more_research": True,
        "final_report": "",
        "next_action": "search",
        "messages": [],
        "token_usage": {"input_tokens": 0, "output_tokens": 0},
    }

    interrupts = ["synthesizer"] if review else None
    with (
        SqliteSaver.from_conn_string(settings.checkpoint_db) as cp,
        structlog.contextvars.bound_contextvars(thread_id=tid),
    ):
        graph = build_graph(checkpointer=cp, interrupt_before=interrupts)
        console.print(f"[bold green]Thread:[/] {tid}")
        console.print(f"[bold green]Query:[/]  {query}\n")

        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                _render_update(node_name, update)

        if review:
            paused = graph.get_state(config)
            if paused.next and "synthesizer" in paused.next:
                graded = paused.values.get("graded_documents") or []
                relevant = [g for g in graded if g.relevance_score >= 0.6 and g.is_grounded]
                console.rule("[bold yellow]Review pause[/]")
                console.print(
                    f"[bold]{len(graded)}[/] graded ({len(relevant)} relevant). "
                    "Press Enter to continue to synthesis, Ctrl-C to abort."
                )
                try:
                    typer.prompt("", default="", show_default=False)
                except (KeyboardInterrupt, EOFError):
                    console.print("[red]Aborted.[/]")
                    return
                # Resume by streaming with None.
                for event in graph.stream(None, config=config, stream_mode="updates"):
                    for node_name, update in event.items():
                        _render_update(node_name, update)

        final = graph.get_state(config)
        report = final.values.get("final_report", "")
        console.rule("[bold green]Run Summary[/]")
        console.print(_render_summary(final.values))
        if report:
            console.rule("[bold green]Final Report[/]")
            console.print(Markdown(report))
            saved = _save_report(tid, report)
            console.print(f"\n[green]Saved report to[/] {saved}")
        else:
            console.print("[bold yellow]No report produced (run interrupted?)[/]")
        console.print(f"[dim]Resume with:[/] agentic-rag resume {tid}")


@app.command()
def resume(thread_id: str = typer.Argument(..., help="Thread ID from a prior run")) -> None:
    """Resume an interrupted research run."""
    settings = get_settings()
    _setup_logging(settings.log_level, settings.log_json)

    config = {"configurable": {"thread_id": thread_id}}
    with (
        SqliteSaver.from_conn_string(settings.checkpoint_db) as cp,
        structlog.contextvars.bound_contextvars(thread_id=thread_id),
    ):
        graph = build_graph(checkpointer=cp)
        for event in graph.stream(None, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                _render_update(node_name, update)

        final = graph.get_state(config)
        report = final.values.get("final_report", "")
        console.rule("[bold green]Run Summary[/]")
        console.print(_render_summary(final.values))
        if report:
            console.rule("[bold green]Final Report[/]")
            console.print(Markdown(report))
            saved = _save_report(thread_id, report)
            console.print(f"\n[green]Saved report to[/] {saved}")


@app.command()
def chat(thread_id: str = typer.Argument(..., help="Thread ID from a prior run")) -> None:
    """Follow-up Q&A over a thread's cached evidence (no web search)."""
    settings = get_settings()
    _setup_logging(settings.log_level, settings.log_json)

    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(settings.checkpoint_db) as cp:
        graph = build_graph(checkpointer=cp)
        state = graph.get_state(config)
        original_query = state.values.get("original_query", "")
    if not original_query:
        console.print(f"[red]No checkpoint found for thread {thread_id}.[/]")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Thread:[/] {thread_id}")
    console.print(f"[bold green]Original query:[/] {original_query}")
    console.print("[dim]Type a follow-up question or blank line to exit.[/]\n")
    while True:
        try:
            question = typer.prompt("you")
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]bye[/]")
            return
        if not question.strip():
            console.print("[dim]bye[/]")
            return
        text, docs = chat_answer(thread_id, original_query, question)
        console.rule()
        console.print(Markdown(text))
        if docs:
            console.print(f"[dim]Drew on {len(docs)} cached doc(s).[/]\n")


@app.command("list-runs")
def list_runs() -> None:
    """List all checkpointed research threads."""
    settings = get_settings()
    db_path = Path(settings.checkpoint_db)
    if not db_path.exists():
        console.print(f"[yellow]No checkpoint DB at {db_path}.[/]")
        return
    conn = sqlite3.connect(str(db_path))
    try:
        rows = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints ORDER BY thread_id"
        ).fetchall()
    except sqlite3.OperationalError as e:
        console.print(f"[red]Failed to read checkpoints table: {e}[/]")
        return
    finally:
        conn.close()
    if not rows:
        console.print("[yellow]No checkpointed runs.[/]")
        return
    console.print("[bold]Checkpointed threads:[/]")
    for (tid,) in rows:
        console.print(f"  - {tid}")


if __name__ == "__main__":
    app()
