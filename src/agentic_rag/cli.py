"""Typer CLI: research / resume / list-runs."""
from __future__ import annotations

import logging
import sqlite3
import uuid
from pathlib import Path

import typer
from langgraph.checkpoint.sqlite import SqliteSaver
from rich.console import Console
from rich.markdown import Markdown

from agentic_rag.config import get_settings
from agentic_rag.graph import build_graph
from agentic_rag.tools.pdf_loader import load_corpus

app = typer.Typer(no_args_is_help=True, add_completion=False)
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )


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
) -> None:
    """Run a new research task."""
    settings = get_settings()
    _setup_logging(settings.log_level)

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
    }

    with SqliteSaver.from_conn_string(settings.checkpoint_db) as cp:
        graph = build_graph(checkpointer=cp)
        console.print(f"[bold green]Thread:[/] {tid}")
        console.print(f"[bold green]Query:[/]  {query}\n")

        for event in graph.stream(initial_state, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                _render_update(node_name, update)

        final = graph.get_state(config)
        report = final.values.get("final_report", "")
        if report:
            console.rule("[bold green]Final Report[/]")
            console.print(Markdown(report))
        else:
            console.print("[bold yellow]No report produced (run interrupted?)[/]")
        console.print(f"\n[dim]Resume with:[/] agentic-rag resume {tid}")


@app.command()
def resume(thread_id: str = typer.Argument(..., help="Thread ID from a prior run")) -> None:
    """Resume an interrupted research run."""
    settings = get_settings()
    _setup_logging(settings.log_level)

    config = {"configurable": {"thread_id": thread_id}}
    with SqliteSaver.from_conn_string(settings.checkpoint_db) as cp:
        graph = build_graph(checkpointer=cp)
        for event in graph.stream(None, config=config, stream_mode="updates"):
            for node_name, update in event.items():
                _render_update(node_name, update)

        final = graph.get_state(config)
        report = final.values.get("final_report", "")
        if report:
            console.rule("[bold green]Final Report[/]")
            console.print(Markdown(report))


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
