"""Tests for CLI helpers (save-to-disk, run summary)."""
from __future__ import annotations

from rich.console import Console

from agentic_rag import cli as cli_mod
from agentic_rag.state import GradedDocument


def test_save_report_creates_file(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "_RUNS_DIR", tmp_path / "runs")
    path = cli_mod._save_report("abc-123", "# Hello\n\nbody")
    assert path.exists()
    assert path.read_text(encoding="utf-8") == "# Hello\n\nbody"
    assert path.name == "abc-123.md"


def test_save_report_overwrites_existing(monkeypatch, tmp_path):
    monkeypatch.setattr(cli_mod, "_RUNS_DIR", tmp_path / "runs")
    cli_mod._save_report("abc", "first")
    cli_mod._save_report("abc", "second")
    path = (tmp_path / "runs" / "abc.md")
    assert path.read_text(encoding="utf-8") == "second"


def _gd(score: float, grounded: bool, url: str = "https://x") -> GradedDocument:
    return GradedDocument(
        content="c", url=url, source="tavily", sub_question="q",
        relevance_score=score, is_grounded=grounded, rationale="r",
    )


def test_run_summary_counts_relevant_only():
    state_values = {
        "iteration_count": 3,
        "raw_documents": [{"url": f"u{i}"} for i in range(7)],
        "graded_documents": [
            _gd(0.9, True, "https://a"),
            _gd(0.5, True, "https://b"),  # below threshold (0.6)
            _gd(0.9, False, "https://c"),  # not grounded
            _gd(0.7, True, "https://d"),
        ],
        "token_usage": {"input_tokens": 1234, "output_tokens": 567},
    }
    table = cli_mod._render_summary(state_values)
    # Render to a string buffer to check the cells.
    console = Console(record=True, width=200)
    console.print(table)
    text = console.export_text()
    assert "Iterations" in text and "3" in text
    assert "Raw documents" in text and "7" in text
    assert "Graded documents" in text and "4 (2 relevant)" in text
    assert "1,234 in / 567 out" in text


def test_run_summary_handles_empty_state():
    table = cli_mod._render_summary({})
    console = Console(record=True, width=200)
    console.print(table)
    text = console.export_text()
    assert "Iterations" in text and "0" in text
    assert "0 in / 0 out" in text


def test_usage_collector_accumulates():
    from types import SimpleNamespace

    from agentic_rag.llm import UsageCollector

    collector = UsageCollector()
    # Mock a LangChain LLMResult shape: .generations is list[list[Generation]]
    # and each generation has .message.usage_metadata.
    msg1 = SimpleNamespace(usage_metadata={"input_tokens": 100, "output_tokens": 20})
    msg2 = SimpleNamespace(usage_metadata={"input_tokens": 50, "output_tokens": 10})
    response = SimpleNamespace(
        generations=[[SimpleNamespace(message=msg1)], [SimpleNamespace(message=msg2)]]
    )
    collector.on_llm_end(response)
    assert collector.as_dict() == {"input_tokens": 150, "output_tokens": 30}
