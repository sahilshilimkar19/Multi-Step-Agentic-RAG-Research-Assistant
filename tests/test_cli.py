"""Tests for CLI helpers (save-to-disk, etc)."""
from __future__ import annotations

from agentic_rag import cli as cli_mod


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
