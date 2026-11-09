"""Every example must run offline with the mock backend."""
import runpy
import sys
from pathlib import Path

import pytest

EX = Path(__file__).resolve().parents[1] / "examples"
SCRIPTS = sorted(p for p in EX.glob("[0-9][0-9]_*.py"))


def test_there_are_twelve_examples():
    assert len(SCRIPTS) == 12


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_example_runs_on_mock(script, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", [str(script), "--mock"])
    monkeypatch.syspath_prepend(str(EX))
    runpy.run_path(str(script), run_name="__main__")
    out = capsys.readouterr().out
    assert "offline mock" in out and ("accuracy" in out or "agreement" in out)
