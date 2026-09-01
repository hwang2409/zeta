from __future__ import annotations

import os
from io import StringIO
from pathlib import Path

import pytest
from rich.console import Console

LIVE_ZETA_HOME = Path.home() / ".zeta"


def test_console_defaults_are_suite_stable() -> None:
    console = Console(file=StringIO())

    assert console.is_terminal
    assert console.color_system == "truecolor"
    assert not console.no_color


def test_test_home_is_isolated() -> None:
    assert Path(os.environ["ZETA_HOME"]) != LIVE_ZETA_HOME


def test_path_home_follows_home_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = tmp_path / "home"
    monkeypatch.setenv("HOME", str(expected))

    assert Path.home() == expected
