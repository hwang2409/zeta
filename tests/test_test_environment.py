from __future__ import annotations

import os
import subprocess
import sys
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


def test_teardown_guard_allows_external_live_history_writer(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    history.write_text("before\n", encoding="utf-8")
    live_home_write_guard.watch(live_home)

    try:
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "Path(__import__('sys').argv[1]).open('a').write('external\\n')"
                ),
                str(history),
            ],
            check=True,
        )

        live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()


def test_teardown_guard_rejects_test_process_live_home_write(
    tmp_path: Path, live_home_write_guard
) -> None:
    live_home = tmp_path / "live-home"
    history = live_home / "history"
    history.parent.mkdir()
    live_home_write_guard.watch(live_home)

    try:
        history.write_text("leak\n", encoding="utf-8")

        with pytest.raises(AssertionError, match="tests wrote to live zeta home"):
            live_home_write_guard.assert_clean()
    finally:
        live_home_write_guard.reset()
