from __future__ import annotations

from io import StringIO

from rich.console import Console


def test_console_defaults_are_suite_stable() -> None:
    console = Console(file=StringIO())

    assert console.is_terminal
    assert console.color_system == "truecolor"
    assert not console.no_color
