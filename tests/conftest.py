from __future__ import annotations

import os
from collections.abc import Generator

import pytest
from rich.console import Console


@pytest.fixture(autouse=True)
def stable_terminal_defaults(
    request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch
) -> Generator[None, None, None]:
    """Keep Rich output stable unless a test explicitly probes the environment."""
    if request.node.get_closest_marker("environment_sensitive") is not None:
        yield
        return

    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    assert os.environ["TERM"] == "xterm-256color"
    assert os.environ["COLORTERM"] == "truecolor"

    original_init = Console.__init__

    def init_with_stable_defaults(self: Console, *args: object, **kwargs: object) -> None:
        if kwargs.get("force_terminal") is not False:
            kwargs.setdefault("force_terminal", True)
            kwargs.setdefault("color_system", "truecolor")
            kwargs.setdefault("no_color", False)
        original_init(self, *args, **kwargs)

    monkeypatch.setattr(Console, "__init__", init_with_stable_defaults)

    yield
