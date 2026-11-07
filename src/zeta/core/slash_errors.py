"""Errors returned by slash input dispatch."""
from __future__ import annotations

import difflib
from collections.abc import Iterable

from ..mcp.prompt_commands import SlashPromptError


def unknown_slash_command(name: str, names: Iterable[str]) -> SlashPromptError:
    suggestion = difflib.get_close_matches(name, list(names), n=1, cutoff=0.6)
    return SlashPromptError(
        f"unknown command /{name}. type // to send it as text."
        + (f" did you mean /{suggestion[0]}?" if suggestion else "")
    )
