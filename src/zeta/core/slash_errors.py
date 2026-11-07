"""Errors returned by slash input dispatch."""
from __future__ import annotations

import asyncio
import difflib
from collections.abc import Iterable
from pathlib import Path

from ..mcp.prompt_commands import SlashPromptError


async def unknown_slash_command(
    value: str, name: str, arguments: str, names: Iterable[str]
) -> SlashPromptError | None:
    """An unknown command is one line, plain-name, non-path input with no arguments or a close match; otherwise send it to the model unchanged."""
    if "\n" in value or "\r" in value or not name or not all(
        "a" <= character <= "z"
        or "A" <= character <= "Z"
        or "0" <= character <= "9"
        or character in "-_"
        for character in name
    ):
        return None
    if await asyncio.to_thread(Path("/" + name).exists):
        return None
    suggestion = difflib.get_close_matches(name, list(names), n=1, cutoff=0.6)
    if arguments and not suggestion:
        return None
    return SlashPromptError(
        f"unknown command /{name}. type // to send it as text."
        + (f" did you mean /{suggestion[0]}?" if suggestion else "")
    )
