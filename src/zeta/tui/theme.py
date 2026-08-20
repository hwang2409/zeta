"""Shared visual tokens for the inline terminal UI."""

from __future__ import annotations

from rich.theme import Theme


ACCENT = "bright_cyan"
ACCENT_DIM = "cyan"
ASSISTANT_BODY = "bright_white"
CODE_BG = "#1c1f2b"
CHROME = "bright_black"
ERROR = "bold red"
OK = "green"
THINKING = "dim italic"
TOOL_RESULT = "dim"
USER_PREFIX = "bold bright_cyan"
PROMPT_ACCENT = "#62d8ff bold"
PROMPT_CHROME = "#677083 #101218"

RICH_THEME = Theme(
    {
        "markdown.paragraph": ASSISTANT_BODY,
        "markdown.h1": f"bold {ACCENT}",
        "markdown.h2": "bold magenta",
        "markdown.h3": "magenta",
        "markdown.link": f"{ACCENT} underline",
        "markdown.link_url": "cyan underline",
        "markdown.code": f"bold {ACCENT} on {CODE_BG}",
        "markdown.code_block": f"{ACCENT_DIM} on {CODE_BG}",
        "markdown.table.border": ACCENT_DIM,
        "markdown.table.header": f"bold {ACCENT}",
    }
)
