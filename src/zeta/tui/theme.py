"""Shared visual tokens for the inline terminal UI."""

from __future__ import annotations

from rich.theme import Theme


ACCENT = "#62d8ff"
DIM = "dim"
BODY = "bright_white"
ERROR = "bold red"
OK = "green"
USER_PREFIX = "bold #62d8ff"
CODE_BG = "#1c1f2b"
CHROME = "#677083"

RICH_THEME = Theme(
    {
        "markdown.paragraph": BODY,
        "markdown.h1": f"bold {ACCENT}",
        "markdown.h2": f"bold {ACCENT}",
        "markdown.h3": ACCENT,
        "markdown.link": f"{ACCENT} underline",
        "markdown.link_url": f"{CHROME} underline",
        "markdown.code": f"bold {ACCENT} on {CODE_BG}",
        "markdown.code_block": f"{ACCENT} on {CODE_BG}",
        "markdown.table.border": CHROME,
        "markdown.table.header": f"bold {ACCENT}",
    }
)
