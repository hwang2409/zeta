"""Shared visual tokens for the full-screen terminal UI."""

from __future__ import annotations

from rich.theme import Theme


SURFACE = "#0b0c0a"
TINT = "#151713"
ACCENT = "#ff8a1f"
DIM = "#8e938b"
BODY = "#f1f2ed"
ERROR = "bold #ff5c57"
CODE_BG = "#252724"
CODE_THEME = "monokai"
CHROME = DIM
CARD_BG = f"on {TINT}"
CARD_BORDER = "#50544d"
COMPOSER_BORDER = "#50544d"
COMPOSER_FOCUS = ACCENT
COMMAND = f"bold {ACCENT}"
RECEIPT = CHROME
THOUGHT = f"italic {CHROME}"
AFFORDANCE = f"dim {DIM}"
USER_ROLE = ACCENT
ASSISTANT_ROLE = CHROME

RICH_THEME = Theme(
    {
        "markdown.paragraph": BODY,
        "markdown.h1": f"bold {BODY}",
        "markdown.h2": f"bold {BODY}",
        "markdown.h3": BODY,
        "markdown.link": f"{BODY} underline",
        "markdown.link_url": f"{CHROME} underline",
        "markdown.code": f"{BODY} on {CODE_BG}",
        "markdown.code_block": f"{BODY} on {CODE_BG}",
        "markdown.table.border": CHROME,
        "markdown.table.header": f"bold {BODY}",
        "zeta.card": CARD_BG,
        "zeta.card.border": CARD_BORDER,
        "zeta.command": COMMAND,
        "zeta.receipt": RECEIPT,
        "zeta.thought": THOUGHT,
        "zeta.affordance": AFFORDANCE,
    }
)
