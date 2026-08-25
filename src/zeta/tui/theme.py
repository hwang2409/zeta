"""Shared visual tokens for the full-screen terminal UI."""

from __future__ import annotations

from rich.theme import Theme


# Keep surfaces transparent so zeta inherits the terminal theme.
SURFACE = ""
TINT = ""
ACCENT = "#ff8a1f"
DIM = "#8e938b"
BODY = "#f1f2ed"
ERROR = "bold #ff5c57"
CODE_BG = None
CODE_THEME = "monokai"
CHROME = DIM
CARD_BG = ""
CARD_BORDER = "#50544d"
COMPOSER_BORDER = "#50544d"
COMPOSER_FOCUS = ACCENT
VIM_STATE = f"bold {ACCENT}"
COMMAND = f"bold {ACCENT}"
RECEIPT = CHROME
THOUGHT = f"italic {CHROME}"
AFFORDANCE = f"dim {DIM}"
USER_ROLE = ACCENT

RICH_THEME = Theme(
    {
        "markdown.paragraph": BODY,
        "markdown.h1": f"bold {BODY}",
        "markdown.h2": f"bold {BODY}",
        "markdown.h3": BODY,
        "markdown.link": f"{BODY} underline",
        "markdown.link_url": f"{CHROME} underline",
        "markdown.code": BODY,
        "markdown.code_block": BODY,
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
