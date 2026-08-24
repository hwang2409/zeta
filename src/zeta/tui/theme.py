"""Shared visual tokens for the inline terminal UI."""

from __future__ import annotations

from rich.theme import Theme


SURFACE = "#3a3533"
TINT = "#302c2a"
ACCENT = "#ff8a1f"
WARM_ACCENT = "#ffb454"
DIM = "#9f9488"
BODY = "#f4e8c8"
ERROR = "bold #ff5c57"
OK = "#a8c76f"
USER_PREFIX = f"bold {ACCENT}"
CODE_BG = "#24211f"
CODE_THEME = "monokai"
CHROME = DIM
CARD_BG = f"on {TINT}"
CARD_BORDER = "#5a4d45"
COMMAND = f"bold {ACCENT}"
RECEIPT = f"dim {ACCENT}"
THOUGHT = f"italic {WARM_ACCENT}"
AFFORDANCE = f"dim {DIM}"
USER_ROLE = ACCENT

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
        "zeta.card": CARD_BG,
        "zeta.card.border": CARD_BORDER,
        "zeta.command": COMMAND,
        "zeta.receipt": RECEIPT,
        "zeta.thought": THOUGHT,
        "zeta.affordance": AFFORDANCE,
    }
)
