"""Rich card for session retention results."""

from __future__ import annotations

from collections.abc import Mapping

from rich.panel import Panel
from rich.text import Text

from .. import theme


def render_retention_card(projection: Mapping[str, object]) -> Panel:
    mode = str(projection["mode"])
    lines = [f"retention · {mode}", f"days: {projection['days']}"]
    if projection.get("preview_id") is not None:
        lines.append(f"preview: {projection['preview_id']}")
    lines.append(f"sessions: {projection['count']}")
    if projection.get("bytes"):
        lines.append(f"bytes: {projection['bytes']}")
    errors = projection.get("errors") or []
    lines.extend(f"error: {error}" for error in errors)
    if projection.get("disable_pending"):
        lines.append("disable pending")
    return Panel(
        Text("\n".join(lines)),
        border_style=theme.WARNING if errors else theme.COMMAND,
        style=theme.CARD_BG,
        padding=(0, 1),
        expand=True,
    )
