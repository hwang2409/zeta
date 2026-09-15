"""Arrow-key model picker behind ``/model``.

Typing a model id by hand is error-prone: ``/model opus`` used to reach the
provider verbatim and 404. ``/model`` now draws a card listing the models
the current provider serves (the static table plus the live catalog once it
loads) and lets the arrow keys choose. ``/model <text>`` narrows the list to
substring matches and switches directly when exactly one remains.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Group
from rich.panel import Panel
from rich.text import Text

from ...core.slash import context_window
from .. import theme

MAX_VISIBLE_ROWS = 12
HINT = "↑/↓ move · enter select · esc cancel · or /model <name>"


def _format_window(tokens: int) -> str:
    if tokens >= 1_000_000 and tokens % 1_000_000 == 0:
        return f"{tokens // 1_000_000}M"
    return f"{tokens // 1_000}K"


@dataclass
class ModelPicker:
    """Selection state for one open picker card."""

    provider: str
    choices: tuple[str, ...]
    current: str
    query: str = ""
    index: int = 0

    def __post_init__(self) -> None:
        if not self.choices:
            raise ValueError("model picker needs at least one choice")
        if self.current in self.choices:
            self.index = self.choices.index(self.current)

    @property
    def selected(self) -> str:
        return self.choices[self.index]

    def move(self, delta: int) -> None:
        self.index = (self.index + delta) % len(self.choices)

    def with_choices(self, choices: tuple[str, ...]) -> ModelPicker:
        """Rebuild after a catalog refresh, keeping the highlighted model."""

        picker = ModelPicker(self.provider, choices, self.current, self.query)
        if self.selected in choices:
            picker.index = choices.index(self.selected)
        return picker

    def _window(self) -> tuple[int, int]:
        total = len(self.choices)
        if total <= MAX_VISIBLE_ROWS:
            return 0, total
        start = min(max(0, self.index - MAX_VISIBLE_ROWS // 2), total - MAX_VISIBLE_ROWS)
        return start, start + MAX_VISIBLE_ROWS

    def render(self) -> Panel:
        header = Text.assemble(
            ("select a model", theme.COMMAND), (f" · {self.provider}", theme.DIM)
        )
        if self.query:
            header.append(f' · matching "{self.query}"', style=theme.DIM)
        rows: list[Text] = [header]
        start, stop = self._window()
        if start:
            rows.append(Text(f"  … {start} more above", style=theme.DIM))
        width = max(len(name) for name in self.choices[start:stop])
        for position in range(start, stop):
            name = self.choices[position]
            highlighted = position == self.index
            line = Text.assemble(
                ("❯ " if highlighted else "  ", theme.ACCENT),
                (name.ljust(width), theme.COMMAND if highlighted else theme.BODY),
            )
            window = context_window(self.provider, name)
            if window:
                line.append(f"  {_format_window(window)}", style=theme.DIM)
            if name == self.current:
                line.append("  current", style=theme.DIM)
            rows.append(line)
        if stop < len(self.choices):
            rows.append(
                Text(f"  … {len(self.choices) - stop} more below", style=theme.DIM)
            )
        rows.append(Text(HINT, style=theme.AFFORDANCE))
        return Panel(
            Group(*rows),
            border_style=theme.ACCENT,
            style=theme.CARD_BG,
            padding=(0, 1),
            expand=True,
        )


__all__ = ["HINT", "MAX_VISIBLE_ROWS", "ModelPicker"]
