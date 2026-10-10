"""View model and bounded control for the decisions popup.

:class:`DecisionsPanel` is a pure, testable view model. It turns an immutable
tuple of :class:`~zeta.attention_decisions.DecisionItem` into finder-style
overlay lines and owns the small amount of interaction state the popup needs:
which item is selected and, while a custom answer is being typed, the answer
text. It never touches storage, the inbox, or a session; the surrounding mixin
performs delivery and fork navigation. Styling uses only the shared theme
tokens, so tool cards and the active palette are untouched.
"""

from __future__ import annotations

from collections.abc import Sequence

from prompt_toolkit.utils import get_cwidth

from ...attention_decisions import DecisionItem
from .. import overlay, theme

# The popup settles near, but not beyond, the shared overlay cap so the text
# screenshots stay stable regardless of the live terminal size.
CONTENT_WIDTH = 68
_WHY_LINES = 2

Fragment = tuple[str, str]
FragmentLine = list[Fragment]


def _cells(text: str) -> int:
    return sum(max(0, get_cwidth(character)) for character in text)


def _wrap(text: str, width: int, limit: int) -> list[str]:
    """Wrap ``text`` to at most ``limit`` lines of ``width`` cells, ellipsised."""

    width = max(1, width)
    words = " ".join(text.split()).split(" ")
    lines: list[str] = []
    current = ""
    for word in words:
        piece = word if not current else f"{current} {word}"
        if _cells(piece) <= width:
            current = piece
            continue
        if current:
            lines.append(current)
        current = word
        if len(lines) == limit:
            break
    if len(lines) < limit and current:
        lines.append(current)
    if not lines:
        return [""]
    if (len(lines) == limit and current and lines[-1] != current) or _cells(
        lines[-1]
    ) > width:
        last = lines[-1]
        while last and _cells(f"{last}…") > width:
            last = last[:-1]
        lines[-1] = f"{last}…"
    return lines


class DecisionsPanel:
    """Interactive state for the open-decisions list and the answer field."""

    def __init__(self) -> None:
        self._items: tuple[DecisionItem, ...] = ()
        self._selected_id: str | None = None
        self._mode: str = "list"
        self._answer: str = ""

    # -- snapshot ingestion -------------------------------------------------

    def set_items(self, items: Sequence[DecisionItem]) -> None:
        self._items = tuple(items)
        ids = [item.record.id for item in self._items]
        if self._selected_id not in ids:
            self._selected_id = ids[0] if ids else None
            if self._mode == "answer":
                self._mode, self._answer = "list", ""

    # -- read-only state ----------------------------------------------------

    @property
    def items(self) -> tuple[DecisionItem, ...]:
        return self._items

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def answer_text(self) -> str:
        return self._answer

    @property
    def selected(self) -> DecisionItem | None:
        for item in self._items:
            if item.record.id == self._selected_id:
                return item
        return None

    # -- navigation ---------------------------------------------------------

    def move(self, delta: int) -> None:
        if self._mode != "list" or not self._items:
            return
        ids = [item.record.id for item in self._items]
        index = ids.index(self._selected_id) if self._selected_id in ids else 0
        index = min(max(0, index + delta), len(ids) - 1)
        self._selected_id = ids[index]

    def pick_option(self, number: int) -> str | None:
        """Return the 1-based offered option for the selection, or ``None``."""

        item = self.selected
        if item is None or number < 1 or number > len(item.record.options):
            return None
        return item.record.options[number - 1]

    # -- answer field -------------------------------------------------------

    def enter_answer(self) -> bool:
        if self.selected is None:
            return False
        self._mode = "answer"
        self._answer = ""
        return True

    def set_answer(self, text: str) -> None:
        self._answer = text

    def exit_answer(self) -> None:
        self._mode = "list"
        self._answer = ""

    # -- rendering ----------------------------------------------------------

    def render_lines(self) -> list[FragmentLine]:
        lines: list[FragmentLine] = [
            overlay.title("Decisions"),
            overlay.rule(),
        ]
        if not self._items:
            lines.append(overlay.value("No open decisions.", theme.DIM))
            lines.append(overlay.rule())
            lines.append(overlay.hint("esc close"))
            return lines
        for item in self._items:
            lines.append(self._list_row(item))
        lines.append(overlay.rule())
        selection = self.selected
        if selection is not None:
            lines.extend(self._detail(selection))
        if self._mode == "answer":
            lines.append(self._answer_line())
            lines.append(overlay.hint("enter send · esc cancel"))
        else:
            lines.append(
                overlay.hint(
                    "↑/↓ select · 1-9 answer · a type answer · enter discuss · esc close"
                )
            )
        return lines

    def _list_row(self, item: DecisionItem) -> FragmentLine:
        selected = item.record.id == self._selected_id
        label = f"{item.project_name} · {item.session_id[:8]}"
        return overlay.row(
            [
                (theme.BODY, item.record.title),
                (theme.DIM, f"  {label}"),
            ],
            selected=selected,
        )

    def _detail(self, item: DecisionItem) -> list[FragmentLine]:
        lines: list[FragmentLine] = []
        for wrapped in _wrap(item.record.why, CONTENT_WIDTH - 2, _WHY_LINES):
            lines.append([overlay.value(f"  {wrapped}", theme.DIM)])
        for index, option in enumerate(item.record.options, start=1):
            lines.append(
                [
                    overlay.value(f"  {index}. ", theme.ACCENT),
                    overlay.value(option),
                ]
            )
        lines.append(overlay.rule())
        return lines

    def _answer_line(self) -> FragmentLine:
        return [
            overlay.value("answer ❯ ", theme.ACCENT),
            overlay.value(self._answer),
            (theme.prompt_toolkit_style(f"bold {theme.ACCENT}"), "▌"),
        ]


__all__ = ["CONTENT_WIDTH", "DecisionsPanel"]
