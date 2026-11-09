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
from ...attention_records import Broken
from .. import overlay, theme
from .shared import wrap_by_cells

# The popup settles near, but not beyond, the shared overlay cap so the text
# screenshots stay stable regardless of the live terminal size.
CONTENT_WIDTH = 68
# A long custom answer is cut in the row; the staged reply keeps the full text.
_STAGED_REPLY_WIDTH = 16

Fragment = tuple[str, str]
FragmentLine = list[Fragment]


def _cells(text: str) -> int:
    return sum(max(0, get_cwidth(character)) for character in text)


def _truncate_text(text: str, width: int) -> str:
    if width <= 1:
        return "…"[:width]
    result = ""
    for character in text:
        if _cells(result + character + "…") > width:
            break
        result += character
    return result + "…"


def _labelled_text(label: str, text: str, width: int, style: str) -> list[FragmentLine]:
    lines = wrap_by_cells(text, width - len(label) - 2)
    return [
        [overlay.value(f"{label}: ", style), overlay.value(line)]
        if index == 0
        else [overlay.value(" " * (len(label) + 2)), overlay.value(line)]
        for index, line in enumerate(lines)
    ]


def _option_lines(index: int, text: str, width: int) -> list[FragmentLine]:
    prefix = f"{index}. "
    return [
        [
            overlay.value(
                prefix if line_index == 0 else " " * len(prefix), theme.ACCENT
            ),
            overlay.value(line),
        ]
        for line_index, line in enumerate(wrap_by_cells(text, width - len(prefix)))
    ]


class DecisionsPanel:
    """Interactive state for the open-decisions list and the answer field."""

    def __init__(self) -> None:
        self._items: tuple[DecisionItem, ...] = ()
        self._staged: dict[str, str] = {}
        self._selected_id: str | None = None
        self._mode: str = "list"
        self._answer: str = ""

    # -- snapshot ingestion -------------------------------------------------

    def set_items(
        self, items: Sequence[DecisionItem], staged: dict[str, str] | None = None
    ) -> None:
        """Show ``items``; ``staged`` maps an item id to its staged reply."""
        self._items = tuple(items)
        self._staged = dict(staged or {})
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
        if (
            item is None
            or isinstance(item.record, Broken)
            or number < 1
            or number > len(item.record.options)
        ):
            return None
        return item.record.options[number - 1]

    # -- answer field -------------------------------------------------------

    def enter_answer(self) -> bool:
        if self.selected is None or isinstance(self.selected.record, Broken):
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
            overlay.title("Attention"),
            overlay.rule(),
        ]
        if not self._items:
            lines.append([overlay.value("No open attention items.", theme.DIM)])
            lines.append(overlay.rule())
            lines.append(overlay.hint("esc close"))
            return lines
        for item in self._items:
            lines.append(self._list_row(item))
        lines.append(overlay.rule())
        selection = self.selected
        if selection is not None:
            lines.extend(self._detail(selection))
        if (
            selection is not None
            and not isinstance(selection.record, Broken)
            and selection.record.kind != "decision"
        ):
            lines.append(overlay.hint("d dismiss"))
        if self._mode == "answer":
            lines.append(self._answer_line())
            lines.append(overlay.hint("enter send · esc cancel"))
        return lines

    def _list_row(self, item: DecisionItem) -> FragmentLine:
        selected = item.record.id == self._selected_id
        if isinstance(item.record, Broken):
            label = f"broken · {item.record.id}"
            title = f"{item.record.path}: {item.record.reason}"
            return overlay.row([(theme.ERROR, title)], selected=selected)
        staged = self._staged.get(item.record.id)
        if staged is not None and _cells(staged) > _STAGED_REPLY_WIDTH:
            staged = _truncate_text(staged, _STAGED_REPLY_WIDTH)
        staged_label = "" if staged is None else f"staged: {staged} · "
        tail = f"{item.project_name} · {item.session_id[:8]}"
        label = f"{item.record.kind} · {staged_label}{tail}"
        title = item.record.title
        available = CONTENT_WIDTH - _cells(label) - 4
        if _cells(title) > available:
            title = _truncate_text(title, available)
        label_style = theme.ACCENT if item.record.kind == "decision" else theme.DIM
        fragments = [(theme.BODY, title), (label_style, f"  {item.record.kind} · ")]
        if staged_label:
            fragments.append((theme.SUCCESS, staged_label))
        fragments.append((label_style, tail))
        return overlay.row(fragments, selected=selected)

    def _detail(self, item: DecisionItem) -> list[FragmentLine]:
        width = CONTENT_WIDTH - 2
        if isinstance(item.record, Broken):
            return [
                overlay.blank(),
                *_labelled_text("State", "broken attention record", width, theme.ERROR),
                *_labelled_text("Id", item.record.id, width, theme.DIM),
                *_labelled_text("Path", item.record.path, width, theme.DIM),
                *_labelled_text("Reason", item.record.reason, width, theme.ERROR),
                overlay.rule(),
            ]
        lines: list[FragmentLine] = [overlay.blank()]
        lines.extend(
            _labelled_text(
                "Kind",
                item.record.kind,
                width,
                theme.ACCENT if item.record.kind == "decision" else theme.DIM,
            )
        )
        lines.extend(
            _labelled_text(
                "Question" if item.record.kind == "decision" else "Topic",
                item.record.title,
                width,
                theme.ACCENT,
            )
        )
        lines.append(overlay.blank())
        lines.extend(_labelled_text("Why", item.record.why, width, theme.DIM))
        if item.record.kind == "decision":
            lines.append(overlay.blank())
            lines.append([overlay.value("Options", theme.ACCENT)])
            for index, option in enumerate(item.record.options, start=1):
                lines.extend(_option_lines(index, option, width))
        if item.record.kind == "decision" and item.record.recommendation:
            lines.append(overlay.blank())
            lines.extend(
                _labelled_text(
                    "Recommendation", item.record.recommendation, width, theme.SUCCESS
                )
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
