"""Pinned todo-list control for the full-screen terminal UI."""

from __future__ import annotations

from prompt_toolkit.formatted_text import StyleAndTextTuples
from prompt_toolkit.layout.controls import (
    GetLinePrefixCallable,
    UIContent,
    UIControl,
)

from ..core.store import ConversationStore
from ..core.todo import TodoItem
from .theme import ACCENT, BODY, DIM


VISIBLE_ROWS = 6


class TodoWidget(UIControl):
    """Render the current session todo list without owning its state."""

    def __init__(self, store: ConversationStore) -> None:
        self.store = store
        self._last_revision = store.todo_revision

    def turn_boundary(self) -> None:
        """Hide a completed receipt when the next model turn begins."""

        self._sync_state()
        if self._is_terminal(self.store.todo_items()):
            self.store.dismiss_todo()

    @property
    def visible(self) -> bool:
        """Return whether the widget has content to render."""

        self._sync_state()
        return bool(self.store.todo_items()) and not self.store.todo_dismissed

    def _sync_state(self) -> None:
        revision = self.store.todo_revision
        if revision != self._last_revision:
            self._last_revision = revision

    @staticmethod
    def _is_terminal(items: list[TodoItem]) -> bool:
        return bool(items) and all(
            item["status"] not in {"pending", "in_progress"} for item in items
        )

    def _render_lines(
        self, width: int, max_height: int | None = None
    ) -> list[StyleAndTextTuples]:
        self._sync_state()
        items = self.store.todo_items()
        if self.store.todo_dismissed:
            return []
        if self._is_terminal(items):
            return [[(f"fg:{DIM}", f"todos done ({len(items)})")]]
        if max_height is not None and max_height <= 0:
            return []
        visible_rows = min(VISIBLE_ROWS, len(items))
        if len(items) > VISIBLE_ROWS and max_height is not None:
            visible_rows = min(visible_rows, max(0, max_height - 1))
        lines = [self._render_item(item, width) for item in items[:visible_rows]]
        remaining = len(items) - visible_rows
        if remaining > 0:
            lines.append([(f"fg:{DIM}", f"+{remaining} more")])
        return lines

    @staticmethod
    def _render_item(item: TodoItem, width: int) -> StyleAndTextTuples:
        glyphs = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
            "canceled": "[-]",
        }
        glyph_style = ACCENT if item["status"] == "in_progress" else DIM
        prefix = f"{glyphs[item['status']]} "
        available = max(1, width - len(prefix))
        content = item["content"]
        if len(content) > available:
            content = content[: max(1, available - 1)] + "…"
        return [(f"fg:{glyph_style}", prefix), (f"fg:{BODY}", content)]

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self._render_lines(max(1, width), max_height=height)

        def get_line(index: int) -> StyleAndTextTuples:
            return lines[index]

        return UIContent(get_line=get_line, line_count=len(lines), show_cursor=False)

    def preferred_width(self, max_available_width: int) -> int | None:
        return max_available_width

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: GetLinePrefixCallable | None,
    ) -> int | None:
        del wrap_lines, get_line_prefix
        return min(max_available_height, len(self._render_lines(max(1, width))))
