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

    def _render_lines(self, width: int) -> list[StyleAndTextTuples]:
        items = self.store.todo_items()
        lines = [self._render_item(item, width) for item in items[:VISIBLE_ROWS]]
        remaining = len(items) - VISIBLE_ROWS
        if remaining > 0:
            lines.append([(f"fg:{DIM}", f"+{remaining} more")])
        return lines

    @staticmethod
    def _render_item(item: TodoItem, width: int) -> StyleAndTextTuples:
        glyphs = {
            "pending": "[ ]",
            "in_progress": "[>]",
            "completed": "[x]",
        }
        glyph_style = ACCENT if item["status"] == "in_progress" else DIM
        prefix = f"{glyphs[item['status']]} "
        available = max(1, width - len(prefix))
        content = item["content"]
        if len(content) > available:
            content = content[: max(1, available - 1)] + "…"
        return [(f"fg:{glyph_style}", prefix), (f"fg:{BODY}", content)]

    def create_content(self, width: int, height: int) -> UIContent:
        lines = self._render_lines(max(1, width))[: max(0, height)]

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
