"""Compact list control for subagent navigation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.controls import UIContent, UIControl

if TYPE_CHECKING:
    from ..agent_card import AgentNavigation

AGENT_LIST_PAGE_SIZE = 5
MAX_AGENT_LIST_ROWS = AGENT_LIST_PAGE_SIZE + 1


class AgentListControl(UIControl):
    """Focusable, compact list of the current agent and its children."""

    def __init__(self, navigator: AgentNavigation) -> None:
        self.navigator = navigator

    @property
    def is_focusable(self) -> bool:
        return True

    def preferred_height(
        self,
        width: int,
        max_available_height: int,
        wrap_lines: bool,
        get_line_prefix: Any,
    ) -> int:
        del width, max_available_height, wrap_lines, get_line_prefix
        self.navigator.refresh()
        return self.navigator.list_height

    def create_content(self, width: int, height: int | None) -> UIContent:
        del height
        self.navigator.refresh()
        entries = self.navigator.entries
        page_index = self.navigator.page_index
        page_start = page_index * AGENT_LIST_PAGE_SIZE
        page_entries = entries[page_start : page_start + AGENT_LIST_PAGE_SIZE]
        has_pager = len(entries) > AGENT_LIST_PAGE_SIZE
        list_focused = self.navigator.list_focused()

        def get_line(index: int) -> list[tuple[str, str]]:
            if index < len(page_entries):
                entry_index = page_start + index
                entry = page_entries[index]
                selected = list_focused and entry_index == self.navigator.selected_index
                marker = ">" if selected else " "
                label = f"{entry.label} · {entry.agent_type} · {entry.state}"
                style = "class:agent-list.selected" if selected else "class:agent-list"
                return [(style, f"{marker} {label}")]

            page_count = self.navigator.page_count
            pager = f"page {page_index + 1}/{page_count} · {len(entries)} agents"
            return [("class:agent-list", pager.rjust(width))]

        return UIContent(
            get_line=get_line,
            line_count=len(page_entries) + int(has_pager),
            cursor_position=Point(x=0, y=self.navigator.selected_index - page_start),
            show_cursor=False,
        )
