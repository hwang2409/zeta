"""Shared terminal layout measurements."""

from __future__ import annotations

from collections.abc import Sequence

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import (
    AnyContainer,
    ConditionalContainer,
    HSplit,
    VSplit,
    Window,
)

from ..core.session import _preview_text
from ..core.store import ConversationStore
from .todo import TodoWidget


CONTENT_MARGIN = 2


def content_width(terminal_width: int) -> int:
    """Return the width between the app's two-column side margins."""

    return max(1, terminal_width - CONTENT_MARGIN * 2)


def resume_picker_line(value: str, width: int) -> str:
    return " " * CONTENT_MARGIN + _preview_text(value, limit=width)


def full_screen_content(
    transcript: AnyContainer,
    composer_rows: Sequence[AnyContainer],
    footer: AnyContainer,
    todo_widget: TodoWidget,
    store: ConversationStore,
) -> VSplit:
    todo_panel = ConditionalContainer(
        Window(content=todo_widget, height=Dimension(min=0, max=7)),
        Condition(lambda: bool(store.todo_items())),
    )
    bottom = HSplit(
        [todo_panel, *composer_rows, footer], height=Dimension(min=4, max=10)
    )
    content = HSplit([transcript, bottom])
    return VSplit(
        [
            Window(width=CONTENT_MARGIN, char=" "),
            content,
            Window(width=CONTENT_MARGIN, char=" "),
        ]
    )
