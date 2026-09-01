"""Shared terminal layout measurements."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from prompt_toolkit.filters import Condition
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import (
    AnyContainer,
    ConditionalContainer,
    Container,
    HSplit,
    VSplit,
    Window,
    to_container,
)
from prompt_toolkit.layout.mouse_handlers import MouseHandler, MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from ..core.session import _preview_text
from ..core.store import ConversationStore
from .todo import TodoWidget


CONTENT_MARGIN = 2


def content_width(terminal_width: int) -> int:
    """Return the width between the app's two-column side margins."""

    return max(1, terminal_width - CONTENT_MARGIN * 2)


def resume_picker_line(value: str, width: int) -> str:
    return " " * CONTENT_MARGIN + _preview_text(value, limit=width)


class WheelRouter(Container):
    """Send wheel ticks over a child region to the transcript instead.

    prompt_toolkit hands a mouse event to whichever window sits under the
    pointer, so a wheel tick over the composer would scroll the input box
    rather than the conversation above it. Wrapping the bottom chrome keeps
    the wheel pointed at the transcript wherever the pointer rests.
    """

    def __init__(
        self,
        content: AnyContainer,
        *,
        on_scroll_up: Callable[[], None],
        on_scroll_down: Callable[[], None],
    ) -> None:
        self.content = to_container(content)
        self._on_scroll_up = on_scroll_up
        self._on_scroll_down = on_scroll_down

    def reset(self) -> None:
        self.content.reset()

    def preferred_width(self, max_available_width: int) -> Dimension:
        return self.content.preferred_width(max_available_width)

    def preferred_height(self, width: int, max_available_height: int) -> Dimension:
        return self.content.preferred_height(width, max_available_height)

    def write_to_screen(
        self,
        screen: Screen,
        mouse_handlers: MouseHandlers,
        write_position: WritePosition,
        parent_style: str,
        erase_bg: bool,
        z_index: int | None,
    ) -> None:
        self.content.write_to_screen(
            screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
        )
        self._claim_wheel(mouse_handlers, write_position)

    def _claim_wheel(
        self, mouse_handlers: MouseHandlers, write_position: WritePosition
    ) -> None:
        """Wrap the handlers the child just registered over its own region."""

        wrapped: dict[int, MouseHandler] = {}
        rows = mouse_handlers.mouse_handlers
        y_range = range(write_position.ypos, write_position.ypos + write_position.height)
        x_range = range(write_position.xpos, write_position.xpos + write_position.width)
        for y in y_range:
            row = rows[y]
            for x in x_range:
                inner = row[x]
                handler = wrapped.get(id(inner))
                if handler is None:
                    handler = wrapped[id(inner)] = self._wheel_handler(inner)
                row[x] = handler

    def _wheel_handler(self, inner: MouseHandler) -> MouseHandler:
        def handle(mouse_event: MouseEvent):
            if mouse_event.event_type is MouseEventType.SCROLL_UP:
                self._on_scroll_up()
                return None
            if mouse_event.event_type is MouseEventType.SCROLL_DOWN:
                self._on_scroll_down()
                return None
            return inner(mouse_event)

        return handle

    def get_children(self) -> list[Container]:
        return [self.content]


def full_screen_content(
    transcript: AnyContainer,
    composer_rows: Sequence[AnyContainer],
    footer: AnyContainer,
    todo_widget: TodoWidget,
    store: ConversationStore,
    *,
    on_scroll_up: Callable[[], None],
    on_scroll_down: Callable[[], None],
) -> VSplit:
    todo_panel = ConditionalContainer(
        Window(content=todo_widget, height=Dimension(min=0, max=7)),
        Condition(lambda: bool(store.todo_items())),
    )
    bottom = WheelRouter(
        HSplit([todo_panel, *composer_rows, footer], height=Dimension(min=4, max=11)),
        on_scroll_up=on_scroll_up,
        on_scroll_down=on_scroll_down,
    )
    content = HSplit([transcript, bottom])
    return VSplit(
        [
            Window(width=CONTENT_MARGIN, char=" "),
            content,
            Window(width=CONTENT_MARGIN, char=" "),
        ]
    )
