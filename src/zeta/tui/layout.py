"""Shared terminal layout measurements."""

from __future__ import annotations

from collections.abc import Callable, Sequence

from prompt_toolkit.enums import DEFAULT_BUFFER
from prompt_toolkit.filters import Condition, has_focus
from prompt_toolkit.layout import Dimension
from prompt_toolkit.layout.containers import (
    AnyContainer,
    ConditionalContainer,
    Container,
    Float,
    FloatContainer,
    HSplit,
    VSplit,
    Window,
    to_container,
)
from prompt_toolkit.layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from prompt_toolkit.layout.mouse_handlers import MouseHandler, MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseEvent, MouseEventType

from ..core.session import _preview_text
from ..core.store import ConversationStore
from .todo import TodoWidget


CONTENT_MARGIN = 2
COMMAND_MENU_ROWS = 12


def detach_completion_menus(container: Container) -> None:
    """Strip prompt-toolkit's default completion floats from a layout subtree.

    PromptSession pins its menus inside the input's own FloatContainer, so in
    the full-screen layout they can only draw within the composer box, where
    a handful of rows squeeze the command list. :func:`command_menu_float`
    re-homes the menu on a container that spans the screen.
    """

    if isinstance(container, FloatContainer):
        container.floats[:] = [
            float_
            for float_ in container.floats
            if not isinstance(
                float_.content, (CompletionsMenu, MultiColumnCompletionsMenu)
            )
        ]
    for child in container.get_children():
        detach_completion_menus(child)


class CommandMenuFloat(Float):
    """Completion menu pinned to the rows directly above the composer chrome.

    prompt-toolkit's cursor-anchored float opens below the cursor whenever
    the rows fit, which here means over the composer and footer. Anchoring
    the float's bottom edge to the chrome's live height keeps the menu above
    the composer however many rows it needs, still aligned to the cursor
    column so it grows out of the `/` that opened it.
    """

    def __init__(self, chrome_height: Callable[[], int], **kwargs: object) -> None:
        self._chrome_height = chrome_height
        super().__init__(**kwargs)  # type: ignore[arg-type]

    @property
    def bottom(self) -> int:
        return self._chrome_height()

    @bottom.setter
    def bottom(self, value: int | None) -> None:
        # Float.__init__ stores its argument here; the live height wins.
        del value


def command_menu_float(chrome_height: Callable[[], int]) -> Float:
    """The slash-command menu, free to cover the transcript above the composer."""

    return CommandMenuFloat(
        chrome_height,
        xcursor=True,
        transparent=True,
        content=CompletionsMenu(
            max_height=COMMAND_MENU_ROWS,
            scroll_offset=1,
            extra_filter=has_focus(DEFAULT_BUFFER),
        ),
    )


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
        # Rows the chrome took on the last paint; the command menu floats
        # directly above it.
        self.height = 0

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
        self.height = write_position.height
        self.content.write_to_screen(
            screen, mouse_handlers, write_position, parent_style, erase_bg, z_index
        )
        self._claim_wheel(mouse_handlers, write_position)

    def _claim_wheel(
        self, mouse_handlers: MouseHandlers, write_position: WritePosition
    ) -> None:
        """Wrap the handlers the child just registered over its own region."""

        # One wrapper per distinct child handler rather than one per cell. The
        # cache holds each wrapper, which closes over its handler, so the ids
        # it is keyed by cannot be reused underneath it.
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
) -> FloatContainer:
    """Transcript over composer chrome, with the command menu floating above it."""

    todo_panel = ConditionalContainer(
        Window(content=todo_widget, height=Dimension(min=0, max=7)),
        Condition(lambda: todo_widget.visible),
    )
    bottom = WheelRouter(
        HSplit([todo_panel, *composer_rows, footer], height=Dimension(min=4, max=11)),
        on_scroll_up=on_scroll_up,
        on_scroll_down=on_scroll_down,
    )
    content = HSplit([transcript, bottom])
    padded = VSplit(
        [
            Window(width=CONTENT_MARGIN, char=" "),
            content,
            Window(width=CONTENT_MARGIN, char=" "),
        ]
    )
    return FloatContainer(padded, floats=[command_menu_float(lambda: bottom.height)])
