"""Shared terminal layout measurements."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from prompt_toolkit.application.current import get_app
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
from rich.cells import cell_len
from ..core.session import _preview_text
from ..core.slash import context_window
from ..core.store import ConversationStore
from .agent_card import AgentNavigation
from .composer import status_formatted_text, vim_state_label
from .render import format_status
from .todo import TODO_PAD_BOTTOM, VISIBLE_ROWS, TodoWidget


CONTENT_MARGIN = 2
COMPOSER_CONTENT_PADDING = 1
COMPOSER_PAD_Y = 1
COMMAND_MENU_ROWS = 12
MAX_CHROME_ROWS = 18
STATUS_CARD_MAX_WIDTH = 72
STATUS_CARD_MARGIN = 2


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


class ComposerPadding(Container):
    """Add collapsible filled rows around the prompt window."""

    def __init__(self, content: AnyContainer) -> None:
        self.content = to_container(content)
        self._pad = Window(char=" ", style="class:text-area")

    def reset(self) -> None:
        self.content.reset()
        self._pad.reset()

    def preferred_width(self, max_available_width: int) -> Dimension:
        return self.content.preferred_width(max_available_width)

    def preferred_height(self, width: int, max_available_height: int) -> Dimension:
        content_height = self.content.preferred_height(width, max_available_height)
        return Dimension(
            min=content_height.min,
            preferred=content_height.preferred + 2 * COMPOSER_PAD_Y,
            max=content_height.max + 2 * COMPOSER_PAD_Y,
        )

    def write_to_screen(
        self,
        screen: Screen,
        mouse_handlers: MouseHandlers,
        write_position: WritePosition,
        parent_style: str,
        erase_bg: bool,
        z_index: int | None,
    ) -> None:
        content_height = self.content.preferred_height(
            write_position.width, write_position.height
        ).preferred
        pad_y = (
            COMPOSER_PAD_Y
            if write_position.height >= content_height + 2 * COMPOSER_PAD_Y
            else 0
        )
        if pad_y:
            self._write_pad(
                screen,
                mouse_handlers,
                write_position,
                parent_style,
                erase_bg,
                z_index,
                write_position.ypos,
            )
        content_height = max(1, write_position.height - 2 * pad_y)
        self.content.write_to_screen(
            screen,
            mouse_handlers,
            WritePosition(
                write_position.xpos,
                write_position.ypos + pad_y,
                write_position.width,
                content_height,
            ),
            parent_style,
            erase_bg,
            z_index,
        )
        if pad_y:
            self._write_pad(
                screen,
                mouse_handlers,
                write_position,
                parent_style,
                erase_bg,
                z_index,
                write_position.ypos + pad_y + content_height,
            )

    def _write_pad(
        self,
        screen: Screen,
        mouse_handlers: MouseHandlers,
        write_position: WritePosition,
        parent_style: str,
        erase_bg: bool,
        z_index: int | None,
        ypos: int,
    ) -> None:
        self._pad.write_to_screen(
            screen,
            mouse_handlers,
            WritePosition(
                write_position.xpos, ypos, write_position.width, COMPOSER_PAD_Y
            ),
            parent_style,
            erase_bg,
            z_index,
        )

    def get_children(self) -> list[Container]:
        return [self.content]


def command_menu_float(chrome_height: Callable[[], int]) -> Float:
    """The completion menu, constrained to the rows above the bottom chrome.

    A cursor-anchored float with only a bottom offset is allowed to extend past
    the top edge when its preferred height is larger than the available space.
    In a short terminal prompt-toolkit then paints the menu over the TODO,
    composer, and footer.  Limit the menu height to the space that its bottom
    anchor actually leaves; one row is still useful for keyboard completion and
    keeps the menu from corrupting the bottom chrome.
    """

    menu = CompletionsMenu(
        max_height=COMMAND_MENU_ROWS,
        scroll_offset=1,
        extra_filter=has_focus(DEFAULT_BUFFER),
    )

    def menu_height() -> int:
        output = get_app().output
        size = output.get_size()
        natural_height = menu.preferred_height(size.columns, size.rows).preferred
        available_height = size.rows - chrome_height()
        return max(1, min(natural_height, COMMAND_MENU_ROWS, available_height))

    return CommandMenuFloat(
        chrome_height,
        xcursor=True,
        height=menu_height,
        transparent=True,
        content=menu,
    )


def content_width(terminal_width: int) -> int:
    """Return the width between the app's two-column side margins."""

    return max(1, terminal_width - CONTENT_MARGIN * 2)


def composer_content_width(terminal_width: int) -> int:
    """Return the width between the composer's one-cell content insets."""

    return max(1, terminal_width - COMPOSER_CONTENT_PADDING * 2)


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
        content_height = self.content.preferred_height(width, max_available_height)
        height = max(content_height.min, min(content_height.preferred, MAX_CHROME_ROWS))
        return Dimension(min=content_height.min, preferred=height, max=height)

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


def status_card_float(
    status_window: AnyContainer, status_active: Callable[[], bool]
) -> Float:
    """Create a centered, content-sized status overlay.

    The explicit callables are important here: without them a float with only
    edge offsets receives the entire space between those edges.  The card is
    allowed to grow with its content, but never beyond a comfortable width or
    the terminal's usable height.  Its window then owns the remaining scroll.
    """

    content = ConditionalContainer(status_window, Condition(status_active))

    def card_width() -> int:
        terminal_width = get_app().output.get_size().columns
        natural_width = status_window.preferred_width(terminal_width).preferred
        return max(
            1,
            min(
                STATUS_CARD_MAX_WIDTH,
                natural_width,
                max(1, terminal_width - STATUS_CARD_MARGIN * 2),
            ),
        )

    def card_height() -> int:
        size = get_app().output.get_size()
        width = card_width()
        natural_height = status_window.preferred_height(width, size.rows).preferred
        return max(1, min(natural_height, max(1, size.rows - STATUS_CARD_MARGIN * 2)))

    return Float(content, width=card_width, height=card_height, z_index=10)


def status_toolbar(app: Any, terminal_width: int | None = None) -> list[tuple[str, str]]:
    """Build the status footer and pad it to the composer's content width."""
    terminal_width = terminal_width or get_app().output.get_size().columns
    width = composer_content_width(terminal_width)
    usage = dict(app._usage)
    usage.setdefault(
        "cache_read_input_tokens",
        app.loop.context_assembler.cache_read_input_tokens_this_session,
    )
    usage.setdefault(
        "cache_creation_input_tokens",
        app.loop.context_assembler.cache_creation_input_tokens_this_session,
    )
    status = format_status(
        app.provider,
        app.model,
        app._loop_state,
        usage,
        app._partial,
        session_id=app.loop.store.session_id[:8],
        token_count=app.loop.context_assembler.token_count,
        retained_tail=app.loop.context_assembler.retained_tail,
        streaming=app._streaming,
        width=width,
        spinner_frame=app._spinner_frame,
        spinner_active=app._spinner_active,
        model_window=context_window(app.provider, app.model),
        vim_state=vim_state_label(app.vim_mode),
        plan_state="PLAN" if app.loop.plan_mode else None,
        background_count=app.loop.tool_registry.background_tasks.running_count,
        undo_available=(
            app._undo_candidate is not None
            and app.active
            and app._loop_state in {"streaming", "compacting", "tool-running", "approval"}
        ),
        transcript_navigation=app._full_screen_active(),
        transcript_search=(
            app._transcript.search_query if app._transcript.search_active else None
        ),
        transcript_match=app._transcript.search_status(),
        transcript_position=app._transcript.position_indicator(),
        copy_notice=app._transcript.copy_notice,
        approval_mode=(
            app._approval_policy.default.value if app._approval_policy is not None else None
        ),
        cwd=app.loop.store.cwd,
    )
    fragments = status_formatted_text(status)
    status_width = cell_len(status.plain)
    if status_width < width:
        fragments.append(("class:status-bar", " " * (width - status_width - 1) + "·"))
    return fragments


def full_screen_content(
    transcript: AnyContainer,
    composer_rows: Sequence[AnyContainer],
    footer: AnyContainer,
    todo_widget: TodoWidget,
    store: ConversationStore,
    *,
    agent_navigation: AgentNavigation | None = None,
    on_scroll_up: Callable[[], None],
    on_scroll_down: Callable[[], None],
    status_window: AnyContainer | None = None,
    status_active: Callable[[], bool] | None = None,
    tasks_window: AnyContainer | None = None,
    tasks_active: Callable[[], bool] | None = None,
) -> FloatContainer:
    """Transcript over composer chrome, with the command menu floating above it."""

    todo_panel = ConditionalContainer(
        Window(
            content=todo_widget,
            height=Dimension(min=0, max=VISIBLE_ROWS + TODO_PAD_BOTTOM + 1),
        ),
        Condition(lambda: todo_widget.visible),
    )

    def agent_list_fits() -> bool:
        if agent_navigation is None or not agent_navigation.list_visible:
            return False
        output_size = get_app().output.get_size()
        required_rows = agent_navigation.list_height + 4
        if todo_widget.visible:
            todo_height = todo_panel.preferred_height(
                output_size.columns, output_size.rows
            ).preferred
            # Leave one extra row so HSplit does not squeeze the visible todo.
            required_rows += todo_height + 1
        if output_size.rows < required_rows:
            if agent_navigation.list_focused():
                agent_navigation.focus_composer()
            return False
        return True

    list_panel = (
        ConditionalContainer(
            agent_navigation.list_window,
            Condition(agent_list_fits),
        )
        if agent_navigation is not None
        else None
    )
    spacer = Window(height=1, char=" ")
    footer = to_container(footer)
    if isinstance(footer, Window):
        footer.width = Dimension(weight=1)
    footer = VSplit(
        [
            Window(width=COMPOSER_CONTENT_PADDING, char=" "),
            footer,
            Window(width=COMPOSER_CONTENT_PADDING, char=" "),
        ]
    )
    padded_composer_rows = list(composer_rows)
    if padded_composer_rows:
        padded_composer_rows[0] = ComposerPadding(padded_composer_rows[0])
    bottom_rows = [spacer, todo_panel, *padded_composer_rows]
    if list_panel is not None:
        bottom_rows.append(list_panel)
    bottom_rows.append(footer)
    bottom = WheelRouter(
        HSplit(bottom_rows),
        on_scroll_up=on_scroll_up,
        on_scroll_down=on_scroll_down,
    )
    transcript_content = VSplit(
        [
            Window(width=CONTENT_MARGIN, char=" "),
            transcript,
            Window(width=CONTENT_MARGIN, char=" "),
        ]
    )
    content = HSplit([transcript_content, bottom])
    if agent_navigation is not None:
        agent_navigation.bind_transcript_layout(content, transcript)
    floats = [command_menu_float(lambda: max(0, bottom.height - 1))]
    if status_window is not None and status_active is not None:
        floats.append(status_card_float(status_window, status_active))
    if tasks_window is not None and tasks_active is not None:
        floats.append(status_card_float(tasks_window, tasks_active))
    return FloatContainer(content, floats=floats)
