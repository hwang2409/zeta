"""The slash-command menu: palette colours, a float over the transcript, click to pick."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

from prompt_toolkit.application.current import set_app
from prompt_toolkit.buffer import CompletionState
from prompt_toolkit.completion import Completion
from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import Container, FloatContainer
from prompt_toolkit.layout.menus import CompletionsMenu, MultiColumnCompletionsMenu
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from rich.console import Console

from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.tui import theme
from zeta.tui.app import FullScreenPromptSession, TUIApp
from zeta.tui.layout import COMMAND_MENU_ROWS, CommandMenuFloat

WIDTH, HEIGHT = 80, 24


def _app(tmp_path: Path) -> tuple[TUIApp, FullScreenPromptSession]:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
        history_path=tmp_path / "history",
    )
    session = app._make_session()
    app._active_session = session
    return app, session


def _completion_menus(container: Container) -> list[type]:
    found: list[type] = []
    if isinstance(container, FloatContainer):
        found.extend(
            type(float_.content)
            for float_ in container.floats
            if isinstance(float_.content, (CompletionsMenu, MultiColumnCompletionsMenu))
        )
    for child in container.get_children():
        found.extend(_completion_menus(child))
    return found


def _render(session: FullScreenPromptSession) -> tuple[Screen, MouseHandlers]:
    screen = Screen()
    handlers = MouseHandlers()
    session.layout.update_parents_relations()
    session.layout.container.write_to_screen(
        screen, handlers, WritePosition(0, 0, WIDTH, HEIGHT), "", False, None
    )
    screen.draw_all_floats()
    return screen, handlers


def _rows(screen: Screen) -> list[str]:
    return [
        "".join(screen.data_buffer[y][x].char for x in range(WIDTH)) for y in range(HEIGHT)
    ]


def _open_menu(session: FullScreenPromptSession) -> None:
    # Needs a running loop: setting text schedules prompt-toolkit's validator.
    buffer = session.default_buffer
    buffer.text = "/mo"
    buffer.cursor_position = 3
    buffer.complete_state = CompletionState(
        original_document=buffer.document,
        completions=[
            Completion("model", start_position=-2, display="/model", display_meta="pick a model"),
            Completion("mcp", start_position=-2, display="/mcp", display_meta="show MCP status"),
        ],
        complete_index=0,
    )


def test_menu_styles_follow_the_palette(tmp_path: Path) -> None:
    app, session = _app(tmp_path)
    palette = theme.active_palette()

    with set_app(session.app):
        rules = dict(app._prompt_style().style_rules)

    assert rules["completion-menu"] == f"bg:{palette.search_bg} fg:{palette.body}"
    assert rules["completion-menu.completion.current"] == (
        f"bg:{palette.accent} fg:{palette.on_accent} bold"
    )
    assert rules["completion-menu.meta.completion"] == (
        f"bg:{palette.search_bg} fg:{palette.dim}"
    )
    assert theme.LIGHT.on_accent == "#ffffff"
    assert theme.DARK.on_accent == "#000000"


def test_full_screen_layout_moves_the_menu_out_of_the_composer(tmp_path: Path) -> None:
    app, session = _app(tmp_path)
    assert CompletionsMenu in _completion_menus(session.layout.container)

    app._install_full_screen_layout(session)

    root = session.layout.container.children[0]
    assert isinstance(root, FloatContainer)
    assert [type(float_.content) for float_ in root.floats] == [CompletionsMenu]
    assert _completion_menus(root.content) == []
    menu = root.floats[0]
    assert isinstance(menu, CommandMenuFloat)
    assert menu.xcursor and not menu.ycursor
    assert menu.content.content.height.max == COMMAND_MENU_ROWS


async def test_menu_sits_directly_above_the_composer_chrome(tmp_path: Path) -> None:
    app, session = _app(tmp_path)
    app._install_full_screen_layout(session)
    _open_menu(session)

    with set_app(session.app):
        screen, _ = _render(session)

    rows = _rows(screen)
    menu_bottom = max(y for y, row in enumerate(rows) if "/model" in row or "/mcp" in row)
    chrome_top = min(
        y for y, row in enumerate(rows) if "╭" in row or "> /mo" in row or "─" in row
    )
    assert menu_bottom + 1 == chrome_top
    assert menu_bottom == HEIGHT - 1 - session.layout.container.children[0].floats[0].bottom


async def test_menu_opens_above_the_composer_in_menu_colours(tmp_path: Path) -> None:
    app, session = _app(tmp_path)
    app._install_full_screen_layout(session)
    _open_menu(session)

    with set_app(session.app):
        screen, _ = _render(session)

    rows = _rows(screen)
    menu_rows = [y for y, row in enumerate(rows) if "/model" in row and "pick a model" in row]
    assert menu_rows, rows
    composer_row = next(y for y, row in enumerate(rows) if "> /mo" in row)
    assert menu_rows[0] < composer_row
    x = rows[menu_rows[0]].index("/model")
    cell = screen.data_buffer[menu_rows[0]][x]
    assert "completion-menu.completion.current" in cell.style
    mcp_row = next(y for y, row in enumerate(rows) if "/mcp" in row)
    assert "completion-menu.completion" in screen.data_buffer[mcp_row][x].style
    assert ".current" not in screen.data_buffer[mcp_row][x].style


async def test_clicking_a_menu_row_picks_that_command(tmp_path: Path) -> None:
    app, session = _app(tmp_path)
    app._install_full_screen_layout(session)
    _open_menu(session)

    with set_app(session.app):
        screen, handlers = _render(session)
        rows = _rows(screen)
        y = next(y for y, row in enumerate(rows) if "/mcp" in row)
        x = rows[y].index("/mcp") + 1
        handlers.mouse_handlers[y][x](
            MouseEvent(
                position=Point(x=x, y=y),
                event_type=MouseEventType.MOUSE_UP,
                button=MouseButton.LEFT,
                modifiers=frozenset(),
            )
        )

    assert session.default_buffer.text == "/mcp"
    assert session.default_buffer.complete_state is None
