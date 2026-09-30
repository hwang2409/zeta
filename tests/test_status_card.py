from io import StringIO

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.text import Text

from zeta.core.fake import FakeBackend
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.skills import SkillCatalog
from zeta.tui.app import TUIApp
from zeta.tui.status_card import StatusCardControl


def test_status_card_is_focusable_and_scrolls_with_a_bounded_offset() -> None:
    card = StatusCardControl()
    card.set_lines([f"line {index}" for index in range(20)])

    assert card.is_focusable
    card.create_content(width=40, height=5)
    card.scroll(3)
    assert card.offset == 3
    card.scroll(100)
    assert card.offset == 15
    card.page(-1)
    assert card.offset == 12
    card.top()
    assert card.offset == 0
    card.bottom()
    assert card.offset == 15


@pytest.mark.asyncio
async def test_status_card_open_close_preserves_composer_and_transcript(
    tmp_path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._transcript.append(Text("existing transcript"))
    units = app._transcript.units
    session.default_buffer.set_document(Document("draft text", 5))
    app.open_status_card()

    assert app.status_card_active
    assert app._transcript.units == units
    assert session.layout.has_focus(app._status_card_window)
    app._status_card.create_content(width=40, height=4)
    app._status_card.bottom()
    assert app._status_card.offset > 0

    app.close_status_card()
    assert not app.status_card_active
    assert session.default_buffer.text == "draft text"
    assert session.default_buffer.cursor_position == 5
    assert session.layout.has_focus(session.default_buffer)
    assert app._transcript.units == units


class _FixedSizeOutput(DummyOutput):
    def __init__(self, width: int, height: int) -> None:
        self._size = Size(rows=height, columns=width)

    def get_size(self) -> Size:
        return self._size


async def _render_status(session, width: int, height: int) -> Screen:
    session.app.output = _FixedSizeOutput(width, height)
    screen = Screen()
    handlers = MouseHandlers()
    session.layout.update_parents_relations()
    with set_app(session.app):
        session.layout.container.write_to_screen(
            screen, handlers, WritePosition(0, 0, width, height), "", False, None
        )
        screen.draw_all_floats()
    return screen


def _card_bounds(screen: Screen, width: int, height: int) -> tuple[int, int, int, int]:
    cells = [
        (x, y)
        for y in range(height)
        for x in range(width)
        if "class:status-card" in screen.data_buffer[y][x].style
    ]
    assert cells
    return (
        min(x for x, _ in cells),
        max(x for x, _ in cells),
        min(y for _, y in cells),
        max(y for _, y in cells),
    )


@pytest.mark.asyncio
async def test_status_card_render_is_centered_and_bounded(tmp_path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._status_card.set_lines(["status", "──────", "x" * 200])
    app._status_card_open = True

    screen = await _render_status(session, 100, 30)
    left, right, top, bottom = _card_bounds(screen, 100, 30)
    assert (left, right, top, bottom) == (14, 85, 13, 15)
    assert right - left + 1 == 72
    assert bottom - top + 1 < 30


@pytest.mark.asyncio
async def test_status_card_small_terminal_scrolls_and_clamps(tmp_path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store, skill_catalog=SkillCatalog.empty()),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=True),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    app._status_card.set_lines([f"line {index}" for index in range(20)])
    app._status_card_open = True

    screen = await _render_status(session, 24, 7)
    left, right, top, bottom = _card_bounds(screen, 24, 7)
    assert (left, right, top, bottom) == (6, 16, 2, 4)
    assert right - left + 1 == 11
    assert bottom - top + 1 == 3

    app._status_card.bottom()
    assert app._status_card.offset == 17
    app._status_card.scroll(100)
    assert app._status_card.offset == 17
    app._status_card.top()
    assert app._status_card.offset == 0


def test_status_card_content_is_width_bounded() -> None:
    card = StatusCardControl()
    card.set_lines(["x" * 100])
    content = card.create_content(width=12, height=3)

    rendered = "".join(text for _, text in content.get_line(0))
    assert len(rendered) == 12
    assert rendered.startswith("│ ")
    assert rendered.endswith(" │")
