from __future__ import annotations

from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.document import Document
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from rich.console import Console

from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import FullScreenPromptSession, TUIApp
from zeta.tui.word_wrap import WordWrapWindow


def _app(tmp_path: Path) -> tuple[TUIApp, FullScreenPromptSession]:
    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions"),
            skill_catalog=SkillCatalog.empty(),
        ),
        provider="fake",
        model="offline",
        console=Console(
            file=StringIO(),
            force_terminal=True,
            color_system="truecolor",
        ),
    )
    session = app._make_session()
    app._active_session = session
    app._install_full_screen_layout(session)
    return app, session


def _render(session: FullScreenPromptSession, width: int) -> Screen:
    screen = Screen(initial_width=width, initial_height=24)
    with set_app(session.app):
        session.layout.update_parents_relations()
        session.layout.container.write_to_screen(
            screen,
            MouseHandlers(),
            WritePosition(0, 0, width, 24),
            "",
            False,
            None,
        )
    return screen


def _composer_rows(screen: Screen, width: int) -> list[int]:
    return [
        y
        for y in range(24)
        if any(
            "class:text-area" in screen.data_buffer[y][x].style for x in range(width)
        )
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("width", "expected_rows"),
    [(40, 5), (80, 2), (120, 2)],
)
async def test_composer_height_tracks_word_wrap(
    tmp_path: Path, width: int, expected_rows: int
) -> None:
    _, session = _app(tmp_path)
    session.default_buffer.set_document(Document("x" * 140))

    rows = _composer_rows(_render(session, width), width)

    assert len(rows) == expected_rows


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_rows"),
    [("", 1), ("short", 1), ("first line\nsecond line", 2)],
)
async def test_composer_height_shrinks_with_deleted_content(
    tmp_path: Path, text: str, expected_rows: int
) -> None:
    _, session = _app(tmp_path)
    session.default_buffer.set_document(Document(text))

    rows = _composer_rows(_render(session, 80), 80)

    assert len(rows) == expected_rows


@pytest.mark.asyncio
async def test_composer_height_caps_and_keeps_cursor_visible(tmp_path: Path) -> None:
    _, session = _app(tmp_path)
    text = " ".join(f"word{index}" for index in range(100))
    session.default_buffer.set_document(Document(text, cursor_position=len(text)))

    screen = _render(session, 80)
    rows = _composer_rows(screen, 80)
    composer_window = next(
        window
        for window in session.layout.find_all_windows()
        if getattr(window.content, "buffer", None) is session.default_buffer
    )
    cursor = screen.get_cursor_position(composer_window)

    assert len(rows) == WordWrapWindow.MAX_COMPOSER_ROWS
    assert cursor.y in rows

    session.default_buffer.set_document(Document())
    assert len(_composer_rows(_render(session, 80), 80)) == 1
