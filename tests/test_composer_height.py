from __future__ import annotations

from io import StringIO
from itertools import pairwise
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
from zeta.tui.layout import content_width
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


def _layout_metrics(
    session: FullScreenPromptSession, width: int
) -> tuple[int, int, int, int, bool]:
    screen = _render(session, width)
    content = session.layout.container.children[0].content.children[1]
    bottom = content.children[1]
    footer_on_bottom = any(
        "status-bar" in screen.data_buffer[23][x].style for x in range(width)
    )
    content_height = bottom.content.preferred_height(
        content_width(width), 24
    ).preferred
    return (
        len(_composer_rows(screen, width)),
        24 - bottom.height,
        bottom.height,
        content_height,
        footer_on_bottom,
    )


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


@pytest.mark.asyncio
async def test_composer_reclaims_transcript_rows_through_submit(tmp_path: Path) -> None:
    app, session = _app(tmp_path)

    states = [
        (80, Document()),
        (80, Document("x" * 140)),
        (40, Document("x" * 140)),
        (40, Document("short")),
    ]
    metrics = []
    for width, document in states:
        session.default_buffer.set_document(document)
        metrics.append(_layout_metrics(session, width))

    app._submit_input(session.default_buffer.text)
    session.default_buffer.reset()
    metrics.append(_layout_metrics(session, 40))

    for _, _, chrome, content_height, footer_on_bottom in metrics:
        assert chrome == content_height
        assert footer_on_bottom

    for previous, current in pairwise(metrics):
        previous_composer, previous_transcript = previous[:2]
        current_composer, current_transcript = current[:2]
        assert current_transcript == previous_transcript - (
            current_composer - previous_composer
        )
