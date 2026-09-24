from __future__ import annotations

from io import StringIO
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Size
from prompt_toolkit.document import Document
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from rich.console import Console
from rich.text import Text

from zeta.core.fake import FakeBackend
from zeta.core.store import ConversationStore
from zeta.loop import AgentLoop
from zeta.skills import SkillCatalog
from zeta.tui.app import FullScreenPromptSession, TUIApp
from zeta.tui.layout import COMPOSER_CONTENT_PADDING, COMPOSER_PAD_Y
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


def _render(
    session: FullScreenPromptSession, width: int, height: int = 24
) -> Screen:
    screen = Screen(initial_width=width, initial_height=height)
    with set_app(session.app):
        session.layout.update_parents_relations()
        session.layout.container.write_to_screen(
            screen,
            MouseHandlers(),
            WritePosition(0, 0, width, height),
            "",
            False,
            None,
        )
    return screen


def _composer_rows(screen: Screen, width: int) -> list[int]:
    return [
        y
        for y in range(len(screen.data_buffer))
        if any(
            "class:text-area" in screen.data_buffer[y][x].style for x in range(width)
        )
    ]


def _layout_metrics(
    session: FullScreenPromptSession, width: int
) -> tuple[int, int, int, int, bool]:
    screen = _render(session, width)
    content = session.layout.container.children[0].content
    bottom = content.children[1]
    footer_on_bottom = any(
        "status-bar" in screen.data_buffer[23][x].style for x in range(width)
    )
    content_height = bottom.preferred_height(width, 24).preferred
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
    [
        (40, 4 + 2 * COMPOSER_PAD_Y),
        (80, 2 + 2 * COMPOSER_PAD_Y),
        (120, 2 + 2 * COMPOSER_PAD_Y),
    ],
)
async def test_composer_height_tracks_word_wrap(
    tmp_path: Path, width: int, expected_rows: int
) -> None:
    _, session = _app(tmp_path)
    session.default_buffer.set_document(Document("x" * 140))

    rows = _composer_rows(_render(session, width), width)

    assert len(rows) == expected_rows


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_width", [80, 120])
@pytest.mark.parametrize(
    ("input_text", "expected_rows"),
    [
        pytest.param("short", 1 + 2 * COMPOSER_PAD_Y, id="one-line"),
        pytest.param("wrapped input " * 9, 2 + 2 * COMPOSER_PAD_Y, id="grown"),
    ],
)
async def test_composer_fill_and_footer_share_terminal_edges(
    tmp_path: Path, terminal_width: int, input_text: str, expected_rows: int
) -> None:
    _, session = _app(tmp_path)
    session.app.output = SimpleNamespace(
        get_size=lambda: Size(rows=24, columns=terminal_width),
    )
    with set_app(session.app):
        session.default_buffer.set_document(Document(input_text))
    screen = _render(session, terminal_width)

    composer_rows = _composer_rows(screen, terminal_width)
    assert len(composer_rows) == expected_rows
    assert all(
        all(
            "class:text-area" in screen.data_buffer[row][column].style
            for column in range(terminal_width)
        )
        for row in composer_rows
    )

    footer_row = next(
        row
        for row in range(len(screen.data_buffer))
        if any(
            "status-bar" in screen.data_buffer[row][column].style
            for column in range(terminal_width)
        )
    )
    footer_text = [
        screen.data_buffer[footer_row][column].char
        for column in range(terminal_width)
    ]
    first_footer_character = next(
        column
        for column in range(terminal_width)
        if footer_text[column] != " "
    )
    prompt_column = next(
        column
        for row in composer_rows
        for column in range(terminal_width)
        if screen.data_buffer[row][column].char == "›"
    )
    prompt_row = next(
        row
        for row in composer_rows
        if screen.data_buffer[row][prompt_column].char == "›"
    )
    if input_text == "short":
        assert composer_rows == list(range(prompt_row - 1, prompt_row + 2))
        assert all(
            not any(
                screen.data_buffer[row][column].char.strip()
                for column in range(terminal_width)
            )
            for row in (composer_rows[0], composer_rows[-1])
        )
    assert first_footer_character == prompt_column == COMPOSER_CONTENT_PADDING
    last_footer_character = next(
        column
        for column in range(terminal_width - 1, -1, -1)
        if footer_text[column] != " "
    )
    assert last_footer_character == terminal_width - COMPOSER_CONTENT_PADDING - 1


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "expected_rows"),
    [
        ("", 1 + 2 * COMPOSER_PAD_Y),
        ("short", 1 + 2 * COMPOSER_PAD_Y),
        ("first line\nsecond line", 2 + 2 * COMPOSER_PAD_Y),
    ],
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

    assert len(rows) == WordWrapWindow.MAX_COMPOSER_ROWS + 2 * COMPOSER_PAD_Y
    assert cursor.y in rows

    session.default_buffer.set_document(Document())
    assert len(_composer_rows(_render(session, 80), 80)) == 1 + 2 * COMPOSER_PAD_Y


@pytest.mark.asyncio
async def test_short_terminal_shrinks_multiline_composer(tmp_path: Path) -> None:
    app, session = _app(tmp_path)
    app._transcript.append(Text("transcript row"))
    text = "\n".join(f"line {index}" for index in range(9))
    session.default_buffer.set_document(Document(text, cursor_position=len(text)))

    screen = _render(session, 80, height=10)
    composer_rows = _composer_rows(screen, 80)
    composer_window = next(
        window
        for window in session.layout.find_all_windows()
        if getattr(window.content, "buffer", None) is session.default_buffer
    )
    cursor = screen.get_cursor_position(composer_window)
    output = "\n".join(
        "".join(screen.data_buffer[y][x].char for x in range(80))
        for y in range(10)
    )

    assert 1 <= len(composer_rows) < WordWrapWindow.MAX_COMPOSER_ROWS
    assert cursor.y in composer_rows
    assert all(
        any(screen.data_buffer[row][column].char.strip() for column in range(80))
        for row in (composer_rows[0], composer_rows[-1])
    )
    assert "transcript row" in output
    assert "status-bar" in "".join(
        screen.data_buffer[9][x].style for x in range(80)
    )
    assert "Window too small" not in output


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
