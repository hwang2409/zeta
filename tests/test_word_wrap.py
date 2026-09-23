from __future__ import annotations

from inspect import signature

import prompt_toolkit
from prompt_toolkit.data_structures import Point
from prompt_toolkit.layout.containers import Window
from prompt_toolkit.layout.controls import UIContent
from prompt_toolkit.layout.screen import Screen, WritePosition

from zeta.tui.key_bindings import FullScreenPromptSession
from zeta.tui.word_wrap import (
    WordWrapWindow,
    _word_wrap_height,
)


def _render(
    text: str, width: int, height: int = 6
) -> tuple[list[str], dict[tuple[int, int], tuple[int, int]]]:
    content = UIContent(
        get_line=lambda _: [("", text), ("", " ")],
        line_count=1,
        cursor_position=Point(x=len(text), y=0),
    )
    screen = Screen(initial_width=width, initial_height=height)
    window = WordWrapWindow(wrap_lines=True)
    _, rowcol_to_yx = window._copy_body(
        content,
        screen,
        WritePosition(xpos=0, ypos=0, width=width, height=height),
        move_x=0,
        width=width,
        wrap_lines=True,
    )
    rows = [
        "".join(screen.data_buffer[row][column].char for column in range(width))
        for row in range(height)
    ]
    return rows, rowcol_to_yx


def _prefix(_lineno: int, wrap_count: int) -> str:
    return "" if wrap_count == 0 else "| "


def test_sentence_wraps_after_spaces() -> None:
    rows, _ = _render("alpha beta gamma", width=12)

    assert rows[:2] == ["alpha beta  ", "gamma       "]


def test_cursor_cell_does_not_displace_word_at_boundary() -> None:
    rows, _ = _render("hello world", width=11)

    assert rows[0] == "hello world"
    assert rows[1].strip() == ""


def test_word_longer_than_window_breaks_at_edge() -> None:
    rows, _ = _render("superlongword", width=5)

    assert rows[:3] == ["super", "longw", "ord  "]


def test_multiple_spaces_keep_the_next_word_whole() -> None:
    rows, _ = _render("a  b c", width=4)

    assert rows[:2] == ["a   ", "b c "]


def test_trailing_space_at_wrap_column_is_a_break_point() -> None:
    rows, _ = _render("abc def", width=4)

    assert rows[:2] == ["abc ", "def "]


def test_cursor_mapping_tracks_character_after_soft_wrap() -> None:
    rows, rowcol_to_yx = _render("alpha beta", width=6)

    assert rows[:2] == ["alpha ", "beta  "]
    assert rowcol_to_yx[(0, 6)] == (1, 0)
    assert rowcol_to_yx[(0, 8)] == (1, 2)


def test_cursor_mapping_includes_continuation_prefix() -> None:
    content = UIContent(
        get_line=lambda _: [("", "alpha beta"), ("", " ")],
        line_count=1,
        cursor_position=Point(x=8, y=0),
    )
    screen = Screen(initial_width=8, initial_height=4)
    window = WordWrapWindow(wrap_lines=True)
    _, rowcol_to_yx = window._copy_body(
        content,
        screen,
        WritePosition(xpos=0, ypos=0, width=8, height=4),
        move_x=0,
        width=8,
        wrap_lines=True,
        get_line_prefix=_prefix,
    )

    assert rowcol_to_yx[(0, 8)] == (1, 4)


def test_prefix_height_matches_word_wrapped_rows() -> None:
    line = [("", "alpha beta gamma")]

    assert _word_wrap_height(line, 0, 12, _prefix) == 2
    assert _word_wrap_height(line, 0, 8, _prefix) == 3


def test_cursor_row_height_uses_full_wrap_plan() -> None:
    line = [("", "alpha beta"), ("", " ")]

    assert _word_wrap_height(line, 0, 8, None, cursor_col=10) == 2
    assert _word_wrap_height(line, 0, 8, None, slice_stop=10, cursor_col=10) == 2

    content = UIContent(
        get_line=lambda _: line,
        line_count=1,
        cursor_position=Point(x=10, y=0),
    )
    window = WordWrapWindow(wrap_lines=True)
    window._scroll_when_linewrapping(content, width=8, height=1)
    assert window.vertical_scroll_2 == 1

    screen = Screen(initial_width=8, initial_height=1)
    _, rowcol_to_yx = window._copy_body(
        content,
        screen,
        WritePosition(xpos=0, ypos=0, width=8, height=1),
        move_x=0,
        width=8,
        vertical_scroll_2=window.vertical_scroll_2,
        wrap_lines=True,
    )
    assert rowcol_to_yx[(0, 10)] == (0, 4)


def test_tabs_use_prompt_toolkit_display_width_and_break_boundary() -> None:
    rows, rowcol_to_yx = _render("aa\tbb", width=5)

    assert rows[:2] == ["aa^I ", "bb   "]
    assert rowcol_to_yx[(0, 2)] == (0, 2)
    assert rowcol_to_yx[(0, 3)] == (1, 0)


def test_word_wrap_override_tracks_pinned_base_signature() -> None:
    base_signature = signature(Window._copy_body)
    override_signature = signature(WordWrapWindow._copy_body)
    parameter_names = (
        "self",
        "ui_content",
        "new_screen",
        "write_position",
        "move_x",
        "width",
        "vertical_scroll",
        "horizontal_scroll",
        "wrap_lines",
        "highlight_lines",
        "vertical_scroll_2",
        "always_hide_cursor",
        "has_focus",
        "align",
        "get_line_prefix",
    )
    assert tuple(base_signature.parameters) == parameter_names
    assert tuple(override_signature.parameters) == parameter_names
    assert [
        (parameter.kind, parameter.default)
        for parameter in base_signature.parameters.values()
    ] == [
        (parameter.kind, parameter.default)
        for parameter in override_signature.parameters.values()
    ]


def test_composer_uses_word_wrap_window_for_pinned_prompt_toolkit() -> None:
    assert prompt_toolkit.__version__ == "3.0.53"

    session = FullScreenPromptSession(multiline=True)
    composer_windows = {
        id(window): window
        for window in session.layout.find_all_windows()
        if getattr(window.content, "buffer", None) is session.default_buffer
    }

    assert len(composer_windows) == 1
    assert isinstance(next(iter(composer_windows.values())), WordWrapWindow)
