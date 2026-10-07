"""Integration tests for the transcript message finder (open, filter, jump)."""

from __future__ import annotations

import time

from rich.text import Text

from zeta.protocol.types import ToolCall
from zeta.tui.transcript import TranscriptWidget
from zeta.tui.transcript.message_finder import Role


def _seeded_transcript() -> tuple[TranscriptWidget, dict[str, int]]:
    transcript = TranscriptWidget()
    indices: dict[str, int] = {}
    first = transcript.append(Text("please run the pytest suite"))
    transcript.mark_user(first)
    indices["user"] = transcript._units.index(first)
    call = ToolCall("c1", "read", {"path": "src/app.py"})
    transcript.start_tool("c1", call, Text("read src/app.py"))
    transcript.finish_tool("c1", Text("read src/app.py"))
    indices["tool"] = len(transcript._units) - 1
    assistant = transcript.append(Text("here is the pytest output summary"))
    indices["assistant"] = transcript._units.index(assistant)
    transcript.create_content(80, 10)
    return transcript, indices


def test_open_finder_lists_messages_with_roles() -> None:
    transcript, _ = _seeded_transcript()
    transcript.open_finder()
    assert transcript.finder_active
    state = transcript.finder_state()
    assert state is not None
    roles = {row.candidate.role for row in state.rows}
    assert Role.USER in roles
    assert Role.TOOL in roles


def test_typing_filters_to_matching_messages() -> None:
    transcript, _ = _seeded_transcript()
    transcript.open_finder()
    transcript.finder_set_query("pytest")
    state = transcript.finder_state()
    assert state is not None
    assert state.rows
    assert all("pytest" in row.candidate.text for row in state.rows)
    # The tool card (no "pytest" text) is filtered out.
    assert all(row.candidate.role is not Role.TOOL for row in state.rows)


def test_accept_jumps_to_selected_message_and_highlights() -> None:
    transcript = TranscriptWidget()
    for index in range(40):
        transcript.append(Text(f"ordinary line number {index}"))
    needle = transcript.append(Text("the unique zebra marker lives here"))
    target = transcript._units.index(needle)
    for index in range(40):
        transcript.append(Text(f"trailing line number {index}"))
    transcript.create_content(80, 6)
    assert transcript.follow_tail  # starts pinned at the tail

    transcript.open_finder()
    transcript.finder_set_query("zebra")
    state = transcript.finder_state()
    assert state is not None and len(state.rows) == 1
    assert transcript.finder_accept()
    assert not transcript.finder_active
    assert transcript.search_active
    assert not transcript.follow_tail
    # The viewport scrolled so the accepted message is the top visible line.
    locations = transcript._locations(80)
    assert locations[transcript.scroll_offset][0] is transcript._units[target]


def test_cancel_restores_the_previous_scroll() -> None:
    transcript = TranscriptWidget()
    for index in range(40):
        unit = transcript.append(Text(f"line {index} with searchable content"))
        if index % 5 == 0:
            transcript.mark_user(unit)
    transcript.create_content(80, 5)
    transcript._set_scroll_offset(7)
    saved = transcript.scroll_offset
    assert not transcript.follow_tail

    transcript.open_finder()
    transcript.finder_set_query("content")
    transcript.finder_move(2)
    transcript.finder_cancel()

    assert not transcript.finder_active
    assert transcript.scroll_offset == saved
    assert not transcript.search_active


def test_preview_shows_selected_message_lines() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("alpha line one\nalpha line two\nbeta tail"))
    transcript.create_content(80, 10)
    transcript.open_finder()
    transcript.finder_set_query("beta")
    preview = transcript.finder_state().preview
    assert "beta tail" in preview


def test_finder_jump_works_on_virtual_history() -> None:
    transcript = TranscriptWidget()
    for index in range(400):
        transcript.append(Text(f"history line {index}"))
    needle = transcript.append(Text("the unique zebra marker on the virtual path"))
    target = transcript._units.index(needle)
    for index in range(400):
        transcript.append(Text(f"tail line {index}"))
    transcript.create_content(100, 10)
    assert transcript._uses_virtual_history()

    transcript.open_finder()
    transcript.finder_set_query("zebra")
    while not transcript.finder_rank_more():
        pass
    state = transcript.finder_state()
    assert state is not None and len(state.rows) == 1
    assert transcript.finder_accept()
    # The virtual viewport is positioned at the accepted message.
    assert transcript._virtual_start is not None
    assert transcript._virtual_start[0] == target


def test_open_finder_on_a_long_session_stays_cheap() -> None:
    transcript = TranscriptWidget()
    for index in range(5000):
        transcript.append(Text(f"message {index} mentioning pytest and fixtures"))
    transcript.create_content(100, 40)

    started = time.perf_counter()
    transcript.open_finder()
    open_seconds = time.perf_counter() - started
    assert transcript.finder_active
    assert open_seconds < 2.0, open_seconds


def test_per_keystroke_ranking_is_bounded_on_a_long_session() -> None:
    transcript = TranscriptWidget()
    for index in range(5000):
        transcript.append(Text(f"message {index} mentioning pytest and fixtures"))
    transcript.create_content(100, 40)
    transcript.open_finder()

    started = time.perf_counter()
    transcript.finder_set_query("pytest")
    first_slice_seconds = time.perf_counter() - started
    # The first keystroke slice does not scan all 5000 messages, keeping the
    # redraw tick responsive.
    assert not transcript.finder_state().complete
    assert first_slice_seconds < 0.02, first_slice_seconds

    # Each continuation slice stays well under the 10 ms ticker gap.
    slices = 0
    while not transcript.finder_rank_more():
        slice_started = time.perf_counter()
        transcript.finder_rank_more()
        assert time.perf_counter() - slice_started < 0.02
        slices += 1
        assert slices < 200
    assert transcript.finder_state().rows
