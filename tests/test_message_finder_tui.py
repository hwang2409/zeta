"""Integration tests for the transcript message finder (open, filter, jump)."""

from __future__ import annotations

import asyncio
import time
from itertools import pairwise
from typing import Any

import pytest
from rich.text import Text

from zeta.protocol.types import ToolCall
from zeta.tui.app import TUIApp
from zeta.tui.transcript import TranscriptWidget
from zeta.tui.transcript.message_finder import Role


async def _open_finder_async(transcript: TranscriptWidget) -> None:
    request = transcript.open_finder()
    assert request is not None
    candidates = await transcript.build_finder_candidates(request)
    assert transcript.finder_publish_candidates(request, candidates)


def _open_finder(transcript: TranscriptWidget) -> None:
    asyncio.run(_open_finder_async(transcript))


def _rank_finder(transcript: TranscriptWidget, query: str) -> None:
    transcript.finder_set_query(query)
    result = transcript.finder_rank()
    assert result is not None
    assert transcript.finder_publish(result)


def _app_for_finder(transcript: TranscriptWidget) -> Any:
    app = object.__new__(TUIApp)
    app._transcript = transcript
    app._finder_prepare_task = None
    app._finder_rank_task = None
    app._invalidate_prompt = lambda: None
    return app


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
    _open_finder(transcript)
    assert transcript.finder_active
    state = transcript.finder_state()
    assert state is not None
    roles = {row.candidate.role for row in state.rows}
    assert Role.USER in roles
    assert Role.TOOL in roles


def test_typing_filters_to_matching_messages() -> None:
    transcript, _ = _seeded_transcript()
    _open_finder(transcript)
    _rank_finder(transcript, "pytest")
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

    _open_finder(transcript)
    _rank_finder(transcript, "zebra")
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

    _open_finder(transcript)
    _rank_finder(transcript, "content")
    transcript.finder_move(2)
    transcript.finder_cancel()

    assert not transcript.finder_active
    assert transcript.scroll_offset == saved
    assert not transcript.search_active


def test_preview_shows_selected_message_lines() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("alpha line one\nalpha line two\nbeta tail"))
    transcript.create_content(80, 10)
    _open_finder(transcript)
    _rank_finder(transcript, "beta")
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

    _open_finder(transcript)
    _rank_finder(transcript, "zebra")
    state = transcript.finder_state()
    assert state is not None and len(state.rows) == 1
    assert transcript.finder_accept()
    # The virtual viewport is positioned at the accepted message.
    assert transcript._virtual_start is not None
    assert transcript._virtual_start[0] == target


@pytest.mark.asyncio
async def test_enter_before_candidate_preparation_keeps_finder_usable() -> None:
    transcript = TranscriptWidget()
    for index in range(400):
        transcript.append(Text(f"history line {index}"))
    target = transcript.append(Text("the unique zebra marker on the virtual path"))
    target_index = transcript._units.index(target)
    for index in range(400):
        transcript.append(Text(f"tail line {index}"))
    transcript.create_content(100, 10)
    app = _app_for_finder(transcript)

    app._finder_open()
    prepare_task = app._finder_prepare_task
    assert prepare_task is not None

    app._finder_accept()

    assert transcript.finder_active
    assert app._finder_prepare_task is prepare_task
    await prepare_task
    state = transcript.finder_state()
    assert state is not None and state.rows

    app._finder_input("zebra")
    rank_task = app._finder_rank_task
    assert rank_task is not None
    await rank_task
    app._finder_accept()

    assert not transcript.finder_active
    assert transcript._virtual_start is not None
    assert transcript._virtual_start[0] == target_index


@pytest.mark.asyncio
async def test_open_does_not_stall_long_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = TranscriptWidget()
    long_tail = " x" * 2_000
    for index in range(5000):
        unit = transcript.append(
            Text(f"message {index} mentioning pytest and fixtures{long_tail}")
        )
        transcript.mark_user(unit)
    transcript.create_content(100, 40)
    extract_parts = transcript._finder_source_parts

    def slow_extract_parts(unit: Any) -> tuple[str, ...]:
        parts = extract_parts(unit)
        deadline = time.thread_time() + 0.00002
        while time.thread_time() < deadline:
            pass
        return parts

    monkeypatch.setattr(transcript, "_finder_source_parts", slow_extract_parts)
    app = _app_for_finder(transcript)
    ticks = [(time.perf_counter(), time.thread_time())]
    running = True

    async def ticker() -> None:
        while running:
            await asyncio.sleep(0.001)
            ticks.append((time.perf_counter(), time.thread_time()))

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    app._finder_open()
    prepare_task = app._finder_prepare_task
    assert prepare_task is not None
    await prepare_task
    running = False
    await ticker_task

    state = transcript.finder_state()
    assert state is not None and state.rows
    gaps = [
        (later_wall - earlier_wall, later_cpu - earlier_cpu)
        for (earlier_wall, earlier_cpu), (later_wall, later_cpu) in pairwise(ticks)
    ]
    # Thread CPU time excludes runner descheduling but includes all work done by
    # the UI loop between 1 ms ticks. The small per-candidate delay models a slow
    # runner and makes the old synchronous scan exceed the 40 ms bound.
    assert gaps and max(cpu_gap for _wall_gap, cpu_gap in gaps) < 0.04, max(gaps)


@pytest.mark.asyncio
async def test_pathological_query_does_not_stall_10ms_ticker() -> None:
    transcript = TranscriptWidget()
    for index in range(50):
        transcript.append(Text(f"{index} " + "a" * 2_000))
    transcript.create_content(100, 40)
    app = _app_for_finder(transcript)
    app._finder_open()
    prepare_task = app._finder_prepare_task
    assert prepare_task is not None
    await prepare_task

    ticks = [time.perf_counter()]
    running = True

    async def ticker() -> None:
        while running:
            await asyncio.sleep(0.01)
            ticks.append(time.perf_counter())

    ticker_task = asyncio.create_task(ticker())
    await asyncio.sleep(0)
    started = time.perf_counter()
    app._finder_input("a" * 50)
    keypress_seconds = time.perf_counter() - started
    rank_task = app._finder_rank_task
    assert rank_task is not None
    await rank_task
    running = False
    await ticker_task

    gaps = [later - earlier for earlier, later in pairwise(ticks)]
    assert keypress_seconds < 0.02, keypress_seconds
    assert gaps and max(gaps) < 0.05, max(gaps)


def test_accept_resolves_target_after_earlier_unit_is_removed() -> None:
    transcript = TranscriptWidget()
    earlier = transcript.append(Text("earlier transient message"))
    target = transcript.append(Text("the stable zebra target"))
    transcript.append(Text("later message"))
    for index in range(5):
        transcript.append(Text(f"tail {index}"))
    transcript.create_content(80, 1)

    _open_finder(transcript)
    state = transcript.finder_state()
    assert state is not None
    selected = next(i for i, row in enumerate(state.rows) if "zebra" in row.candidate.text)
    transcript.finder_move(selected)
    transcript.remove(earlier)

    assert transcript.finder_accept()
    locations = transcript._locations(80)
    assert locations[transcript.scroll_offset][0] is target


def test_accept_removed_target_jumps_to_nearest_surviving_unit() -> None:
    transcript = TranscriptWidget()
    earlier = transcript.append(Text("earlier message"))
    target = transcript.append(Text("the removed zebra target"))
    following = transcript.append(Text("following message"))
    for index in range(5):
        transcript.append(Text(f"tail {index}"))
    transcript.create_content(80, 1)

    _open_finder(transcript)
    state = transcript.finder_state()
    assert state is not None
    selected = next(i for i, row in enumerate(state.rows) if "zebra" in row.candidate.text)
    transcript.finder_move(selected)
    transcript.remove(earlier)
    transcript.remove(target)

    assert transcript.finder_accept()
    assert not transcript.finder_active
    locations = transcript._locations(80)
    assert locations[transcript.scroll_offset][0] is following


@pytest.mark.parametrize("filler_count", [5, 400])
def test_accept_removed_matching_target_keeps_nearest_survivor(
    filler_count: int,
) -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("an earlier zebra match"))
    for index in range(filler_count):
        transcript.append(Text(f"filler {index}"))
    target = transcript.append(Text("the selected zebra target is removed"))
    following = transcript.append(Text("nearest surviving message"))
    for index in range(filler_count):
        transcript.append(Text(f"tail {index}"))
    transcript.create_content(80, 1)

    _open_finder(transcript)
    _rank_finder(transcript, "zebra")
    state = transcript.finder_state()
    assert state is not None
    selected = next(
        index
        for index, row in enumerate(state.rows)
        if row.candidate.key == target.key
    )
    transcript.finder_move(selected)
    transcript.remove(target)

    assert transcript.finder_accept()
    following_index = transcript._units.index(following)
    if transcript._uses_virtual_history():
        assert transcript._virtual_start is not None
        assert transcript._virtual_start[0] == following_index
    else:
        locations = transcript._locations(80)
        assert locations[transcript.scroll_offset][0] is following
    assert not transcript.search_active


@pytest.mark.parametrize("filler_count", [5, 400])
def test_accept_highlights_target_and_continues_navigation(
    filler_count: int,
) -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("an earlier zebra match"))
    for index in range(filler_count):
        transcript.append(Text(f"filler {index}"))
    target = transcript.append(Text("the selected zebra target"))
    for index in range(filler_count):
        transcript.append(Text(f"tail {index}"))
    later = transcript.append(Text("a later zebra match"))
    transcript.create_content(80, 1)

    _open_finder(transcript)
    _rank_finder(transcript, "zebra")
    state = transcript.finder_state()
    assert state is not None
    selected = next(
        index
        for index, row in enumerate(state.rows)
        if row.candidate.key == target.key
    )
    transcript.finder_move(selected)

    assert transcript.finder_accept()
    assert transcript.search_active
    assert transcript.search_query == "zebra"
    assert transcript.next_search_match()
    if transcript._uses_virtual_history():
        assert transcript._virtual_start is not None
        assert transcript._virtual_start[0] == transcript._units.index(later)
    else:
        locations = transcript._locations(80)
        assert locations[transcript.scroll_offset][0] is later
    assert transcript.previous_search_match()
    if transcript._uses_virtual_history():
        assert transcript._virtual_start is not None
        assert transcript._virtual_start[0] == transcript._units.index(target)
    else:
        locations = transcript._locations(80)
        assert locations[transcript.scroll_offset][0] is target


@pytest.mark.asyncio
async def test_virtual_accept_stays_on_target_while_search_index_builds() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("an earlier zebra match"))
    for index in range(70):
        transcript.append(Text(f"filler {index}"))
    target = transcript.append(Text("the selected zebra target"))
    for index in range(70):
        transcript.append(Text(f"tail {index}"))
    later = transcript.append(Text("a later zebra match"))
    transcript.create_content(80, 1)

    await _open_finder_async(transcript)
    _rank_finder(transcript, "zebra")
    state = transcript.finder_state()
    assert state is not None
    selected = next(
        index
        for index, row in enumerate(state.rows)
        if row.candidate.key == target.key
    )
    transcript.finder_move(selected)

    assert transcript.finder_accept()
    assert transcript._virtual_start is not None
    assert transcript._virtual_start[0] == transcript._units.index(target)
    deadline = asyncio.get_running_loop().time() + 5
    while not transcript._virtual_search_complete:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.001)
    assert transcript._virtual_search_complete
    assert transcript._virtual_start is not None
    assert transcript._virtual_start[0] == transcript._units.index(target)
    assert transcript.next_search_match()
    assert transcript._virtual_start is not None
    assert transcript._virtual_start[0] == transcript._units.index(later)


@pytest.mark.asyncio
@pytest.mark.parametrize("forward", [True, False])
async def test_removed_virtual_search_anchor_keeps_navigation_working(
    forward: bool,
) -> None:
    transcript = TranscriptWidget()
    earlier = transcript.append(Text("an earlier zebra match"))
    for index in range(199):
        transcript.append(Text(f"filler {index}"))
    target = transcript.append(Text("the selected zebra target"))
    transcript.append(Text("nearest surviving message"))
    for index in range(198):
        transcript.append(Text(f"tail {index}"))
    later = transcript.append(Text("a later zebra match"))
    transcript.create_content(80, 1)

    await _open_finder_async(transcript)
    _rank_finder(transcript, "zebra")
    state = transcript.finder_state()
    assert state is not None
    selected = next(
        index
        for index, row in enumerate(state.rows)
        if row.candidate.key == target.key
    )
    transcript.finder_move(selected)

    assert transcript.finder_accept()
    transcript.remove(target)
    viewport = transcript._virtual_start
    transcript.create_content(80, 1)
    deadline = asyncio.get_running_loop().time() + 5
    while not transcript._virtual_search_complete:
        assert asyncio.get_running_loop().time() < deadline
        await asyncio.sleep(0.001)

    assert transcript._virtual_start == viewport
    navigate = (
        transcript.next_search_match if forward else transcript.previous_search_match
    )
    assert navigate()
    assert transcript._virtual_start is not None
    expected = later if forward else earlier
    assert transcript._virtual_start[0] == transcript._units.index(expected)


@pytest.mark.parametrize("forward", [True, False])
def test_removed_eager_search_anchor_keeps_navigation_working(
    forward: bool,
) -> None:
    transcript = TranscriptWidget()
    earlier = transcript.append(Text("an earlier zebra match"))
    target = transcript.append(Text("the selected zebra target"))
    following = transcript.append(Text("nearest surviving message"))
    later = transcript.append(Text("a later zebra match"))
    for index in range(5):
        transcript.append(Text(f"tail {index}"))
    transcript.create_content(80, 1)

    _open_finder(transcript)
    _rank_finder(transcript, "zebra")
    state = transcript.finder_state()
    assert state is not None
    selected = next(
        index
        for index, row in enumerate(state.rows)
        if row.candidate.key == target.key
    )
    transcript.finder_move(selected)

    assert transcript.finder_accept()
    transcript.remove(target)
    viewport_unit = transcript._locations(80)[transcript.scroll_offset][0]
    assert viewport_unit is following
    navigate = (
        transcript.next_search_match if forward else transcript.previous_search_match
    )
    assert navigate()
    expected = later if forward else earlier
    assert transcript._locations(80)[transcript.scroll_offset][0] is expected


def test_removed_search_anchor_resolves_when_indexing_has_not_started(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("an earlier zebra match"))
    target = transcript.append(Text("the selected zebra target"))
    following = transcript.append(Text("nearest surviving message"))
    later = transcript.append(Text("a later zebra match"))
    for index in range(5):
        transcript.append(Text(f"tail {index}"))
    transcript.create_content(80, 1)
    transcript.jump_to_index(transcript._units.index(target))

    original_focus = transcript._focus_search_match
    monkeypatch.setattr(transcript, "_focus_search_match", lambda: None)
    transcript.begin_search()
    transcript.update_search("zebra", anchor_key=target.key)
    transcript.remove(target)
    monkeypatch.setattr(transcript, "_focus_search_match", original_focus)

    viewport_unit = transcript._locations(80)[transcript.scroll_offset][0]
    assert viewport_unit is following
    assert transcript.next_search_match()
    assert transcript._locations(80)[transcript.scroll_offset][0] is later


def test_candidate_request_does_not_carry_transcript_data() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("plain candidate"))

    request = transcript.open_finder()

    assert request is not None
    assert isinstance(request.generation, int)
    assert not hasattr(request, "sources")
    assert not hasattr(request, "units")
