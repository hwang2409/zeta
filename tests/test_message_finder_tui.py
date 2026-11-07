"""Integration tests for the transcript message finder (open, filter, jump)."""

from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace
from typing import Any

import pytest
from prompt_toolkit.styles import Style
from rich.text import Text

from zeta.core.store import ConversationStore
from zeta.protocol.types import ToolCall
from zeta.tui.agent_card import AgentNavigation
from zeta.tui.app import TUIApp
from zeta.tui.transcript import FinderControl, TranscriptWidget, transcript_finder
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


def _visible_transcript_text(transcript: TranscriptWidget) -> str:
    content = transcript.create_content(80, 1)
    return "".join(fragment[1] for fragment in content.get_line(0))


def _track_virtual_search_publication(
    transcript: TranscriptWidget, monkeypatch: pytest.MonkeyPatch
) -> asyncio.Event:
    published = asyncio.Event()
    continue_search = transcript._continue_virtual_search_index

    def track(key: tuple[int, int, str], *, bounded: bool = True) -> None:
        continue_search(key, bounded=bounded)
        if transcript._virtual_search_complete:
            published.set()

    # test-contract: no public seam reports incremental search-index completion.
    monkeypatch.setattr(transcript, "_continue_virtual_search_index", track)
    return published


def _app_for_finder(
    transcript: TranscriptWidget, *, child: TranscriptWidget | None = None
) -> Any:
    app = object.__new__(TUIApp)
    app._transcript = transcript
    app._agent_navigation = SimpleNamespace(
        child_view_active=child is not None,
        transcript_control=SimpleNamespace(transcript=child),
    )
    app._active_finder = transcript
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


class _ThreadRecordingRenderable:
    def __init__(
        self,
        threads: list[int],
        *,
        entered: threading.Event | None = None,
        release: threading.Event | None = None,
    ) -> None:
        self._threads = threads
        self._entered = entered
        self._release = release

    def __deepcopy__(self, memo: dict[int, Any]) -> _ThreadRecordingRenderable:
        del memo
        return self

    def __rich_console__(self, console: Any, options: Any) -> Any:
        del console, options
        self._threads.append(threading.get_ident())
        if self._entered is not None:
            self._entered.set()
        if self._release is not None:
            assert self._release.wait(timeout=5)
        yield Text("older rendered candidate")


@pytest.mark.asyncio
async def test_newest_candidates_publish_before_older_render_finishes() -> None:
    ui_thread = threading.get_ident()
    render_threads: list[int] = []
    entered = threading.Event()
    release = threading.Event()
    transcript = TranscriptWidget()
    transcript.append(
        _ThreadRecordingRenderable(
            render_threads,
            entered=entered,
            release=release,
        )
    )
    for index in range(40):
        transcript.append(Text(f"new candidate {index}"))
    app = _app_for_finder(transcript)
    partial_states: list[tuple[str, ...]] = []

    def record_state() -> None:
        state = transcript.finder_state()
        if state is not None and state.rows and not state.complete:
            partial_states.append(tuple(row.candidate.text for row in state.rows))

    app._invalidate_prompt = record_state
    safety_release = threading.Timer(1.0, release.set)
    safety_release.start()
    app._finder_open()
    prepare_task = app._finder_prepare_task
    assert prepare_task is not None
    initial_content = FinderControl(transcript.finder_state).create_content(80, 20)
    initial_paint = "".join(
        fragment[1]
        for line_index in range(initial_content.line_count)
        for fragment in initial_content.get_line(line_index)
    )
    assert "searching…" in initial_paint
    assert "no matches" not in initial_paint
    await asyncio.wait_for(asyncio.to_thread(entered.wait, 2), timeout=3)

    assert partial_states
    assert partial_states[0][0] == "new candidate 39"
    assert ui_thread not in render_threads
    content = FinderControl(transcript.finder_state).create_content(80, 20)
    painted = "".join(
        fragment[1]
        for line_index in range(content.line_count)
        for fragment in content.get_line(line_index)
    )
    assert "no matches" not in painted

    app._finder_input("candidate 3")
    app._finder_input("candidate 39")
    release.set()
    safety_release.cancel()
    await prepare_task
    while app._finder_rank_task is not None:
        await app._finder_rank_task
    state = transcript.finder_state()
    assert state is not None and state.complete
    assert [row.candidate.text for row in state.rows] == ["new candidate 39"]


def test_open_finder_lists_messages_with_roles() -> None:
    transcript, _ = _seeded_transcript()
    _open_finder(transcript)
    assert transcript.finder_active
    state = transcript.finder_state()
    assert state is not None
    roles = {row.candidate.role for row in state.rows}
    assert Role.USER in roles
    assert Role.TOOL in roles


def _assert_fragment_styles_parse(content: Any) -> None:
    style = Style([])
    for line_index in range(content.line_count):
        for fragment_style, _text, *_handler in content.get_line(line_index):
            style.get_attrs_for_style_str(fragment_style)


def test_selected_finder_row_uses_valid_prompt_toolkit_styles() -> None:
    transcript, _ = _seeded_transcript()
    _open_finder(transcript)
    state = transcript.finder_state()
    assert state is not None and state.rows

    content = FinderControl(lambda: state).create_content(80, 20)

    _assert_fragment_styles_parse(content)
    selected_styles = {
        fragment_style
        for line_index in range(content.line_count)
        for fragment_style, _text, *_handler in content.get_line(line_index)
        if "bg:" in fragment_style
    }
    assert selected_styles
    assert all(" on " not in style for style in selected_styles)


def test_representative_transcript_fragment_styles_parse() -> None:
    transcript, _ = _seeded_transcript()
    for index in range(128):
        transcript.append(Text(f"history row {index}"))
    transcript.begin_search()
    transcript.update_search("pytest")

    _assert_fragment_styles_parse(transcript.create_content(80, 10))


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
async def test_child_view_finder_stays_in_child_transcript() -> None:
    root = TranscriptWidget()
    root.append(Text("root-only searchable marker"))
    root.create_content(80, 5)
    child = TranscriptWidget()
    child.append(Text("child-only searchable marker one"))
    child.append(Text("child-only searchable marker two"))
    child.create_content(80, 5)
    app = _app_for_finder(root, child=child)

    app._finder_open()
    assert app._finder_prepare_task is not None
    await app._finder_prepare_task

    assert child.finder_active
    state = child.finder_state()
    assert state is not None
    assert [row.candidate.text for row in state.rows] == [
        "child-only searchable marker two",
        "child-only searchable marker one",
    ]
    app._finder_input("child-only")
    assert app._finder_rank_task is not None
    await app._finder_rank_task
    app._finder_accept()
    assert child.search_active
    assert child.search_status() == (2, 2)
    child.next_search_match()
    assert child.search_status() == (1, 2)
    child.previous_search_match()
    assert child.search_status() == (2, 2)
    child.end_search()
    assert not child.search_active
    assert not root.search_active


@pytest.mark.asyncio
async def test_switching_from_child_to_root_cancels_stale_finder_work(
    tmp_path: Any,
) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    navigation = AgentNavigation(store)
    child_path = store.session_dir / "agents" / "child"
    child_path.mkdir(parents=True)
    navigation.current_path = child_path
    child = navigation.transcript_control.transcript
    child.append(Text("child-only searchable marker"))
    child.create_content(80, 5)
    root = TranscriptWidget()
    root.append(Text("root-only searchable marker"))
    root.create_content(80, 5)
    app = _app_for_finder(root, child=child)
    app._agent_navigation = navigation
    navigation._on_view_change = app._finder_cancel

    app._finder_open()
    prepare_task = app._finder_prepare_task
    assert prepare_task is not None
    navigation.current_path = navigation.root_path
    navigation._switch_transcript()
    await asyncio.sleep(0)

    assert prepare_task.cancelled()
    assert not child.finder_active
    assert app._active_finder is root
    store.close()


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
async def test_open_yields_within_the_candidate_work_budget(
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
    extracted = 0
    batch_sizes: list[int] = []
    extract_parts = transcript._finder_source_parts
    original_sleep = transcript_finder.asyncio.sleep

    def record_extract(unit: Any) -> tuple[str, ...]:
        nonlocal extracted
        extracted += 1
        return extract_parts(unit)

    async def record_yield(_delay: float) -> None:
        nonlocal extracted
        batch_sizes.append(extracted)
        extracted = 0
        await original_sleep(0)

    monkeypatch.setattr(transcript, "_finder_source_parts", record_extract)
    monkeypatch.setattr(transcript_finder.asyncio, "sleep", record_yield)
    app = _app_for_finder(transcript)
    app._finder_open()
    prepare_task = app._finder_prepare_task
    assert prepare_task is not None
    await prepare_task

    state = transcript.finder_state()
    assert state is not None and state.rows
    assert batch_sizes
    assert max(batch_sizes) <= transcript_finder._FINDER_PREPARE_BATCH_SIZE


@pytest.mark.asyncio
async def test_pathological_query_ranks_off_the_ui_thread(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = TranscriptWidget()
    for index in range(50):
        transcript.append(Text(f"{index} " + "a" * 2_000))
    transcript.create_content(100, 40)
    app = _app_for_finder(transcript)
    app._finder_open()
    prepare_task = app._finder_prepare_task
    assert prepare_task is not None
    await prepare_task
    ui_thread = threading.get_ident()
    ranking_threads: list[int] = []
    rank = transcript.finder_rank

    def record_rank(*, candidate_limit: int | None = None) -> Any:
        ranking_threads.append(threading.get_ident())
        return rank(candidate_limit=candidate_limit)

    monkeypatch.setattr(transcript, "finder_rank", record_rank)
    app._finder_input("a" * 50)
    rank_task = app._finder_rank_task
    assert rank_task is not None
    await rank_task

    assert ranking_threads
    assert ui_thread not in ranking_threads


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
async def test_virtual_accept_stays_on_target_while_search_index_builds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("an earlier zebra match"))
    for index in range(70):
        transcript.append(Text(f"filler {index}"))
    target = transcript.append(Text("the selected zebra target"))
    for index in range(70):
        transcript.append(Text(f"tail {index}"))
    transcript.append(Text("a later zebra match"))
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
    search_published = _track_virtual_search_publication(transcript, monkeypatch)

    assert transcript.finder_accept()
    assert "the selected zebra target" in _visible_transcript_text(transcript)
    await asyncio.wait_for(search_published.wait(), 10)
    assert "the selected zebra target" in _visible_transcript_text(transcript)
    assert transcript.next_search_match()
    assert "a later zebra match" in _visible_transcript_text(transcript)


@pytest.mark.asyncio
@pytest.mark.parametrize("forward", [True, False])
async def test_removed_virtual_search_anchor_keeps_navigation_working(
    forward: bool, monkeypatch: pytest.MonkeyPatch
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
    search_published = _track_virtual_search_publication(transcript, monkeypatch)

    assert transcript.finder_accept()
    transcript.remove(target)
    visible_after_removal = _visible_transcript_text(transcript)
    await asyncio.wait_for(search_published.wait(), 10)

    assert _visible_transcript_text(transcript) == visible_after_removal
    navigate = (
        transcript.next_search_match if forward else transcript.previous_search_match
    )
    assert navigate()
    expected = later if forward else earlier
    assert expected.value.plain in _visible_transcript_text(transcript)


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


@pytest.mark.asyncio
async def test_cancelled_rank_stops_after_one_candidate_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from zeta.tui.transcript import message_finder
    from zeta.tui.transcript.message_finder import Candidate, MessageFinder

    entered = threading.Event()
    release = threading.Event()
    finished = threading.Event()
    errors: list[AssertionError] = []
    call_count = 0
    original_match_query = message_finder.match_query

    def counted_match_query(query: Any, text: str) -> Any:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            entered.set()
            assert release.wait(timeout=5)
        assert call_count <= 16, (
            "ranking continued beyond one candidate batch after cancellation"
        )
        return original_match_query(query, text)

    candidates = tuple(
        Candidate(index, index, Role.ASSISTANT, "#0", f"candidate {index}")
        for index in range(3_016)
    )
    transcript = TranscriptWidget()
    transcript._finder = MessageFinder(candidates)
    original_rank = transcript.finder_rank

    def recording_rank(*, candidate_limit: int | None = None) -> Any:
        try:
            return original_rank(candidate_limit=candidate_limit)
        except AssertionError as exc:
            errors.append(exc)
            return None
        finally:
            finished.set()

    monkeypatch.setattr(message_finder, "match_query", counted_match_query)
    monkeypatch.setattr(transcript, "finder_rank", recording_rank)
    app = _app_for_finder(transcript)

    app._finder_input("candidate")
    assert await asyncio.to_thread(entered.wait, 2)
    app._finder_cancel()
    release.set()
    assert await asyncio.to_thread(finished.wait, 2)

    assert not errors, str(errors[0]) if errors else ""
    assert call_count == 16
