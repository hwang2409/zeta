from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock

import pytest
from prompt_toolkit.data_structures import Point
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.text import Text

from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from zeta.tui import agent_card as agent_card_module
from zeta.tui import checkpoints as checkpoints_module
from zeta.tui import composer as composer_module
from zeta.tui import render as render_module
from zeta.tui import theme
from zeta.tui.agent_card import AgentNavigation, AgentTranscriptControl
from zeta.tui.app import TUIApp
from zeta.tui.cards import agent_sync as agent_sync_module
from zeta.tui.composer import TurnConsumerMixin
from zeta.tui.key_bindings import FullScreenPromptSession
from zeta.tui.render import render_markdown, render_thought_live
from zeta.tui.transcript import AnchoredSelection, TranscriptPresenter, TranscriptWidget
from zeta.tui.transcript.streaming_text import StreamingText
from zeta.tui.transcript.transcript_search import find_matches


def _streaming_transcript() -> tuple[TranscriptWidget, TranscriptPresenter]:
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        Mock(),
        lambda: True,
        transcript.append,
    )
    return transcript, presenter


def _streaming_app() -> tuple[TUIApp, TranscriptWidget]:
    transcript, presenter = _streaming_transcript()
    app = TUIApp.__new__(TUIApp)
    app.provider = "codex"
    app._presenter = presenter
    app._stream_kind = app._stream_identity = None
    app._assistant_chunks = []
    app._thinking_chunks = []
    app._thinking_started_at = app._thinking_duration = None
    app._partial = ""
    app._streaming = False
    return app, transcript


def test_discard_retry_removes_full_screen_attempt_and_invalidates_caches() -> None:
    app, transcript = _streaming_app()
    transcript.begin_search()
    transcript.update_search("failed")

    app._prepare_stream_event(StreamEvent(StreamEventType.MESSAGE_START))
    thinking = StreamEvent(
        StreamEventType.MESSAGE_UPDATE,
        content=ThinkingContent("failed thought"),
    )
    app._prepare_stream_event(thinking)
    app._consume_text(thinking)
    text = StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="failed answer")
    app._prepare_stream_event(text)
    app._consume_text(text)

    assert "failed thought" in _content_text(transcript, 80, 20)
    assert "failed answer" in _content_text(transcript, 80, 20)
    assert transcript.search_status()[1] == 2

    app._prepare_stream_event(
        StreamEvent(StreamEventType.ASSISTANT_RESET)
    )

    assert "failed" not in _content_text(transcript, 80, 20)
    assert transcript.search_status() == (0, 0)
    assert not transcript._unit_search_cache


def _transcript(messages: int) -> TranscriptWidget:
    transcript = TranscriptWidget()
    for index in range(messages):
        transcript.append(
            Markdown(
                f"message {index}\n\n"
                "A paragraph with **markdown** and `code` that stays visible."
            )
        )
        transcript.append_blank()
    return transcript


def test_keystroke_invalidation_is_not_delayed_by_output_frame_cap() -> None:
    with create_pipe_input() as pipe:
        session = FullScreenPromptSession(
            input=pipe, output=DummyOutput(), multiline=True
        )

        assert session.app.min_redraw_interval is None


def test_stream_invalidation_does_not_add_an_idle_trailing_paint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    callbacks: list[Callable[[], None]] = []
    loop = Mock()
    loop.call_later.side_effect = lambda _delay, callback: (
        callbacks.append(callback) or Mock()
    )
    monkeypatch.setattr(composer_module.asyncio, "get_running_loop", lambda: loop)
    app = TUIApp.__new__(TUIApp)
    app._stream_invalidation_handle = None
    app._stream_invalidation_pending = False
    app._invalidate_prompt = Mock()

    app._invalidate_stream_prompt()
    callbacks.pop(0)()

    assert app._invalidate_prompt.call_count == 1


def test_repaints_do_not_rescan_terminal_agent_sessions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="root")
    for index in range(40):
        child = ConversationStore(store.session_dir / "agents", session_id=str(index))
        child.agent_lifecycle_path.write_text(
            json.dumps({"description": f"agent {index}", "state": "completed"})
        )
        child.close()
    navigation = AgentNavigation(store)
    metadata = Mock(wraps=agent_card_module._agent_metadata)
    monkeypatch.setattr(agent_card_module, "_agent_metadata", metadata)

    for _ in range(20):
        assert not navigation.list_visible

    assert metadata.call_count == 0


@pytest.mark.asyncio
async def test_agent_spinner_refresh_does_no_child_file_io_on_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    reads: list[int] = []
    original = agent_sync_module.AgentTranscriptSource._refresh_path

    def recording_refresh(*args: object, **kwargs: object) -> object:
        reads.append(threading.get_ident())
        return original(*args, **kwargs)

    monkeypatch.setattr(
        agent_sync_module.AgentTranscriptSource, "_refresh_path", recording_refresh
    )
    transcript = TranscriptWidget()
    for index in range(8):
        child = ConversationStore(tmp_path / "agents", session_id=str(index))
        child.append_message(
            Message(MessageRole.ASSISTANT, [TextContent("child output" * 1_000)])
        )
        call = ToolCall(
            f"agent-{index}",
            "agent",
            {"prompt": "inspect", "description": f"agent {index}"},
        )
        start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
        transcript.start_tool(call.id, call, render_module.render_event(start), start)
        update = StreamEvent(
            StreamEventType.TOOL_EXECUTION_UPDATE,
            tool_call=call,
            data={"child_session_path": str(child.session_dir)},
        )
        transcript.update_tool(call.id, Text("turn 1"), update)
        child.close()

    loop_thread = threading.get_ident()
    for _ in range(5):
        transcript.refresh_active_agents()
    assert reads == []

    await transcript.refresh_agent_transcripts()
    assert reads
    assert set(reads) == {reads[0]}
    assert loop_thread not in reads


@pytest.mark.asyncio
async def test_production_agent_refresh_keeps_event_loop_responsive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    template = ConversationStore(tmp_path / "template", session_id="0")
    template.append_many(
        (
            "message",
            {
                "message": Message(
                    MessageRole.ASSISTANT,
                    [TextContent("child output line")],
                ).to_dict()
            },
        )
        for _ in range(22_000)
    )
    template.close()
    fixture_size = template.path.stat().st_size
    assert 4_000_000 < fixture_size < 6_000_000

    transcript = TranscriptWidget()
    for index in range(8):
        child_path = tmp_path / "agents" / str(index) / "0"
        shutil.copytree(template.session_dir, child_path)
        call = ToolCall(
            f"agent-{index}",
            "agent",
            {"prompt": "inspect", "description": f"agent {index}"},
        )
        start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
        transcript.start_tool(call.id, call, render_module.render_event(start), start)
        transcript.update_tool(
            call.id,
            Text("turn 1"),
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_UPDATE,
                tool_call=call,
                data={"child_session_path": str(child_path)},
            ),
        )

    reads: list[int] = []
    original_refresh = agent_sync_module.AgentTranscriptSource._refresh_path

    def recording_refresh(*args: object, **kwargs: object) -> object:
        reads.append(threading.get_ident())
        return original_refresh(*args, **kwargs)

    monkeypatch.setattr(
        agent_sync_module.AgentTranscriptSource, "_refresh_path", recording_refresh
    )
    loop_thread = threading.get_ident()
    await transcript.refresh_agent_transcripts()

    assert reads
    assert loop_thread not in reads


def test_bounded_agent_source_reads_only_active_branch_tail(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "agents", session_id="child")
    first = store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("first")])
    )
    store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("abandoned")])
    )
    store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("new branch")]),
        parent_id=first.id,
    )
    store.close()

    source = agent_sync_module.AgentTranscriptSource(
        store.session_dir, message_limit=2
    )
    snapshot = source.refresh().transcript(store.session_dir)

    assert snapshot is not None
    assert [message["content"][0]["text"] for _, message in snapshot.messages] == [
        "first",
        "new branch",
    ]
    assert source._stores == {}
    source.close()
    with pytest.raises(RuntimeError, match="closed"):
        source.refresh()


@pytest.mark.asyncio
async def test_finished_agent_cards_release_transcript_sources(
    tmp_path: Path,
) -> None:
    await asyncio.to_thread(lambda: None)
    baseline_tasks = set(asyncio.all_tasks())
    fd_root = Path("/dev/fd") if Path("/dev/fd").is_dir() else Path("/proc/self/fd")
    baseline = len(os.listdir(fd_root))
    transcript = TranscriptWidget()

    for index in range(100):
        child = ConversationStore(tmp_path / "agents", session_id=str(index))
        child.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"answer {index}")])
        )
        child.close()
        call = ToolCall(
            f"agent-{index}",
            "agent",
            {"prompt": "inspect", "description": f"agent {index}"},
        )
        start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
        transcript.start_tool(call.id, call, render_module.render_event(start), start)
        transcript.update_tool(
            call.id,
            Text("turn 1"),
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_UPDATE,
                tool_call=call,
                data={"child_session_path": str(child.session_dir)},
            ),
        )
        end = StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "done",
                structured_content={"child_session_path": str(child.session_dir)},
            ),
        )
        transcript.finish_tool(call.id, render_module.render_event(end), end)

    final_card = transcript._card_units[(None, "agent-99")].card
    pending = asyncio.all_tasks() - baseline_tasks - {asyncio.current_task()}
    await asyncio.gather(*pending)

    assert len(os.listdir(fd_root)) == baseline
    assert all(
        unit.card.transcript_source is None
        for unit in transcript._card_units.values()
    )
    assert final_card._tail == ("assistant: answer 99",)


def test_follow_tail_redraw_does_not_rebuild_location_map() -> None:
    counts: list[int] = []
    for size in (100, 2_000):
        transcript = _transcript(size)
        original = transcript._locations
        locations = Mock(wraps=original)
        transcript._locations = locations
        transcript.create_content(100, 30)
        transcript.create_content(100, 30)
        counts.append(locations.call_count)
    assert counts == [0, 0]


def test_virtual_streaming_tail_cost_is_bounded_by_viewport() -> None:
    render_counts: list[int] = []
    for size in (2_000, 20_000):
        transcript = _transcript(size)
        stream = StreamingText(theme.BODY, palette_role="body")
        unit = transcript.append(stream)
        stream.append("x" * 50_000)
        transcript.touch(unit)
        transcript.create_content(100, 30)
        render = Mock(wraps=transcript._render_unit)
        transcript._render_unit = render

        for _ in range(20):
            stream.append(" next")
            transcript.touch(unit)
            transcript.create_content(100, 30)

        render_counts.append(render.call_count)

    assert render_counts == [0, 0]


def test_follow_tail_rendered_output_remains_available() -> None:
    transcript = _transcript(8)
    transcript.create_content(100, 30)
    assert "message 7" in "\n".join(transcript.lines(100))
    assert "markdown" in "\n".join(transcript.lines(100))


def _content_text(transcript: TranscriptWidget, width: int, height: int) -> str:
    content = transcript.create_content(width, height)
    return "\n".join(
        "".join(fragment[1] for fragment in content.get_line(index))
        for index in range(content.line_count)
    )


def test_resume_renders_only_visible_units_eagerly() -> None:
    render_counts: list[int] = []
    for size in (500, 2_000):
        transcript = _transcript(size)
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered
        transcript.create_content(100, 30)
        render_counts.append(rendered.call_count)

    assert render_counts[0] == render_counts[1]
    assert render_counts[0] < 30


def test_virtual_search_indexes_visible_text_inside_rich_containers() -> None:
    transcript = TranscriptWidget()
    for index in range(128):
        transcript.append(Text(f"filler {index}"))
    transcript.append(
        Panel(
            Group(
                Text("tool header: NEEDLE"),
                Markdown("agent output with **needle** and wide 界🙂"),
                Text("status: needle complete"),
            ),
            title="agent needle card",
        )
    )
    transcript.create_content(48, 10)

    transcript.begin_search()
    transcript.update_search("needle")

    assert transcript.search_status() == (1, 4)


@pytest.mark.asyncio
async def test_uncached_search_index_builds_in_bounded_batches() -> None:
    transcript = TranscriptWidget()
    for index in range(120):
        transcript.append(Text(f"filler {index}"))
    for index in range(8):
        transcript.append(Panel(Text(f"panel needle {index}")))
    transcript.create_content(79, 10)
    original = transcript._search_rendered

    def slow_search_render(unit: object, width: int) -> str:
        if width == 79:
            time.sleep(0.005)
        return original(unit, width)

    transcript._search_rendered = slow_search_render
    transcript.begin_search()
    transcript.update_search("needle")

    assert not transcript._virtual_search_complete
    while not transcript._virtual_search_complete:
        await asyncio.sleep(0.001)
    assert transcript.search_status() == (1, 8)


def test_search_index_does_not_render_history() -> None:
    counts: list[int] = []
    for size in (500, 2_000):
        transcript = _transcript(size)
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered
        transcript.create_content(100, 30)
        rendered.reset_mock()

        transcript.begin_search()
        transcript.update_search("message 42")

        counts.append(rendered.call_count)
        assert transcript.search_status() == (1, 11)
    assert counts == [0, 0]


def _virtual_search_transcript(*values: Text) -> TranscriptWidget:
    transcript = TranscriptWidget()
    for index in range(128):
        transcript.append(Text(f"filler {index}"))
    for value in values:
        transcript.append(value)
    transcript.create_content(40, 6)
    return transcript


def test_virtual_search_focuses_match_deep_in_one_unit() -> None:
    transcript = _virtual_search_transcript(
        Text("\n".join([*(f"long line {index}" for index in range(81)), "deep needle"]))
    )

    transcript.begin_search()
    transcript.update_search("needle")
    content = _content_text(transcript, 40, 6)

    assert "deep needle" in content


@pytest.mark.parametrize("width", [40, 80])
@pytest.mark.parametrize(
    ("markup", "query"),
    [
        ("| first | second |\n| --- | --- |\n| target | table target |", "target"),
        ("- list target\n  - nested list target", "list target"),
        ("**emphasis target** before *emphasis target*", "emphasis target"),
        ("`code target` before `code target`", "code target"),
        ("target " + ("wrapped words " * 12) + "paragraph target at the end", "target"),
    ],
)
def test_virtual_markdown_search_uses_rendered_line_coordinates(
    width: int, markup: str, query: str
) -> None:
    transcript = TranscriptWidget()
    for index in range(128):
        transcript.append(Text(f"filler {index}"))
    unit = transcript.append(render_markdown(markup))
    transcript.create_content(width, 6)

    rendered = transcript._search_rendered(unit, width)
    expected = find_matches(Text.from_ansi(rendered).plain.splitlines(), query)
    assert expected

    transcript.begin_search()
    transcript.update_search(query)
    for match_index, match in enumerate(expected):
        content = transcript.create_content(width, 6)
        highlighted = "".join(
            text
            for line in range(content.line_count)
            for style, text in content.get_line(line)
            if theme.prompt_toolkit_style(theme.SEARCH_CURRENT) in style
        )

        target = (128, match.first_line)
        assert transcript._virtual_start == min(
            target, transcript._virtual_tail_start(width, 6)
        )
        assert highlighted == query
        if match_index + 1 < len(expected):
            assert transcript.next_search_match()


def test_virtual_search_generation_uses_exact_query() -> None:
    transcript = _virtual_search_transcript(Text("only sharp s: ß"))
    transcript.begin_search()

    transcript.update_search("ß")
    assert transcript.search_status() == (1, 1)
    sharp_s_key = transcript._virtual_search_key

    transcript.update_search("SS")
    assert transcript.search_status() == (0, 0)
    assert transcript._virtual_search_key != sharp_s_key

    transcript.update_search("only")
    ordinary_key = transcript._virtual_search_key
    transcript.update_search("ONLY")
    assert transcript.search_status() == (1, 1)
    assert transcript._virtual_search_key != ordinary_key


@pytest.mark.asyncio
async def test_progressive_search_cache_mutations_stay_on_loop_thread() -> None:
    mutation_threads: list[int] = []

    class RecordingWidths(OrderedDict[int, None]):
        def __setitem__(self, key: int, value: None) -> None:
            mutation_threads.append(threading.get_ident())
            super().__setitem__(key, value)

        def move_to_end(self, key: int, last: bool = True) -> None:
            mutation_threads.append(threading.get_ident())
            super().move_to_end(key, last)

        def popitem(self, last: bool = True) -> tuple[int, None]:
            mutation_threads.append(threading.get_ident())
            return super().popitem(last)

    class RecordingCache(dict[int, dict[int, tuple[int, str]]]):
        def setdefault(
            self, key: int, default: dict[int, tuple[int, str]] | None = None
        ) -> dict[int, tuple[int, str]]:
            mutation_threads.append(threading.get_ident())
            return super().setdefault(key, default or {})

    transcript = TranscriptWidget()
    unit = transcript.append(Panel(Text("thread needle")))
    transcript._unit_search_widths = RecordingWidths()
    transcript._unit_search_cache = RecordingCache()
    transcript._content_width = 73
    key = (transcript._revision, 73, "needle")
    transcript._virtual_search_key = key
    transcript._virtual_search_cursor = len(transcript._units)

    await transcript._render_search_unit_async(key, unit)

    assert mutation_threads
    assert set(mutation_threads) == {threading.get_ident()}


@pytest.mark.asyncio
async def test_progressive_search_discards_revised_unit_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = TranscriptWidget()
    stream = StreamingText(theme.BODY, palette_role="body")
    stream.append("old needle")
    unit = transcript.append(stream)
    transcript._content_width = 61
    key = (transcript._revision, 61, "needle")
    transcript._virtual_search_key = key
    transcript._virtual_search_cursor = len(transcript._units)
    started = threading.Event()
    release = threading.Event()
    original = transcript._render_search_snapshot

    def blocked_render(snapshot: object, width: int, rich_theme: object) -> str:
        started.set()
        assert release.wait(timeout=2)
        return original(snapshot, width, rich_theme)

    monkeypatch.setattr(transcript, "_render_search_snapshot", blocked_render)
    task = asyncio.create_task(transcript._render_search_unit_async(key, unit))
    while not started.is_set():
        await asyncio.sleep(0)

    stream.append(" revised")
    transcript.touch(unit)
    release.set()
    await task

    assert transcript._unit_search_cache.get(unit.key) is None


@pytest.mark.asyncio
async def test_progressive_search_survives_updates_and_resizes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = TranscriptWidget()
    units = [
        transcript.append(Panel(Text(f"item {index} needle")))
        for index in range(140)
    ]
    original = transcript._render_search_snapshot

    def slowed_render(snapshot: object, width: int, rich_theme: object) -> str:
        time.sleep(0.0005)
        return original(snapshot, width, rich_theme)

    monkeypatch.setattr(transcript, "_render_search_snapshot", slowed_render)
    transcript.create_content(80, 8)
    transcript.begin_search()
    transcript.update_search("needle")

    for index, width in enumerate((81, 63, 77, 54, 80)):
        transcript.replace(units[index], Panel(Text(f"updated {index} without match")))
        transcript.append(Panel(Text(f"appended {index} needle")))
        transcript.create_content(width, 8)
        transcript.search_status()
        await asyncio.sleep(0.002)

    transcript.create_content(80, 8)
    transcript.search_status()
    while not transcript._virtual_search_complete:
        await asyncio.sleep(0.001)

    expected = sum(
        len(
            find_matches(
                transcript._searchable_text(unit, 80).splitlines(), "needle"
            )
        )
        for unit in transcript._units
        if unit is not None
    )
    assert transcript.search_status() == (1, expected)
    assert expected == 140


def test_unit_search_cache_keeps_at_most_two_widths_per_unit() -> None:
    transcript = TranscriptWidget()
    units = [transcript.append(Text(f"unit {index}")) for index in range(1_500)]
    transcript._search_rendered = lambda unit, width: f"{unit.key} at {width}"

    for width in range(40, 80):
        for unit in units:
            transcript._searchable_text(unit, width)

    assert sum(len(entries) for entries in transcript._unit_search_cache.values()) <= (
        len(units) * 2
    )


def test_virtual_search_next_restyles_occurrences_in_the_same_unit() -> None:
    current = theme.prompt_toolkit_style(theme.SEARCH_CURRENT)
    match = theme.prompt_toolkit_style(theme.SEARCH_MATCH)
    transcript = _virtual_search_transcript(Text("needle between needle"))
    transcript.begin_search()
    transcript.update_search("needle")

    first = transcript.create_content(40, 6)
    first_styles = [
        style
        for line in range(first.line_count)
        for style, text in first.get_line(line)
        for _character in text
        if style in {current, match}
    ]
    assert transcript.next_search_match()
    second = transcript.create_content(40, 6)
    second_styles = [
        style
        for line in range(second.line_count)
        for style, text in second.get_line(line)
        for _character in text
        if style in {current, match}
    ]

    assert first_styles == [current] * 6 + [match] * 6
    assert second_styles == [match] * 6 + [current] * 6


def test_virtual_search_does_not_match_through_rendered_wrapping() -> None:
    transcript = _virtual_search_transcript(Text("prefix needletoken suffix"))
    transcript.create_content(7, 6)
    transcript.begin_search()
    transcript.update_search("needletoken")

    content = transcript.create_content(7, 6)
    highlighted = "".join(
        text
        for line in range(content.line_count)
        for style, text in content.get_line(line)
        if style == theme.prompt_toolkit_style(theme.SEARCH_CURRENT)
    )

    assert transcript.search_status() == (0, 0)
    assert highlighted == ""


def _threshold_transcript() -> TranscriptWidget:
    transcript = TranscriptWidget()
    for index in range(127):
        transcript.append(Text(f"threshold line {index}"))
    transcript.create_content(40, 10)
    transcript.page_up()
    transcript.create_content(40, 10)
    return transcript


def _visible_text(transcript: TranscriptWidget, width: int, height: int) -> str:
    content = transcript.create_content(width, height)
    start = 0 if transcript._uses_virtual_history() else transcript.scroll_offset
    return "\n".join(
        "".join(text for _, text in content.get_line(line))
        for line in range(start, min(start + height, content.line_count))
    )


def test_virtual_threshold_transition_preserves_off_tail_view() -> None:
    transcript = _threshold_transcript()
    before = _visible_text(transcript, 40, 10)

    transcript.append(Text("threshold line 127"))
    after = _visible_text(transcript, 40, 10)

    assert after == before


def _top_virtual_location(
    transcript: TranscriptWidget, width: int = 40, height: int = 10
) -> tuple[object | None, int]:
    transcript.create_content(width, height)
    return transcript._virtual_locations[0]


def _location_order(
    transcript: TranscriptWidget, location: tuple[object | None, int]
) -> tuple[int, int]:
    unit, offset = location
    return (transcript._units.index(unit), offset) if unit is not None else (-1, offset)


def _wrapped_tail_transcript() -> TranscriptWidget:
    transcript = TranscriptWidget()
    for index in range(127):
        transcript.append(Text(f"history {index}"))
    transcript.append(Text("\n".join(f"tail row {index}" for index in range(40))))
    return transcript


def test_scroll_up_from_bottom_never_moves_down_with_estimated_heights() -> None:
    transcript = _wrapped_tail_transcript()
    before = _top_virtual_location(transcript)

    transcript.mouse_handler(
        MouseEvent(
            position=Point(x=0, y=0),
            event_type=MouseEventType.SCROLL_UP,
            button=MouseButton.NONE,
            modifiers=frozenset(),
        )
    )
    after = _top_virtual_location(transcript)

    assert after[0] is not None
    assert _location_order(transcript, after) < _location_order(transcript, before)
    assert not transcript.follow_tail


def test_scroll_up_while_streaming_leaves_follow_mode_and_moves_up() -> None:
    transcript, presenter = _streaming_transcript()
    for index in range(127):
        transcript.append(Text(f"history {index}"))
    presenter.append_assistant("\n".join(f"stream row {index}" for index in range(40)))
    before = _top_virtual_location(transcript)
    before_start = transcript._virtual_start

    transcript.page_up()
    assert not transcript.follow_tail
    presenter.append_assistant("\nnew streamed row")
    after = _top_virtual_location(transcript)
    after_start = transcript._virtual_start

    assert before_start is not None and after_start is not None
    assert after_start < before_start
    assert _location_order(transcript, after)[0] <= _location_order(transcript, before)[0]


def test_scroll_anchor_stable_when_heights_above_are_corrected() -> None:
    transcript = TranscriptWidget()
    call = ToolCall("child-refresh", "agent", {"prompt": "inspect"})
    start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    transcript.start_tool(call.id, call, Text("child running"), start)
    transcript._virtual_unit_lines(0, 40)
    for index in range(126):
        transcript.append(Text(f"history {index}"))
    transcript.append(Text("\n".join(f"tail row {index}" for index in range(40))))
    _top_virtual_location(transcript)
    transcript.page_up()
    anchor = _top_virtual_location(transcript)

    child = next(iter(transcript._card_units.values()))
    child.card.refresh = Mock(
        return_value=Text("\n".join(f"child row {index}" for index in range(20)))
    )
    transcript.refresh_active_agents()
    transcript._virtual_unit_lines(0, 40)
    corrected = _top_virtual_location(transcript)

    assert corrected == anchor


def test_virtual_wheel_down_is_a_noop_at_tail() -> None:
    transcript = TranscriptWidget()
    for index in range(200):
        transcript.append(Text(f"line {index}"))
    content = transcript.create_content(80, 10)
    tail = transcript.scroll_offset

    transcript.scroll_down()
    assert transcript._pending_virtual_scroll == 0
    content = transcript.create_content(80, 10)

    assert transcript.scroll_offset == tail
    assert transcript.follow_tail
    assert content.line_count >= 10


def test_virtual_page_down_clamps_after_scrolling_up() -> None:
    transcript = TranscriptWidget()
    for index in range(200):
        transcript.append(Text(f"line {index}"))
    transcript.create_content(80, 10)
    transcript.scroll_up()
    transcript.create_content(80, 10)

    transcript.page_down()
    content = transcript.create_content(80, 10)

    assert transcript.follow_tail
    assert content.line_count >= 10
    assert transcript.scroll_offset == transcript._estimated_total(80) - 10


def test_virtual_resize_reclamps_near_tail() -> None:
    transcript = TranscriptWidget()
    for index in range(200):
        transcript.append(Text(f"line {index}"))
    transcript.create_content(80, 10)
    transcript.page_up()
    transcript.create_content(80, 10)
    transcript.scroll_down()
    transcript.create_content(80, 10)

    content = transcript.create_content(80, 20)

    assert transcript.scroll_offset == 180
    assert transcript.follow_tail
    assert content.line_count >= 20


def test_virtual_search_at_last_line_keeps_full_viewport() -> None:
    transcript = TranscriptWidget()
    for index in range(200):
        transcript.append(Text(f"line {index}"))
    transcript.create_content(80, 10)

    transcript.begin_search()
    transcript.update_search("line 199")
    content = transcript.create_content(80, 10)

    assert transcript.scroll_offset == 190
    assert not transcript.follow_tail
    assert content.line_count >= 10
    assert "line 199" in _visible_text(transcript, 80, 10)


def test_agent_transcript_scroll_paths_stay_clamped_at_tail() -> None:
    control = AgentTranscriptControl()
    for index in range(200):
        control.transcript.append(Text(f"agent line {index}"))
    control.create_content(80, 10)
    tail = control.offset

    control.scroll(3)  # Child-view mouse-wheel callback.
    control.create_content(80, 10)
    assert control.offset == tail

    control.half_page(1)  # Child-view Ctrl-D callback.
    content = control.create_content(80, 10)
    assert control.offset == tail
    assert content.line_count >= 10


def test_virtual_streaming_growth_follows_tail_unless_scrolled_up() -> None:
    transcript, presenter = _streaming_transcript()
    for index in range(127):
        transcript.append(Text(f"history {index}"))
    presenter.append_assistant("stream start")
    transcript.create_content(80, 10)

    presenter.append_assistant("\nstream tail")
    content = transcript.create_content(80, 10)
    assert transcript.follow_tail
    assert "stream tail" in _content_text(transcript, 80, 10)
    assert content.line_count >= 10

    transcript.page_up()
    transcript.create_content(80, 10)
    held_start = transcript._virtual_start
    presenter.append_assistant("\nheld tail")
    transcript.create_content(80, 10)

    assert not transcript.follow_tail
    assert transcript._virtual_start == held_start


def test_virtual_card_collapse_reclamps_near_tail() -> None:
    transcript = TranscriptWidget()
    for index in range(127):
        transcript.append(Text(f"filler {index}"))
    call = ToolCall("read-collapse", "read", {"path": "large.py"})
    start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    end = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "\n".join(f"source line {index}" for index in range(80)),
        ),
    )
    transcript.start_tool(call.id, call, render_module.render_event(start))
    transcript.finish_tool(call.id, render_module.render_event(end), end)
    transcript.create_content(80, 10)
    transcript.page_up()
    transcript.create_content(80, 10)

    assert transcript.toggle_latest_agent()
    content = transcript.create_content(80, 10)

    assert transcript._virtual_start == transcript._virtual_tail_start(80, 10)
    assert transcript.follow_tail
    assert content.line_count >= 10


def test_virtual_threshold_transition_preserves_selection() -> None:
    transcript = _threshold_transcript()
    top = transcript.scroll_offset
    anchor = transcript._anchor_for((top, 0))
    extent = transcript._anchor_for((top + 1, 5))
    transcript._selection = AnchoredSelection(anchor, extent, False)
    before = transcript.selection_text()

    transcript.append(Text("threshold line 127"))
    transcript.create_content(40, 10)

    assert transcript.selection_text() == before
    assert transcript.selection is not None


def test_virtual_threshold_transition_preserves_active_search() -> None:
    transcript = _threshold_transcript()
    transcript.begin_search()
    transcript.update_search("threshold line 42")
    before = _visible_text(transcript, 40, 10)

    transcript.append(Text("threshold line 127"))
    after = _visible_text(transcript, 40, 10)

    assert after == before
    assert transcript.search_status() == (1, 1)


def test_scrolled_paint_cost_is_independent_of_unit_count() -> None:
    render_counts: list[int] = []
    for size in (2_000, 20_000):
        transcript = _transcript(size)
        transcript.create_content(100, 30)
        for _ in range(10):
            transcript.page_up()
            transcript.create_content(100, 30)
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered
        for _ in range(100):
            transcript.create_content(100, 30)
        render_counts.append(rendered.call_count)

    assert render_counts[1] <= render_counts[0]


def test_virtual_position_indicator_uses_consistent_line_estimates() -> None:
    transcript = TranscriptWidget()
    for index in range(128):
        transcript.append(
            Text("\n".join(f"unit {index} row {row}" for row in range(10)))
        )
    transcript.create_content(40, 10)
    for _ in range(12):
        transcript.page_up()
        transcript.create_content(40, 10)

    indicator = transcript.position_indicator()

    assert indicator is not None
    current, total = (
        int(value) for value in re.fullmatch(r"line (\d+)/~(\d+)", indicator).groups()
    )
    assert current <= total


def test_resize_renders_only_a_viewport() -> None:
    counts: list[int] = []
    for size in (500, 2_000):
        transcript = _transcript(size)
        transcript.create_content(100, 30)
        transcript.page_up()
        transcript.create_content(100, 30)
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered

        transcript.create_content(60, 30)

        counts.append(rendered.call_count)
    assert counts[0] == counts[1]
    assert counts[0] < 30


def test_off_tail_append_does_not_revisit_visible_history() -> None:
    counts: list[int] = []
    for size in (500, 2_000):
        transcript = _transcript(size)
        transcript.create_content(100, 30)
        transcript.page_up()
        transcript.create_content(100, 30)
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered

        transcript.append(Text("new streaming tail"))
        transcript.create_content(100, 30)

        counts.append(rendered.call_count)
    assert counts == [0, 0]


@pytest.mark.parametrize("width", [13, 40, 80])
def test_virtual_tail_is_golden_equivalent_with_wide_text(width: int) -> None:
    transcript = TranscriptWidget()
    for index in range(80):
        transcript.append(Text(f"row {index}: wide 界🙂 café e\u0301"))
        transcript.append(Markdown(f"message **{index}** with `code`"))
        transcript.append_blank()
    expected = transcript._parsed_lines(width)[-24:]

    actual = _content_text(transcript, width, 24).splitlines()
    expected_text = ["".join(text for _, text in line) for line in expected]
    assert actual == expected_text


def test_page_up_renders_only_the_entering_viewport_units() -> None:
    counts: list[int] = []
    for size in (500, 2_000):
        transcript = _transcript(size)
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered
        transcript.create_content(100, 30)
        eager_count = rendered.call_count

        transcript.page_up()
        transcript.create_content(100, 30)

        counts.append(rendered.call_count - eager_count)

    assert counts[0] == counts[1]
    assert counts[0] < 30


def test_resume_defers_markdown_parsing_outside_viewport(
    monkeypatch,
) -> None:
    original_parse = render_module._MARKDOWN.parse
    parse = Mock(wraps=original_parse)
    monkeypatch.setattr(render_module._MARKDOWN, "parse", parse)
    transcript = TranscriptWidget()
    for index in range(500):
        transcript.append(render_markdown(f"message **{index}**"))
        transcript.append_blank()

    transcript.create_content(72, 24)

    assert parse.call_count < 24


def test_resume_does_not_sanitize_entire_tool_result_before_bounded_render(
    monkeypatch,
) -> None:
    sanitize = Mock(wraps=checkpoints_module.strip_terminal_controls)
    monkeypatch.setattr(checkpoints_module, "strip_terminal_controls", sanitize)
    message = Message(
        MessageRole.TOOL_RESULT,
        tool_result=ToolResult("call", "output\n" * 10_000),
    )

    sanitized = checkpoints_module._sanitize_replayed_message(message)

    assert sanitized.tool_result is message.tool_result
    sanitize.assert_not_called()


def test_resume_output_matches_eager_render() -> None:
    transcript = _transcript(500)
    eager_lines = transcript._parsed_lines(72)
    expected = "\n".join(
        "".join(fragment[1] for fragment in line) for line in eager_lines[-24:]
    )

    lazy = _transcript(500)
    assert _content_text(lazy, 72, 24) == expected
    lazy.page_up()
    assert lazy.lines(72) == transcript.lines(72)


def _line_transcript() -> TranscriptWidget:
    transcript = TranscriptWidget()
    for index in range(200):
        unit = transcript.append(Text(f"line {index}"))
        if index % 50 == 0:
            transcript.mark_user(unit)
    return transcript


def test_lazy_tail_previous_user_message_navigates_from_tail() -> None:
    transcript = _line_transcript()
    transcript.create_content(80, 10)

    assert transcript.previous_user_message()
    assert transcript.scroll_offset == 150


def test_lazy_tail_next_user_message_stays_at_tail() -> None:
    transcript = _line_transcript()
    transcript.create_content(80, 10)

    assert not transcript.next_user_message()
    assert transcript.follow_tail


def test_lazy_tail_page_up_continues_from_visible_position() -> None:
    transcript = _line_transcript()
    transcript.create_content(80, 10)

    transcript.page_up()
    content = transcript.create_content(80, 10)

    assert transcript.scroll_offset == 180
    assert "line 180" in "".join(text for _, text in content.get_line(0))


@pytest.mark.parametrize("size", [2_000, 20_000])
def test_streaming_assistant_paint_work_is_bounded_per_delta(size: int) -> None:
    app, transcript = _streaming_app()
    delta = "abcdefghij" * 10
    event = StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=delta)
    app._consume_text(event)
    stream = transcript._units[-1]
    assert stream is not None
    wrapped_characters = 0
    original = stream.value._wrapped

    def counted_wrap(console, value, width, style):
        nonlocal wrapped_characters
        wrapped_characters += len(value)
        return original(console, value, width, style)

    stream.value._wrapped = counted_wrap
    for _ in range(size // len(delta) - 1):
        app._consume_text(event)
        transcript.create_content(80, 24)

    assert wrapped_characters <= (size // len(delta)) * 5_000


def test_streaming_assistant_paint_work_is_history_independent() -> None:
    counts: list[int] = []
    for history_size in (100, 2_000, 10_000):
        transcript = _transcript(history_size)
        presenter = TranscriptPresenter(
            transcript,
            Mock(),
            lambda: True,
            transcript.append,
        )
        rendered = Mock(wraps=transcript._render_unit)
        transcript._render_unit = rendered
        for _ in range(20):
            presenter.append_assistant("abcdefghij" * 10)
            transcript.create_content(80, 24)
        counts.append(rendered.call_count)

    assert max(counts) - min(counts) <= 2


@pytest.mark.parametrize("width", [7, 13, 40, 80])
@pytest.mark.parametrize(
    "value",
    [
        "plain words wrap exactly like rich text does",
        "wide: 界🙂 café e\u0301 and tabs\there",
        "markup [bold]is literal[/bold] and ansi \x1b[31m stays literal",
        "first line\n\nsecond line with trailing spaces   \nlast",
    ],
)
def test_streaming_assistant_live_output_matches_text(value: str, width: int) -> None:
    expected = TranscriptWidget()
    expected.append(Text(value, style=theme.BODY))
    transcript, presenter = _streaming_transcript()
    for start in range(0, len(value), 3):
        presenter.append_assistant(value[start : start + 3])
        transcript.create_content(width, 40)

    assert _content_text(transcript, width, 40) == _content_text(expected, width, 40)
    assert transcript.lines(width) == expected.lines(width)


@pytest.mark.parametrize("width", [4, 7, 11, 23])
def test_streaming_carriage_returns_match_text_across_stable_tail(width: int) -> None:
    value = "abc\rdef" * 100
    expected = TranscriptWidget()
    expected.append(Text(value, style=theme.BODY))
    transcript, presenter = _streaming_transcript()

    for character in value:
        presenter.append_assistant(character)
        transcript.create_content(width, 6)

    assert presenter._assistant_stream is not None
    assert presenter._assistant_stream.plain == value
    assert (
        _content_text(transcript, width, 6).splitlines()
        == _content_text(expected, width, 6).splitlines()[-6:]
    )
    assert transcript.lines(width) == expected.lines(width)


def test_streaming_text_restyles_cached_lines_after_palette_switch() -> None:
    original = theme.active_palette()
    theme.set_active_palette(theme.DARK)
    try:
        transcript, presenter = _streaming_transcript()
        presenter.append_assistant("palette colored text " * 100)
        transcript.create_content(12, 6)

        theme.set_active_palette(theme.LIGHT)
        content = transcript.create_content(12, 6)
        styles = {
            fragment[0]
            for line in range(content.line_count)
            for fragment in content.get_line(line)
            if fragment[1]
        }

        assert any(theme.LIGHT.body in style for style in styles)
        assert all(theme.DARK.body not in style for style in styles)
    finally:
        theme.set_active_palette(original)


@pytest.mark.parametrize(
    ("chunks", "intermediate"),
    [
        (["\x1b[", "31mred"], ["", "red"]),
        (["\x1b]0;", "secret", "\x07visible"], ["", "", "visible"]),
        (["*", "*bold", "**"], ["", "", "bold"]),
    ],
)
def test_streaming_thinking_holds_incomplete_constructs(
    chunks: list[str], intermediate: list[str]
) -> None:
    app, transcript = _streaming_app()
    app.provider = "codex"

    for chunk, expected in zip(chunks, intermediate, strict=True):
        app._consume_text(
            StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=ThinkingContent(chunk),
            )
        )
        assert Text.from_ansi("\n".join(transcript.lines(80))).plain == expected

    final = TranscriptWidget()
    final.append(render_thought_live("".join(chunks), provider="codex"))
    assert transcript.lines(80) == final.lines(80)


def test_streaming_thinking_uses_the_incremental_unit() -> None:
    transcript, presenter = _streaming_transcript()
    rendered = Mock(wraps=transcript._render_unit)
    transcript._render_unit = rendered
    for _ in range(200):
        presenter.append_thinking("reasoning delta ")
        transcript.create_content(80, 24)

    assert rendered.call_count == 0
    assert "reasoning delta" in _content_text(transcript, 80, 24)


@pytest.mark.asyncio
async def test_streaming_repaints_are_coalesced_and_final_paint_is_immediate() -> None:
    consumer = TurnConsumerMixin()
    paints = Mock()
    consumer._invalidate_prompt = paints

    for _ in range(1_000):
        consumer._invalidate_stream_prompt()

    assert paints.call_count == 1
    consumer._finish_stream_invalidation()
    assert paints.call_count == 2


def test_lazy_tail_disabled_when_max_lines_set() -> None:
    marker = "[custom older output]"

    def limited_transcript() -> TranscriptWidget:
        transcript = TranscriptWidget(max_lines=5)
        transcript.set_line_limit_marker(marker)
        for index in range(128):
            transcript.append(Text(f"line {index}"))
        return transcript

    expected = limited_transcript()._parsed_lines(80)
    transcript = limited_transcript()
    content = transcript.create_content(80, 10)
    actual = [content.get_line(index) for index in range(content.line_count)]

    assert not transcript._lazy_viewport
    assert actual == ([[]] * (10 - len(expected))) + expected
    assert marker in "".join(text for _, text in expected[0])
    assert content.line_count == 10


def test_long_stream_page_up_visits_every_stream_line() -> None:
    transcript = _transcript(128)
    stream = StreamingText(theme.BODY, palette_role="body")
    unit = transcript.append(stream)
    stream.append("\n".join(f"stream-{index}" for index in range(30)))
    transcript.touch(unit)
    transcript.create_content(80, 5)

    visible: set[str] = set()
    for _ in range(8):
        text = _content_text(transcript, 80, 5)
        visible.update(re.findall(r"stream-\d+", text))
        transcript.page_up()

    assert visible == {f"stream-{index}" for index in range(30)}


@pytest.mark.asyncio
async def test_agent_completion_before_first_refresh_publishes_final_tail(
    tmp_path: Path,
) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("final child answer")])
    )
    transcript = TranscriptWidget()
    call = ToolCall("agent-1", "agent", {"prompt": "inspect", "description": "short"})
    start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    transcript.start_tool(call.id, call, render_module.render_event(start), start)
    transcript.update_tool(
        call.id,
        Text("turn 1"),
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_UPDATE,
            tool_call=call,
            data={"child_session_path": str(child.session_dir)},
        ),
    )
    end = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "done",
            structured_content={"child_session_path": str(child.session_dir)},
        ),
    )
    transcript.finish_tool(call.id, render_module.render_event(end), end)
    await asyncio.sleep(0.05)

    assert "final child answer" in Text.from_ansi(transcript.render(100)).plain
    assert "child transcript unavailable" not in Text.from_ansi(
        transcript.render(100)
    ).plain


@pytest.mark.asyncio
async def test_grandchild_append_updates_parent_agent_card(tmp_path: Path) -> None:
    parent = ConversationStore(tmp_path / "agents", session_id="1")
    grandchild = ConversationStore(parent.session_dir / "agents", session_id="1")
    nested = ToolCall("nested", "agent", {"prompt": "nested"})
    parent.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(nested)])
    )
    parent.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(
                nested.id,
                "running",
                structured_content={"child_session_path": str(grandchild.session_dir)},
            ),
        )
    )
    grandchild.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("nested first")])
    )
    transcript = TranscriptWidget()
    call = ToolCall("agent-1", "agent", {"prompt": "inspect"})
    start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    transcript.start_tool(call.id, call, render_module.render_event(start), start)
    transcript.update_tool(
        call.id,
        Text("turn 1"),
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_UPDATE,
            tool_call=call,
            data={"child_session_path": str(parent.session_dir)},
        ),
    )
    await transcript.refresh_agent_transcripts()
    grandchild.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("nested appended")])
    )

    await transcript.refresh_agent_transcripts()

    assert "nested appended" in Text.from_ansi(transcript.render(100)).plain


@pytest.mark.asyncio
async def test_oversized_far_branch_clears_existing_agent_tail(tmp_path: Path) -> None:
    from zeta.tui.cards.agent import AgentCard

    store = ConversationStore(tmp_path / "agents", session_id="oversized")
    first = store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("first")])
    )
    payload = "x" * 13_000
    for index in range(1_500):
        store.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"old-{index}-{payload}")])
        )
    assert store.path.stat().st_size > 16 * 1024 * 1024

    card = AgentCard(ToolCall("agent-oversized", "agent", {"prompt": "inspect"}))
    card.set_child_session_path(str(store.session_dir))
    assert await card.refresh_tail()
    assert any("old-1499" in line for line in card._tail)

    store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("NEW-BRANCH")]),
        parent_id=first.id,
    )
    assert await card.refresh_tail()

    assert card._tail == (
        "child transcript unavailable: history exceeds bounded scan",
    )
    console = Console(record=True, width=100)
    console.print(card.current())
    rendered = console.export_text()
    assert "old-1499" not in rendered
    assert "history exceeds bounded scan" in rendered
    card.release_transcript_source()
    store.close()


@pytest.mark.asyncio
async def test_failing_agent_source_does_not_stop_batch_or_spinner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    from zeta.tui.cards.agent import AgentCard

    cards = []
    for index in range(3):
        child = ConversationStore(tmp_path / "agents", session_id=str(index))
        child.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"child-{index}")])
        )
        child.close()
        card = AgentCard(ToolCall(f"agent-{index}", "agent", {"prompt": "inspect"}))
        card.set_child_session_path(str(child.session_dir))
        cards.append(card)

    failing_path = cards[1].transcript_source.path
    original_refresh = agent_sync_module.AgentTranscriptSource.refresh

    def fail_one_source(self, *, recursive: bool = False):
        if self.path == failing_path:
            raise OSError("broken child")
        return original_refresh(self, recursive=recursive)

    monkeypatch.setattr(
        agent_sync_module.AgentTranscriptSource,
        "refresh",
        fail_one_source,
    )

    class Presenter:
        has_active_agent = True

        def __init__(self) -> None:
            self.calls = 0

        async def refresh_active_agent_transcripts(self) -> None:
            self.calls += 1
            await agent_sync_module.refresh_agent_cards(cards)

    class Spinner(TurnConsumerMixin):
        pass

    monkeypatch.setattr(composer_module, "SPINNER_INTERVAL", 0.001)
    monkeypatch.setattr(composer_module, "AGENT_TRANSCRIPT_REFRESH_INTERVAL", 0.0)
    caplog.set_level("WARNING", logger=agent_sync_module.__name__)
    spinner = Spinner()
    spinner._presenter = Presenter()
    spinner._spinner_reset = asyncio.Event()
    spinner._spinner_active = True
    spinner._spinner_frame = 0
    spinner._invalidate_prompt = lambda: None
    task = asyncio.create_task(spinner._pulse_spinner())
    try:
        for _ in range(100):
            if spinner._spinner_frame >= 3:
                break
            await asyncio.sleep(0.005)
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    assert spinner._spinner_frame >= 3
    assert spinner._presenter.calls >= 3
    assert cards[0]._tail == ("assistant: child-0",)
    assert cards[1]._tail == ("child transcript unavailable: refresh failed",)
    assert cards[2]._tail == ("assistant: child-2",)
    errors = [record for record in caplog.records if "broken child" in record.message]
    assert len(errors) == 1
    for card in cards:
        card.release_transcript_source()
