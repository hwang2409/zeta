from __future__ import annotations

import asyncio
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import time
import uuid
from collections.abc import AsyncIterator, Callable, Sequence
from dataclasses import replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import set_app
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from prompt_toolkit.data_structures import Size
from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from zeta.core.approval import ApprovalPolicy
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.tools import ToolStreamPublisher
from zeta.tui.app import FullScreenPromptSession, TUIApp
from zeta.tui.composer import (
    build_key_bindings,
    history_for,
    parse_input,
)
from zeta.tui.layout import content_width
from zeta.tui.render import (
    MarkdownStream,
    collapse_thought,
    format_status,
    format_thought,
    render_code,
    render_event,
    render_markdown,
    render_tool_progress,
    tool_render_mode,
)
from zeta.tui.theme import ACCENT, BODY, DIM, RICH_THEME
from zeta.tui.transcript import TranscriptWidget
from zeta.types import (
    CompletionBackend,
    ErrorInfo,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
    ThinkingContent,
)


class GateBackend(CompletionBackend):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        index = len(self.calls)
        self.calls.append(list(messages))
        if index == 0:
            self.started.set()
            await self.release.wait()
        response = f"reply {index}\n"
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=response)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(response)]),
        )


class QueueOrderBackend(CompletionBackend):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()
        self.calls: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        index = len(self.calls)
        self.calls.append(list(messages))
        if index == 0:
            yield StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=TextContent("| name | value |\n| --- | --- |"),
            )
            self.started.set()
            await self.release.wait()
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(f"reply {index}")]),
        )


class ErrorBackend(CompletionBackend):
    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("| name | value |\n| --- | --- |"),
        )
        raise RuntimeError("boom")


class EventBackend(CompletionBackend):
    def __init__(self, event: StreamEvent) -> None:
        self.event = event

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("| name | value |\n| --- | --- |"),
        )
        yield self.event


class BlockingToolBackend(CompletionBackend):
    def __init__(self, call: ToolCall) -> None:
        self.call = call

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        blocks = [TextContent("using tool"), ToolUseContent(self.call)]
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[0])
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class StreamingToolBackend(CompletionBackend):
    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self.tool_update = asyncio.Event()
        self.allow_message_end = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ToolUseContent(self.call),
        )
        self.tool_update.set()
        await self.allow_message_end.wait()
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(
                MessageRole.ASSISTANT,
                [ToolUseContent(self.call)],
            ),
        )


class OrderedToolBackend(CompletionBackend):
    def __init__(self, calls: list[ToolCall]) -> None:
        self.calls = calls
        self.index = 0

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        call_index = self.index
        self.index += 1
        if call_index in {0, 2}:
            call = self.calls[call_index // 2]
            text = f"assistant {call_index // 2 + 1}\n"
            blocks = [TextContent(text), ToolUseContent(call)]
        else:
            blocks = [TextContent(f"assistant after tool {call_index}\n")]
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[0])
        if len(blocks) > 1:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[1])
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class SlowSecondCompletionBackend(CompletionBackend):
    def __init__(self, call: ToolCall) -> None:
        self.call = call
        self.second_started = asyncio.Event()
        self.release_second = asyncio.Event()
        self.index = 0

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        index = self.index
        self.index += 1
        if index == 0:
            blocks = [TextContent("before tool\n"), ToolUseContent(self.call)]
        else:
            blocks = [TextContent("after tool")]
            self.second_started.set()
            await self.release_second.wait()
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[0])
        if len(blocks) > 1:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[1])
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


async def wait_until(check: Callable[[], bool]) -> None:
    for _ in range(100):
        if check():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition did not become true")


def app_session(app: TUIApp, pipe: PipeInput) -> PromptSession[str]:
    return PromptSession(
        input=pipe,
        output=DummyOutput(),
        key_bindings=build_key_bindings(
            on_interrupt=app.abort_active,
            on_exit=app.request_exit,
        ),
        multiline=True,
    )


def fast_vi_timeouts(session: PromptSession[str]) -> None:
    session.app.ttimeoutlen = 0.02
    session.app.timeoutlen = 0.02


def paced_vi_timeouts(session: PromptSession[str]) -> None:
    session.app.ttimeoutlen = 0.02
    session.app.timeoutlen = 0.5


def renderable_plain(renderable: object) -> str:
    if hasattr(renderable, "plain"):
        return renderable.plain
    inner = getattr(renderable, "renderable", None)
    if inner is not None and hasattr(inner, "plain"):
        return inner.plain
    raise AssertionError(f"renderable has no plain text: {renderable!r}")


def test_render_event_compacts_tool_call_and_result() -> None:
    call = ToolCall("call-1", "read", {"path": "README.md", "extra": "x"})
    start = render_event(StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call))
    result = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, "first line\nsecond line"),
        )
    )

    assert start is not None
    assert "read" in renderable_plain(start)
    assert "README.md" in renderable_plain(start)
    assert result is not None
    assert result.plain == "⏺ read README.md"


def test_render_event_shows_tool_output_update() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_UPDATE,
            delta="hello\n",
            data={"stream": "stdout"},
        )
    )

    assert rendered is not None
    assert rendered.plain == "  ↳ [stdout] hello\n"


def test_transcript_follows_tail_until_scrolled_up() -> None:
    transcript = TranscriptWidget()
    for index in range(8):
        transcript.append(Text(f"line {index}"))

    transcript.create_content(80, 3)
    assert transcript.follow_tail
    tail_offset = transcript.scroll_offset

    transcript.append(Text("new tail"))
    transcript.create_content(80, 3)
    assert transcript.follow_tail
    assert transcript.scroll_offset == tail_offset + 1

    transcript.page_up()
    held_offset = transcript.scroll_offset
    assert not transcript.follow_tail
    transcript.append(Text("while scrolled"))
    transcript.create_content(80, 3)
    assert transcript.scroll_offset == held_offset
    assert not transcript.follow_tail

    transcript.page_down()
    transcript.page_down()
    transcript.create_content(80, 3)
    assert transcript.follow_tail
    assert transcript.scroll_offset == len(transcript.lines(80)) - 3


def test_transcript_resize_preserves_anchor_and_tail_reentry() -> None:
    transcript = TranscriptWidget()
    for index in range(8):
        transcript.append(Text(f"item-{index} abcdefgh"))

    transcript.create_content(20, 3)
    transcript.page_up()
    transcript.create_content(20, 3)
    wide_anchor = transcript._line_locations[transcript.scroll_offset][0]
    transcript.create_content(10, 3)
    assert transcript._line_locations[transcript.scroll_offset][0] is wide_anchor
    assert not transcript.follow_tail

    transcript = TranscriptWidget()
    for index in range(8):
        transcript.append(Text(f"item-{index} abcdefgh"))
    transcript.create_content(10, 3)
    transcript.page_up()
    transcript.create_content(10, 3)
    transcript.create_content(20, 3)
    assert transcript.follow_tail
    transcript.append(Text("new tail"))
    transcript.create_content(20, 3)
    assert transcript.scroll_offset == len(transcript._parsed_lines(20)) - 3


def test_transcript_resize_preserves_offset_in_long_wrapped_unit() -> None:
    source = " ".join(f"token-{index:03}" for index in range(500))
    transcript = TranscriptWidget()
    transcript.append(Text(source))

    transcript.create_content(80, 3)
    transcript.page_up()
    transcript.create_content(80, 3)
    wide_line = transcript.lines(80)[transcript.scroll_offset]
    anchor_offset = source.index(transcript._strip_padding(wide_line))

    transcript.create_content(20, 3)
    narrow_line = transcript.lines(20)[transcript.scroll_offset]
    mapped_offset = source.index(transcript._strip_padding(narrow_line))

    assert mapped_offset == anchor_offset
    assert not transcript.follow_tail


def test_transcript_resize_round_trip_preserves_canonical_token() -> None:
    source = " ".join(f"token-{index:03}" for index in range(500))
    transcript = TranscriptWidget()
    transcript.append(Text(source))
    transcript.create_content(20, 3)
    for _ in range(10):
        transcript.scroll_up()
    transcript.create_content(20, 3)
    initial_token = transcript.lines(20)[transcript.scroll_offset].strip().split()[0]

    transcript.create_content(80, 3)
    transcript.create_content(20, 3)

    round_trip_token = transcript.lines(20)[transcript.scroll_offset].strip().split()[0]
    assert round_trip_token == initial_token
    assert not transcript.follow_tail


def test_transcript_resize_cycles_have_zero_anchor_drift() -> None:
    source = " ".join(f"token-{index:03}" for index in range(500))
    transcript = TranscriptWidget()
    transcript.append(Text(source))
    transcript.create_content(20, 3)
    for _ in range(10):
        transcript.scroll_up()
    transcript.create_content(20, 3)
    initial_anchor = transcript._anchor

    for _ in range(3):
        transcript.create_content(80, 3)
        transcript.create_content(20, 3)
        assert transcript._anchor == initial_anchor
        assert transcript._line_locations[transcript.scroll_offset] == initial_anchor


def test_transcript_same_width_repaint_preserves_blank_anchor() -> None:
    transcript = TranscriptWidget()
    for index in range(20):
        transcript.append(Text(f"line-{index}"))
        transcript.append_blank()

    transcript.create_content(80, 3)
    transcript._set_scroll_offset(33)
    initial_anchor = transcript._anchor
    assert initial_anchor is not None
    assert initial_anchor[0] is not None
    assert initial_anchor[0].value is None

    transcript.create_content(80, 3)

    assert transcript.scroll_offset == 33
    assert transcript._line_locations[transcript.scroll_offset] == initial_anchor


def test_transcript_resize_cycles_preserve_blank_anchor() -> None:
    transcript = TranscriptWidget()
    for index in range(20):
        transcript.append(Text(f"line-{index} " + "x" * 40))
        transcript.append_blank()

    transcript.create_content(80, 3)
    transcript._set_scroll_offset(33)
    initial_anchor = transcript._anchor
    assert initial_anchor is not None
    assert initial_anchor[0] is not None
    assert initial_anchor[0].value is None

    for width in (20, 80, 20, 80):
        transcript.create_content(width, 3)
        assert transcript._anchor == initial_anchor
        assert transcript._line_locations[transcript.scroll_offset] == initial_anchor


def test_transcript_resize_preserves_anchor_in_unbroken_unit() -> None:
    source = "".join(f"{index:03}" for index in range(500))
    transcript = TranscriptWidget()
    transcript.append(Text(source))
    transcript.create_content(20, 3)
    for _ in range(10):
        transcript.scroll_up()
    transcript.create_content(20, 3)
    initial_anchor = transcript._anchor

    assert len(transcript.units) == 1
    assert len(transcript.lines(20)) > 10
    for width in (80, 20, 80, 20, 80, 20):
        transcript.create_content(width, 3)

    assert transcript._anchor == initial_anchor
    assert transcript._line_locations[transcript.scroll_offset] == initial_anchor


def test_transcript_parsed_cache_is_bounded_and_revision_scoped() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("line"))

    for width in (20, 30, 40, 50):
        transcript._parsed_lines(width)
    assert len(transcript._parsed_cache) == 3

    transcript.append(Text("new line"))
    assert not transcript._parsed_cache


def test_transcript_cache_uses_stable_keys_after_tool_discard() -> None:
    transcript = TranscriptWidget()
    call = ToolCall("old", "read", {"path": "old.txt"})
    transcript.start_tool(call.id, call, Text("old"))
    transcript.render(80)
    old_unit = transcript._units[-1]
    assert old_unit is not None
    old_key = old_unit.key

    transcript.discard_tools()
    assert old_key not in transcript._render_cache

    replacement = ToolCall("new", "read", {"path": "new.txt"})
    transcript.start_tool(replacement.id, replacement, Text("new"))
    new_unit = transcript._units[-1]
    assert new_unit is not None
    assert new_unit.key != old_key
    assert "new" in transcript.render(80)


def test_tool_output_strips_terminal_controls() -> None:
    call = ToolCall("ansi-1", "bash", {})
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, "a\x1b[2Jb\x1b]0;title\x07c"),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "abc" in plain
    assert "2J" not in plain
    assert "title" not in plain


@pytest.mark.asyncio
async def test_streamed_tool_output_is_not_repeated_at_end(tmp_path: Path) -> None:
    call = ToolCall("call-1", "bash", {"cmd": "printf chunk"})
    app = TUIApp(
        AgentLoop(
            FakeBackend(
                [
                    ScriptedTurn(tool_calls=[call]),
                    ScriptedTurn([TextContent("done")]),
                ]
            ),
            ConversationStore(tmp_path),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    rendered: list[object] = []
    app._print = rendered.append

    await app._consume_turn("prompt")

    expected_start = render_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    )
    expected_end = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, "stdout:\nchunk\nstderr:\n"),
        )
    )
    assert expected_start is not None
    assert expected_end is not None
    start_idx = next(
        index
        for index, item in enumerate(rendered)
        if renderable_plain(item) == renderable_plain(expected_start)
    )
    end_idx = next(
        index
        for index, item in enumerate(rendered)
        if renderable_plain(item) == renderable_plain(expected_end)
    )
    tool_region = "\n".join(
        renderable_plain(item) for item in rendered[start_idx : end_idx + 1]
    )
    assert tool_region == f"{renderable_plain(expected_start)}\n{renderable_plain(expected_end)}"
    assert "  ↳ [stdout] chunk" not in tool_region


@pytest.mark.asyncio
async def test_tui_overflow_final_render_is_authoritative(tmp_path: Path) -> None:
    call = ToolCall("burst-1", "stream", {})
    final_text = "".join(f"chunk-{index}\n" for index in range(200))

    async def stream(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: ToolStreamPublisher,
    ) -> str:
        del arguments, abort_signal
        for index in range(200):
            publisher.publish(f"chunk-{index}\n", "stdout")
        return final_text

    app = TUIApp(
        AgentLoop(
            FakeBackend([ScriptedTurn(tool_calls=[call])]),
            ConversationStore(tmp_path),
            tools={"stream": stream},
            max_turns=1,
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    rendered: list[object] = []
    app._print = rendered.append

    await app._consume_turn("prompt")

    expected = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, final_text),
        )
    )
    assert expected is not None
    expected_start = render_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    )
    assert expected_start is not None
    start_idx = next(
        index
        for index, item in enumerate(rendered)
        if renderable_plain(item) == renderable_plain(expected_start)
    )
    end_idx = next(
        index
        for index, item in enumerate(rendered)
        if renderable_plain(item) == renderable_plain(expected)
    )
    tool_region = "\n".join(
        renderable_plain(item) for item in rendered[start_idx : end_idx + 1]
    )
    assert tool_region == f"{renderable_plain(expected_start)}\n{renderable_plain(expected)}"
    for index in range(200):
        assert f"  ↳ [stdout] chunk-{index}" not in tool_region


@pytest.mark.asyncio
async def test_tui_cancel_replaces_streamed_region_with_canceled_render(
    tmp_path: Path,
) -> None:
    call = ToolCall("cancel-tui", "stream", {})

    async def stream(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: ToolStreamPublisher,
    ) -> str:
        del arguments, abort_signal
        publisher.publish("first\n", "stdout")
        await asyncio.Event().wait()
        return "unreachable"

    store = ConversationStore(tmp_path)
    app = TUIApp(
        AgentLoop(
            FakeBackend([ScriptedTurn(tool_calls=[call])]),
            store,
            tools={"stream": stream},
            max_turns=1,
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    rendered: list[object] = []
    app._print = rendered.append
    task = asyncio.create_task(app._consume_turn("prompt"))
    app._active_task = task

    await wait_until(lambda: app._tool_region is not None)
    app.abort_active()
    await task

    expected = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "tool execution canceled",
                is_error=True,
            ),
        )
    )
    assert expected is not None
    expected_start = render_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    )
    assert expected_start is not None
    start_idx = next(
        index
        for index, item in enumerate(rendered)
        if renderable_plain(item) == renderable_plain(expected_start)
    )
    end_idx = next(
        index
        for index, item in enumerate(rendered)
        if renderable_plain(item) == renderable_plain(expected)
    )
    tool_region = "\n".join(
        renderable_plain(item) for item in rendered[start_idx : end_idx + 1]
    )
    assert tool_region == f"{renderable_plain(expected_start)}\n{renderable_plain(expected)}"
    assert "  ↳ [stdout] first" not in tool_region
    assert app._loop_state == "interrupted"
    results = [
        message.tool_result
        for message in store.messages()
        if message.tool_result
    ]
    assert len(results) == 1
    assert results[0] is not None
    assert results[0].content == "tool execution canceled"


def test_render_event_preserves_multiline_tool_result_formatting() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_result=ToolResult("call-1", "first\n\n  third"),
        )
    )

    assert rendered is not None
    assert "first" in renderable_plain(rendered)
    assert "third" in renderable_plain(rendered)


def test_render_event_shows_tool_result_truncation_metadata() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_result=ToolResult(
                "call-1",
                "abcd\n[truncated: 4 of 8 chars shown]",
                content_blocks=[
                    {
                        "type": "text",
                        "text": "abcd",
                        "truncated": True,
                        "full_size": 8,
                    }
                ],
            ),
        )
    )

    assert rendered is not None
    assert "[truncated; full_size=8]" in renderable_plain(rendered)


def test_render_event_shows_non_text_tool_block_placeholders() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_result=ToolResult(
                "call-1",
                "",
                content_blocks=[
                    {
                        "type": "text",
                        "text": "answer",
                        "truncated": False,
                        "full_size": 6,
                    },
                    {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
                    {
                        "type": "resource",
                        "resource": {"uri": "file:///tmp/note.txt", "text": "note"},
                    },
                ],
            ),
        )
    )

    assert rendered is not None
    rendered_text = renderable_plain(rendered)
    assert "answer" in rendered_text
    assert "[image block]" in rendered_text
    assert "[resource: file:///tmp/note.txt]" in rendered_text


def test_render_helpers_use_the_zeta_palette() -> None:
    markdown = render_markdown("# heading")
    code = render_code("print('hi')", "python")
    start = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_START,
            tool_call=ToolCall("call-1", "bash", {"cmd": "printf hi"}),
        )
    )

    assert markdown.style == BODY
    assert code.background_color is None
    assert start is not None
    styled_text = start if hasattr(start, "spans") else start.renderable
    assert any(ACCENT in str(span.style) for span in styled_text.spans)


def test_render_event_error_is_visible() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("backend_error", "provider stopped"),
        )
    )

    assert rendered is not None
    assert rendered.plain == "[error] provider stopped"


def test_tool_render_mode_keeps_receipt_rule_in_one_place() -> None:
    read = ToolCall("read-1", "read", {"path": "README.md"})
    short = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=read,
        tool_result=ToolResult(read.id, "one\ntwo"),
    )
    long_read = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=read,
        tool_result=ToolResult(read.id, "one\ntwo\nthree"),
    )
    glob = ToolCall("glob-1", "glob", {"pattern": "**/*.py"})
    search = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=glob,
        tool_result=ToolResult(glob.id, "\n".join(f"file-{i}.py" for i in range(38))),
    )
    bash = ToolCall("bash-1", "bash", {"cmd": "printf hi"})
    generic = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=bash,
        tool_result=ToolResult(bash.id, "hi"),
    )

    assert tool_render_mode(short) == "receipt"
    assert tool_render_mode(long_read) == "card"
    assert tool_render_mode(search) == "receipt"
    assert tool_render_mode(generic) == "card"


def test_long_single_line_read_uses_a_cropped_card() -> None:
    call = ToolCall("read-long", "read", {"path": "README.md"})
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(call.id, "x" * 240),
    )

    assert tool_render_mode(event) == "card"
    rendered = render_event(event)
    assert rendered is not None
    assert "x" * 240 in renderable_plain(rendered)


def test_receipt_mode_forces_errors_and_non_text_results_into_cards() -> None:
    call = ToolCall("read-special", "read", {"path": "missing.txt"})
    error = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(call.id, "permission denied", is_error=True),
    )
    mixed = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "answer\n[image block]",
            content_blocks=[
                {"type": "text", "text": "answer", "truncated": False, "full_size": 6},
                {"type": "image", "data": "aGVsbG8=", "mimeType": "image/png"},
            ],
        ),
    )

    assert tool_render_mode(error) == "card"
    assert tool_render_mode(mixed) == "card"
    assert "permission denied" in renderable_plain(render_event(error))
    assert "[image block]" in renderable_plain(render_event(mixed))


def test_tool_card_truncates_at_fifteen_lines() -> None:
    call = ToolCall("long-1", "bash", {"cmd": "seq 30"})
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, "\n".join(f"line-{i}" for i in range(20))),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "line-0" in plain and "line-14" in plain
    assert "line-15" not in plain
    assert "… +5 lines" in plain


def test_live_tool_preview_reuses_the_fifteen_line_limit() -> None:
    call = ToolCall("live-1", "bash", {"cmd": "seq 20"})
    rendered = render_tool_progress(call, "\n".join(f"line-{i}" for i in range(20)))
    plain = renderable_plain(rendered)

    assert "line-14" in plain
    assert "line-15" not in plain
    assert "… +5 lines" in plain


def test_thought_collapses_to_first_sentence_and_keeps_duration() -> None:
    assert collapse_thought("Plan first. Hide the rest.") == "Plan first."
    assert collapse_thought("Use e.g. this value. Hide the rest.") == "Use e.g. this value."
    assert collapse_thought("") == ""
    assert collapse_thought("No final punctuation") == "No final punctuation"
    rendered = format_thought("Plan first. Hide the rest.", 2.7)

    assert rendered.plain == "✱ thought · Plan first. · 2.7s"
    assert "italic" in str(rendered.style)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ('Use U.S. defaults. Hide the rest.', "Use U.S. defaults."),
        ('Run at 5 p.m. today. Hide the rest.', "Run at 5 p.m. today."),
        ('It said "Done." Then continue.', 'It said "Done."'),
    ],
)
def test_thought_sentence_detection_handles_initialisms_and_quotes(
    value: str, expected: str
) -> None:
    assert collapse_thought(value) == expected


@pytest.mark.parametrize(
    "state", ["streaming", "tool-running", "idle", "interrupted", "compacting"]
)
def test_status_bar_supports_all_session_states(state: str) -> None:
    rendered = format_status(
        "fake",
        "offline",
        state,
        session_id="abcdef123456",
        token_count=12,
        width=120,
    )

    assert state in rendered.plain
    assert "abcdef12" in rendered.plain


@pytest.mark.parametrize("state", ["tool-running", "interrupted", "compacting"])
def test_special_status_states_override_spinner(state: str) -> None:
    rendered = format_status(
        "fake",
        "offline",
        state,
        streaming=True,
        spinner_active=True,
        spinner_frame=2,
    )

    if state == "tool-running":
        assert rendered.plain.startswith("● tool-running")
    else:
        assert rendered.plain.startswith(state)
    assert "esc interrupt" not in rendered.plain


def test_markdown_stream_highlights_complete_fence() -> None:
    stream = MarkdownStream()
    output = []
    output.extend(stream.consume("```python"))
    output.extend(stream.consume("print('hi')"))
    assert len(output) == 2
    assert type(output[1]).__name__ == "Syntax"
    output.extend(stream.consume("```"))

    assert len(output) == 3


def test_markdown_stream_requires_matching_four_backtick_fence() -> None:
    stream = MarkdownStream()

    assert stream.consume("````python")
    inner = stream.consume("```")
    assert type(inner[0]).__name__ == "Syntax"
    assert stream.language == "python"

    closing = stream.consume("````")
    assert closing[0].plain == "````"
    assert stream.language is None


def test_truncated_response_notice_is_dim() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.MESSAGE_END,
            data={"truncated": True},
        )
    )

    assert isinstance(rendered, Text)
    assert rendered.plain == "response truncated (stream ended early)"
    assert rendered.style == DIM

    transcript = TranscriptWidget()
    transcript.append(rendered)
    for width in (40, 80):
        lines = transcript.lines(width)
        assert sum(line.count(rendered.plain) for line in lines) == 1
        assert all(cell_len(Text.from_ansi(line).plain) <= width for line in lines)

    output = StringIO()
    Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=40,
        theme=RICH_THEME,
    ).print(rendered)
    escaped = output.getvalue()
    assert escaped.count(rendered.plain) == 1
    assert not re.search(r"\x1b\[[0-9;]*48(?:;[0-9;]*)?m", escaped)


@pytest.mark.asyncio
async def test_truncated_response_notice_is_printed_once(tmp_path: Path) -> None:
    class TruncatedBackend(CompletionBackend):
        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            yield StreamEvent(StreamEventType.MESSAGE_START)
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT),
                data={"truncated": True},
            )

    output = StringIO()
    app = TUIApp(
        AgentLoop(
            TruncatedBackend(),
            ConversationStore(tmp_path / "sessions"),
            tool_schemas=[],
        ),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )

    await app._consume_turn("hello")

    assert output.getvalue().count("response truncated (stream ended early)") == 1
    assert "no response" not in output.getvalue()


def test_markdown_stream_keeps_text_after_fence_inside_code_block() -> None:
    stream = MarkdownStream()
    stream.consume("```python")

    output = stream.consume("```still code")

    assert len(output) == 1
    assert isinstance(output[0], Syntax)
    assert output[0].code == "```still code"
    assert stream.language == "python"


def test_stream_kind_switch_flushes_assistant_before_thinking(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    rendered = []
    app._print = rendered.append

    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("| name | value |\n| --- | --- |"),
        )
    )
    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent("plan\n"),
        )
    )
    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("after\n"),
        )
    )

    assert isinstance(rendered[0], Table)
    assert rendered[1].plain.startswith("✱ thought · plan · ")
    assert rendered[1].plain.endswith("s")


def test_thought_duration_uses_local_monotonic_lifecycle_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter((10.0, 10.25))
    monkeypatch.setattr("zeta.tui.app.time.monotonic", lambda: next(ticks))
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    rendered: list[object] = []
    app._print = rendered.append

    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent("plan"),
        )
    )
    app._flush_pending_stream()

    assert rendered[0].plain == "✱ thought · plan · 0.2s"


def test_assistant_renderables_share_one_logical_unit(tmp_path: Path) -> None:
    output = StringIO()
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )

    app._print_committed(["first paragraph", "second paragraph"])
    app._print_committed(["```python", "print('hi')", "```"])
    rendered = "\n".join(line.rstrip() for line in output.getvalue().splitlines())

    assert "  first paragraph\n  second paragraph" in rendered
    assert "first paragraph\n\nsecond paragraph" not in rendered
    assert "print('hi')\n\n" not in rendered


@pytest.mark.asyncio
async def test_error_flushes_assistant_before_error(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(ErrorBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    rendered = []

    def capture(renderable: object | None) -> None:
        if renderable is not None:
            rendered.append(renderable)

    app._print = capture

    await app._consume_turn("prompt")

    table_index = next(
        index for index, item in enumerate(rendered) if isinstance(item, Table)
    )
    error_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "plain", None) == "[error] boom"
    )
    assert table_index < error_index


@pytest.mark.asyncio
async def test_verbose_error_flushes_before_raw_error(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(ErrorBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        verbose=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    rendered = []

    def capture(renderable: object | None) -> None:
        if renderable is not None:
            rendered.append(renderable)

    app._print = capture

    await app._consume_turn("prompt")

    table_index = next(
        index for index, item in enumerate(rendered) if isinstance(item, Table)
    )
    raw_error_index = next(
        index
        for index, item in enumerate(rendered)
        if '"type": "error"' in getattr(item, "plain", "")
    )
    pretty_error_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "plain", None) == "[error] boom"
    )
    assert table_index < raw_error_index < pretty_error_index


@pytest.mark.parametrize(
    ("event", "raw_marker"),
    [
        (
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_START,
                tool_call=ToolCall("call-1", "read", {}),
            ),
            "tool_execution_start",
        ),
        (
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=ToolCall("call-1", "read", {}),
                tool_result=ToolResult("call-1", "done"),
            ),
            "tool_execution_end",
        ),
        (
            StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=ThinkingContent("plan\n"),
            ),
            "plan",
        ),
    ],
)
@pytest.mark.asyncio
async def test_verbose_transition_flushes_before_raw_event(
    tmp_path: Path,
    event: StreamEvent,
    raw_marker: str,
) -> None:
    app = TUIApp(
        AgentLoop(EventBackend(event), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        verbose=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    rendered = []

    def capture(renderable: object | None) -> None:
        if renderable is not None:
            rendered.append(renderable)

    app._print = capture

    await app._consume_turn("prompt")

    table_index = next(
        index for index, item in enumerate(rendered) if isinstance(item, Table)
    )
    raw_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "plain", "").startswith("{")
        and raw_marker in item.plain
    )
    assert table_index < raw_index


@pytest.mark.asyncio
async def test_queued_user_output_waits_for_assistant_flush(tmp_path: Path) -> None:
    backend = QueueOrderBackend()
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )
    with create_pipe_input() as pipe:
        run_task = asyncio.create_task(app.run(app_session(app, pipe)))
        pipe.send_text("first\r")
        await backend.started.wait()
        pipe.send_text("second\r")
        await wait_until(lambda: app.queued_messages == ("second",))
        backend.release.set()
        await wait_until(lambda: len(backend.calls) == 2)
        pipe.send_text("\x04")
        await run_task

    rendered = output.getvalue()
    assert rendered.index("name") < rendered.index("▌ second")


def test_main_exits_on_ctrl_d_at_empty_prompt(tmp_path: Path) -> None:
    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env["ZETA_HOME"] = str(tmp_path / "zeta-home")
    env["TERM"] = "xterm-256color"
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from zeta.tui.app import main; raise SystemExit(main(['--provider', 'fake']))",
        ],
        stdin=slave_fd,
        stdout=slave_fd,
        stderr=slave_fd,
        env=env,
        close_fds=True,
    )
    os.close(slave_fd)
    try:
        output = bytearray()
        deadline = time.monotonic() + 5
        while " > ".encode() not in output and time.monotonic() < deadline:
            ready, _, _ = select.select(
                [master_fd],
                [],
                [],
                max(0, deadline - time.monotonic()),
            )
            if ready:
                output.extend(os.read(master_fd, 4096))
        assert " > ".encode() in output

        os.write(master_fd, b"\x04")
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if not ready:
                if process.poll() is not None:
                    break
                continue
            try:
                output.extend(os.read(master_fd, 4096))
            except OSError:
                break
        assert process.wait(timeout=5) == 0
        assert b"\x1b[?1049h" in output
        assert b"\x1b[?1049l" in output
        assert output.count(b"\x1b[?1049h") == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master_fd)


def test_main_import_compatibility() -> None:
    from zeta.cli import main as cli_main
    from zeta.tui import main as tui_main
    from zeta.tui.app import main as app_main

    assert tui_main is cli_main
    assert app_main is cli_main


def test_tui_import_does_not_load_cli() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import zeta.tui; raise SystemExit('zeta.cli' in sys.modules)",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_markdown_stream_renders_complete_table() -> None:
    stream = MarkdownStream()

    assert stream.consume("| name | value |") == []
    assert stream.consume("| --- | --- |") == []
    assert stream.consume("| one | two |") == []

    output = stream.consume("after")
    assert len(output) == 2
    assert isinstance(output[0], Table)
    assert output[0].columns[0].header == "name"
    assert output[0].columns[1].header == "value"
    assert isinstance(output[1], Markdown)


def test_status_includes_provider_state_and_usage() -> None:
    status = format_status("fake", "offline", "streaming", {"input_tokens": 2}, "partial")

    assert "streaming" in status.plain
    assert "2 (0%)" in status.plain


def test_footer_builder_formats_context_usage_and_hints() -> None:
    footer = format_status(
        "openai",
        "gpt-5.4",
        "idle",
        token_count=18_600,
        model_window=200_000,
        session_id="abcdef12",
    )

    assert footer.plain == (
        "idle  18.6K (9%)  /status · ctrl+c interrupt · ctrl+d quit · abcdef12"
    )


def test_footer_shows_vim_state_and_degrades_as_a_whole_segment() -> None:
    footer = format_status(
        "fake",
        "offline",
        "idle",
        token_count=14,
        session_id="abcdef12",
        width=80,
        vim_state="NORMAL",
    )
    assert footer.plain.startswith("NORMAL  idle")

    narrow = format_status(
        "fake",
        "offline",
        "idle",
        token_count=14,
        session_id="abcdef12",
        width=30,
        vim_state="NORMAL",
    )
    assert "NORMAL" not in narrow.plain
    assert "idle" in narrow.plain


def test_status_bar_includes_session_context_and_streaming_indicator() -> None:
    status = format_status(
        "codex",
        "gpt-5.4",
        "streaming",
        {"input_tokens": 5, "output_tokens": 121},
        session_id="abc12345",
        token_count=42,
        retained_tail=8,
        streaming=True,
        width=120,
        spinner_frame=1,
    )

    assert len(status.plain) <= 120
    assert "ctrl+c interrupt" in status.plain
    assert "42 (0%)" in status.plain
    assert "/status" in status.plain
    assert "ctrl+d quit" in status.plain


def test_status_bar_drops_whole_segments_at_narrow_widths() -> None:
    for width in range(10, 81):
        status = format_status(
            "fake",
            "offline",
            "idle",
            token_count=14,
            session_id="abcdef12",
            width=width,
        )

        assert cell_len(status.plain) <= width
        assert "/stat" not in status.plain or "/status" in status.plain


def test_status_bar_fits_segments_and_pulses() -> None:
    statuses = [
        format_status(
            "codex",
            "gpt-5.4",
            "streaming",
            {"input_tokens": 5, "output_tokens": 121},
            session_id="abc12345",
            retained_tail=8,
            streaming=True,
            width=width,
            spinner_frame=frame,
        )
        for width, frame in ((80, 0), (120, 1), (200, 2))
    ]

    assert all(len(status.plain) <= width for status, width in zip(statuses, (80, 120, 200)))
    assert all("ctrl+c interrupt" in status.plain and "5 (0%)" in status.plain for status in statuses)
    assert all(marker in statuses[index].plain for index, marker in enumerate(("·", "•", "●")))
    assert all(value in statuses[1].plain for value in ("/status", "ctrl+d quit"))
    assert all(value in statuses[2].plain for value in ("abc12345", "/status"))

    narrow = format_status(
        "provider-with-a-long-name",
        "model-with-a-long-name-that-does-not-fit",
        "streaming",
        {"input_tokens": 5, "output_tokens": 121},
        session_id="abcdef1234567890",
        retained_tail=8,
        streaming=True,
        width=80,
    )
    assert len(narrow.plain) <= 80
    assert "ctrl+c interrupt" in narrow.plain
    assert "5 (0%)" in narrow.plain

    cleared = format_status(
        "codex",
        "gpt-5.4",
        "idle",
        session_id="abc12345",
        retained_tail=8,
        streaming=False,
        width=80,
    )
    assert "idle" in cleared.plain
    assert "abc12345" in cleared.plain


def test_full_screen_layout_pins_composer_and_footer(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._install_full_screen_layout(session)

    root = session.layout.container
    assert len(root.children) == 1
    padded = root.children[0]
    assert padded.__class__.__name__ == "VSplit"
    assert padded.children[0].__class__.__name__ == "Window"
    content = padded.children[1]
    assert content.__class__.__name__ == "HSplit"
    assert content.children[0].__class__.__name__ == "Window"
    bottom = content.children[1]
    assert bottom.__class__.__name__ == "HSplit"
    assert bottom.children[-1].__class__.__name__ == "ConditionalContainer"


@pytest.mark.asyncio
async def test_full_screen_steady_state_paint_does_not_clear_screen(
    tmp_path: Path,
) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        history_path=tmp_path / "history",
    )
    session = app._make_session()
    app._install_full_screen_layout(session)
    output_text = StringIO()
    output = Vt100_Output(
        output_text,
        lambda: Size(rows=45, columns=200),
    )
    session.app.output = output
    session.app.renderer.output = output
    session.app.renderer.full_screen = True

    with set_app(session.app):
        session.app.renderer.render(session.app, session.app.layout)
        output_text.seek(0)
        output_text.truncate(0)
        app._spinner_active = True
        app._spinner_frame = 1
        session.app.renderer.render(session.app, session.app.layout)

    steady_state = output_text.getvalue()
    assert "\x1b[J" not in steady_state
    assert "\x1b[2J" not in steady_state


@pytest.mark.parametrize(
    ("terminal_width", "right_segments"),
    [(120, ("/status", "ctrl+c interrupt", "ctrl+d quit", "abcdef12")),
     (80, ("/status", "ctrl+c interrupt", "ctrl+d quit", "abcdef12")),
     (40, ("abcdef12",))],
)
def test_full_screen_footer_fits_content_column(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    terminal_width: int,
    right_segments: tuple[str, ...],
) -> None:
    app = TUIApp(
        AgentLoop(
            GateBackend(),
            ConversationStore(tmp_path / "sessions", session_id="abcdef123456"),
        ),
        provider="fake",
        model="offline",
    )
    output = SimpleNamespace(
        get_size=lambda: Size(rows=24, columns=terminal_width),
    )
    monkeypatch.setattr("zeta.tui.app.get_app", lambda: SimpleNamespace(output=output))

    footer = "".join(value for _, value in app._status_toolbar())
    content_width = terminal_width - 4

    assert cell_len(footer) <= content_width
    assert all(segment in footer for segment in right_segments)
    if terminal_width == 40:
        assert all(
            segment not in footer
            for segment in ("/status", "ctrl+c interrupt")
        )


def test_rich_rendering_does_not_paint_terminal_background() -> None:
    output = StringIO()
    console = Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=80,
        theme=RICH_THEME,
    )
    console.print(render_markdown("# heading\n\n`inline`"))
    console.print(render_code("print('hi')", "python"))
    console.print(
        render_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_START,
                tool_call=ToolCall("call-1", "bash", {"cmd": "pwd"}),
            )
        )
    )

    assert not re.search(r"\x1b\[[0-9;]*48(?:;[0-9;]*)?m", output.getvalue())


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
@pytest.mark.parametrize(("columns", "rows"), [(200, 45), (80, 24)])
def test_full_screen_pty_keeps_padded_margins_clean(
    tmp_path: Path, columns: int, rows: int
) -> None:
    session = f"zeta-pty-{uuid.uuid4().hex[:10]}"
    zeta = Path(sys.executable).with_name("zeta")
    env = os.environ.copy()
    env["ZETA_HOME"] = str(tmp_path / "zeta-home")
    env["TERM"] = "xterm-256color"
    subprocess.run(
        [
            "tmux",
            "new-session",
            "-d",
            "-s",
            session,
            "-x",
            str(columns),
            "-y",
            str(rows),
            str(zeta),
            "--provider",
            "fake",
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        check=True,
    )
    try:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            capture = subprocess.run(
                ["tmux", "capture-pane", "-t", session, "-p"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            if " > type a message..." in capture:
                break
            time.sleep(0.05)
        else:
            pytest.fail("zeta did not render the full-screen prompt")

        subprocess.run(
            ["tmux", "send-keys", "-t", session, "hello", "Enter"],
            check=True,
        )
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            capture = subprocess.run(
                ["tmux", "capture-pane", "-t", session, "-p"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
            if "you said: hello" in capture:
                break
            time.sleep(0.05)
        else:
            pytest.fail("fake provider response did not render")

        plain = subprocess.run(
            ["tmux", "capture-pane", "-t", session, "-p"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        escaped = subprocess.run(
            ["tmux", "capture-pane", "-e", "-t", session, "-p"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        assert any(line.startswith("  ▌ hello") for line in plain)
        assert all(not line[:2].strip() for line in plain)
        assert not re.search(r"\x1b\[[0-9;]*48(?:;[0-9;]*)?m", escaped)
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)


@pytest.mark.parametrize(("width", "height"), [(120, 40), (80, 24), (40, 12)])
def test_transcript_visual_snapshot_is_compact_and_bottom_aligned(
    width: int, height: int
) -> None:
    transcript = TranscriptWidget()
    call = ToolCall("visual", "bash", {"cmd": "pwd"})
    transcript.append(Text.assemble(("▌ ", ACCENT), ("inspect the session", BODY)))
    transcript.append_blank()
    transcript.append(render_markdown("## result\n\n1. first item\n2. second item"))
    transcript.append_blank()
    transcript.append(
        render_event(
            StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
        )
    )
    transcript.append_blank()
    transcript.append(
        render_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=ToolResult(call.id, "done"),
            )
        )
    )
    transcript.append_blank()
    transcript.append(
        Text(
            "[approval pending] request-1: bash; type approve request-1 or deny request-1",
            style=ACCENT,
        )
    )

    lines = transcript.lines(width)
    plain_lines = [Text.from_ansi(line).plain for line in lines]
    assert all(line == line.rstrip() for line in plain_lines)
    assert all(len(line) <= width for line in plain_lines)
    assert all(line.strip() != "|" for line in plain_lines)

    content = transcript.create_content(width, height)
    visible = [
        "".join(fragment[1] for fragment in content.get_line(index))
        for index in range(content.line_count)
    ]
    parsed = transcript._parsed_lines(width)
    prefix = max(0, height - len(parsed))
    assert visible[prefix:] == [
        "".join(fragment[1] for fragment in line) for line in parsed
    ]
    assert all(not line for line in visible[:prefix])
    assert "request-1" in visible[-1]


def test_full_screen_transcript_drops_markdown_list_placeholder_row(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._active_session = app._make_session()
    output = SimpleNamespace(get_size=lambda: Size(rows=24, columns=40))
    monkeypatch.setattr("zeta.tui.app.get_app", lambda: SimpleNamespace(output=output))

    app._append_transcript(render_markdown("1. first item"))

    assert app._transcript_lines
    assert app._transcript_lines[0].strip() == "1 first item"


def test_full_screen_transcript_reflows_logical_text_at_narrow_widths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    columns = 40
    output = SimpleNamespace(get_size=lambda: Size(rows=12, columns=columns))
    monkeypatch.setattr("zeta.tui.app.get_app", lambda: SimpleNamespace(output=output))

    app._append_transcript(Text("abcdefghijk"))
    wide_lines = app._transcript_lines
    columns = 8
    narrow_lines = app._transcript_lines

    assert wide_lines
    assert narrow_lines
    plain = Text.from_ansi("\n".join(narrow_lines)).plain.replace(" ", "").replace("\n", "")
    assert "abcdefghijk" in plain


def test_app_status_prefers_latest_provider_usage(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(
            GateBackend(),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
    )
    app._update_usage(
        StreamEvent(
            StreamEventType.MESSAGE_END,
            data={"usage": {"input_tokens": 5, "output_tokens": 121}},
        )
    )

    toolbar = app._status_toolbar()
    plain = "".join(value for _, value in toolbar)
    assert "5 (0%)" in plain
    assert "/status" in plain
    assert "\n" not in plain


def test_status_toolbar_preserves_vim_state_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    output = SimpleNamespace(get_size=lambda: Size(rows=24, columns=80))
    monkeypatch.setattr(
        "zeta.tui.app.get_app",
        lambda: SimpleNamespace(output=output),
    )
    monkeypatch.setattr(
        "zeta.tui.composer.get_app",
        lambda: SimpleNamespace(output=output),
    )

    toolbar = app._status_toolbar()

    assert any(
        text == "INSERT" and "bold" in style and "#ff8a1f" in style
        for style, text in toolbar
    )


def test_status_toolbar_does_not_advance_spinner_frame(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._spinner_frame = 0
    app._streaming = False
    app._status_toolbar()
    assert app._spinner_frame == 0

    app._streaming = True
    app._status_toolbar()
    assert app._spinner_frame == 0


@pytest.mark.asyncio
async def test_spinner_pulses_on_timer(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._streaming = True
    app._spinner_active = True
    task = asyncio.create_task(app._pulse_spinner())
    await asyncio.sleep(0.45)
    app._streaming = False
    app._spinner_active = False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert app._spinner_frame >= 2


@pytest.mark.asyncio
async def test_spinner_restarts_for_completion_after_tool(tmp_path: Path) -> None:
    call = ToolCall("call-1", "read", {})
    backend = SlowSecondCompletionBackend(call)
    tool_started = asyncio.Event()

    async def slow_tool(_: dict[str, object]) -> str:
        tool_started.set()
        await asyncio.sleep(0.8)
        return "tool result"

    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions"),
            tools={"read": slow_tool},
        ),
        provider="fake",
        model="offline",
    )

    turn = asyncio.create_task(app._consume_turn("prompt"))

    async def observe_tool_spinner() -> tuple[int, int, str]:
        await tool_started.wait()
        frame_before = app._spinner_frame
        await asyncio.sleep(0.75)
        frame_after = app._spinner_frame
        toolbar = "".join(value for _, value in app._status_toolbar())
        return frame_before, frame_after, toolbar

    frame_before, frame_after, during_tool_toolbar = await asyncio.wait_for(
        observe_tool_spinner(), timeout=2.0
    )
    await backend.second_started.wait()
    starting_frame = app._spinner_frame
    await asyncio.sleep(0.06)
    first_provider_frame = app._spinner_frame
    await asyncio.sleep(0.39)
    ending_frame = app._spinner_frame
    backend.release_second.set()
    await turn

    assert app._streaming is False
    assert frame_after > frame_before, "spinner did not advance during tool execution"
    assert any(marker in during_tool_toolbar for marker in ("·", "•", "●"))
    assert starting_frame == 0
    assert first_provider_frame == 0
    assert app._spinner_frame >= 1
    assert ending_frame - starting_frame >= 2


@pytest.mark.asyncio
async def test_app_abort_enters_interrupted_state(tmp_path: Path) -> None:
    backend = GateBackend()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    states: list[str] = []
    app._invalidate_prompt = lambda: states.append(app._loop_state)
    turn = asyncio.create_task(app._consume_turn("prompt"))
    app._active_task = turn
    await backend.started.wait()

    app.abort_active()
    await asyncio.gather(turn, return_exceptions=True)

    assert "interrupted" in states
    assert app._loop_state == "interrupted"
    assert "interrupted" in format_status(
        app.provider,
        app.model,
        app._loop_state,
        session_id="session",
    ).plain


@pytest.mark.asyncio
async def test_app_surfaces_compacting_state_before_provider_output(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    assembler = app.loop.context_assembler
    assemble = assembler.assemble
    states: list[str] = []

    async def compacted_assemble(*args: object, **kwargs: object) -> list[Message]:
        messages = await assemble(*args, **kwargs)
        assert assembler.last_context is not None
        assembler.last_context = replace(assembler.last_context, compacted=True)
        return messages

    assembler.assemble = compacted_assemble  # type: ignore[method-assign]
    assembler.needs_compaction = lambda: True  # type: ignore[method-assign]
    app._invalidate_prompt = lambda: states.append(app._loop_state)

    await app._consume_turn("prompt")

    assert "compacting" in states
    compacting_index = states.index("compacting")
    assert "streaming" in states[compacting_index + 1 :]


@pytest.mark.asyncio
async def test_empty_completion_prints_neutral_fallback(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(
            FakeBackend([ScriptedTurn()]),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._print_user("empty turn")
    await app._consume_turn("empty turn")

    rendered = app.console.file.getvalue()
    assert "empty turn" in rendered
    assert "no response" in rendered


@pytest.mark.asyncio
async def test_full_screen_separates_user_and_assistant_units(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(
            FakeBackend([ScriptedTurn([TextContent("answer")])]),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._active_session = app._make_session()

    app._print_user("prompt")
    await app._consume_turn("prompt")

    units = app._transcript.units
    assert len(units) == 3
    assert units[0] is not None
    assert units[1] is None
    assert units[2] is not None
    assert renderable_plain(units[0]) == "▌ prompt"
    rendered = app._transcript.render(80)
    assert "zeta" in rendered
    assert "answer" in rendered


@pytest.mark.asyncio
async def test_full_session_preserves_assistant_tool_user_order(tmp_path: Path) -> None:
    calls = [ToolCall("call-1", "read", {}), ToolCall("call-2", "read", {})]
    backend = OrderedToolBackend(calls)
    output = StringIO()
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions"),
            tools={"read": lambda _: "tool result"},
        ),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )

    app._print_user("first user")
    await app._consume_turn("first user")
    app._print_user("second user")
    await app._consume_turn("second user")

    rendered = output.getvalue()
    first_result = rendered.index("⏺ read")
    second_result = rendered.rindex("⏺ read")
    markers = [
        rendered.index("▌ first user"),
        rendered.index("assistant 1"),
        first_result,
        rendered.index("assistant after tool 1"),
        rendered.index("▌ second user"),
        rendered.index("assistant 2"),
        second_result,
    ]
    assert markers == sorted(markers)


@pytest.mark.asyncio
async def test_visual_snapshot_fake_turn_has_cards_receipt_and_thought(tmp_path: Path) -> None:
    bash = ToolCall("bash-visual", "bash", {"cmd": "seq 24"})
    read = ToolCall("read-visual", "read", {"path": "README.md", "limit": 120})
    backend = FakeBackend(
        [
            ScriptedTurn(
                content=[ThinkingContent("Plan the inspection. More reasoning stays collapsed.")],
                tool_calls=[bash],
            ),
            ScriptedTurn(tool_calls=[read]),
            ScriptedTurn(content=[TextContent("finished")]),
        ]
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions"),
            tools={
                "bash": lambda _: "\n".join(f"line-{i}" for i in range(24)),
                "read": lambda _: "title\nbody",
            },
        ),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False, width=72),
    )

    app._print_user("inspect the session")
    await app._consume_turn("inspect the session")
    snapshot = "\n".join(
        line.rstrip() for line in output.getvalue().splitlines()
    ).strip()
    snapshot = re.sub(
        r"(✱ thought · Plan the inspection\.) · \d+\.\ds",
        r"\1",
        snapshot,
    )
    lines = snapshot.splitlines()
    panel_lines = [
        line for line in lines if line.startswith(("  ╭", "  │", "  ╰"))
    ]
    assert "▌ inspect the session" in snapshot
    assert "✱ thought · Plan the inspection." in snapshot
    assert "⏺ read README.md [limit=120]" in snapshot
    assert "finished" in snapshot
    assert len(panel_lines) == 23
    assert all(cell_len(line) <= 72 for line in panel_lines)
    assert all(
        cell_len(line) == 70
        for line in panel_lines
        if line.startswith(("  ╭", "  ╰"))
    )


@pytest.mark.parametrize("terminal_width", [200, 120, 80, 40])
def test_transcript_units_share_the_padded_content_edges(terminal_width: int) -> None:
    width = content_width(terminal_width)
    transcript = TranscriptWidget()
    call = ToolCall("visual", "bash", {"cmd": "printf output"})
    transcript.append(Text("assistant prose that wraps at the shared edge."))
    transcript.append(
        render_markdown(
            "> quoted markdown\n\n```python\nprint(\"hello\")\n```"
        )
    )
    transcript.append(
        render_event(
            StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
        )
    )
    transcript.append(
        render_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=ToolResult(call.id, "done"),
            )
        )
    )
    transcript.append(
        Text(
            "[approval pending] request-1: bash; type approve request-1 or deny request-1",
            style=ACCENT,
        )
    )

    lines = [Text.from_ansi(line).plain for line in transcript.lines(width)]
    assert all(cell_len(line) <= width for line in lines)
    panel_lines = [
        line
        for line in lines
        if line.startswith(("╭", "│", "╰"))
    ]
    assert panel_lines
    assert all(cell_len(line) == width for line in panel_lines)
    assert any(line.startswith("╭") and line.endswith("╮") for line in panel_lines)
    assert any(line.startswith("╰") and line.endswith("╯") for line in panel_lines)


@pytest.mark.asyncio
async def test_run_delivers_queued_follow_up_after_current_turn(tmp_path: Path) -> None:
    backend = GateBackend()
    store = ConversationStore(tmp_path / "sessions")
    with create_pipe_input() as pipe:
        app = TUIApp(
            AgentLoop(backend, store),
            provider="fake",
            model="offline",
            console=Console(file=StringIO(), force_terminal=False),
        )
        run_task = asyncio.create_task(app.run(app_session(app, pipe)))
        pipe.send_text("first\r")
        await backend.started.wait()
        pipe.send_text("second\r")
        backend.release.set()
        await wait_until(lambda: len(backend.calls) == 2)
        pipe.send_text("\x04")
        await run_task

    user_texts = [
        block.text
        for message in store.messages()
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert user_texts == ["first", "second"]
    assert [message.role for message in store.messages()] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.USER,
        MessageRole.ASSISTANT,
    ]


@pytest.mark.asyncio
async def test_run_abort_persists_cancelled_tool_result(tmp_path: Path) -> None:
    tool_started = asyncio.Event()
    call = ToolCall("call-1", "block", {})
    backend = BlockingToolBackend(call)

    async def block(arguments: dict[str, object]) -> str:
        tool_started.set()
        await asyncio.Event().wait()
        return "unreachable"

    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(backend, store, tools={"block": block})
    with create_pipe_input() as pipe:
        app = TUIApp(
            loop,
            provider="fake",
            model="offline",
            console=Console(file=StringIO(), force_terminal=False),
        )
        run_task = asyncio.create_task(app.run(app_session(app, pipe)))
        pipe.send_text("run\r")
        await tool_started.wait()
        pipe.send_text("\x03")
        await wait_until(
            lambda: len(store.messages()) == 3
            and store.messages()[-1].tool_result is not None
        )
        pipe.send_text("draft")
        pipe.send_text("\x04")
        await run_task

    result = store.messages()[-1].tool_result
    assert result is not None
    assert result.tool_call_id == call.id
    assert result.content == "tool execution canceled"
    assert result.is_error


@pytest.mark.asyncio
async def test_run_abort_during_streamed_tool_call_pairs_result(tmp_path: Path) -> None:
    call = ToolCall("partial-call", "block", {})
    backend = StreamingToolBackend(call)
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(backend, store, tools={"block": lambda _: "unused"})
    with create_pipe_input() as pipe:
        app = TUIApp(
            loop,
            provider="fake",
            model="offline",
            console=Console(file=StringIO(), force_terminal=False),
        )
        run_task = asyncio.create_task(app.run(app_session(app, pipe)))
        pipe.send_text("run\r")
        await backend.tool_update.wait()
        pipe.send_text("\x03")
        await wait_until(
            lambda: len(store.messages()) == 3
            and store.messages()[-1].tool_result is not None
        )
        pipe.send_text("\x04")
        await run_task

    messages = store.messages()
    calls = [
        block.tool_call
        for block in messages[1].content
        if isinstance(block, ToolUseContent)
    ]
    results = [message.tool_result for message in messages if message.tool_result]
    assert [call.id for call in calls] == [result.tool_call_id for result in results]
    assert all(result.content == "tool execution canceled" for result in results)


@pytest.mark.asyncio
async def test_composer_submits_enter_and_keeps_ctrl_j_multiline() -> None:
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(on_interrupt=lambda: None, on_exit=lambda: None),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("line one")
        pipe.send_text("\x0a")
        pipe.send_text("line two")
        pipe.send_text("\r")
        assert await task == "line one\nline two"


@pytest.mark.asyncio
async def test_vi_composer_motions_move_and_delete_the_current_line() -> None:
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
            ),
            multiline=True,
        )
        fast_vi_timeouts(session)
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("hello")
        await wait_until(lambda: session.app.current_buffer.text == "hello")
        pipe.send_text("\x1b")
        await wait_until(
            lambda: session.app.vi_state.input_mode.value == "vi-navigation"
        )
        pipe.send_text("0")
        await wait_until(lambda: session.app.current_buffer.cursor_position == 0)
        pipe.send_text("$")
        await wait_until(lambda: session.app.current_buffer.cursor_position == 4)
        pipe.send_text("dd")
        await wait_until(lambda: session.app.current_buffer.text == "")
        session.app.exit()
        await task


@pytest.mark.asyncio
async def test_vi_composer_normal_enter_submits_without_losing_buffer() -> None:
    submitted: list[str] = []
    with create_pipe_input() as pipe:
        session: PromptSession[str] | None = None

        def submit(value: str) -> None:
            submitted.append(value)
            assert session is not None
            session.app.exit()

        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_submit=submit,
            ),
            multiline=True,
        )
        fast_vi_timeouts(session)
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("keep this buffer")
        await wait_until(lambda: session is not None and session.app.current_buffer.text == "keep this buffer")
        pipe.send_text("\x1b")
        await wait_until(
            lambda: session is not None
            and session.app.vi_state.input_mode.value == "vi-navigation"
        )
        pipe.send_text("\r")
        await task

    assert submitted == ["keep this buffer"]


@pytest.mark.asyncio
async def test_full_screen_vi_escape_enters_normal_mode_with_low_latency() -> None:
    with create_pipe_input() as pipe:
        session = FullScreenPromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
            ),
            multiline=True,
        )
        assert session.app.ttimeoutlen <= 0.1
        assert session.app.timeoutlen >= 0.5
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        started = time.monotonic()
        pipe.send_text("\x1b")
        await wait_until(
            lambda: session.app.vi_state.input_mode.value == "vi-navigation"
        )
        elapsed = time.monotonic() - started
        session.app.exit()
        await task

    assert elapsed < 0.1


@pytest.mark.asyncio
async def test_vi_composer_paced_dd_and_gg_commands() -> None:
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
            ),
            multiline=True,
        )
        paced_vi_timeouts(session)
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("first line\nsecond line")
        await wait_until(lambda: session.app.current_buffer.text == "first line\nsecond line")
        pipe.send_text("\x1b")
        await wait_until(
            lambda: session.app.vi_state.input_mode.value == "vi-navigation"
        )
        pipe.send_text("d")
        await asyncio.sleep(0.1)
        pipe.send_text("d")
        await wait_until(lambda: session.app.current_buffer.text == "first line\n")
        pipe.send_text("g")
        await asyncio.sleep(0.1)
        pipe.send_text("g")
        await wait_until(lambda: session.app.current_buffer.cursor_position == 0)
        session.app.exit()
        await task


@pytest.mark.asyncio
async def test_vi_composer_ctrl_c_interrupts_in_insert_and_normal_modes() -> None:
    interrupts: list[None] = []
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: interrupts.append(None),
                on_exit=lambda: None,
            ),
            multiline=True,
        )
        fast_vi_timeouts(session)
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("draft")
        await wait_until(lambda: session.app.current_buffer.text == "draft")
        pipe.send_text("\x03")
        await wait_until(lambda: len(interrupts) == 1)
        assert session.app.current_buffer.text == ""
        pipe.send_text("draft")
        await wait_until(lambda: session.app.current_buffer.text == "draft")
        pipe.send_text("\x1b")
        await wait_until(
            lambda: session.app.vi_state.input_mode.value == "vi-navigation"
        )
        pipe.send_text("\x03")
        await wait_until(lambda: len(interrupts) == 2)
        assert session.app.current_buffer.text == ""
        session.app.exit()
        await task


@pytest.mark.asyncio
async def test_vi_composer_paste_in_insert_mode_and_wrapped_multiline_submission() -> None:
    submitted: list[str] = []
    long_line = "wrapped " + "x" * 160
    with create_pipe_input() as pipe:
        session: PromptSession[str] | None = None

        def submit(value: str) -> None:
            submitted.append(value)
            assert session is not None
            session.app.exit()

        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_submit=submit,
            ),
            multiline=True,
        )
        fast_vi_timeouts(session)
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("\x1b[200~pasted line 1\npasted line 2\x1b[201~")
        await wait_until(
            lambda: session is not None
            and session.app.current_buffer.text == "pasted line 1\npasted line 2"
        )
        pipe.send_text("\x1b[200~\n" + long_line + "\x1b[201~\r")
        await task

    assert submitted == ["pasted line 1\npasted line 2\n" + long_line]


@pytest.mark.asyncio
async def test_vi_composer_history_down_returns_to_empty_buffer(tmp_path: Path) -> None:
    history = history_for(tmp_path / "history")
    submitted: list[str] = []

    with create_pipe_input() as pipe:
        session: PromptSession[str] | None = None

        def submit(value: str) -> None:
            submitted.append(value)
            assert session is not None
            session.app.exit()

        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            history=history,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_submit=submit,
            ),
            multiline=True,
        )
        fast_vi_timeouts(session)
        first = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("first\r")
        await first
        second = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("second\r")
        await second
        third = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("\x1b[A")
        await wait_until(lambda: session.app.current_buffer.text == "second")
        pipe.send_text("\x1b[B")
        await wait_until(lambda: session.app.current_buffer.text == "")
        session.app.exit()
        await third

    assert submitted == ["first", "second"]


def test_vim_slash_command_toggles_both_directions(tmp_path: Path) -> None:
    session = PromptSession()
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        session=session,
        vim_mode=True,
    )
    app._invalidate_prompt = lambda: None
    session.editing_mode = EditingMode.VI

    assert app.slash_vim("off") == "vim mode: off"
    assert app.vim_mode is False
    assert session.editing_mode is EditingMode.EMACS
    assert app.slash_vim("on") == "vim mode: on"
    assert app.vim_mode is True
    assert session.editing_mode is EditingMode.VI


@pytest.mark.asyncio
async def test_vi_composer_can_show_approval_prompt_while_in_normal_mode(
    tmp_path: Path,
) -> None:
    call = ToolCall("approval-1", "danger", {})
    store = ConversationStore(tmp_path / "sessions")
    policy = ApprovalPolicy(default="ask", store=store)
    app = TUIApp(
        AgentLoop(
            BlockingToolBackend(call),
            store,
            tools={"danger": lambda _: "done"},
            approval_policy=policy,
        ),
        provider="fake",
        model="offline",
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=app.abort_active,
                on_exit=app.request_exit,
            ),
            multiline=True,
        )
        fast_vi_timeouts(session)
        run_task = asyncio.create_task(app.run(session))
        pipe.send_text("run\r")
        await wait_until(lambda: bool(app.pending_approvals))
        pipe.send_text("\x1b")
        await wait_until(
            lambda: session.app.vi_state.input_mode.value == "vi-navigation"
        )
        assert app.pending_approvals[0].request_id == "approval-1"
        pipe.send_text("\x04")
        await run_task


@pytest.mark.asyncio
@pytest.mark.parametrize("newline", ["\x1b[27;2;13~", "\x1b\r", "\x0a"])
async def test_vi_composer_modified_enter_inserts_newline(newline: str) -> None:
    with create_pipe_input() as pipe:
        session: PromptSession[str] | None = None
        submitted: list[str] = []

        def submit(value: str) -> None:
            submitted.append(value)
            assert session is not None
            session.app.exit()

        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_submit=submit,
            ),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text(f"line one{newline}line two\r")
        await task

    assert submitted == ["line one\nline two"]


@pytest.mark.asyncio
async def test_vi_composer_escape_and_insert_update_native_state() -> None:
    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
            ),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("\x1b")
        for _ in range(300):
            if session.app.vi_state.input_mode.value == "vi-navigation":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("vi escape did not enter navigation mode")
        pipe.send_text("i")
        for _ in range(100):
            if session.app.vi_state.input_mode.value == "vi-insert":
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("vi insert key did not enter insert mode")
        session.app.exit()
        await task


@pytest.mark.asyncio
async def test_composer_submit_writes_file_history_and_up_replays_it(
    tmp_path: Path,
) -> None:
    history = history_for(tmp_path / "history")
    submitted: list[str] = []

    with create_pipe_input() as pipe:
        first_session: PromptSession[str] | None = None

        def submit_first(value: str) -> None:
            submitted.append(value)
            assert first_session is not None
            first_session.app.exit()

        first_session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            history=history,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_submit=submit_first,
            ),
            multiline=True,
        )
        first_task = asyncio.create_task(first_session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("remember me\r")
        await first_task

        second_session: PromptSession[str] | None = None

        def submit_second(value: str) -> None:
            submitted.append(value)
            assert second_session is not None
            second_session.app.exit()

        second_session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            history=history_for(tmp_path / "history"),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_submit=submit_second,
            ),
            multiline=True,
        )
        second_task = asyncio.create_task(second_session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("\x1b[A\r")
        await second_task

    assert submitted == ["remember me", "remember me"]


@pytest.mark.asyncio
async def test_vi_composer_history_up_works_from_insert_mode_on_empty_buffer(
    tmp_path: Path,
) -> None:
    history = history_for(tmp_path / "history")
    submitted: list[str] = []

    with create_pipe_input() as pipe:
        session: PromptSession[str] | None = None

        def submit(value: str) -> None:
            submitted.append(value)
            assert session is not None
            session.app.exit()

        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=EditingMode.VI,
            history=history,
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_submit=submit,
            ),
            multiline=True,
        )
        first = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("remember me\r")
        await first

        second = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("\x1b[A\r")
        await second

    assert submitted == ["remember me", "remember me"]


@pytest.mark.parametrize("value", ["", "  \n  "])
def test_parse_input_rejects_blank_turns(value: str) -> None:
    assert parse_input(value) is None
