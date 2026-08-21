from __future__ import annotations

import asyncio
import os
import pty
import select
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Callable, Sequence
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console
from rich.syntax import Syntax
from rich.table import Table

from zeta.loop import AgentLoop
from zeta.store import ConversationStore
from zeta.tui.app import TUIApp
from zeta.tui.composer import build_key_bindings, parse_input
from zeta.tui.render import (
    MarkdownStream,
    format_status,
    render_code,
    render_event,
    render_markdown,
)
from zeta.tui.theme import ACCENT, BODY, CODE_BG
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

    assert start is not None and start.plain.startswith("▸ read(")
    assert "README.md" in start.plain
    assert result is not None
    assert result.plain == "  ↳ [tool result] first line\n  ↳ second line"


def test_render_event_preserves_multiline_tool_result_formatting() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_result=ToolResult("call-1", "first\n\n  third"),
        )
    )

    assert rendered is not None
    assert rendered.plain == "  ↳ [tool result] first\n  ↳ \n  ↳   third"


def test_render_helpers_use_the_zeta_palette() -> None:
    markdown = render_markdown("# heading")
    code = render_code("print('hi')", "python")
    start = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_START,
            tool_call=ToolCall("call-1", "read", {}),
        )
    )

    assert markdown.style == BODY
    assert code.background_color == CODE_BG
    assert start is not None
    assert any(span.style == ACCENT for span in start.spans)


def test_render_event_error_is_visible() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("backend_error", "provider stopped"),
        )
    )

    assert rendered is not None
    assert rendered.plain == "[error] provider stopped"


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
    assert rendered[1].plain == "[thinking] plan"


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
    assert rendered.index("name") < rendered.index("[user] second")


def test_main_exits_on_ctrl_d_at_empty_prompt(tmp_path: Path) -> None:
    master_fd, slave_fd = pty.openpty()
    env = os.environ.copy()
    env["ZETA_HOME"] = str(tmp_path / "zeta-home")
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
        while b"you > " not in output and time.monotonic() < deadline:
            ready, _, _ = select.select(
                [master_fd],
                [],
                [],
                max(0, deadline - time.monotonic()),
            )
            if ready:
                output.extend(os.read(master_fd, 4096))
        assert b"you > " in output

        os.write(master_fd, b"\x04")
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master_fd)


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
    assert output[1].__class__.__name__ == "Markdown"


def test_status_includes_provider_state_and_usage() -> None:
    status = format_status("fake", "offline", "streaming", {"input_tokens": 2}, "partial")

    assert status.plain == " fake/offline  streaming  tokens in=2 out=0  |  partial"


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
    assert "mode streaming" in status.plain
    assert "tok 5/121" in status.plain
    assert "codex/gpt-5.4" in status.plain
    assert "s:abc12" in status.plain
    assert "tail 8" in status.plain


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
    assert all("mode streaming" in status.plain and "tok 5/121" in status.plain for status in statuses)
    assert all(marker in statuses[index].plain for index, marker in enumerate(("·", "•", "●")))
    assert all(value in statuses[1].plain for value in ("codex/gpt-5.4", "s:abc12", "tail 8"))
    assert all(value in statuses[2].plain for value in ("codex/gpt-5.4", "s:abc12", "tail 8"))
    assert "  |  " in statuses[2].plain

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
    assert "mode streaming" in narrow.plain
    assert "tok 5/121" in narrow.plain

    cleared = format_status(
        "codex",
        "gpt-5.4",
        "idle",
        session_id="abc12345",
        retained_tail=8,
        streaming=False,
        width=80,
    )
    assert not any(marker in cleared.plain for marker in ("·", "•", "●"))


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
    assert "tok 5/121" in plain


@pytest.mark.asyncio
async def test_spinner_pulses_on_timer(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._streaming = True
    task = asyncio.create_task(app._pulse_spinner())
    await asyncio.sleep(0.45)
    app._streaming = False
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)

    assert app._spinner_frame >= 2


@pytest.mark.asyncio
async def test_spinner_restarts_for_completion_after_tool(tmp_path: Path) -> None:
    call = ToolCall("call-1", "read", {})
    backend = SlowSecondCompletionBackend(call)
    app = TUIApp(
        AgentLoop(
            backend,
            ConversationStore(tmp_path / "sessions"),
            tools={"read": lambda _: "tool result"},
        ),
        provider="fake",
        model="offline",
    )

    turn = asyncio.create_task(app._consume_turn("prompt"))
    await backend.second_started.wait()
    starting_frame = app._spinner_frame
    await asyncio.sleep(0.45)
    ending_frame = app._spinner_frame
    backend.release_second.set()
    await turn

    assert app._streaming is False
    assert ending_frame - starting_frame >= 2


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
    first_result = rendered.index("[tool result] tool result")
    second_result = rendered.rindex("[tool result] tool result")
    markers = [
        rendered.index("[user] first user"),
        rendered.index("assistant 1"),
        rendered.index("▸ read("),
        first_result,
        rendered.index("assistant after tool 1"),
        rendered.index("[user] second user"),
        rendered.index("assistant 2"),
        second_result,
    ]
    assert markers == sorted(markers)


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
        task = asyncio.create_task(session.prompt_async("you > "))
        await asyncio.sleep(0)
        pipe.send_text("line one")
        pipe.send_text("\x0a")
        pipe.send_text("line two")
        pipe.send_text("\r")
        assert await task == "line one\nline two"


@pytest.mark.parametrize("value", ["", "  \n  "])
def test_parse_input_rejects_blank_turns(value: str) -> None:
    assert parse_input(value) is None
