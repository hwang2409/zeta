from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable, Sequence
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from zeta.loop import AgentLoop
from zeta.store import ConversationStore
from zeta.tui.app import TUIApp
from zeta.tui.composer import build_key_bindings, parse_input
from zeta.tui.render import MarkdownStream, format_status, render_event
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

    assert start is not None and "[tool] read" in start.plain
    assert "README.md" in start.plain
    assert result is not None and result.plain == "[tool result] first line"


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


def test_status_includes_provider_state_and_usage() -> None:
    status = format_status("fake", "offline", "streaming", {"input_tokens": 2}, "partial")

    assert status.plain == " fake/offline  streaming  tokens in=2 out=0  |  partial"


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
