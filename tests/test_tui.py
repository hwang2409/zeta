from __future__ import annotations

import asyncio
from io import StringIO
from pathlib import Path

import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from zeta.fake import FakeBackend, ScriptedTurn
from zeta.loop import AgentLoop
from zeta.store import ConversationStore
from zeta.tui.app import TUIApp
from zeta.tui.composer import build_key_bindings, parse_input
from zeta.tui.render import MarkdownStream, format_status, render_event
from zeta.types import (
    ErrorInfo,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolResult,
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
    assert len(output) == 1
    output.extend(stream.consume("```"))

    assert any(type(item).__name__ == "Syntax" for item in output)


def test_status_includes_provider_state_and_usage() -> None:
    status = format_status("fake", "offline", "streaming", {"input_tokens": 2}, "partial")

    assert status.plain == " fake/offline  streaming  tokens in=2 out=0  |  partial"


@pytest.mark.asyncio
async def test_commit_on_newline_and_follow_up_queue(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first line\npartial")], delay=0.001),
            ScriptedTurn([TextContent("second line\n")]),
        ]
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
        history_path=tmp_path / "history",
    )

    app._start_turn("one")
    app._queued.append("two")
    assert app._active_task is not None
    await app._active_task
    app._active_task = None
    app._start_turn(app._queued.popleft())
    assert app._active_task is not None
    await app._active_task

    rendered = output.getvalue()
    assert "first line" in rendered
    assert "partial" in rendered
    assert len(backend.calls) == 2


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
