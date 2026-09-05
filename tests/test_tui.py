from __future__ import annotations

import ast
import asyncio
import base64
import json
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
from textwrap import dedent
from types import SimpleNamespace

import httpx
import pytest
from prompt_toolkit import PromptSession
from prompt_toolkit.application.current import set_app
from prompt_toolkit.data_structures import Point, Size
from prompt_toolkit.enums import EditingMode
from prompt_toolkit.input import PipeInput, create_pipe_input
from prompt_toolkit.keys import Keys
from prompt_toolkit.layout.controls import UIContent
from prompt_toolkit.layout.mouse_handlers import MouseHandlers
from prompt_toolkit.layout.screen import Screen, WritePosition
from prompt_toolkit.mouse_events import MouseButton, MouseEvent, MouseEventType
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.output.vt100 import Vt100_Output
from rich.cells import cell_len
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text

from zeta.core.approval import ApprovalPolicy
from zeta.core.context import ContextAssembler
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.mcp import MCPPrompt, MCPPromptArgument
from zeta.persistence import DraftPersistence, history_for
from zeta.providers.anthropic import (
    AnthropicBackend,
    AnthropicCredentialStore,
    OAuthTokens,
)
from zeta.providers.codex import DEFAULT_CODEX_MODEL, CodexBackend, CodexCredentialStore
from zeta.tools import ToolStreamPublisher
from zeta.tui.agent_card import AgentCard
from zeta.tui.app import FullScreenPromptSession, TUIApp, background_notice
from zeta.tui.composer import (
    UndoCandidate,
    build_key_bindings,
    parse_input,
)

PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
)
from zeta.tui.layout import content_width
from zeta.tui.render import (
    _render_tool_output,
    format_status,
    format_thought,
    is_retryable_error,
    render_agent_progress,
    render_agent_receipt,
    render_code,
    render_event,
    render_line,
    render_markdown,
    render_thought,
    render_thought_live,
    render_tool_progress,
    tool_render_mode,
)
from zeta.tui.theme import ACCENT, BODY, DIM, ERROR, RICH_THEME
from zeta.tui.transcript import TranscriptPresenter, TranscriptWidget
from zeta.types import (
    CompletionBackend,
    ErrorInfo,
    ImageContent,
    Message,
    MessageRole,
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolSchema,
    ToolUseContent,
)


def _test_console(output: StringIO | None = None, *, width: int = 80) -> Console:
    return Console(
        file=output if output is not None else StringIO(),
        force_terminal=True,
        color_system="truecolor",
        no_color=False,
        width=width,
        theme=RICH_THEME,
    )


def test_mcp_background_notice_is_dim_in_forced_terminal() -> None:
    output = StringIO()
    console = _test_console(output)
    rendered: list[Text] = []

    def print_notice(value: Text) -> None:
        rendered.append(value)
        console.print(value)

    background_notice(
        SimpleNamespace(_print=print_notice, _invalidate_prompt=lambda: None),
        "mcp · server mounted",
    )

    assert rendered[0].style == DIM
    assert "\x1b[" in output.getvalue()


def test_mcp_slash_error_uses_error_style(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path), skip_mcp_mount=True),
        provider="fake",
        model="offline",
    )
    rendered: list[Text] = []
    app._presenter.print_unit = lambda value: rendered.append(value)

    app._print_system("mcp error: unknown MCP server: absent")

    assert rendered[0].style == ERROR


def test_background_agent_card_survives_parent_tool_completion() -> None:
    call = ToolCall(
        "background-1",
        "agent",
        {
            "prompt": "inspect",
            "description": "background research",
            "background": True,
        },
    )
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        _test_console(),
        lambda: True,
        lambda renderable: transcript.append(renderable) if renderable else None,
    )
    presenter.handle_tool_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call),
        aborted=False,
    )
    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "background agent started",
                structured_content={"status": "running"},
            ),
        ),
        aborted=False,
    )
    assert presenter.has_active_agent

    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_UPDATE,
            tool_call=call,
            delta="turn 1: thinking",
        ),
        aborted=False,
    )
    assert "thinking" in Text.from_ansi(transcript.render(120)).plain

    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "child complete",
                structured_content={"status": "completed", "turns_used": 1},
            ),
        ),
        aborted=False,
    )
    assert not presenter.has_active_agent
    assert "completed" in Text.from_ansi(transcript.render(120)).plain


def _contains_background_sgr(value: str) -> bool:
    return any(
        parameter == "48"
        for sequence in re.findall(r"\x1b\[([0-9;]*)m", value)
        for parameter in sequence.split(";")
    )


def _codex_access_token() -> str:
    def encode(value: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return ".".join(
        (
            encode({"alg": "none"}),
            encode(
                {
                    "exp": 4_000_000_000,
                    "https://api.openai.com/auth": {
                        "chatgpt_account_id": "account-test"
                    },
                }
            ),
            encode({"signature": "fixture"}),
        )
    )


def _codex_sse(events: list[dict[str, object]]) -> str:
    return "".join(
        f"event: {value['type']}\ndata: {json.dumps(value)}\n\n" for value in events
    )


def _codex_event(event_type: str, **values: object) -> dict[str, object]:
    return {"type": event_type, **values}


def _codex_reasoning_events(items: list[tuple[str, str]]) -> list[dict[str, object]]:
    events = [_codex_event("response.created", response={"id": "response-test"})]
    for index, (summary, raw) in enumerate(items):
        item_id = f"reasoning-{index}"
        events.append(
            _codex_event(
                "response.output_item.added",
                output_index=index,
                item={"type": "reasoning", "id": item_id},
            )
        )
        if summary:
            events.extend(
                [
                    _codex_event(
                        "response.reasoning_summary_part.added",
                        output_index=index,
                        summary_index=0,
                        part={"type": "summary_text"},
                    ),
                    _codex_event(
                        "response.reasoning_summary_text.delta",
                        output_index=index,
                        summary_index=0,
                        delta=summary,
                    ),
                    _codex_event(
                        "response.reasoning_summary_text.done",
                        output_index=index,
                        summary_index=0,
                        text=summary,
                    ),
                    _codex_event(
                        "response.reasoning_summary_part.done",
                        output_index=index,
                        summary_index=0,
                        part={"type": "summary_text", "text": summary},
                    ),
                ]
            )
        if raw:
            events.extend(
                [
                    _codex_event(
                        "response.content_part.added",
                        output_index=index,
                        content_index=0,
                        part={"type": "reasoning_text"},
                    ),
                    _codex_event(
                        "response.reasoning_text.delta",
                        output_index=index,
                        content_index=0,
                        delta=raw,
                    ),
                    _codex_event(
                        "response.reasoning_text.done",
                        output_index=index,
                        content_index=0,
                        text=raw,
                    ),
                    _codex_event(
                        "response.content_part.done",
                        output_index=index,
                        content_index=0,
                        part={"type": "reasoning_text"},
                    ),
                ]
            )
        events.append(
            _codex_event(
                "response.output_item.done",
                output_index=index,
                item={
                    "type": "reasoning",
                    "id": item_id,
                    "status": "completed",
                    "summary": (
                        [{"type": "summary_text", "text": summary}]
                        if summary
                        else []
                    ),
                    "content": (
                        [{"type": "reasoning_text", "text": raw}] if raw else []
                    ),
                },
            )
        )
    events.append(_codex_event("response.completed"))
    return events


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
            message=Message(
                MessageRole.ASSISTANT,
                [
                    TextContent(
                        "| name | value |\n| --- | --- |"
                        if index == 0
                        else f"reply {index}"
                    )
                ],
            ),
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


class AbortThenSuccessBackend(CompletionBackend):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.calls = 0

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        call = self.calls
        self.calls += 1
        yield StreamEvent(StreamEventType.MESSAGE_START)
        if call == 0:
            yield StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=TextContent("partial response"),
            )
            self.started.set()
            await asyncio.Event().wait()
        response = "second response"
        yield StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent(response),
        )
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(response)]),
        )


class ErrorThenSuccessBackend(CompletionBackend):
    def __init__(self) -> None:
        self.calls = 0
        self.request_messages: list[list[Message]] = []

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        call = self.calls
        self.calls += 1
        self.request_messages.append(list(messages))
        yield StreamEvent(StreamEventType.MESSAGE_START)
        if call == 0:
            yield StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=TextContent("partial response"),
            )
            raise RuntimeError("boom")
        response = "second response"
        yield StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent(response),
        )
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(response)]),
        )


class AlwaysErrorBackend(CompletionBackend):
    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        yield StreamEvent(StreamEventType.MESSAGE_START)
        raise TimeoutError("provider timeout")


class WaitingFailureBackend(CompletionBackend):
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("partial response"),
        )
        self.started.set()
        await self.release.wait()
        raise ConnectionError("network disconnected")


class RaisedErrorBackend(CompletionBackend):
    def __init__(self, error: Exception) -> None:
        self.error = error

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        del messages, tool_schemas
        raise self.error
        yield


class GhostPreviewBackend(CompletionBackend):
    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("ghost preview"),
        )
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT),
        )


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
    try:
        async with asyncio.timeout(30):
            while not check():
                await asyncio.sleep(0.01)
    except TimeoutError as exc:
        raise AssertionError("condition did not become true") from exc


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
    if hasattr(renderable, "code"):
        return renderable.code
    children = getattr(renderable, "renderables", None)
    if children is not None:
        return "\n".join(renderable_plain(child) for child in children)
    inner = getattr(renderable, "renderable", None)
    if inner is not None and hasattr(inner, "plain"):
        return inner.plain
    if inner is not None:
        return renderable_plain(inner)
    raise AssertionError(f"renderable has no plain text: {renderable!r}")


def renderable_spans(renderable: object) -> list[object]:
    spans = getattr(renderable, "spans", None)
    if spans is not None:
        return list(spans)
    children = getattr(renderable, "renderables", None)
    if children is not None:
        return [span for child in children for span in renderable_spans(child)]
    inner = getattr(renderable, "renderable", None)
    return renderable_spans(inner) if inner is not None else []


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


def test_transcript_search_highlights_matches_and_wraps() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("alpha target"))
    transcript.append(Text("target between"))
    transcript.append(Text("omega target"))
    transcript.create_content(80, 2)

    transcript.begin_search()
    transcript.update_search("target")
    assert transcript.search_status() == (1, 3)
    assert "target" in Text.from_ansi(transcript.render(80)).plain
    assert "\x1b[" in transcript.render(80)

    assert transcript.next_search_match()
    assert transcript.search_status() == (2, 3)
    assert transcript.next_search_match()
    assert transcript.search_status() == (3, 3)
    assert transcript.next_search_match()
    assert transcript.search_status() == (1, 3)
    assert transcript.previous_search_match()
    assert transcript.search_status() == (3, 3)

    transcript.end_search()
    assert transcript.search_status() is None
    assert "\x1b[" not in transcript.render(80)


def test_transcript_search_resize_back_refreshes_current_match_style() -> None:
    def match_styles(content: UIContent) -> tuple[str, str]:
        first, second = (
            next(
                style
                for style, text in content.get_line(line)
                if text == "t" and style
            )
            for line in (1, 2)
        )
        return first, second

    transcript = TranscriptWidget()
    transcript.append(Text("first target"))
    transcript.append(Text("second target"))
    transcript.begin_search()
    transcript.update_search("target")

    initial = transcript.create_content(60, 3)
    initial_styles = match_styles(initial)

    transcript.create_content(30, 3)
    assert transcript.next_search_match()
    assert transcript.search_status() == (2, 2)

    resized = transcript.create_content(60, 3)
    resized_styles = match_styles(resized)

    assert resized_styles == initial_styles[::-1]


def test_transcript_search_stays_anchored_at_the_tail_until_closed() -> None:
    transcript = TranscriptWidget()
    for index in range(20):
        transcript.append(Text(f"line {index}"))
    transcript.create_content(80, 3)

    transcript.begin_search()
    transcript.update_search("line 19")
    transcript.create_content(80, 3)

    assert not transcript.follow_tail
    assert transcript.position_indicator() == "line 18/20"

    transcript.append(Text("new tail"))
    transcript.create_content(80, 3)
    assert not transcript.follow_tail
    assert transcript.scroll_offset == 17

    transcript.end_search()
    assert not transcript.follow_tail


def test_transcript_user_jumps_skip_non_user_units() -> None:
    transcript = TranscriptWidget()
    first_user = transcript.append(Text("first user"))
    transcript.mark_user(first_user)
    transcript.append(Text("tool receipt"))
    transcript.append(Text("assistant thought"))
    second_user = transcript.append(Text("second user"))
    transcript.mark_user(second_user)
    transcript.append(Text("assistant answer"))
    transcript.create_content(80, 2)
    transcript._set_scroll_offset(0)

    assert transcript.next_user_message()
    assert transcript.scroll_offset == 3
    assert transcript.previous_user_message()
    assert transcript.scroll_offset == 0
    assert not transcript.previous_user_message()


def test_transcript_user_jump_targets_only_the_start_of_each_message() -> None:
    transcript = TranscriptWidget()
    first = transcript.append(Text("first\ncontinuation\nlast"))
    transcript.mark_user(first)
    transcript.append(Text("assistant"))
    second = transcript.append(Text("second"))
    transcript.mark_user(second)
    transcript.create_content(80, 1)
    transcript._set_scroll_offset(0)

    assert transcript.next_user_message()
    assert transcript.scroll_offset == 4
    assert transcript.previous_user_message()
    assert transcript.scroll_offset == 0


def test_transcript_position_indicator_hides_at_pinned_tail() -> None:
    transcript = TranscriptWidget()
    for index in range(8):
        transcript.append(Text(f"line {index}"))
    transcript.create_content(80, 3)

    assert transcript.position_indicator() is None
    transcript._set_scroll_offset(0)
    assert transcript.position_indicator() == "line 1/8"
    transcript._set_scroll_offset(4)
    assert transcript.position_indicator() == "line 5/8"
    transcript._set_scroll_offset(5)
    assert transcript.position_indicator() is None


def test_transcript_paging_reuses_rendered_content_for_large_sessions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = TranscriptWidget()
    for index in range(2_000):
        transcript.append(Text(f"line {index}"))
    transcript.create_content(80, 20)

    calls = 0
    original_render = transcript.render

    def counted_render(width: int) -> str:
        nonlocal calls
        calls += 1
        return original_render(width)

    monkeypatch.setattr(transcript, "render", counted_render)
    for _ in range(100):
        transcript.page_up()
        transcript.create_content(80, 20)
    assert calls == 0


def test_transcript_paging_reuses_locations_by_width_and_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcript = TranscriptWidget()
    for index in range(2_000):
        transcript.append(Text(f"line {index}"))
    transcript.create_content(80, 20)

    calls = 0
    original_compute = transcript._compute_locations

    def counted_compute(width: int) -> list[tuple[object, int]]:
        nonlocal calls
        calls += 1
        return original_compute(width)  # type: ignore[return-value]

    monkeypatch.setattr(transcript, "_compute_locations", counted_compute)
    for _ in range(100):
        transcript.page_up()
        transcript.create_content(80, 20)
    assert calls == 0

    transcript.create_content(40, 20)
    assert calls == 1
    transcript.append(Text("new line"))
    transcript.create_content(40, 20)
    assert calls == 2


def test_transcript_locations_cache_is_bounded_by_width() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("line"))

    for width in range(40, 141):
        transcript._locations(width)

    assert len(transcript._locations_cache) == 3


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


def test_transcript_completion_swap_preserves_scrolled_anchor() -> None:
    source = "\n".join(f"token-{index:03}" for index in range(200))
    transcript = TranscriptWidget()
    unit = transcript.append(Text(source))

    transcript.create_content(30, 5)
    transcript._set_scroll_offset(65)
    transcript.create_content(30, 5)
    assert (
        Text.from_ansi(transcript.lines(30)[transcript.scroll_offset]).plain
        == "token-065"
    )

    transcript.replace(unit, render_markdown(source))
    transcript.create_content(30, 5)
    visible = transcript.lines(30)[
        transcript.scroll_offset : transcript.scroll_offset + 5
    ]

    assert "token-065" in " ".join(Text.from_ansi(line).plain for line in visible)
    assert transcript._line_locations[transcript.scroll_offset][0] is unit



def _render_layout(
    session: FullScreenPromptSession, mouse_handlers: MouseHandlers
) -> None:
    """Lay the full-screen tree out over an 80x24 terminal."""

    session.layout.update_parents_relations()
    session.layout.container.write_to_screen(
        Screen(),
        mouse_handlers,
        WritePosition(xpos=0, ypos=0, width=80, height=24),
        "",
        False,
        None,
    )


def _wheel(event_type: MouseEventType, *, x: int, y: int) -> MouseEvent:
    return MouseEvent(
        position=Point(x=x, y=y),
        event_type=event_type,
        button=MouseButton.NONE,
        modifiers=frozenset(),
    )


def test_full_screen_session_enables_mouse_reporting(tmp_path: Path) -> None:
    app = _test_tui_app(ConversationStore(tmp_path / "sessions"), StringIO())

    session = app._make_session()

    assert isinstance(session, FullScreenPromptSession)
    assert session.app.mouse_support()


def test_mouse_reporting_leaves_out_pointer_motion(tmp_path: Path) -> None:
    app = _test_tui_app(ConversationStore(tmp_path / "sessions"), StringIO())
    session = app._make_session()
    written: list[str] = []
    session.app.output.write_raw = written.append

    session.app.output.enable_mouse_support()

    assert "\x1b[?1000h" in written  # clicks and the wheel
    assert "\x1b[?1006h" in written  # SGR coordinates
    assert "\x1b[?1003h" not in written  # every pointer move


async def test_wheel_scrolls_transcript_from_transcript_and_composer(tmp_path: Path) -> None:
    app = _test_tui_app(ConversationStore(tmp_path / "sessions"), StringIO())
    session = app._make_session()
    app._install_full_screen_layout(session)
    for index in range(120):
        app._transcript.append(Text(f"line {index}"))
    mouse_handlers = MouseHandlers()
    with set_app(session.app):
        _render_layout(session, mouse_handlers)
        over_transcript = mouse_handlers.mouse_handlers[5][10]
        over_composer = mouse_handlers.mouse_handlers[23][10]
        tail_offset = app._transcript.scroll_offset

        over_transcript(_wheel(MouseEventType.SCROLL_UP, x=10, y=5))
        assert app._transcript.scroll_offset == tail_offset - 3

        over_composer(_wheel(MouseEventType.SCROLL_UP, x=10, y=23))
        assert app._transcript.scroll_offset == tail_offset - 6

        over_composer(_wheel(MouseEventType.SCROLL_DOWN, x=10, y=23))
        assert app._transcript.scroll_offset == tail_offset - 3


async def test_composer_still_receives_non_wheel_mouse_events(tmp_path: Path) -> None:
    app = _test_tui_app(ConversationStore(tmp_path / "sessions"), StringIO())
    session = app._make_session()
    app._install_full_screen_layout(session)
    mouse_handlers = MouseHandlers()
    with set_app(session.app):
        _render_layout(session, mouse_handlers)
        over_composer = mouse_handlers.mouse_handlers[23][10]

        result = over_composer(_wheel(MouseEventType.MOUSE_MOVE, x=10, y=23))

    assert result is NotImplemented

def test_transcript_parsed_cache_is_bounded_and_revision_scoped() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("line"))

    for width in (20, 30, 40, 50):
        transcript._parsed_lines(width)
    assert len(transcript._parsed_cache) == 3

    transcript.append(Text("new line"))
    assert not transcript._parsed_cache


def test_transcript_search_cache_is_bounded_by_width() -> None:
    transcript = TranscriptWidget()
    transcript.append(Text("target"))
    transcript.begin_search()
    transcript.update_search("target")

    for width in range(40, 141):
        transcript._search_matches(width)

    assert len(transcript._search_cache) == 3


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
    assert "[truncated; full_size=8 bytes]" in renderable_plain(rendered)


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


def test_image_placeholders_use_dim_style() -> None:
    rendered = _render_tool_output(
        "[image block] media_type=image/png bytes=70\nanswer"
    )

    assert any(
        span.start == 0 and span.style == DIM
        for span in rendered.spans
    )


def test_render_helpers_use_the_zeta_palette() -> None:
    markdown = render_line("# heading")
    code = render_code("print('hi')", "python")
    start = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_START,
            tool_call=ToolCall("call-1", "bash", {"cmd": "printf hi"}),
        )
    )

    assert markdown.style == BODY
    assert code.background_color == "default"
    assert start is not None
    assert any(ACCENT in str(span.style) for span in renderable_spans(start))


def test_command_tool_card_highlights_extracted_command() -> None:
    command = 'python3 -c "import fastapi; print(fastapi.__version__)"'
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("exec-1", "exec", {"command": command}),
            tool_result=ToolResult("exec-1", ""),
        )
    )

    assert rendered is not None
    header = rendered.renderable
    assert isinstance(header.renderables[1], Syntax)
    assert header.renderables[1].code == command
    plain = renderable_plain(rendered)
    assert command in plain
    assert '{"command"' not in plain


@pytest.mark.parametrize("width", [80, 200])
def test_command_tool_card_keeps_header_left_aligned(width: int) -> None:
    command = "cat /tmp/example.txt"
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("bash-1", "bash", {"cmd": command}),
            tool_result=ToolResult("bash-1", ""),
        )
    )

    assert rendered is not None
    output = StringIO()
    Console(file=output, force_terminal=False, width=width).print(rendered.renderable)

    assert output.getvalue().splitlines()[0] == f"bash {command}"


def test_command_syntax_has_no_background_sgr() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("exec-3", "exec", {"command": "printf hi"}),
            tool_result=ToolResult("exec-3", ""),
        )
    )
    assert rendered is not None

    output = StringIO()
    Console(
        file=output,
        force_terminal=True,
        color_system="truecolor",
        width=100,
    ).print(rendered.renderable)

    assert "\x1b[48" not in output.getvalue()


def test_tool_card_renders_nested_arguments_without_json_escapes() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall(
                "tool-1",
                "custom",
                {
                    "payload": {"path": "/tmp/a\nb", "quote": "don't"},
                    "count": 3,
                },
            ),
            tool_result=ToolResult("tool-1", ""),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "count=3" in plain
    assert "path=/tmp/a b" in plain
    assert "quote=don't" in plain
    assert r"\n" not in plain


def test_tool_card_omits_empty_output_sections_and_exit_codes() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("bash-1", "bash", {"cmd": "true"}),
            tool_result=ToolResult(
                "bash-1",
                "exit_code: 0\nstdout:\nstderr:\n",
            ),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "true" in plain
    assert "stdout:" not in plain
    assert "stderr:" not in plain
    assert "exit_code:" not in plain


@pytest.mark.parametrize("exit_code", [0, 7])
def test_tool_card_never_displays_exit_codes(exit_code: int) -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("exec-2", "exec", {"command": "false"}),
            tool_result=ToolResult(
                "exec-2",
                f"exit_code: {exit_code}\nstdout:\nstderr:\nfailed",
                is_error=exit_code != 0,
            ),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "failed" in plain
    assert "exit_code:" not in plain


def test_tool_card_renders_only_nonempty_output_sections() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("bash-2", "bash", {"cmd": "printf err >&2"}),
            tool_result=ToolResult(
                "bash-2",
                "exit_code: 0\nstdout:\nstderr:\nerr",
            ),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "stderr:\nerr" in plain
    assert "stdout:" not in plain


def test_tool_card_renders_nonempty_result_section_without_misnesting() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("tool-2", "exec", {"command": "run"}),
            tool_result=ToolResult(
                "tool-2",
                "stdout:\nout\nstderr:\nresult:\nanswer",
            ),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "stdout:\nout" in plain
    assert "result:\nanswer" in plain
    assert "stderr:" not in plain


def test_tool_card_preserves_timeout_preamble_before_stdout() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("tool-5", "exec", {"command": "sleep 1"}),
            tool_result=ToolResult(
                "tool-5",
                "timed out after 30s\nstdout:\npartial output",
                is_error=True,
            ),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "timed out after 30s" in plain
    assert "stdout:\npartial output" in plain
    assert plain.index("timed out after 30s") < plain.index("stdout:")


def test_tool_card_preserves_preamble_without_sections() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("tool-6", "exec", {"command": "sleep 1"}),
            tool_result=ToolResult(
                "tool-6", "timed out after 30s", is_error=True
            ),
        )
    )

    assert rendered is not None
    assert "timed out after 30s" in renderable_plain(rendered)


def test_tool_card_preserves_preamble_before_multiple_sections() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("tool-7", "exec", {"command": "run"}),
            tool_result=ToolResult(
                "tool-7",
                "warning\nstdout:\nout\nstderr:\nerr",
            ),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "warning" in plain
    assert "stdout:\nout" in plain
    assert "stderr:\nerr" in plain
    assert plain.index("warning") < plain.index("stdout:") < plain.index("stderr:")


def test_tool_card_keeps_trailing_text_in_active_section() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("tool-8", "exec", {"command": "run"}),
            tool_result=ToolResult(
                "tool-8",
                "stdout:\nout\nstderr:\nerr\ntrailing detail",
            ),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "stderr:\nerr\ntrailing detail" in plain
    assert "result:" not in plain


@pytest.mark.parametrize("position", ["leading", "middle", "trailing"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_tool_card_omits_exit_codes_in_any_section_position(
    position: str, exit_code: int
) -> None:
    sections = {
        "leading": f"exit_code: {exit_code}\nstdout:\nout",
        "middle": f"stdout:\nout\nexit_code: {exit_code}\nstderr:\nerr",
        "trailing": f"stdout:\nout\nexit_code: {exit_code}",
    }
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("tool-3", "exec", {"command": "run"}),
            tool_result=ToolResult("tool-3", sections[position]),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "exit_code:" not in plain


@pytest.mark.parametrize("position", ["leading", "middle", "trailing"])
@pytest.mark.parametrize("exit_code", [0, 7])
def test_tool_card_preserves_unlabeled_output_around_exit_codes(
    position: str, exit_code: int
) -> None:
    outputs = {
        "leading": f"exit_code: {exit_code}\nbefore\nafter",
        "middle": f"before\nexit_code: {exit_code}\nafter",
        "trailing": f"before\nafter\nexit_code: {exit_code}",
    }
    rendered = render_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=ToolCall("tool-4", "exec", {"command": "run"}),
            tool_result=ToolResult("tool-4", outputs[position]),
        )
    )

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert "before\nafter" in plain
    assert "exit_code:" not in plain


def test_render_event_error_is_visible() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("backend_error", "provider stopped"),
        )
    )

    assert rendered is not None
    assert isinstance(rendered, Panel)
    assert "provider failure · backend_error" in renderable_plain(rendered)
    assert "reason: provider stopped" in renderable_plain(rendered)


def test_render_error_card_bounds_and_labels_json_payload() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.ERROR,
            error=ErrorInfo("stream_error", '{"message":"' + "x" * 1_000 + '"}'),
        )
    )

    assert isinstance(rendered, Panel)
    plain = renderable_plain(rendered)
    assert "provider failure · stream_error" in plain
    assert "payload · json" in plain
    assert len(plain) < 600
    assert "retry: ctrl+y" in plain


@pytest.mark.parametrize("code", ["max_turns", "ui_error"])
def test_non_provider_errors_are_not_retryable(code: str) -> None:
    event = StreamEvent(
        StreamEventType.ERROR,
        error=ErrorInfo(code, "do not retry"),
    )

    rendered = render_event(event)

    assert rendered is not None
    plain = renderable_plain(rendered)
    assert f"error · {code}" in plain
    assert "provider failure" not in plain
    assert "retry: ctrl+y" not in plain
    assert is_retryable_error(event.error) is False


@pytest.mark.asyncio
@pytest.mark.parametrize("full_screen", [False, True])
async def test_provider_failure_card_is_visible_in_both_modes(
    tmp_path: Path, full_screen: bool
) -> None:
    app = TUIApp(
        AgentLoop(ErrorBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._active_session = app._make_session() if full_screen else None
    rendered: list[object] = []
    app._print = rendered.append

    await app._consume_turn("prompt")

    plain = "\n".join(renderable_plain(item) for item in rendered)
    assert "| name | value |" in plain
    assert "provider failure · backend_error" in plain
    assert "reason: boom" in plain
    assert "retry: ctrl+y" in plain


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.ConnectError(
            "connect failed", request=httpx.Request("GET", "https://test.invalid")
        ),
        httpx.ReadError(
            "read failed", request=httpx.Request("GET", "https://test.invalid")
        ),
        TimeoutError("deadline exceeded"),
        ValueError("malformed provider event"),
    ],
    ids=["connect", "read", "timeout", "malformed"],
)
async def test_transport_failure_shapes_render_error_cards(
    tmp_path: Path, error: Exception
) -> None:
    app = TUIApp(
        AgentLoop(
            RaisedErrorBackend(error),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    rendered: list[object] = []
    app._print = rendered.append

    await app._consume_turn("prompt")

    plain = "\n".join(renderable_plain(item) for item in rendered)
    assert "provider failure · " in plain
    assert str(error) in plain


@pytest.mark.asyncio
async def test_retry_reuses_user_message_and_keeps_partial_output(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    backend = ErrorThenSuccessBackend()
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._active_session = None
    app._print_user("prompt")

    await app._consume_turn("prompt")
    assert app.retry_available()
    app.retry_failed_turn()
    assert app._active_task is not None
    await app._active_task

    messages = store.messages()
    assert [message.role for message in messages] == [
        MessageRole.USER,
        MessageRole.ASSISTANT,
        MessageRole.ASSISTANT,
    ]
    assert sum(message.role is MessageRole.USER for message in messages) == 1
    assert [message.content[0].text for message in messages[1:]] == [
        "partial response",
        "second response",
    ]
    assert sum(
        message.role is MessageRole.USER for message in backend.request_messages[1]
    ) == 1


@pytest.mark.asyncio
async def test_retry_failure_renders_a_fresh_error_card(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(
            AlwaysErrorBackend(),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._active_session = None

    await app._consume_turn("prompt")
    app.retry_failed_turn()
    assert app._active_task is not None
    await app._active_task

    rendered = Text.from_ansi(app.console.file.getvalue()).plain
    assert rendered.count("provider failure · timeout") == 2


@pytest.mark.asyncio
async def test_compaction_failure_renders_reason_and_retry_succeeds(
    tmp_path: Path,
) -> None:
    class CompactionBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("compaction provider disconnected")
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
            )

    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("old")]))
    backend = CompactionBackend()
    assembler = ContextAssembler(
        store,
        token_budget=100,
        retained_tail=1,
        system_prompt="",
        token_counter=lambda message: 60 if message.role is MessageRole.USER else 1,
        backend=backend,
    )
    app = TUIApp(
        AgentLoop(backend, store, context_assembler=assembler),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._consume_turn("new")

    rendered = Text.from_ansi(app.console.file.getvalue()).plain
    assert app.retry_available()
    assert "provider failure · backend_error" in rendered
    assert "summary completion failed" in rendered
    assert store.turn_in_flight() is False

    app.retry_failed_turn()
    assert app._active_task is not None
    await app._active_task

    assert backend.calls == 3
    assert store.messages()[-1].content[0].text == "done"


@pytest.mark.asyncio
async def test_compaction_error_event_aborts_and_preserves_source(
    tmp_path: Path,
) -> None:
    class PartialThenErrorBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            self.calls += 1
            if self.calls == 1:
                yield StreamEvent(
                    StreamEventType.MESSAGE_UPDATE,
                    content=TextContent("half a summary"),
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("stream_error", "compaction stream aborted"),
                )
                return
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
            )

    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("old")]))
    baseline = [entry.id for entry in store.replay()]
    backend = PartialThenErrorBackend()
    assembler = ContextAssembler(
        store,
        token_budget=100,
        retained_tail=1,
        system_prompt="",
        token_counter=lambda message: 60 if message.role is MessageRole.USER else 1,
        backend=backend,
    )
    app = TUIApp(
        AgentLoop(backend, store, context_assembler=assembler),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    await app._consume_turn("new")

    rendered = Text.from_ansi(app.console.file.getvalue()).plain
    assert app.retry_available()
    assert "provider failure · stream_error" in rendered
    assert "compaction stream aborted" in rendered
    assert store.compaction_marker_count() == 0
    assert [entry.id for entry in store.replay()][: len(baseline)] == baseline
    assert store.turn_in_flight() is False

    app.retry_failed_turn()
    assert app._active_task is not None
    await app._active_task

    assert backend.calls == 3
    assert store.messages()[-1].content[0].text == "done"


@pytest.mark.asyncio
async def test_manual_compact_error_event_aborts_and_preserves_source(
    tmp_path: Path,
) -> None:
    class PartialThenErrorBackend(CompletionBackend):
        def __init__(self) -> None:
            self.calls = 0

        async def complete(
            self,
            messages: Sequence[Message],
            tool_schemas: Sequence[ToolSchema],
        ) -> AsyncIterator[StreamEvent]:
            del messages, tool_schemas
            self.calls += 1
            if self.calls == 1:
                yield StreamEvent(
                    StreamEventType.MESSAGE_UPDATE,
                    content=TextContent("half a summary"),
                )
                yield StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("stream_error", "compaction stream aborted"),
                )
                return
            yield StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(MessageRole.ASSISTANT, [TextContent("summary text")]),
            )

    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("old")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("tail")]))
    baseline = [entry.id for entry in store.replay()]
    backend = PartialThenErrorBackend()
    assembler = ContextAssembler(
        store,
        token_budget=10_000,
        retained_tail=1,
        system_prompt="",
        backend=backend,
    )
    app = TUIApp(
        AgentLoop(backend, store, context_assembler=assembler),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    reply = await app.slash_compact()

    assert reply.startswith("compact failed:")
    assert "compaction stream aborted" in reply
    assert store.compaction_marker_count() == 0
    assert [entry.id for entry in store.replay()] == baseline

    retry = await app.slash_compact()

    assert backend.calls == 2
    assert store.compaction_marker_count() == 1
    assert retry.startswith("compacted entries")


@pytest.mark.asyncio
async def test_resumed_failed_turn_renders_and_retries_without_duplication(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("prompt")]))
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("partial response")],
            metadata={
                "turn_failed": True,
                "turn_error": {"code": "backend_error", "message": "boom"},
            },
        )
    )
    output = StringIO()
    backend = ErrorThenSuccessBackend()
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )

    app._rebuild_transcript()

    assert app.retry_available()
    rendered = Text.from_ansi(output.getvalue()).plain
    assert "partial response" in rendered
    assert "provider failure · backend_error" in rendered
    assert "retry: ctrl+y" in rendered

    app.retry_failed_turn()
    assert app._active_task is not None
    await app._active_task

    assert [message.role for message in store.messages()].count(MessageRole.USER) == 1
    assert [
        message.role
        for message in backend.request_messages[0]
        if message.role is not MessageRole.SYSTEM
    ] == [MessageRole.USER]


def test_resumed_failure_followed_by_success_is_not_retryable(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("prompt")]))
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("partial")],
            metadata={
                "turn_failed": True,
                "turn_error": {"code": "backend_error", "message": "boom"},
            },
        )
    )
    store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("recovered")])
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._rebuild_transcript()

    assert not app.retry_available()


def test_resumed_failure_followed_by_new_user_is_not_retryable(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("first")]))
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            metadata={
                "turn_failed": True,
                "turn_error": {"code": "backend_error", "message": "boom"},
            },
        )
    )
    store.append_message(Message(MessageRole.USER, [TextContent("second")]))
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._rebuild_transcript()

    assert not app.retry_available()


@pytest.mark.asyncio
async def test_draft_survives_provider_failure(tmp_path: Path) -> None:
    backend = WaitingFailureBackend()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._active_session = session
    task = asyncio.create_task(app._consume_turn("prompt"))
    app._active_task = task
    await backend.started.wait()
    session.app.current_buffer.insert_text("draft while streaming")
    backend.release.set()
    await task

    assert session.app.current_buffer.text == "draft while streaming"
    failed_message = app.loop.store.messages()[-1]
    assert failed_message.metadata["turn_failed"] is True


@pytest.mark.asyncio
async def test_submitted_revision_does_not_clear_a_rapid_new_draft(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    draft_path = tmp_path / "draft"
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        draft_path=draft_path,
    )
    session = app._make_session()
    app._active_session = session
    session.default_buffer.insert_text("sent prompt")
    app._submit_input("sent prompt")
    session.default_buffer.reset()
    session.default_buffer.insert_text("new rapid draft")

    await asyncio.sleep(0.25)

    assert app._draft.load() == "new rapid draft"
    if app._active_task is not None:
        await app._active_task
    await app.loop.close()


@pytest.mark.asyncio
async def test_rapid_buffer_sends_keep_the_newest_persisted_draft(
    tmp_path: Path,
) -> None:
    backend = GateBackend()
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        draft_path=tmp_path / "draft",
    )
    session = app._make_session()
    app._active_session = session
    session.default_buffer.insert_text("first")
    app._submit_input(session.default_buffer.text)
    session.default_buffer.reset()
    session.default_buffer.insert_text("second")
    app._submit_input(session.default_buffer.text)
    session.default_buffer.reset()
    session.default_buffer.insert_text("third draft")
    await asyncio.sleep(0.25)

    await backend.started.wait()
    backend.release.set()
    await app._active_task
    await wait_until(lambda: len(backend.calls) == 2)
    await app._active_task

    assert session.default_buffer.text == "third draft"
    assert app._draft.load() == "third draft"
    await app.loop.close()


@pytest.mark.asyncio
async def test_rapid_buffer_sends_keep_each_staged_image_owned(
    tmp_path: Path,
) -> None:
    backend = GateBackend()
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    first_image = store.session_dir / "clipboard-first.png"
    second_image = store.session_dir / "clipboard-second.png"
    first_image.write_bytes(PNG)
    second_image.write_bytes(PNG)
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
    )
    session = app._make_session()
    app._active_session = session
    app._pending_attachments.append(first_image)
    app._pending_attachment_tokens["[Image #1]"] = first_image
    session.default_buffer.insert_text("first [Image #1]")
    app._submit_input(session.default_buffer.text)
    session.default_buffer.reset()
    app._pending_attachments.append(second_image)
    app._pending_attachment_tokens["[Image #1]"] = second_image
    session.default_buffer.insert_text("second [Image #1]")
    app._submit_input(session.default_buffer.text)
    session.default_buffer.reset()

    await backend.started.wait()
    backend.release.set()
    await app._active_task
    await wait_until(lambda: len(backend.calls) == 2)
    await app._active_task

    assert first_image.exists()
    assert second_image.exists()
    user_messages = [
        message
        for message in store.messages()
        if message.role is MessageRole.USER
    ]
    assert user_messages[0].content[1].path == str(first_image)
    assert user_messages[1].content[1].path == str(second_image)
    await app.loop.close()


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


def test_agent_rendering_stays_on_one_status_line() -> None:
    call = ToolCall(
        "agent-1",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(call.id, "done"),
    )

    rendered = render_event(event)
    assert rendered is not None
    assert not isinstance(rendered, Panel)
    assert "task research" in renderable_plain(rendered)
    start = render_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    )
    assert isinstance(start, Panel)
    progress = render_tool_progress(call, "task research: thinking")
    assert isinstance(progress, Panel)


def test_agent_running_card_shows_step_elapsed_and_turns() -> None:
    call = ToolCall(
        "agent-running",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )

    rendered = render_agent_progress(
        call,
        "task research: turn 3: tool: read {\"path\":\"README.md\"}",
        elapsed_seconds=4.2,
    )

    plain = renderable_plain(rendered)
    assert "task research · 4.2s · 3 turns" in plain
    assert "tool: read" in plain
    assert "README.md" in plain


def test_agent_receipts_show_success_and_canceled_status() -> None:
    call = ToolCall(
        "agent-receipt",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    success = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "done",
            structured_content={"turns_used": 2, "child_session_path": ""},
        ),
        data={"elapsed_seconds": 1.5},
    )
    canceled = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "tool execution canceled",
            is_error=True,
            is_canceled=True,
        ),
        data={"elapsed_seconds": 0.4, "depth": 2},
    )

    assert "2 turns · 1.5s · ok" in render_agent_receipt(success).plain
    canceled_plain = render_agent_receipt(canceled).plain
    assert "0 turns · 0.4s · canceled" in canceled_plain
    assert "error=false" in canceled_plain
    assert "canceled=true" in canceled_plain
    assert "error=true" not in canceled_plain
    assert "depth 2" in canceled_plain


def test_unstructured_agent_error_renders_as_error() -> None:
    call = ToolCall(
        "agent-error",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(call.id, "setup exploded", is_error=True),
    )

    rendered = render_agent_receipt(event)

    assert "error=true" in rendered.plain
    assert "canceled=false" in rendered.plain


def test_background_start_event_reaches_presenter_with_depth(tmp_path: Path) -> None:
    output = StringIO()
    app = _test_tui_app(ConversationStore(tmp_path), output)
    call = ToolCall(
        "background-nested",
        "agent",
        {"prompt": "inspect", "description": "nested research"},
    )

    app._handle_background_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_START,
            tool_call=call,
            data={"depth": 2},
        )
    )

    assert "depth 2" in output.getvalue()


def test_typed_agent_cards_and_receipts_show_type() -> None:
    call = ToolCall(
        "typed-agent",
        "agent",
        {
            "prompt": "inspect",
            "description": "task research",
            "agent_type": "explore",
        },
    )
    progress = AgentCard.render_progress(call, "turn 1: thinking")
    assert progress is not None
    assert "explore · task research" in renderable_plain(progress)

    receipt = AgentCard.render_receipt(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "done",
                structured_content={
                    "turns_used": 1,
                    "child_session_path": "",
                    "agent_type": "explore",
                },
            ),
        )
    )
    assert receipt is not None
    assert "explore · task research" in receipt.plain


def test_nested_agent_card_shows_depth_and_reaches_grandchild_tail(
    tmp_path: Path,
) -> None:
    grandchild = ConversationStore(tmp_path / "agents", session_id="1")
    grandchild.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("grandchild receipt")])
    )
    child = ConversationStore(tmp_path / "agents", session_id="child")
    nested = ToolCall(
        "grandchild-call",
        "agent",
        {"prompt": "inspect", "description": "grandchild"},
    )
    child.append_message(Message(MessageRole.ASSISTANT, [ToolUseContent(nested)]))
    child.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [TextContent("grandchild complete")],
            tool_result=ToolResult(
                nested.id,
                "grandchild complete",
                structured_content={
                    "turns_used": 1,
                    "child_session_path": str(grandchild.session_dir),
                    "depth": 2,
                },
            ),
        )
    )
    call = ToolCall(
        "child-call",
        "agent",
        {"prompt": "inspect", "description": "child"},
    )
    rendered = AgentCard.render_expanded(
        call,
        elapsed_seconds=1.0,
        turns_used=1,
        child_session_path=str(child.session_dir),
    )

    assert rendered is not None
    plain = Text.from_ansi(renderable_plain(rendered)).plain
    assert "depth 1" in plain
    assert "grandchild receipt" in plain


@pytest.mark.asyncio
async def test_nested_agent_lifecycle_reaches_tui_with_depth(
    tmp_path: Path,
) -> None:
    nested = ToolCall(
        "grandchild-call",
        "agent",
        {"prompt": "inspect", "description": "grandchild"},
    )
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[
                    ToolCall(
                        "child-call",
                        "agent",
                        {"prompt": "inspect", "description": "child"},
                    )
                ]
            ),
            ScriptedTurn(tool_calls=[nested]),
            ScriptedTurn([TextContent("grandchild complete")]),
            ScriptedTurn([TextContent("child complete")]),
        ]
    )
    events = [
        event
        async for event in AgentLoop(
            backend, ConversationStore(tmp_path), max_turns=1
        ).run_turn("start")
    ]

    starts = [
        event
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_START
        and event.tool_call is not None
        and event.tool_call.id == nested.id
    ]
    ends = [
        event
        for event in events
        if event.type is StreamEventType.TOOL_EXECUTION_END
        and event.tool_call is not None
        and event.tool_call.id == nested.id
    ]
    assert len(starts) == 1
    assert len(ends) == 1
    assert starts[0].data["depth"] == 2
    assert ends[0].tool_result is not None
    assert ends[0].tool_result.structured_content is not None
    assert ends[0].tool_result.structured_content["depth"] == 2
    rendered = AgentCard.render_start(starts[0])
    assert rendered is not None
    assert "depth 2" in renderable_plain(rendered)


def test_nested_lifecycle_keys_keep_identical_foreground_and_background_ids() -> None:
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        _test_console(),
        lambda: True,
        lambda renderable: None,
    )
    foreground = ToolCall(
        "same-id",
        "agent",
        {"prompt": "inspect", "description": "foreground"},
    )
    background = ToolCall(
        "same-id",
        "agent",
        {"prompt": "inspect", "description": "background"},
    )
    foreground_scope = "root:foreground"
    background_scope = "root:background"

    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_START,
            tool_call=foreground,
            data={"agent_instance_id": foreground_scope},
        ),
        aborted=False,
    )
    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_START,
            tool_call=background,
            data={"agent_instance_id": background_scope},
        ),
        aborted=False,
    )
    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=background,
            tool_result=ToolResult(
                background.id,
                "background agent started",
                structured_content={
                    "status": "running",
                    "child_session_path": "/tmp/background",
                },
            ),
            data={"agent_instance_id": background_scope},
        ),
        aborted=False,
    )
    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=foreground,
            tool_result=ToolResult(foreground.id, "foreground complete"),
            data={"agent_instance_id": foreground_scope},
        ),
        aborted=False,
    )

    assert (background_scope, background.id) in transcript._tools
    assert (foreground_scope, foreground.id) not in transcript._tools
    assert presenter.has_active_agent

    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=background,
            tool_result=ToolResult(background.id, "background complete"),
            data={"agent_instance_id": background_scope},
        ),
        aborted=False,
    )

    assert not transcript._tools
    assert not presenter.has_active_agent


def test_typed_agent_receipt_keeps_type_on_transcript_replay(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path)
    call = ToolCall(
        "replayed-agent",
        "agent",
        {
            "prompt": "inspect",
            "description": "task research",
            "agent_type": "plan",
        },
    )
    store.append_message(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)])
    )
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(
                call.id,
                "done",
                structured_content={
                    "turns_used": 2,
                    "child_session_path": "",
                    "agent_type": "plan",
                },
            ),
        )
    )
    app = TUIApp(AgentLoop(FakeBackend([]), store), provider="fake", model="offline")
    app._active_session = app._make_session()

    app._rebuild_transcript()

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert "plan · task research" in rendered


def test_agent_card_expansion_reads_bounded_child_tail(tmp_path: Path) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    for index in range(25):
        child.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"child line {index}")])
        )
    call = ToolCall(
        "agent-expand",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "done",
            structured_content={
                "turns_used": 3,
                "child_session_path": str(child.session_dir),
            },
        ),
    )
    transcript = TranscriptWidget()
    transcript.start_tool(call.id, call, render_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    ))
    transcript.finish_tool(call.id, render_event(event), event)

    assert transcript.toggle_latest_agent()
    rendered = Text.from_ansi(transcript.render(120)).plain
    assert "child line 4" not in rendered
    assert "child line 5" in rendered
    assert "child line 24" in rendered
    assert rendered.count("child line ") == 20


def test_agent_cards_remain_in_sequential_order() -> None:
    transcript = TranscriptWidget()
    for index in range(2):
        call = ToolCall(
            f"agent-{index}",
            "agent",
            {"prompt": "inspect", "description": f"task {index}"},
        )
        event = StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "done",
                structured_content={"turns_used": index + 1, "child_session_path": ""},
            ),
        )
        transcript.start_tool(call.id, call, render_event(
            StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
        ))
        transcript.finish_tool(call.id, render_event(event), event)

    rendered = Text.from_ansi(transcript.render(120)).plain
    assert rendered.index("task 0") < rendered.index("task 1")
    assert len(transcript._agent_units) == 2


def test_non_full_screen_agent_cards_keep_interleaved_child_streams() -> None:
    calls = [
        ToolCall(
            f"agent-{index}",
            "agent",
            {"prompt": "inspect", "description": f"task {index}"},
        )
        for index in (1, 2)
    ]
    output: list[object] = []
    presenter = TranscriptPresenter(
        TranscriptWidget(),
        _test_console(),
        lambda: False,
        output.append,
    )
    for call in calls:
        presenter.handle_tool_event(
            StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call),
            aborted=False,
        )
    for index, call in enumerate(calls, start=1):
        presenter.handle_tool_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_UPDATE,
                tool_call=call,
                delta=f"step-{index}",
                data={"child_session_path": f"/tmp/child-{index}"},
            ),
            aborted=False,
        )

    units = presenter._tool_region_units
    assert units[(None, "agent-1")].card._child_session_path == "/tmp/child-1"
    assert units[(None, "agent-2")].card._child_session_path == "/tmp/child-2"
    assert "step-1" in "\n".join(units[(None, "agent-1")].output)
    assert "step-2" in "\n".join(units[(None, "agent-2")].output)

    for index, call in enumerate(calls, start=1):
        presenter.handle_tool_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_END,
                tool_call=call,
                tool_result=ToolResult(
                    call.id,
                    f"done-{index}",
                    structured_content={
                        "turns_used": index,
                        "child_session_path": f"/tmp/child-{index}",
                    },
                ),
            ),
            aborted=False,
        )
    rendered = "\n".join(renderable_plain(item) for item in output)
    assert "task 1" in rendered
    assert "task 2" in rendered


def test_running_agent_card_can_expand_and_read_live_tail(tmp_path: Path) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("live tail")]))
    call = ToolCall(
        "agent-running-expand",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    update = StreamEvent(
        StreamEventType.TOOL_EXECUTION_UPDATE,
        tool_call=call,
        delta="turn 1: thinking",
        data={"stream": "stdout", "child_session_path": str(child.session_dir)},
    )
    transcript = TranscriptWidget()
    transcript.start_tool(call.id, call, render_event(start))
    transcript.update_tool(call.id, Text(update.delta), update)

    assert transcript.toggle_latest_agent()
    rendered = Text.from_ansi(transcript.render(120)).plain
    assert "live tail" in rendered
    assert "1 turns" in rendered


def test_background_agent_card_stores_path_at_start_and_expands(
    tmp_path: Path,
) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("live tail")]))
    call = ToolCall(
        "agent-background-expand",
        "agent",
        {
            "prompt": "inspect",
            "description": "background research",
            "background": True,
        },
    )
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        _test_console(),
        lambda: True,
        lambda renderable: None,
    )

    presenter.handle_tool_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call),
        aborted=False,
    )
    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "background agent started",
                structured_content={
                    "status": "running",
                    "child_session_path": str(child.session_dir),
                },
            ),
        ),
        aborted=False,
    )

    assert transcript.toggle_latest_agent()
    rendered = Text.from_ansi(transcript.render(120)).plain
    assert "live tail" in rendered


def test_presenter_refreshes_live_agent_cards_in_full_screen() -> None:
    call = ToolCall(
        "agent-refresh",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    transcript = TranscriptWidget()
    transcript.start_tool(
        call.id,
        call,
        render_event(StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)),
    )
    presenter = TranscriptPresenter(
        transcript,
        _test_console(),
        lambda: True,
        lambda renderable: None,
    )

    presenter.refresh_active_agents()

    assert transcript._tools[(None, call.id)].revision == 1


def test_background_agent_refresh_invalidates_transcript_search(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call = ToolCall(
        "agent-background-refresh",
        "agent",
        {"prompt": "inspect", "description": "background research"},
    )
    transcript = TranscriptWidget()
    presenter = TranscriptPresenter(
        transcript,
        _test_console(),
        lambda: True,
        lambda renderable: None,
    )
    presenter.handle_tool_event(
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call),
        aborted=False,
    )
    presenter.handle_tool_event(
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(
                call.id,
                "background agent started",
                structured_content={"status": "running"},
            ),
        ),
        aborted=False,
    )
    transcript.begin_search()
    transcript.update_search("fresh target")
    assert transcript.search_status() == (0, 0)

    unit = transcript._tools[(None, call.id)]
    monkeypatch.setattr(unit.card, "refresh", lambda: Text("fresh target"))
    presenter.refresh_active_agents()

    assert transcript.search_status() == (1, 1)


def test_agent_card_toggle_is_symmetric_during_and_after_execution(
    tmp_path: Path,
) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("live tail")]))
    call = ToolCall(
        "agent-toggle",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    start = StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)
    update = StreamEvent(
        StreamEventType.TOOL_EXECUTION_UPDATE,
        tool_call=call,
        delta="turn 1: thinking",
        data={"child_session_path": str(child.session_dir)},
    )
    transcript = TranscriptWidget()
    transcript.start_tool(call.id, call, render_event(start))
    transcript.update_tool(call.id, Text(update.delta), update)

    assert transcript.toggle_latest_agent()
    assert "collapse: ctrl+x ctrl+o" in transcript.render(120)
    assert transcript.toggle_latest_agent()
    assert "expand: ctrl+x ctrl+o" in transcript.render(120)

    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "done",
            structured_content={
                "turns_used": 1,
                "child_session_path": str(child.session_dir),
            },
        ),
    )
    transcript.finish_tool(call.id, render_event(event), event)
    assert transcript.toggle_latest_agent()
    assert "collapse: ctrl+x ctrl+o" in transcript.render(120)
    assert transcript.toggle_latest_agent()
    rendered = Text.from_ansi(transcript.render(120)).plain
    assert "expand: ctrl+x ctrl+o" in rendered
    assert "1 turns · 0.0s · ok" in rendered


def test_canceled_agent_card_can_expand_with_persisted_child_tail(tmp_path: Path) -> None:
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.append_message(Message(MessageRole.USER, [TextContent("cancelled task")]))
    call = ToolCall(
        "agent-canceled-expand",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "tool execution canceled",
            is_error=True,
            structured_content={
                "turns_used": 1,
                "child_session_path": str(child.session_dir),
            },
        ),
    )
    transcript = TranscriptWidget()
    transcript.start_tool(call.id, call, render_event(StreamEvent(
        StreamEventType.TOOL_EXECUTION_START, tool_call=call
    )))
    transcript.finish_tool(call.id, render_event(event), event)

    assert transcript.toggle_latest_agent()
    assert "cancelled task" in Text.from_ansi(transcript.render(120)).plain


def test_recovered_canceled_agent_card_expands_with_child_tail(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path, session_id="parent")
    call = ToolCall(
        "agent-recovered",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    child = ConversationStore(
        store.session_dir / "agents", session_id="1", cwd=store.cwd
    )
    child.mark_agent_parent(call.id)
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("saved child tail")]))
    store.allocate_agent_index()
    store.register_agent_child(
        call,
        child_session_path=str(child.session_dir),
        description="task research",
    )
    store.update_agent_child_turns(call.id, 2)

    AgentLoop(FakeBackend([]), store, max_turns=1)
    result = next(message.tool_result for message in store.messages() if message.tool_result)
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=result,
    )
    transcript = TranscriptWidget()
    transcript.start_tool(
        call.id,
        call,
        render_event(StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call)),
    )
    transcript.finish_tool(call.id, render_event(event), event)

    assert transcript.toggle_latest_agent()
    rendered = Text.from_ansi(transcript.render(120)).plain
    assert "saved child tail" in rendered
    assert "2 turns" in rendered


def test_finished_agent_card_keeps_elapsed_time_after_clock_moves(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.tui import agent_card

    clock = iter((100.0, 105.0, 205.0))
    monkeypatch.setattr(agent_card.time, "monotonic", lambda: next(clock))
    call = ToolCall(
        "agent-clock",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    child = ConversationStore(tmp_path / "agents", session_id="1")
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("done")]))
    event = StreamEvent(
        StreamEventType.TOOL_EXECUTION_END,
        tool_call=call,
        tool_result=ToolResult(
            call.id,
            "done",
            structured_content={
                "turns_used": 1,
                "child_session_path": str(child.session_dir),
            },
        ),
    )
    transcript = TranscriptWidget()
    transcript.start_tool(call.id, call, render_event(StreamEvent(
        StreamEventType.TOOL_EXECUTION_START, tool_call=call
    )))
    transcript.finish_tool(call.id, render_event(event), event)
    assert transcript.toggle_latest_agent()

    rendered = Text.from_ansi(transcript.render(120)).plain
    assert "5.0s" in rendered
    assert "105.0s" not in rendered


def test_agent_rendering_dispatch_stays_in_agent_card_seam() -> None:
    root = Path(__file__).parents[1] / "src" / "zeta" / "tui"
    allowed_path = Path("tui") / "agent_card.py"
    violations: list[str] = []

    def docstring_constants(tree: ast.Module) -> set[ast.Constant]:
        return {
            node.body[0].value
            for node in ast.walk(tree)
            if isinstance(
                node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
            )
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
            and isinstance(node.body[0].value.value, str)
        }

    for path in root.rglob("*.py"):
        if path.relative_to(root.parent) == allowed_path:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        ignored = docstring_constants(tree)
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Constant)
                and node.value == "agent"
                and node not in ignored
            ):
                violations.append(f"{path}:{node.lineno}")

    assert violations == []


def test_agent_card_binding_preserves_native_ctrl_o_in_both_edit_modes() -> None:
    bindings = build_key_bindings(
        on_interrupt=lambda: None,
        on_exit=lambda: None,
        on_toggle_agent=lambda: None,
    )
    assert not bindings.get_bindings_for_keys((Keys.ControlO,))
    assert bindings.get_bindings_for_keys((Keys.ControlX, Keys.ControlO))
    for mode in (EditingMode.EMACS, EditingMode.VI):
        session = PromptSession(key_bindings=bindings, editing_mode=mode)
        assert not session.app.key_bindings.get_bindings_for_keys((Keys.ControlO,))


def test_transcript_navigation_bindings_are_full_screen_only() -> None:
    bindings = build_key_bindings(
        on_interrupt=lambda: None,
        on_exit=lambda: None,
        on_search_start=lambda: None,
        search_active=lambda: False,
        on_search_input=lambda _value: None,
        on_search_end=lambda: None,
        on_previous_user=lambda: None,
        on_next_user=lambda: None,
    )

    assert bindings.get_bindings_for_keys((Keys.ControlF,))
    assert bindings.get_bindings_for_keys((Keys.ControlUp,))
    assert bindings.get_bindings_for_keys((Keys.ControlDown,))
    inline = PromptSession(key_bindings=bindings)
    with set_app(inline.app):
        assert not bindings.get_bindings_for_keys((Keys.ControlF,))[-1].filter()


@pytest.mark.asyncio
async def test_transcript_search_query_accepts_navigation_key_text() -> None:
    active = False
    values: list[str] = []
    actions: list[str] = []

    def start_search() -> None:
        nonlocal active
        active = True

    def end_search() -> None:
        nonlocal active
        active = False

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_search_start=start_search,
                search_active=lambda: active,
                on_search_input=values.append,
                on_search_next=lambda: actions.append("next"),
                on_search_previous=lambda: actions.append("previous"),
                on_search_end=end_search,
            ),
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0.05)
        session.app.full_screen = True
        pipe.send_text("\x06nN\x12")
        await wait_until(lambda: values == ["n", "nN", "nN\x12"])
        pipe.send_text("\rN\x1b")
        await wait_until(lambda: actions == ["next", "previous"])
        session.app.exit()
        await task


@pytest.mark.asyncio
@pytest.mark.parametrize("editing_mode", [EditingMode.EMACS, EditingMode.VI])
async def test_history_search_acceptance_survives_transcript_search(
    editing_mode: EditingMode,
    tmp_path: Path,
) -> None:
    history = history_for(tmp_path / "history")
    history.append_string("history target")
    transcript_search_active = False
    actions: list[str] = []

    def start_search() -> None:
        nonlocal transcript_search_active
        transcript_search_active = True

    def end_search() -> None:
        nonlocal transcript_search_active
        transcript_search_active = False

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            editing_mode=editing_mode,
            history=history_for(tmp_path / "history"),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_search_start=start_search,
                search_active=lambda: transcript_search_active,
                on_search_input=lambda _value: None,
                on_search_next=lambda: actions.append("next"),
                on_search_end=end_search,
            ),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0.05)
        session.app.full_screen = True
        pipe.send_text("\x06query\r")
        await wait_until(lambda: actions == ["next"])
        pipe.send_text("\x12history")
        await wait_until(lambda: session.search_buffer.text == "history")
        pipe.send_text("\r")
        await wait_until(
            lambda: session.default_buffer.text == "history target"
        )
        assert actions == ["next"]
        session.app.exit()
        await task


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


def test_thought_renders_full_trace_with_header_and_duration() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent("Plan first.\nHide the rest."),
            data={"duration": 2.7},
        )
    )

    assert isinstance(rendered, Text)
    assert rendered.plain == "✱ thought · 2.7s\nPlan first.\nHide the rest."
    assert all("italic" in span.style for span in rendered.spans)


def test_thought_header_has_duration_only() -> None:
    assert format_thought(2.7).plain == "✱ thought · 2.7s"


def test_redacted_thought_renders_as_collapsed_line_with_duration() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=RedactedThinkingContent("opaque"),
            data={"duration": 1.25},
        )
    )

    assert isinstance(rendered, Text)
    assert rendered.plain == "✱ thought · redacted · 1.2s"


@pytest.mark.asyncio
async def test_anthropic_redacted_thinking_reaches_tui_stream(tmp_path: Path) -> None:
    stream = "\n".join(  # noqa: FLY002 - construct the SSE fixture clearly
        [
            'data: {"type":"message_start","message":{}}',
            "",
            'data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}',
            "",
            'data: {"type":"content_block_stop","index":0}',
            "",
            'data: {"type":"message_stop"}',
            "",
        ]
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=stream,
            request=request,
        )

    credentials = AnthropicCredentialStore(tmp_path / "zeta.json")
    credentials.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = AnthropicBackend(
        client=client,
        token_store=credentials,
        base_url="https://test.invalid/v1/messages",
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="anthropic",
        model="claude-sonnet-4-6",
        console=Console(file=output, force_terminal=False),
    )

    await app._consume_turn("hello")
    await client.aclose()

    rendered = output.getvalue()
    assert "✱ thought · redacted" in rendered
    assert "no response" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["output-items", "summary-and-raw"])
async def test_codex_thought_blocks_reach_tui_as_separate_units(
    tmp_path: Path, case: str
) -> None:
    if case == "output-items":
        events = _codex_reasoning_events([("first", ""), ("second", "")])
        expected = ("first", "second")
    else:
        events = _codex_reasoning_events([("summary", "raw")])
        expected = ("summary", "raw")

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=_codex_sse(events),
            request=request,
        )

    credentials = CodexCredentialStore(tmp_path / "codex.json")
    credentials.save(OAuthTokens(_codex_access_token(), "refresh-fixture", 4_000_000_000))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    output = StringIO()
    app = TUIApp(
        AgentLoop(
            CodexBackend(
                client=client,
                token_store=credentials,
                base_url="https://test.invalid/codex/responses",
            ),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="codex",
        model=DEFAULT_CODEX_MODEL,
        console=Console(file=output, force_terminal=False),
    )

    await app._consume_turn("hello")
    await client.aclose()

    rendered = output.getvalue()
    assert rendered.count("✱ thought ·") == 2
    assert all(text in rendered for text in expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("redacted_first", [True, False])
async def test_anthropic_mixed_thinking_blocks_reach_tui_as_separate_lines(
    tmp_path: Path, redacted_first: bool
) -> None:
    blocks = (
        [("redacted_thinking", '"data":"opaque"'), ("thinking", '"thinking":"plan"')]
        if redacted_first
        else [("thinking", '"thinking":"plan"'), ("redacted_thinking", '"data":"opaque"')]
    )
    lines = [
        'data: {"type":"message_start","message":{}}',
        "",
    ]
    for index, (kind, value) in enumerate(blocks):
        lines.extend(
            [
                f'data: {{"type":"content_block_start","index":{index},"content_block":{{"type":"{kind}",{value}}}}}',
                "",
            ]
        )
        if kind == "thinking":
            lines.extend(
                [
                    f'data: {{"type":"content_block_delta","index":{index},"delta":{{"type":"thinking_delta","thinking":"plan"}}}}',
                    "",
                    f'data: {{"type":"content_block_delta","index":{index},"delta":{{"type":"signature_delta","signature":"sig-{index}"}}}}',
                    "",
                ]
            )
        lines.extend(
            [
                f'data: {{"type":"content_block_stop","index":{index}}}',
                "",
            ]
        )
    lines.extend(['data: {"type":"message_stop"}', ""])

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text="\n".join(lines),
            request=request,
        )

    credentials = AnthropicCredentialStore(tmp_path / "zeta.json")
    credentials.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = AnthropicBackend(
        client=client,
        token_store=credentials,
        base_url="https://test.invalid/v1/messages",
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="anthropic",
        model="claude-sonnet-4-6",
        console=Console(file=output, force_terminal=False),
    )

    await app._consume_turn("hello")
    await client.aclose()

    rendered = output.getvalue()
    assert rendered.count("✱ thought · redacted") == 1
    assert rendered.count("✱ thought ·") == 2
    assert rendered.count("\n  plan") == 1


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
    assert not _contains_background_sgr(escaped)


def test_retry_notice_is_dim() -> None:
    rendered = render_event(
        StreamEvent(
            StreamEventType.RETRY,
            data={"text": "retrying (1/3) in 1s — 503 server error"},
        )
    )

    assert isinstance(rendered, Text)
    assert rendered.plain == "retrying (1/3) in 1s — 503 server error"
    assert rendered.style == DIM


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


def test_stream_kind_switch_keeps_partial_assistant_transient(tmp_path: Path) -> None:
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

    assert rendered[0].plain.startswith("✱ thought · ")
    assert rendered[0].plain.endswith("\nplan\n")
    assert all("| name | value |" not in item.plain for item in rendered)


def test_thought_stream_is_visible_before_completion(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()

    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent("first\n"),
        )
    )
    assert [Text.from_ansi(line).plain for line in app._transcript.lines(120)] == [
        "first"
    ]

    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent("second"),
        )
    )
    assert [Text.from_ansi(line).plain for line in app._transcript.lines(120)] == [
        "first",
        "second",
    ]

    app._flush_pending_stream()
    rendered = [Text.from_ansi(line).plain for line in app._transcript.lines(120)]
    assert rendered[0].startswith("✱ thought · ")
    assert rendered[1:] == ["first", "second"]


def test_adjacent_signed_thought_blocks_keep_separate_units(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()

    for content in (
        ThinkingContent("block-one", "sig-1"),
        ThinkingContent("block-two", "sig-2"),
    ):
        app._consume_text(
            StreamEvent(StreamEventType.MESSAGE_UPDATE, content=content)
        )
    app._flush_pending_stream()

    rendered = [Text.from_ansi(line).plain for line in app._transcript.lines(120)]
    assert sum(line.startswith("✱ thought ·") for line in rendered) == 2
    assert rendered.count("block-one") == 1
    assert rendered.count("block-two") == 1


def test_thought_commit_keeps_logical_lines_for_resize(tmp_path: Path) -> None:
    source = "logical-line-that-reflows"
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()

    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent(source),
        )
    )
    app._flush_pending_stream()

    assert any(
        source in getattr(unit, "plain", "")
        for unit in app._transcript.units
        if unit
    )
    narrow = [Text.from_ansi(line).plain for line in app._transcript.lines(18)]
    wide = [Text.from_ansi(line).plain for line in app._transcript.lines(80)]
    assert len([line for line in narrow if line]) > len(wide)
    assert wide[-1] == source


def test_thought_updates_keep_active_unit_after_interleaved_output(
    tmp_path: Path,
) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()

    app._presenter.start_thinking(render_thought_live("partial"))
    app._transcript.append(Text("tool output"))
    app._presenter.update_thinking(render_thought_live("complete"))
    app._presenter.finish_thinking(render_thought("complete", 1.0))

    rendered = [Text.from_ansi(line).plain for line in app._transcript.lines(120)]
    assert sum(line.startswith("✱ thought ·") for line in rendered) == 1
    assert rendered.count("complete") == 1
    assert "partial" not in rendered
    assert "tool output" in rendered


def test_thought_duration_uses_local_monotonic_lifecycle_clock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ticks = iter((10.0, 10.25))
    monkeypatch.setattr("zeta.tui.app.time.monotonic", lambda: next(ticks))
    output = StringIO()
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )
    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent("plan"),
        )
    )
    app._flush_pending_stream()

    rendered = "\n".join(
        line.strip() for line in Text.from_ansi(output.getvalue()).plain.splitlines()
    )
    assert "✱ thought · 0.2s\nplan" in rendered


def test_assistant_renderables_share_one_logical_unit(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )

    app._active_session = app._make_session()
    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("first paragraph\nsecond paragraph\n```python\nprint('hi')\n```"),
        )
    )
    app._finish_message(
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(
                MessageRole.ASSISTANT,
                [TextContent("first paragraph\nsecond paragraph\n```python\nprint('hi')\n```")],
            ),
        )
    )
    rendered = [Text.from_ansi(line).plain for line in app._transcript.lines(120)]
    assert rendered[:2] == ["first paragraph second paragraph", ""]
    assert "print('hi')" in "\n".join(rendered)


@pytest.mark.asyncio
async def test_error_flushes_assistant_before_error(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(ErrorBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._active_session = app._make_session()
    rendered = []

    def capture(renderable: object | None) -> None:
        if renderable is not None:
            rendered.append(renderable)

    app._print = capture

    await app._consume_turn("prompt")

    line_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "plain", "").startswith("| name | value |")
    )
    error_index = next(
        index
        for index, item in enumerate(rendered)
        if "provider failure · backend_error" in renderable_plain(item)
    )
    assert line_index < error_index


@pytest.mark.asyncio
async def test_verbose_error_flushes_before_raw_error(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(ErrorBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        verbose=True,
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._active_session = app._make_session()
    rendered = []

    def capture(renderable: object | None) -> None:
        if renderable is not None:
            rendered.append(renderable)

    app._print = capture

    await app._consume_turn("prompt")

    line_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "plain", "").startswith("| name | value |")
    )
    raw_error_index = next(
        index
        for index, item in enumerate(rendered)
        if '"type": "error"' in getattr(item, "plain", "")
    )
    pretty_error_index = next(
        index
        for index, item in enumerate(rendered)
        if "provider failure · backend_error" in renderable_plain(item)
    )
    assert line_index < raw_error_index < pretty_error_index


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
    app._active_session = app._make_session()
    rendered = []

    def capture(renderable: object | None) -> None:
        if renderable is not None:
            rendered.append(renderable)

    app._print = capture

    await app._consume_turn("prompt")

    line_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "plain", "").startswith("| name | value |")
    )
    raw_index = next(
        index
        for index, item in enumerate(rendered)
        if getattr(item, "plain", "").startswith("{")
        and raw_marker in item.plain
    )
    assert line_index < raw_index


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
        while b" > " not in output and time.monotonic() < deadline:
            ready, _, _ = select.select(
                [master_fd],
                [],
                [],
                max(0, deadline - time.monotonic()),
            )
            if ready:
                output.extend(os.read(master_fd, 4096))
        assert b" > " in output

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
        assert b"\x1b[0 q" in output
        assert output.count(b"\x1b[?1049h") == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master_fd)


def test_main_pty_emits_vim_cursor_shapes_and_resets_on_toggle(
    tmp_path: Path,
) -> None:
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
    output = bytearray()

    def read_until(needle: bytes, start: int = 0) -> None:
        deadline = time.monotonic() + 5
        while needle not in output[start:] and time.monotonic() < deadline:
            ready, _, _ = select.select(
                [master_fd],
                [],
                [],
                max(0, deadline - time.monotonic()),
            )
            if ready:
                try:
                    output.extend(os.read(master_fd, 4096))
                except OSError:
                    break
        assert needle in output[start:]

    try:
        read_until(b" > ")
        read_until(b"\x1b[6 q")

        os.write(master_fd, b"\x1b")
        read_until(b"\x1b[2 q")

        start = len(output)
        os.write(master_fd, b"i")
        read_until(b"\x1b[6 q", start)
        time.sleep(0.1)

        start = len(output)
        os.write(master_fd, b"/vim off\r")
        read_until(b"vim mode: off", start)
        read_until(b"\x1b[0 q", start)

        os.write(master_fd, b"\x04")
        deadline = time.monotonic() + 5
        while process.poll() is None and time.monotonic() < deadline:
            ready, _, _ = select.select([master_fd], [], [], 0.1)
            if ready:
                try:
                    output.extend(os.read(master_fd, 4096))
                except OSError:
                    break
        assert process.wait(timeout=5) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        os.close(master_fd)


def test_main_pty_normal_command_then_queued_enter_submits(
    tmp_path: Path,
) -> None:
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
    output = bytearray()

    def read_until(needle: bytes, start: int = 0) -> None:
        deadline = time.monotonic() + 5
        while needle not in output[start:] and time.monotonic() < deadline:
            ready, _, _ = select.select(
                [master_fd],
                [],
                [],
                max(0, deadline - time.monotonic()),
            )
            if ready:
                try:
                    output.extend(os.read(master_fd, 4096))
                except OSError:
                    break
        assert needle in output[start:]

    try:
        read_until(b" > ")
        os.write(master_fd, b"abc")
        read_until(b"abc")

        start = len(output)
        os.write(master_fd, b"\x1bx\r")
        read_until(b"you said: ab", start)
    finally:
        if process.poll() is None:
            os.write(master_fd, b"\x04")
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
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
        check=False,
    )

    assert result.returncode == 0, result.stderr


def test_completed_markdown_fixture_renders_the_capability_set() -> None:
    source = dedent(
        """
        # heading

        **bold** *italic* ***both*** ~~strike~~ `code` [link](https://example.com)

        - [x] done
          - nested
        1. ordered
        2. second

        > quote
        >
        > > nested quote

        | left | center | right |
        | :--- | :----: | ----: |
        | one | two | 3 |

        ```python
        print("python")
        ```

        ```ts
        const value = 1
        ```

        ```diff
        - old
        + new
        ```

        ---

        escaped \\*literal\\* and a hard break\\
        next line
        """
    ).strip()
    output = StringIO()
    rendered_console = _test_console(output, width=72)
    rendered_console.print(render_markdown(source))
    rendered = output.getvalue()
    plain = Text.from_ansi(rendered).plain

    assert "heading" in plain
    assert "bold" in plain and "italic" in plain and "both" in plain
    assert "strike" in plain and "code" in plain
    assert "link (https://example.com)" in plain
    assert "[x] done" in plain and "nested" in plain
    assert "│ quote" in plain and "│ │ nested quote" in plain
    assert "left" in plain and "center" in plain and "right" in plain
    assert "print(\"python\")" in plain and "const value = 1" in plain
    assert "old" in plain and "new" in plain
    assert "literal*" in plain and "hard break\nnext line" in plain
    assert "\x1b[9m" in rendered
    assert not _contains_background_sgr(rendered)


@pytest.mark.parametrize(
    ("source", "expected"),
    [("<div>raw html</div>", "<div>raw html</div>"), ("    indented code", "indented code")],
)
def test_completed_markdown_keeps_visible_literal_blocks(
    source: str, expected: str
) -> None:
    output = StringIO()
    _test_console(output).print(render_markdown(source))

    assert expected in Text.from_ansi(output.getvalue()).plain


def test_completed_message_replaces_streaming_unit_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    source = "# completed\n\n- item"
    calls: list[str] = []
    real_renderer = render_markdown

    def spy(value: str):
        calls.append(value)
        return real_renderer(value)

    monkeypatch.setattr("zeta.tui.app.render_markdown", spy)
    app._consume_text(
        StreamEvent(StreamEventType.MESSAGE_UPDATE, content=TextContent(source))
    )
    unit = app._transcript._units[0]
    assert isinstance(unit.value, Text)
    assert unit.value.plain == source

    app._finish_message(
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(source)]),
        )
    )

    assert calls == [source]
    assert app._transcript._units[0] is unit
    assert type(unit.value).__name__ == "MarkdownDocument"
    assert "completed" in "\n".join(app._transcript.lines(80))


def test_mixed_assistant_tool_transcript_commits_each_text_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    call = ToolCall("mixed-1", "read", {"path": "README.md"})
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    calls: list[str] = []
    real_renderer = render_markdown

    def spy(value: str):
        calls.append(value)
        return real_renderer(value)

    monkeypatch.setattr("zeta.tui.app.render_markdown", spy)
    before = "**before tool**"
    after = "**after tool**"
    events = [
        StreamEvent(StreamEventType.MESSAGE_START),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent(before),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ToolUseContent(call),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(
                MessageRole.ASSISTANT,
                [TextContent(before), ToolUseContent(call)],
            ),
        ),
        StreamEvent(StreamEventType.TOOL_EXECUTION_START, tool_call=call),
        StreamEvent(
            StreamEventType.TOOL_EXECUTION_END,
            tool_call=call,
            tool_result=ToolResult(call.id, "tool result"),
        ),
        StreamEvent(StreamEventType.MESSAGE_START),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent(after),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(after)]),
        ),
    ]
    for event in events:
        app._prepare_stream_event(event)
        app._handle_tool_event(event)
        if event.type is StreamEventType.MESSAGE_UPDATE:
            app._consume_text(event)
        elif event.type is StreamEventType.MESSAGE_END:
            app._finish_message(event)

    plain = "\n".join(
        Text.from_ansi(line).plain for line in app._transcript.lines(80)
    )
    assert plain.count("before tool") == 1
    assert plain.count("after tool") == 1
    assert calls == [before, after]
    assert type(app._transcript.units[0]).__name__ == "MarkdownDocument"
    before_line = next(
        line for line in app._transcript.lines(80) if "before tool" in line
    )
    assert "\x1b[1;" in before_line


def test_hostile_markdown_is_bounded_and_falls_back_or_renders() -> None:
    source = "\n".join(
        [
            "*" * 10000,
            "`" * 10000,
            *(f"- item {index}" for index in range(10000)),
            ">" * 500 + " tail",
        ]
    )
    started = time.monotonic()
    document = render_markdown(source)
    output = StringIO()
    _test_console(output, width=80).print(document)
    assert time.monotonic() - started < 5


def test_large_markdown_table_falls_back_within_render_budget() -> None:
    source = "| name | value |\n| --- | --- |\n" + "\n".join(
        f"| row-{index} | value |" for index in range(10_000)
    )
    started = time.monotonic()
    output = StringIO()
    _test_console(output).print(render_markdown(source))

    assert time.monotonic() - started < 2
    assert "| row-9999 | value |" in Text.from_ansi(output.getvalue()).plain


def test_full_stream_preserves_inline_literals(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    source = "foo_bar_baz\n\\*literal\\*\n`a_b_c`\n**bold** mid _text_\n"

    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent(source),
        )
    )
    app._finish_message(
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(source)]),
        )
    )

    rendered = [Text.from_ansi(line).plain for line in app._transcript.lines(120)]
    assert rendered == ["foo_bar_baz *literal* a_b_c bold mid text"]


def test_markdown_stream_preserves_model_line_structure(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    source = [
        "pick what sounds fun:",
        "",
        "1. **take a walk** — get some air.",
        "2. **read a book** — settle in.",
        "3. **make a meal** — try a recipe.",
        "4. **play a game** — choose one.",
        "5. **call a friend** — catch up.",
    ]

    app._print_committed(source)
    app._flush_markdown()

    rendered = [Text.from_ansi(line).plain for line in app._transcript.lines(120)]
    assert rendered == [
        "pick what sounds fun:",
        "",
        "1. take a walk — get some air.",
        "2. read a book — settle in.",
        "3. make a meal — try a recipe.",
        "4. play a game — choose one.",
        "5. call a friend — catch up.",
    ]


def test_completed_message_renders_final_message_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    source = (
        "1. **Round 2 (parallel):** use **four** workers, then "
        "**single `fallback`**."
    )
    partial = source[: source.index("`fallback`") + 1]
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()

    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent(partial),
        )
    )
    app._finish_message(
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent(source)]),
        )
    )

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert rendered == (
        "1. Round 2 (parallel): use four workers, then single fallback."
    )
    assert "**" not in rendered


def test_completed_message_rerenders_text_split_by_thinking(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    events = [
        StreamEvent(StreamEventType.MESSAGE_START),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("before **"),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent("plan"),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("bold** after"),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(
                MessageRole.ASSISTANT,
                [
                    TextContent("before **"),
                    ThinkingContent("plan"),
                    TextContent("bold** after"),
                ],
            ),
        ),
    ]
    for event in events:
        app._prepare_stream_event(event)
        if event.type is StreamEventType.MESSAGE_UPDATE:
            app._consume_text(event)
        elif event.type is StreamEventType.MESSAGE_END:
            app._finish_message(event)

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert "**" not in rendered
    assert "before bold after" in rendered


def test_completed_message_rerenders_text_split_by_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    call = ToolCall("split-1", "read", {"path": "README.md"})
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    events = [
        StreamEvent(StreamEventType.MESSAGE_START),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("before **"),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ToolUseContent(call),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("bold** after"),
        ),
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(
                MessageRole.ASSISTANT,
                [
                    TextContent("before **"),
                    ToolUseContent(call),
                    TextContent("bold** after"),
                ],
            ),
        ),
    ]
    for event in events:
        app._prepare_stream_event(event)
        if event.type is StreamEventType.MESSAGE_UPDATE:
            app._consume_text(event)
        elif event.type is StreamEventType.MESSAGE_END:
            app._finish_message(event)

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert "**" not in rendered
    assert "before bold after" in rendered


@pytest.mark.asyncio
async def test_rebuild_transcript_matches_character_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    source_parts = [
        TextContent("before **"),
        ThinkingContent(""),
        TextContent("bold** after"),
    ]
    app = TUIApp(
        AgentLoop(
            FakeBackend(
                [ScriptedTurn(content=source_parts)]
            ),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    app._print_user("prompt")

    await app._consume_turn("prompt")

    live = app._transcript.render(120)
    app._rebuild_transcript()
    replayed = app._transcript.render(120)
    assert replayed == live


@pytest.mark.asyncio
@pytest.mark.parametrize("full_screen", [False, True])
async def test_run_replays_resumed_transcript_before_prompt(
    tmp_path: Path, full_screen: bool
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("remembered user")]))
    call = ToolCall("resume-read", "read", {"path": "README.md"})
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [TextContent("remembered assistant"), ToolUseContent(call)],
        )
    )
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(call.id, "remembered tool output"),
        )
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=True, color_system="truecolor"),
    )

    with create_pipe_input() as pipe:
        if full_screen:
            session = FullScreenPromptSession(
                input=pipe,
                output=DummyOutput(),
                key_bindings=build_key_bindings(
                    on_interrupt=app.abort_active,
                    on_exit=app.request_exit,
                ),
                multiline=True,
            )
        else:
            session = app_session(app, pipe)
        run_task = asyncio.create_task(app.run(session))
        await asyncio.sleep(0.05)
        pipe.send_text("\x04")
        await run_task

    rendered = (
        app._transcript.render(120)
        if full_screen
        else Text.from_ansi(output.getvalue()).plain
    )
    rendered = Text.from_ansi(rendered).plain
    assert "▌ remembered user" in rendered
    assert "remembered assistant" in rendered
    assert "read" in rendered
    assert "README.md" in rendered


@pytest.mark.asyncio
async def test_run_keeps_mcp_startup_notice_after_transcript_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = TUIApp(
        AgentLoop(
            FakeBackend([]),
            ConversationStore(tmp_path / "sessions"),
            skip_mcp_mount=True,
        ),
        provider="fake",
        model="offline",
    )

    async def ensure_mcp_servers() -> None:
        assert app.loop._mcp_notice_sink is not None
        app.loop._mcp_notice_sink("mcp · prompts unavailable")

    monkeypatch.setattr(app.loop, "ensure_mcp_servers", ensure_mcp_servers)
    with create_pipe_input() as pipe:
        session = FullScreenPromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=build_key_bindings(
                on_interrupt=app.abort_active,
                on_exit=app.request_exit,
            ),
            multiline=True,
        )
        run_task = asyncio.create_task(app.run(session))
        await asyncio.sleep(0.05)
        pipe.send_text("\x04")
        await run_task

    assert "prompts unavailable" in Text.from_ansi(
        app._transcript.render(120)
    ).plain


def test_rebuild_renders_compaction_marker_as_chrome(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(Message(MessageRole.USER, [TextContent("old user")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("old reply")]))
    store.append_compaction_marker("provider summary", 1, 2)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()

    app._rebuild_transcript()

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert "[compaction marker: entries 1–2]" in rendered
    assert "provider summary" not in rendered


@pytest.mark.asyncio
@pytest.mark.parametrize("full_screen", [False, True])
async def test_run_keeps_empty_resumed_transcript_blank(
    tmp_path: Path, full_screen: bool
) -> None:
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=True, color_system="truecolor"),
    )

    with create_pipe_input() as pipe:
        session = (
            FullScreenPromptSession(
                input=pipe,
                output=DummyOutput(),
                key_bindings=build_key_bindings(
                    on_interrupt=app.abort_active,
                    on_exit=app.request_exit,
                ),
                multiline=True,
            )
            if full_screen
            else app_session(app, pipe)
        )
        run_task = asyncio.create_task(app.run(session))
        await asyncio.sleep(0.05)
        pipe.send_text("\x04")
        await run_task

    assert app._transcript.units == ()
    assert "[error]" not in output.getvalue()


def _representative_fork_store(tmp_path: Path) -> ConversationStore:
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(
        Message(
            MessageRole.USER,
            [
                TextContent("inspect notes.txt and [Image #1]"),
                TextContent(
                    "[file: /tmp/notes.txt · 12 bytes]\nsecret contents",
                    "/tmp/notes.txt",
                    12,
                ),
                ImageContent(
                    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                    "image/png",
                    "/tmp/image.png",
                    68,
                ),
            ],
        )
    )
    read_call = ToolCall("resume-read", "read", {"path": "README.md"})
    agent_call = ToolCall(
        "resume-agent",
        "agent",
        {"prompt": "inspect", "description": "task research"},
    )
    store.append_message(
        Message(
            MessageRole.ASSISTANT,
            [
                TextContent("kept reply"),
                ToolUseContent(read_call),
                ToolUseContent(agent_call),
            ],
        )
    )
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(read_call.id, "stdout:\nread output"),
        )
    )
    child = ConversationStore(store.session_dir / "agents", session_id="1")
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("child result")]))
    store.append_message(
        Message(
            MessageRole.TOOL_RESULT,
            [],
            tool_result=ToolResult(
                agent_call.id,
                "done",
                structured_content={
                    "turns_used": 2,
                    "child_session_path": str(child.session_dir),
                },
            ),
        )
    )
    store.append_compaction_marker("provider summary", 1, 4)
    store._append_row("warning", {"message": "filtered warning"})
    checkpoint = store.append_checkpoint("base")
    store.append_message(Message(MessageRole.USER, [TextContent("abandoned branch")]))
    store.append_message(
        Message(MessageRole.ASSISTANT, [TextContent("abandoned reply")])
    )
    store.append_fork(str(checkpoint.seq))
    store.append_message(Message(MessageRole.USER, [TextContent("new branch")]))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent("new reply")]))
    store.append_checkpoint("saved")
    return store


def _display_session(
    app: TUIApp, pipe: PipeInput, full_screen: bool
) -> PromptSession[str]:
    if not full_screen:
        return app_session(app, pipe)
    return FullScreenPromptSession(
        input=pipe,
        output=DummyOutput(),
        key_bindings=build_key_bindings(
            on_interrupt=app.abort_active,
            on_exit=app.request_exit,
        ),
        multiline=True,
    )


def _rendered_display(app: TUIApp, output: StringIO, full_screen: bool) -> str:
    return app._transcript.render(120) if full_screen else output.getvalue()


def _test_tui_app(store: ConversationStore, output: StringIO) -> TUIApp:
    return TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(
            file=output,
            force_terminal=True,
            color_system="truecolor",
        ),
    )


async def _run_reopened_session(
    store: ConversationStore, full_screen: bool
) -> str:
    reopened = ConversationStore(store.root_dir, session_id=store.session_id)
    output = StringIO()
    app = _test_tui_app(reopened, output)
    with create_pipe_input() as pipe:
        session = _display_session(app, pipe, full_screen)
        run_task = asyncio.create_task(app.run(session))
        await asyncio.sleep(0.05)
        pipe.send_text("\x04")
        await asyncio.wait_for(run_task, timeout=2)
    return _rendered_display(app, output, full_screen)


@pytest.mark.asyncio
@pytest.mark.parametrize("full_screen", [False, True])
async def test_resume_replay_matches_slash_fork_render(
    tmp_path: Path, full_screen: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    store = _representative_fork_store(tmp_path)
    fork_output = StringIO()
    fork_app = _test_tui_app(store, fork_output)
    with create_pipe_input() as pipe:
        fork_app._active_session = _display_session(fork_app, pipe, full_screen)
        assert fork_app.slash_fork("saved") == (
            "forked to checkpoint 'saved' at seq 13"
        )
        forked = _rendered_display(fork_app, fork_output, full_screen)

    resumed = await _run_reopened_session(store, full_screen)
    assert resumed == forked


@pytest.mark.asyncio
@pytest.mark.parametrize("full_screen", [False, True])
async def test_resume_replays_only_the_forked_branch(
    tmp_path: Path, full_screen: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    store = _representative_fork_store(tmp_path)
    fork_output = StringIO()
    fork_app = _test_tui_app(store, fork_output)
    with create_pipe_input() as pipe:
        fork_app._active_session = _display_session(fork_app, pipe, full_screen)
        assert fork_app.slash_fork("saved") == (
            "forked to checkpoint 'saved' at seq 13"
        )

    rendered = Text.from_ansi(
        await _run_reopened_session(store, full_screen)
    ).plain
    assert "kept reply" in rendered
    assert "new branch" in rendered
    assert "abandoned branch" not in rendered
    assert "filtered warning" not in rendered
    assert "[compaction marker: entries 1–4]" in rendered
    assert "checkpoint 'saved'" in rendered
    assert "forked to checkpoint 'saved'" in rendered
    assert "task research · 2 turns" in rendered
    assert "read" in rendered


@pytest.mark.asyncio
async def test_aborted_turn_does_not_reorder_the_next_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    backend = AbortThenSuccessBackend()
    app = TUIApp(
        AgentLoop(backend, ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    app._print_user("first prompt")
    first_turn = asyncio.create_task(app._consume_turn("first prompt"))
    app._active_task = first_turn
    await backend.started.wait()

    app.abort_active()
    await asyncio.gather(first_turn, return_exceptions=True)
    app._print_user("second prompt")
    await app._consume_turn("second prompt")

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    markers = [
        rendered.index("▌ first prompt"),
        rendered.index("partial response"),
        rendered.index("[aborted]"),
        rendered.index("▌ second prompt"),
        rendered.index("second response"),
    ]
    assert markers == sorted(markers)


@pytest.mark.asyncio
async def test_failed_turn_does_not_reorder_the_next_reply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    app = TUIApp(
        AgentLoop(
            ErrorThenSuccessBackend(),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    app._print_user("first prompt")
    await app._consume_turn("first prompt")
    app._print_user("second prompt")
    await app._consume_turn("second prompt")

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    markers = [
        rendered.index("▌ first prompt"),
        rendered.index("partial response"),
        rendered.index("provider failure · backend_error"),
        rendered.index("▌ second prompt"),
        rendered.index("second response"),
    ]
    assert markers == sorted(markers)


def test_empty_final_message_removes_streamed_assistant_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    app._prepare_stream_event(StreamEvent(StreamEventType.MESSAGE_START))
    app._consume_text(
        StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=TextContent("ghost **text**"),
        )
    )
    app._finish_message(
        StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT),
        )
    )

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert "ghost" not in rendered
    assert "text" not in rendered


@pytest.mark.parametrize("boundary", ["thinking", "tool"])
def test_inline_message_commits_canonical_text_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    call = ToolCall("inline-split", "read", {"path": "README.md"})
    middle = (
        ThinkingContent("plan")
        if boundary == "thinking"
        else ToolUseContent(call)
    )
    message = Message(
        MessageRole.ASSISTANT,
        [TextContent("before **"), middle, TextContent("bold** after")],
    )
    output = StringIO()
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )
    app._prepare_stream_event(StreamEvent(StreamEventType.MESSAGE_START))
    for block in message.content:
        event = StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        app._prepare_stream_event(event)
        app._consume_text(event)
    app._finish_message(StreamEvent(StreamEventType.MESSAGE_END, message=message))

    plain = Text.from_ansi(output.getvalue()).plain
    assert plain.count("before bold after") == 1
    assert "**" not in plain
    if boundary == "thinking":
        assert plain.count("plan") == 1


def test_rebuild_user_attachment_hides_file_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    store = ConversationStore(tmp_path / "sessions")
    store.append_message(
        Message(
            MessageRole.USER,
            [
                TextContent("inspect @notes.txt"),
                TextContent(
                    "[file: /tmp/notes.txt · 12 bytes]\nsecret contents",
                    "/tmp/notes.txt",
                    12,
                ),
            ],
        )
    )
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()

    app._rebuild_transcript()

    rendered = Text.from_ansi(app._transcript.render(120)).plain
    assert "▌ inspect @notes.txt" in rendered
    assert "file · /tmp/notes.txt · 12 bytes" in rendered
    assert "secret contents" not in rendered


def test_markdown_stream_keeps_model_blank_lines_without_inserting_more(
    tmp_path: Path,
) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    app._active_session = app._make_session()
    source = ["intro", "", "1. first", "", "", "2. second", "after"]

    app._print_committed(source)
    app._flush_markdown()

    rendered = [Text.from_ansi(line).plain for line in app._transcript.lines(120)]
    assert rendered == ["intro", "", "1. first", "", "2. second after"]


@pytest.mark.parametrize("value", ["*unclosed", "**unclosed", "_unclosed", "__unclosed"])
def test_render_line_keeps_unmatched_emphasis_literal(value: str) -> None:
    rendered = render_line(value)

    assert rendered.plain == value


def test_render_line_keeps_emphasis_markers_literal() -> None:
    rendered = render_line("1. **bold** and *italic* foo_bar_baz")

    assert rendered.plain == "1. bold and italic foo_bar_baz"


def test_render_line_styles_code_span() -> None:
    rendered = render_line("`code_span`")

    assert rendered.plain == "code_span"
    assert rendered.spans


def test_render_line_keeps_unmatched_backtick_literal() -> None:
    rendered = render_line("before `unmatched")

    assert rendered.plain == "before `unmatched"


def test_full_stream_hostile_inline_markers_finish_within_timeout() -> None:
    script = dedent(
        """
        from pathlib import Path
        from tempfile import TemporaryDirectory

        from zeta.core.fake import FakeBackend
        from zeta.core.loop import AgentLoop
        from zeta.core.store import ConversationStore
        from zeta.tui.app import TUIApp
        from zeta.types import StreamEvent, StreamEventType, TextContent

        with TemporaryDirectory() as directory:
            app = TUIApp(
                AgentLoop(FakeBackend([]), ConversationStore(Path(directory))),
                provider="fake",
                model="offline",
            )
            app._active_session = app._make_session()
            app._consume_text(
                StreamEvent(
                    StreamEventType.MESSAGE_UPDATE,
                    content=TextContent(
                        "\\n".join(
                            [
                                "*unclosed",
                                "**unclosed",
                                "_unclosed",
                                "__unclosed",
                                "before `unmatched",
                            ]
                        )
                    ),
                )
            )
            app._flush_pending_stream()
        print("finished")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=5,
        check=True,
    )

    assert result.stdout.strip() == "finished"


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
    assert "ctrl+u undo" not in footer.plain

    undo_footer = format_status(
        "openai",
        "gpt-5.4",
        "streaming",
        token_count=18_600,
        model_window=200_000,
        session_id="abcdef12",
        undo_available=True,
    )
    assert "ctrl+u undo" in undo_footer.plain


def test_footer_shows_transcript_navigation_and_search_state() -> None:
    footer = format_status(
        "fake",
        "offline",
        "idle",
        token_count=18,
        transcript_navigation=True,
        transcript_search="target",
        transcript_match=(2, 5),
        transcript_position="line 14/80",
    )

    assert 'find "target" 2/5' in footer.plain
    assert "line 14/80" in footer.plain
    assert "ctrl+f find" in footer.plain
    assert "enter/n next" in footer.plain
    assert "N prev" in footer.plain
    assert "esc close" in footer.plain

    narrow = format_status(
        "fake",
        "offline",
        "idle",
        token_count=18,
        width=50,
        transcript_navigation=True,
        transcript_search="target",
        transcript_match=(2, 5),
        transcript_position="line 14/80",
    )
    assert 'find "target" 2/5' in narrow.plain
    assert "line 14/80" in narrow.plain


def test_narrow_footer_search_survives_a_tail_anchored_transcript() -> None:
    """position_indicator() is None while the viewport follows the tail."""

    narrow = format_status(
        "fake",
        "offline",
        "idle",
        token_count=18,
        width=50,
        transcript_navigation=True,
        transcript_search="target",
        transcript_match=(2, 5),
        transcript_position=None,
    )
    assert 'find "target" 2/5' in narrow.plain


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
    wheel_router = content.children[1]
    assert wheel_router.__class__.__name__ == "WheelRouter"
    bottom = wheel_router.content
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
    console = _test_console(output)
    console.print(render_line("# heading\n\n`inline`"))
    console.print(render_code("print('hi')", "python"))
    console.print(
        render_event(
            StreamEvent(
                StreamEventType.TOOL_EXECUTION_START,
                tool_call=ToolCall("call-1", "bash", {"cmd": "pwd"}),
            )
        )
    )

    assert not _contains_background_sgr(output.getvalue())


def test_test_console_forces_truecolor_output() -> None:
    output = StringIO()
    console = _test_console(output)

    console.print(Text("probe", style=ACCENT))

    assert console.is_terminal
    assert console.color_system == "truecolor"
    assert not console.no_color
    assert "\x1b[" in output.getvalue()


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
            "sh",
            "-c",
            'exec env ZETA_HOME="$1" TERM="$2" COLORTERM="$3" "$4" --provider fake',
            "zeta-pane",
            str(tmp_path / "zeta-home"),
            "xterm-256color",
            "truecolor",
            str(zeta),
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
        assert not _contains_background_sgr(escaped)
        assert list((tmp_path / "zeta-home" / "sessions").iterdir())
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
    transcript.append(render_line("## result\n\n1. first item\n2. second item"))
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


def test_full_screen_transcript_preserves_markdown_list_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = TUIApp(
        AgentLoop(GateBackend(), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=_test_console(),
    )
    app._active_session = app._make_session()
    output = SimpleNamespace(get_size=lambda: Size(rows=24, columns=40))
    monkeypatch.setattr("zeta.tui.app.get_app", lambda: SimpleNamespace(output=output))

    app._append_transcript(render_line("1. first item"))

    assert app._transcript_lines
    assert Text.from_ansi(app._transcript_lines[0]).plain.strip() == "1. first item"


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
    try:
        await wait_until(lambda: app._spinner_frame >= 2)
    finally:
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
        await wait_until(lambda: app._spinner_frame > frame_before)
        frame_after = app._spinner_frame
        toolbar = "".join(value for _, value in app._status_toolbar())
        return frame_before, frame_after, toolbar

    frame_before, frame_after, during_tool_toolbar = await asyncio.wait_for(
        observe_tool_spinner(), timeout=2.0
    )
    await backend.second_started.wait()
    starting_frame = app._spinner_frame
    first_provider_frame = app._spinner_frame
    await wait_until(lambda: app._spinner_frame - starting_frame >= 2)
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
async def test_empty_completion_removes_preview_region_before_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    app = TUIApp(
        AgentLoop(
            GhostPreviewBackend(),
            ConversationStore(tmp_path / "sessions"),
        ),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._active_session = app._make_session()
    app._print_user("prompt")

    await app._consume_turn("prompt")

    assert [Text.from_ansi(line).plain for line in app._transcript.lines(120)] == [
        "▌ prompt",
        "",
        "no response",
    ]


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
    assert "answer" in Text.from_ansi(rendered).plain


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
                content=[
                    ThinkingContent(
                        "Plan the inspection.\nMore reasoning stays visible."
                    )
                ],
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
    lines = snapshot.splitlines()
    panel_lines = [
        line for line in lines if line.startswith(("  ╭", "  │", "  ╰"))
    ]
    assert "▌ inspect the session" in snapshot
    assert "✱ thought ·" in snapshot
    assert "Plan the inspection.\n  More reasoning stays visible." in snapshot
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
        Text("> quoted markdown\n\n```python\nprint(\"hello\")\n```")
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
@pytest.mark.parametrize(
    ("command", "expected"),
    [
        ("/checkpoint", "checkpoint_count"),
        ("/compact", "compact: nothing to compact"),
        ("/fork", "no checkpoints on the active branch"),
        ("/model offline", "model: offline"),
        ("/plan on", "plan mode: on"),
    ],
)
async def test_control_command_does_not_reject_rapid_follow_up(
    tmp_path: Path, command: str, expected: str
) -> None:
    backend = FakeBackend([ScriptedTurn(content=[TextContent("done")])])
    store = ConversationStore(tmp_path / "sessions")
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )

    app._submit_input(command)
    app._submit_input("probe")
    await wait_until(lambda: len(backend.calls) == 1)
    await app._active_task

    if expected == "checkpoint_count":
        assert store.checkpoint_count() == 1
    else:
        assert expected in output.getvalue()
    await app.loop.close()


@pytest.mark.asyncio
async def test_submission_queue_preserves_rapid_enter_order(tmp_path: Path) -> None:
    backend = GateBackend()
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submit_input("first")
    app._submit_input("second")
    await backend.started.wait()

    backend.release.set()
    await app._active_task
    await wait_until(lambda: len(backend.calls) == 2)
    await app._active_task

    user_texts = [
        block.text
        for message in store.messages()
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert user_texts == ["first", "second"]
    await app.loop.close()


@pytest.mark.asyncio
async def test_submission_undo_cancels_only_the_exact_rapid_enter(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submit_input("same")
    app.undo_sent_turn()
    app._submit_input("same")
    await asyncio.sleep(0)
    await asyncio.gather(app._active_task, return_exceptions=True)
    await asyncio.sleep(0)
    if app._active_task is not None:
        await app._active_task

    user_texts = [
        block.text
        for message in store.messages()
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert user_texts == ["same"]
    await app.loop.close()


@pytest.mark.asyncio
async def test_undo_restores_double_slash_source_text(tmp_path: Path) -> None:
    backend = GateBackend()
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._active_session = session

    raw = "  //status  \n"
    await app._handle_prompt_value(raw)
    await backend.started.wait()
    app.undo_sent_turn()
    await asyncio.gather(app._active_task, return_exceptions=True)

    assert session.app.current_buffer.text == raw
    await app._handle_prompt_value(raw)
    await app._active_task

    assert len(backend.calls) == 2
    resent = [
        block.text
        for message in backend.calls[-1]
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert resent[-1] == "/status"
    await app.loop.close()


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
async def test_full_screen_vi_escape_enter_chord_keeps_insert_mode() -> None:
    submitted: list[str] = []
    with create_pipe_input() as pipe:
        session: FullScreenPromptSession | None = None

        def submit(value: str) -> None:
            submitted.append(value)
            assert session is not None
            session.app.exit()

        session = FullScreenPromptSession(
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
        pipe.send_text("line one\x1b\rline two\r")
        await task

    assert submitted == ["line one\nline two"]


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
        await wait_until(lambda: session.app.current_buffer.text == "first line")
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
async def test_vi_composer_history_down_restores_multiline_entry_cursor(
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
        fast_vi_timeouts(session)
        first = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("first line\nsecond line\r")
        await first
        second = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("other first\nother second\r")
        await second
        third = asyncio.create_task(session.prompt_async(" ❯ "))
        await asyncio.sleep(0)
        pipe.send_text("\x1b[A")
        await wait_until(
            lambda: session.app.current_buffer.text == "other first\nother second"
        )
        pipe.send_text("\x1b[A")
        await wait_until(
            lambda: session.app.current_buffer.text == "first line\nsecond line"
        )
        pipe.send_text("\x1b[B")
        await wait_until(
            lambda: session.app.current_buffer.text == "other first\nother second"
            and session.app.current_buffer.cursor_position == len("other first\nother second")
        )
        session.app.exit()
        await third

    assert submitted == ["first line\nsecond line", "other first\nother second"]


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


@pytest.mark.asyncio
async def test_approval_shortcuts_only_fire_on_an_empty_composer() -> None:
    answered: list[str] = []

    def bindings() -> object:
        return build_key_bindings(
            on_interrupt=lambda: None,
            on_exit=lambda: None,
            on_approve=lambda: answered.append("approve"),
            on_deny=lambda: answered.append("deny"),
            approval_active=lambda: True,
        )

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            key_bindings=bindings(),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0)
        pipe.send_text("deny 3")
        await asyncio.sleep(0.05)

        assert session.default_buffer.text == "deny 3"
        assert answered == []

        session.default_buffer.reset()
        pipe.send_text("y")
        await asyncio.sleep(0.05)

        assert answered == ["approve"]
        assert session.default_buffer.text == ""
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("value", ["", "  \n  "])
def test_parse_input_rejects_blank_turns(value: str) -> None:
    assert parse_input(value) is None


@pytest.mark.asyncio
async def test_composer_history_is_bounded_and_keeps_multiline_entries(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history"
    history = history_for(path)
    for index in range(1001):
        history.append_string(f"prompt {index}\nline two")

    assert len(history.get_strings()) == 1000
    assert history.get_strings()[0] == "prompt 1\nline two"
    assert history.get_strings()[-1] == "prompt 1000\nline two"
    reopened = history_for(path)
    loaded: list[str] = []
    async for entry in reopened.load():
        loaded.append(entry)
    assert loaded[:2] == ["prompt 1000\nline two", "prompt 999\nline two"]


@pytest.mark.asyncio
async def test_composer_history_merges_updates_from_active_instances(
    tmp_path: Path,
) -> None:
    path = tmp_path / "history"
    first = history_for(path)
    second = history_for(path)
    async for _entry in first.load():
        pass
    async for _entry in second.load():
        pass

    first.append_string("first session prompt")
    second.append_string("second session prompt")

    reopened = history_for(path)
    async for _entry in reopened.load():
        pass

    assert reopened.get_strings() == [
        "first session prompt",
        "second session prompt",
    ]


@pytest.mark.asyncio
async def test_draft_persistence_round_trip_and_clear(tmp_path: Path) -> None:
    persistence = DraftPersistence(tmp_path / "draft", delay=0.01)
    persistence.schedule("line one\nline two")
    await asyncio.sleep(0.02)
    assert persistence.load() == "line one\nline two"

    persistence.clear()
    assert not (tmp_path / "draft").exists()


def test_draft_persistence_restores_attachment_metadata(tmp_path: Path) -> None:
    staged = tmp_path / "clipboard-image.png"
    persistence = DraftPersistence(tmp_path / "draft")
    persistence.schedule(
        "inspect [Image #1]",
        attachment_tokens={"[Image #1]": staged},
        next_image_token=2,
    )
    persistence.flush()

    state = DraftPersistence(tmp_path / "draft").load_state()

    assert state.text == "inspect [Image #1]"
    assert state.attachment_tokens == (("[Image #1]", staged),)
    assert state.next_image_token == 2


@pytest.mark.asyncio
async def test_history_and_draft_store_attachment_refs_without_payloads(
    tmp_path: Path,
) -> None:
    image = tmp_path / "image.png"
    text_file = tmp_path / "notes.txt"
    image.write_bytes(PNG)
    text_file.write_text("private attachment text", encoding="utf-8")
    history_path = tmp_path / "history"
    draft_path = tmp_path / "draft"
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
        history_path=history_path,
        draft_path=draft_path,
    )
    app._draft.schedule("inspect @./image.png @./notes.txt")
    app._draft.flush()

    await app._handle_prompt_value("inspect @./image.png @./notes.txt")
    assert app._active_task is not None
    await app._active_task

    payload = base64.b64encode(PNG)
    for path in (history_path, draft_path):
        contents = path.read_bytes() if path.exists() else b""
        assert payload not in contents
        assert b"private attachment text" not in contents
    assert isinstance(store.messages()[0].content[1], ImageContent)
    assert isinstance(store.messages()[0].content[2], TextContent)
    await app.loop.close()


@pytest.mark.asyncio
async def test_mcp_prompt_result_does_not_activate_attachment_syntax(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    secret = tmp_path / "secret.txt"
    secret.write_text("must not be sent", encoding="utf-8")
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([ScriptedTurn([TextContent("done")])]), store),
        provider="fake",
        model="offline",
    )
    app._slash_commands.set_mcp_prompts(
        [("fake:review", "fake", MCPPrompt("review"))]
    )

    async def resolve(_name: str, _arguments: dict[str, str]) -> str:
        return f"@./{secret.name} $ARGUMENTS !`echo unsafe`"

    async def ensure_mcp_servers() -> None:
        return None

    monkeypatch.setattr(app.loop, "ensure_mcp_servers", ensure_mcp_servers)
    monkeypatch.setattr(app.loop, "slash_mcp_prompt", resolve)
    await app._handle_prompt_value("/fake:review")
    assert app._active_task is not None
    await app._active_task

    message = store.messages()[0]
    assert message.content == [
        TextContent(f"@./{secret.name} $ARGUMENTS !`echo unsafe`")
    ]
    await app.loop.close()


@pytest.mark.asyncio
async def test_mcp_prompt_error_restores_draft_and_attachments(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged = tmp_path / "clipboard-prompt.png"
    staged.write_bytes(PNG)
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
    )
    session = app._make_session()
    app._active_session = session
    app._slash_commands.set_mcp_prompts(
        [
            (
                "fake:review",
                "fake",
                MCPPrompt(
                    "review",
                    arguments=(MCPPromptArgument("topic", required=True),),
                ),
            )
        ]
    )
    rendered: list[Text] = []
    app._print_unit = rendered.append

    async def fail(_name: str, _arguments: dict[str, str]) -> str:
        raise RuntimeError("prompt unavailable")

    async def ensure_mcp_servers() -> None:
        return None

    monkeypatch.setattr(app.loop, "ensure_mcp_servers", ensure_mcp_servers)
    monkeypatch.setattr(app.loop, "slash_mcp_prompt", fail)
    raw = "/fake:review topic"
    session.default_buffer.insert_text(raw)
    app._pending_attachments.append(staged)
    app._pending_attachment_tokens["[Image #1]"] = staged
    app._next_image_token = 2
    app._submit_input(raw)
    session.default_buffer.reset()

    await app._handle_prompt_value(raw)

    assert session.default_buffer.text == raw
    assert app._pending_attachments == [staged]
    assert app._pending_attachment_tokens == {"[Image #1]": staged}
    assert rendered and rendered[-1].style == ERROR
    assert "prompt unavailable" in rendered[-1].plain
    await app.loop.close()


@pytest.mark.asyncio
async def test_ctrl_r_search_uses_cross_session_history(tmp_path: Path) -> None:
    history = history_for(tmp_path / "history")
    history.append_string("older session prompt")
    history.append_string("newer session prompt")

    with create_pipe_input() as pipe:
        session = PromptSession(
            input=pipe,
            output=DummyOutput(),
            history=history_for(tmp_path / "history"),
            key_bindings=build_key_bindings(
                on_interrupt=lambda: None,
                on_exit=lambda: None,
                on_retry=lambda: None,
                retry_available=lambda: True,
            ),
            multiline=True,
        )
        task = asyncio.create_task(session.prompt_async(" > "))
        await asyncio.sleep(0.05)
        pipe.send_text("\x12older")
        await wait_until(lambda: session.search_buffer.text == "older")
        pipe.send_text("\r")
        await wait_until(
            lambda: session.default_buffer.text == "older session prompt"
        )
        session.app.exit()
        await task


@pytest.mark.asyncio
async def test_app_draft_round_trip_and_send_clear(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    first = TUIApp(
        AgentLoop(FakeBackend([]), store),
        provider="fake",
        model="offline",
        history_path=tmp_path / "history",
    )
    first_session = first._make_session()
    first_session.default_buffer.insert_text("unsent draft")
    await asyncio.sleep(0.25)

    reopened_store = ConversationStore(
        store.root_dir, session_id=store.session_id, cwd=store.cwd
    )
    second = TUIApp(
        AgentLoop(FakeBackend([]), reopened_store),
        provider="fake",
        model="offline",
        history_path=tmp_path / "history",
    )
    second_session = second._make_session()
    assert second_session.default_buffer.text == "unsent draft"

    second._record_prompt("sent prompt")
    assert not (store.session_dir / "draft").exists()


@pytest.mark.asyncio
async def test_undo_restores_submission_before_turn_creation(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._active_session = session
    session.default_buffer.insert_text("sent too early")
    app._submit_input(session.default_buffer.text)
    session.default_buffer.reset()

    app.undo_sent_turn()
    await asyncio.sleep(0)

    assert session.default_buffer.text == "sent too early"
    assert app._active_task is None
    await app.loop.close()


@pytest.mark.asyncio
async def test_pending_undo_keeps_a_new_buffer_draft(tmp_path: Path) -> None:
    app = TUIApp(
        AgentLoop(FakeBackend([]), ConversationStore(tmp_path / "sessions")),
        provider="fake",
        model="offline",
    )
    session = app._make_session()
    app._active_session = session
    session.default_buffer.insert_text("sent")
    app._submit_input(session.default_buffer.text)
    session.default_buffer.reset()
    session.default_buffer.insert_text("new draft")

    app.undo_sent_turn()
    await asyncio.sleep(0)
    if app._active_task is not None:
        await asyncio.gather(app._active_task, return_exceptions=True)

    assert session.default_buffer.text == "new draft"
    await app.loop.close()


@pytest.mark.asyncio
async def test_undo_restores_text_and_aborts_streaming_turn(tmp_path: Path) -> None:
    backend = AbortThenSuccessBackend()
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._active_session = session
    task = asyncio.create_task(app._consume_turn("sent message"))
    app._active_task = task
    await backend.started.wait()

    app._undo_candidate = UndoCandidate("sent message")
    app.undo_sent_turn()
    await asyncio.gather(task, return_exceptions=True)

    assert session.app.current_buffer.text == "sent message"
    assert app._loop_state == "interrupted"
    app.undo_sent_turn()
    assert session.app.current_buffer.text == "sent message"
    assert "undo unavailable" in "\n".join(app._transcript.lines(120))


@pytest.mark.asyncio
async def test_undo_keeps_a_draft_typed_during_streaming(tmp_path: Path) -> None:
    backend = AbortThenSuccessBackend()
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    session = app._make_session()
    app._active_session = session
    task = asyncio.create_task(app._consume_turn("sent message"))
    app._active_task = task
    await backend.started.wait()
    session.default_buffer.insert_text("new draft")
    app._undo_candidate = UndoCandidate("sent message")

    app.undo_sent_turn()
    await asyncio.gather(task, return_exceptions=True)
    await asyncio.sleep(0.25)

    assert session.app.current_buffer.text == "new draft"
    assert app._draft.load() == "new draft"
    assert "sent text: sent message" in "\n".join(app._transcript.lines(120))
    await app.loop.close()


@pytest.mark.asyncio
async def test_undo_restores_staged_image_for_resubmission(tmp_path: Path) -> None:
    backend = AbortThenSuccessBackend()
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    staged = store.session_dir / "clipboard-image.png"
    staged.write_bytes(PNG)
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._pending_attachments.append(staged)
    app._pending_attachment_tokens["[Image #1]"] = staged
    await app._handle_prompt_value("inspect [Image #1]")
    assert app._active_task is not None
    await backend.started.wait()

    app.undo_sent_turn()
    await asyncio.gather(app._active_task, return_exceptions=True)

    assert app._pending_attachment_tokens == {"[Image #1]": staged}
    assert app._pending_attachments == [staged]
    await app._handle_prompt_value("inspect [Image #1]")
    assert app._active_task is not None
    await app._active_task

    user_messages = [
        message
        for message in store.messages()
        if message.role is MessageRole.USER
    ]
    assert isinstance(user_messages[-1].content[1], ImageContent)
    assert user_messages[-1].content[1].data == base64.b64encode(PNG).decode()
    await app.loop.close()


@pytest.mark.asyncio
async def test_undo_restores_next_image_token_after_deleted_token(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first.png"
    second = tmp_path / "second.png"
    pasted = tmp_path / "pasted.png"
    first.write_bytes(PNG)
    second.write_bytes(PNG)
    pasted.write_bytes(PNG)
    backend = AbortThenSuccessBackend()
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    app = TUIApp(
        AgentLoop(backend, store),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )
    app._pending_attachments[:] = [first, second]
    app._pending_attachment_tokens.update(
        {"[Image #1]": first, "[Image #2]": second}
    )
    app._next_image_token = 3

    await app._handle_prompt_value("inspect [Image #2]")
    await backend.started.wait()
    app.undo_sent_turn()
    await asyncio.gather(app._active_task, return_exceptions=True)

    monkeypatch.setattr("zeta.tui.composer.paste_image", lambda _: pasted)
    assert app.slash_paste("") == "[Image #3]"
    assert app._pending_attachment_tokens == {
        "[Image #2]": second,
        "[Image #3]": pasted,
    }
    await app.loop.close()
