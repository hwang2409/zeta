"""Tests for ZETA-82 mid-turn steering + composer escape hatches."""

from __future__ import annotations

import asyncio
import os
import stat
from collections.abc import AsyncIterator, Sequence
from io import StringIO
from pathlib import Path
from typing import Any

import pytest
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.console import Console

from zeta.core.approval import ApprovalDecision, ApprovalPolicy
from zeta.core.fake import FakeBackend
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.tui.app import FullScreenPromptSession, TUIApp
from zeta.tui.composer import build_key_bindings, parse_submission
from zeta.tui.key_bindings import DEFAULTS, resolve_keybindings
from zeta.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)


# --- fakes ------------------------------------------------------------------


class SteerToolBackend(CompletionBackend):
    """Turn 1: assistant + one tool_use (blocks in-flight). Turn 2: tail text.

    Test hooks:
    * ``turn1_streaming`` — set once the first turn's provider stream opens.
    * ``release_turn1`` — awaited before yielding turn 1's MESSAGE_END. The
      caller sets this after enqueuing steering so the drain runs at the
      boundary between the tool_result and turn 2's provider call.
    """

    def __init__(self, tool_call: ToolCall) -> None:
        self.tool_call = tool_call
        self.calls: list[list[Message]] = []
        self.turn1_streaming = asyncio.Event()
        self.release_turn1 = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        index = len(self.calls)
        self.calls.append(list(messages))
        if index == 0:
            self.turn1_streaming.set()
            await self.release_turn1.wait()
            blocks = [TextContent("working"), ToolUseContent(self.tool_call)]
        else:
            blocks = [TextContent(f"reply {index}")]
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[0])
        if len(blocks) > 1:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=blocks[1])
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
            data={"usage": {"input_tokens": 1, "output_tokens": 1}},
        )


async def _drive(coro: asyncio.Task[Any]) -> None:
    try:
        await coro
    except asyncio.CancelledError:
        pass


# --- parse_submission -------------------------------------------------------


def test_parse_submission_defaults_to_steering() -> None:
    assert parse_submission("hello") == ("hello", True, None)


def test_parse_submission_backslash_marks_follow_up() -> None:
    assert parse_submission("\\ hey") == ("hey", False, None)
    # Only the marker + one separating space is stripped so the user's
    # composer whitespace survives into undo.
    assert parse_submission("\\   hey  ") == ("  hey  ", False, None)
    assert parse_submission("\\hey") == ("hey", False, None)


def test_parse_submission_double_backslash_escapes_to_literal_prefix() -> None:
    text, steer, passthrough = parse_submission("\\\\path")
    assert text == "\\path"
    assert steer is True
    assert passthrough is None


def test_parse_submission_bang_routes_to_passthrough() -> None:
    assert parse_submission("!ls -la") == (None, True, "ls -la")


def test_parse_submission_double_bang_is_repeat_last() -> None:
    assert parse_submission("!!") == (None, True, "")
    assert parse_submission("!!  rerun") == (None, True, "rerun")


def test_parse_submission_blank_returns_none() -> None:
    assert parse_submission("") == (None, True, None)
    assert parse_submission("   \n  ") == (None, True, None)


def test_open_editor_default_is_ctrl_x_ctrl_e() -> None:
    assert DEFAULTS["open-editor"] == ("c-x", "c-e")


def test_open_editor_binding_can_be_remapped() -> None:
    resolved = resolve_keybindings({"open-editor": "c-e"})
    assert resolved["open-editor"] == ("c-e",)


# --- loop-level steering ----------------------------------------------------


def test_agent_loop_rejects_non_user_steering(tmp_path: Path) -> None:
    loop = AgentLoop(FakeBackend([]), ConversationStore(tmp_path))
    with pytest.raises(ValueError, match="user role"):
        loop.steer(Message(MessageRole.ASSISTANT, [TextContent("bad")]))


def test_agent_loop_abort_clears_steering(tmp_path: Path) -> None:
    loop = AgentLoop(FakeBackend([]), ConversationStore(tmp_path))
    loop.steer(Message(MessageRole.USER, [TextContent("later")]))
    assert loop.has_pending_steering is True
    loop.abort()
    assert loop.has_pending_steering is False


@pytest.mark.asyncio
async def test_steer_delivers_between_tool_pair_and_next_provider_call(
    tmp_path: Path,
) -> None:
    """Contract: steering never splits a tool_call/tool_result pair.

    Enqueue a steer while turn 1 streams; verify the store's ordering is
    ``user, assistant(tool_use), tool_result, user_steer, assistant`` — the
    steer message lands strictly between the completed tool pair and the
    next assistant response.
    """

    tool_call = ToolCall("call-1", "noop", {})

    async def noop_tool(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: object,
    ) -> str:
        del arguments, abort_signal, publisher
        return "ok"

    backend = SteerToolBackend(tool_call)
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(backend, store, tools={"noop": noop_tool}, max_turns=3)

    events: list[StreamEvent] = []

    async def run() -> None:
        async for event in loop.run_turn("prompt"):
            events.append(event)

    task = asyncio.create_task(run())
    await backend.turn1_streaming.wait()
    loop.steer(Message(MessageRole.USER, [TextContent("steer-1")]))
    backend.release_turn1.set()
    await task

    roles = [message.role for message in store.messages()]
    texts = [
        block.text
        for message in store.messages()
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]

    # The steering user message lands strictly after the tool_use assistant
    # AND after its tool_result — never between them.
    assert roles == [
        MessageRole.USER,         # initial prompt
        MessageRole.ASSISTANT,    # tool_use turn
        MessageRole.TOOL_RESULT,  # tool_result
        MessageRole.USER,         # steering-injected
        MessageRole.ASSISTANT,    # response to steering
    ]
    assert texts == ["prompt", "steer-1"]
    # The second provider call MUST have seen the steer message in context.
    assert backend.calls[1][-1].role is MessageRole.USER
    assert backend.calls[1][-1].content[0].text == "steer-1"
    # The loop yielded a USER_STEERING event so UI layers see the injection.
    steering_events = [
        event for event in events if event.type is StreamEventType.USER_STEERING
    ]
    assert len(steering_events) == 1
    assert steering_events[0].message is not None
    assert steering_events[0].message.content[0].text == "steer-1"


@pytest.mark.asyncio
async def test_multiple_steers_deliver_in_order_at_one_boundary(
    tmp_path: Path,
) -> None:
    tool_call = ToolCall("call-2", "noop", {})

    async def noop_tool(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: object,
    ) -> str:
        del arguments, abort_signal, publisher
        return "ok"

    backend = SteerToolBackend(tool_call)
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(backend, store, tools={"noop": noop_tool}, max_turns=3)

    async def run() -> None:
        async for _event in loop.run_turn("prompt"):
            pass

    task = asyncio.create_task(run())
    await backend.turn1_streaming.wait()
    loop.steer(Message(MessageRole.USER, [TextContent("a")]))
    loop.steer(Message(MessageRole.USER, [TextContent("b")]))
    backend.release_turn1.set()
    await task

    user_texts = [
        block.text
        for message in store.messages()
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert user_texts == ["prompt", "a", "b"]


# --- pipeline + TUI integration --------------------------------------------


async def _wait(check) -> None:
    async with asyncio.timeout(30):
        while not check():
            await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_pipeline_routes_default_submission_as_steer(tmp_path: Path) -> None:
    tool_call = ToolCall("call-3", "noop", {})

    async def noop_tool(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: object,
    ) -> str:
        del arguments, abort_signal, publisher
        return "ok"

    backend = SteerToolBackend(tool_call)
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store, tools={"noop": noop_tool}, max_turns=3),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submit_input("prompt")
    await backend.turn1_streaming.wait()
    app._submit_input("second thoughts")
    await _wait(lambda: app.loop.has_pending_steering)
    backend.release_turn1.set()
    await _wait(lambda: not app._submissions.active)
    if app._active_task is not None:
        await _drive(app._active_task)
    await app.loop.close()

    user_texts = [
        block.text
        for message in store.messages()
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert user_texts == ["prompt", "second thoughts"]
    # Only ONE provider run_turn call was made — steering never spawned a new
    # top-level turn, it injected into the running one.
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_backslash_prefix_defers_to_after_turn_end(tmp_path: Path) -> None:
    tool_call = ToolCall("call-4", "noop", {})

    async def noop_tool(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: object,
    ) -> str:
        del arguments, abort_signal, publisher
        return "ok"

    backend = SteerToolBackend(tool_call)
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store, tools={"noop": noop_tool}, max_turns=3),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submit_input("prompt")
    await backend.turn1_streaming.wait()
    app._submit_input("\\ follow-up")
    # Follow-up submissions never inject into the running turn.
    await asyncio.sleep(0.05)
    assert app.loop.has_pending_steering is False
    backend.release_turn1.set()
    await _wait(lambda: len(backend.calls) >= 3)
    await _wait(lambda: not app.active)
    if app._active_task is not None:
        await _drive(app._active_task)
    await app.loop.close()

    user_texts = [
        block.text
        for message in store.messages()
        if message.role is MessageRole.USER
        for block in message.content
        if isinstance(block, TextContent)
    ]
    assert user_texts == ["prompt", "follow-up"]


@pytest.mark.asyncio
async def test_abort_drops_pending_steering(tmp_path: Path) -> None:
    tool_call = ToolCall("call-5", "noop", {})

    async def noop_tool(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: object,
    ) -> str:
        del arguments, abort_signal, publisher
        return "ok"

    backend = SteerToolBackend(tool_call)
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store, tools={"noop": noop_tool}, max_turns=3),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submit_input("prompt")
    await backend.turn1_streaming.wait()
    app._submit_input("ghost")
    await _wait(lambda: app.loop.has_pending_steering)
    app.abort_active()
    await _drive(app._active_task or asyncio.sleep(0))
    assert app.loop.has_pending_steering is False
    await app.loop.close()


# --- cache-prefix stability -------------------------------------------------


@pytest.mark.asyncio
async def test_steering_preserves_cache_prefix_of_running_turn(
    tmp_path: Path,
) -> None:
    """ZETA-39 governance: steer messages APPEND to the tail; the previous
    turn's request bytes remain a prefix of the next turn's request, so
    ``cache_read_input_tokens`` reflects that full prefix reuse."""

    tool_call = ToolCall("call-6", "noop", {})

    async def noop_tool(
        arguments: dict[str, object],
        abort_signal: object,
        publisher: object,
    ) -> str:
        del arguments, abort_signal, publisher
        return "ok"

    backend = SteerToolBackend(tool_call)
    store = ConversationStore(tmp_path / "sessions")
    loop = AgentLoop(backend, store, tools={"noop": noop_tool}, max_turns=3)

    async def run() -> None:
        async for _event in loop.run_turn("prompt"):
            pass

    task = asyncio.create_task(run())
    await backend.turn1_streaming.wait()
    loop.steer(Message(MessageRole.USER, [TextContent("steer")]))
    backend.release_turn1.set()
    await task

    # Every message in the first provider call is also in the second, in
    # the same order at the head: the cached prefix (system prompt + tool
    # schemas + prior turns) is byte-stable, and steering messages append to
    # the tail. This is the ZETA-39 governance for prompt caching.
    first_messages = [message.to_dict() for message in backend.calls[0]]
    second_messages = [message.to_dict() for message in backend.calls[1]]
    assert second_messages[: len(first_messages)] == first_messages
    # The tail added between calls is the assistant tool-use turn, the
    # tool_result, and the steer user message — nothing else.
    tail = second_messages[len(first_messages):]
    tail_roles = [entry["role"] for entry in tail]
    assert tail_roles == ["assistant", "tool_result", "user"]
    steer_user = tail[-1]
    assert steer_user["content"][0]["text"] == "steer"


# --- external editor --------------------------------------------------------


@pytest.mark.asyncio
async def test_open_editor_binding_wires_to_buffer_open_in_editor() -> None:
    captured: list[str] = []
    fired = asyncio.Event()

    def fake_open(self: Buffer) -> Any:
        captured.append(self.text)
        fired.set()
        loop = asyncio.get_running_loop()
        future: asyncio.Future[None] = loop.create_future()
        future.set_result(None)
        return future

    original = Buffer.open_in_editor
    Buffer.open_in_editor = fake_open  # type: ignore[method-assign]
    try:
        with create_pipe_input() as pipe:
            session = FullScreenPromptSession(
                input=pipe,
                output=DummyOutput(),
                key_bindings=build_key_bindings(
                    on_interrupt=lambda: None,
                    on_exit=lambda: None,
                    on_submit=lambda _value: None,
                ),
                multiline=True,
            )
            task = asyncio.create_task(session.prompt_async(" > "))
            await asyncio.sleep(0)
            pipe.send_text("draft\x18\x05")  # ...draft, Ctrl+X, Ctrl+E
            async with asyncio.timeout(5):
                await fired.wait()
            session.app.exit()
            await task
    finally:
        Buffer.open_in_editor = original  # type: ignore[method-assign]

    assert captured == ["draft"]


def test_open_in_editor_round_trip_replaces_buffer_text(tmp_path: Path) -> None:
    """End-to-end: a scripted $EDITOR overwrites the temp file; the buffer
    picks up the new content on return."""

    editor_script = tmp_path / "editor.sh"
    editor_script.write_text(
        "#!/bin/sh\nprintf 'edited via script' > \"$1\"\n"
    )
    editor_script.chmod(editor_script.stat().st_mode | stat.S_IXUSR)

    buffer = Buffer()
    buffer.text = "before"
    old_editor = os.environ.get("EDITOR")
    old_visual = os.environ.get("VISUAL")
    os.environ["EDITOR"] = str(editor_script)
    os.environ.pop("VISUAL", None)
    try:
        filename, cleanup = buffer._editor_simple_tempfile()
        try:
            ok = buffer._open_file_in_editor(filename)
            assert ok is True
            replaced = Path(filename).read_text()
            assert replaced == "edited via script"
        finally:
            cleanup()
    finally:
        if old_editor is None:
            os.environ.pop("EDITOR", None)
        else:
            os.environ["EDITOR"] = old_editor
        if old_visual is not None:
            os.environ["VISUAL"] = old_visual


# --- `!cmd` passthrough -----------------------------------------------------


@pytest.mark.asyncio
async def test_passthrough_runs_shell_without_provider_call(tmp_path: Path) -> None:
    backend = FakeBackend([])
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store, max_turns=1),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submit_input("!printf hi")
    await _wait(
        lambda: (
            app._active_task is not None
            and app._active_task.done()
        )
        or bool(app._macro_receipts)
    )
    if app._active_task is not None:
        await _drive(app._active_task)

    assert backend.calls == []
    receipts = list(app._macro_receipts)
    assert receipts, "passthrough must record a receipt"
    assert receipts[0].startswith("ran !printf hi")
    assert receipts[0].endswith("exit 0")
    await app.loop.close()


@pytest.mark.asyncio
async def test_passthrough_repeat_uses_last_command(tmp_path: Path) -> None:
    backend = FakeBackend([])
    store = ConversationStore(tmp_path / "sessions")
    app = TUIApp(
        AgentLoop(backend, store, max_turns=1),
        provider="fake",
        model="offline",
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submit_input("!true")
    await _wait(lambda: len(app._macro_receipts) == 1)
    if app._active_task is not None:
        await _drive(app._active_task)
    app._macro_receipts.clear()

    app._submit_input("!!")
    await _wait(lambda: len(app._macro_receipts) == 1)
    if app._active_task is not None:
        await _drive(app._active_task)

    assert list(app._macro_receipts) == ["ran !true, exit 0"]
    await app.loop.close()


@pytest.mark.asyncio
async def test_passthrough_repeat_without_history_notes_error(tmp_path: Path) -> None:
    backend = FakeBackend([])
    store = ConversationStore(tmp_path / "sessions")
    output = StringIO()
    app = TUIApp(
        AgentLoop(backend, store, max_turns=1),
        provider="fake",
        model="offline",
        console=Console(file=output, force_terminal=False),
    )

    app._submit_input("!!")
    await _wait(
        lambda: "no previous !" in output.getvalue()
        or (app._active_task is not None and app._active_task.done())
    )
    if app._active_task is not None:
        await _drive(app._active_task)
    assert "no previous !" in output.getvalue()
    assert backend.calls == []
    await app.loop.close()


@pytest.mark.asyncio
async def test_passthrough_respects_approval_policy(tmp_path: Path) -> None:
    backend = FakeBackend([])
    store = ConversationStore(tmp_path / "sessions")
    policy = ApprovalPolicy(default=ApprovalDecision.DENY, store=store)
    app = TUIApp(
        AgentLoop(backend, store, approval_policy=policy, max_turns=1),
        provider="fake",
        model="offline",
        approval_policy=policy,
        console=Console(file=StringIO(), force_terminal=False),
    )

    app._submit_input("!echo blocked")
    await _wait(lambda: bool(app._macro_receipts))
    if app._active_task is not None:
        await _drive(app._active_task)
    receipts = list(app._macro_receipts)
    assert receipts, "denied passthrough still emits a receipt"
    assert "denied" in receipts[0]
    assert backend.calls == []
    await app.loop.close()
