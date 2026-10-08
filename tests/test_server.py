"""Real-socket tests for the native frontend server."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import stat
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from zeta.config.settings import SettingsError
from zeta.core.approval import ApprovalPolicy, ApprovalRequest
from zeta.core.session import SessionManager, SessionMetadata
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    FAILED_TURN_ERROR,
    FAILED_TURN_MARKER,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolUseContent,
    with_message_origin,
)
from zeta.server import ZetaServer
from zeta.server.approval_lifecycle import ApprovalLifecycle
from zeta.server.protocol import MAX_FRAME_BYTES, MAX_REQUEST_ID_BYTES, FrameCodec
from zeta.server.server import _approval_display_fields, _Client
from zeta.server.slash_commands import ServerSlashSession

TIMEOUT = 3


def test_removed_fake_provider_has_clear_serve_error(tmp_path: Path) -> None:
    with pytest.raises(
        SettingsError,
        match="the fake provider was removed; choose claude, codex or ollama",
    ):
        ZetaServer(home=tmp_path, cwd=tmp_path, provider="fake", port=0)


@pytest.mark.asyncio
async def test_removed_fake_provider_has_clear_new_session_error(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, cwd=tmp_path, provider="codex", port=0)
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        response = (
            await _request(
                reader,
                writer,
                2,
                "new_session",
                {"provider": "fake"},
            )
        )[-1]
        assert response["error"] == {
            "code": -32602,
            "message": "the fake provider was removed; choose claude, codex or ollama",
        }
    finally:
        await _close(server, writer)


@pytest.fixture(autouse=True)
def _controlled_terminal_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)


def test_delegated_approval_payload_includes_execution_context() -> None:
    request = ApprovalRequest(
        "child-write",
        ToolCall("child-write", "write", {"path": "src/file.py"}),
        effective_cwd="/worktree",
        resolved_path="/worktree/src/file.py",
    )

    assert _approval_display_fields(request) == {
        "approval_display": {
            "effective_cwd": "/worktree",
            "resolved_path": "/worktree/src/file.py",
        }
    }


def _socket_path(tmp_path: Path) -> Path:
    # macOS limits Unix socket paths to 104 bytes; pytest's tmp_path is longer.
    return Path("/tmp") / f"zeta-{tmp_path.name}.sock"


async def _connect(server: ZetaServer):
    await asyncio.wait_for(server.start(), TIMEOUT)
    if server.port is None:
        return await asyncio.wait_for(
            asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT
        )
    return await asyncio.wait_for(asyncio.open_connection("127.0.0.1", server.port), TIMEOUT)


async def _read(reader: asyncio.StreamReader) -> dict[str, Any]:
    line = await asyncio.wait_for(reader.readline(), TIMEOUT)
    assert line, "server closed the socket"
    return json.loads(line)


async def _read_raw(reader: asyncio.StreamReader) -> bytes:
    line = await asyncio.wait_for(reader.readline(), TIMEOUT)
    assert line, "server closed the socket"
    return line


async def _request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request_id: str | int,
    method: str,
    params: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    writer.write(
        (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params or {},
                }
            )
            + "\n"
        ).encode()
    )
    await asyncio.wait_for(writer.drain(), TIMEOUT)
    frames: list[dict[str, Any]] = []
    while True:
        frame = await _read(reader)
        frames.append(frame)
        if frame.get("id") == request_id:
            return frames


async def _frames_until_event(
    reader: asyncio.StreamReader, name: str
) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while True:
        frame = await _read(reader)
        frames.append(frame)
        if frame.get("params", {}).get("event") == name:
            return frames


async def _event(reader: asyncio.StreamReader, name: str) -> dict[str, Any]:
    frames = await _frames_until_event(reader, name)
    return frames[-1]["params"]


def _named_events(
    frames: list[dict[str, Any]], name: str
) -> list[dict[str, Any]]:
    return [
        frame["params"]
        for frame in frames
        if frame.get("params", {}).get("event") == name
    ]


async def _frames_until_eof(reader: asyncio.StreamReader) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    while line := await asyncio.wait_for(reader.readline(), TIMEOUT):
        frames.append(json.loads(line))
    return frames


async def _close(server: ZetaServer, writer: asyncio.StreamWriter) -> None:
    writer.close()
    await asyncio.wait_for(writer.wait_closed(), TIMEOUT)
    await asyncio.wait_for(server.close(), TIMEOUT)


class FailingWakeBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tool_schemas):
        del messages, tool_schemas
        self.calls += 1
        if False:
            yield
        raise RuntimeError("boom")


class DisconnectThenSucceedBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()

    async def complete(self, messages, tool_schemas):
        del messages, tool_schemas
        self.calls += 1
        if self.calls == 1:
            self.started.set()
            await asyncio.Event().wait()
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("retried")]),
        )


class AbortTwiceThenSucceedBackend:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.started = [asyncio.Event(), asyncio.Event()]

    async def complete(self, messages, tool_schemas):
        del tool_schemas
        self.calls.append(messages)
        call_index = len(self.calls) - 1
        if call_index < len(self.started):
            self.started[call_index].set()
            await asyncio.Event().wait()
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("retried")]),
        )


class BlockingThenCaptureBackend:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.started = asyncio.Event()

    async def complete(self, messages, tool_schemas):
        del tool_schemas
        self.calls.append(messages)
        if len(self.calls) == 1:
            self.started.set()
            await asyncio.Event().wait()
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
        )


class BackgroundChildAndForegroundToolBackend:
    def __init__(self, command: str) -> None:
        self.command = command
        self.child_started = asyncio.Event()
        self.release_child = asyncio.Event()

    async def complete(self, messages, tool_schemas):
        del tool_schemas
        if any(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in messages
        ):
            blocks = [TextContent("notification handled")]
        else:
            last_user = next(
                (
                    block.text
                    for message in reversed(messages)
                    if message.role is MessageRole.USER
                    for block in message.content
                    if isinstance(block, TextContent)
                ),
                "",
            )
            has_tool_result = any(
                message.role is MessageRole.TOOL_RESULT for message in messages
            )
            if last_user == "start child" and not has_tool_result:
                blocks = [
                    ToolUseContent(
                        ToolCall(
                            "background-child",
                            "agent",
                            {
                                "prompt": "child work",
                                "description": "background child",
                                "background": True,
                            },
                        )
                    )
                ]
            elif last_user == "child work":
                self.child_started.set()
                await self.release_child.wait()
                blocks = [TextContent("child complete")]
            elif last_user == "run slowly":
                blocks = [
                    ToolUseContent(
                        ToolCall("slow-tool", "bash", {"command": self.command})
                    )
                ]
            else:
                blocks = [TextContent("done")]
        yield StreamEvent(StreamEventType.MESSAGE_START)
        for block in blocks:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=block)
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, blocks),
        )


class MidstreamRetryBackend:
    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, messages, tool_schemas):
        del messages, tool_schemas
        self.calls += 1
        if self.calls == 1:
            yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="discard me")
            raise ConnectionError("provider connection dropped")
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="keep me")
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("keep me")]),
        )


async def _provider_retry_frames(
    tmp_path: Path, *, features: list[str] | None
) -> tuple[MidstreamRetryBackend, dict[str, Any], list[dict[str, Any]]]:
    backend = MidstreamRetryBackend()
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        hello_params: dict[str, object] = {"protocol_version": "1.1"}
        if features is not None:
            hello_params["features"] = features
        hello = (await _request(reader, writer, 1, "hello", hello_params))[-1][
            "result"
        ]
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        frames = await _request(reader, writer, 3, "send", {"text": "hello"})
        while not any(
            frame.get("params", {}).get("event") == "agent_end" for frame in frames
        ):
            frames.append(await _read(reader))
        return backend, hello, frames
    finally:
        await _close(server, writer)


async def _ready(server: ZetaServer):
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
    await _request(reader, writer, 2, "new_session", {"provider": "codex"})
    return reader, writer


@pytest.mark.asyncio
async def test_server_streams_scripted_turn_over_real_socket(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    assert server.runtime._test_scripted_provider is True
    reader, writer = await _ready(server)
    try:
        frames = await _request(reader, writer, 3, "send", {"text": "hello"})
        assert frames[-1]["result"]["accepted"] is True
        assert (await _event(reader, "assistant_delta"))["delta"]
        committed = await _event(reader, "assistant_message")
        assert committed["message"]["role"] == "assistant"
        await _event(reader, "turn_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_server_streams_display_safe_thinking_body_without_metadata(
    tmp_path: Path,
) -> None:
    sentinel = "raw-reasoning-metadata-sentinel"

    class MetadataBackend(FakeBackend):
        async def complete(self, messages, tool_schemas):
            async for event in super().complete(messages, tool_schemas):
                if event.type is StreamEventType.MESSAGE_END and event.message is not None:
                    yield replace(
                        event,
                        message=Message(
                            event.message.role,
                            event.message.content,
                            metadata={"codex_output_items": [{"text": sentinel}]},
                        ),
                    )
                else:
                    yield event

    backend = MetadataBackend([ScriptedTurn([ThinkingContent("summary")])])
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "hello"})
        await _event(reader, "assistant_delta")
        raw_frame = await _read_raw(reader)
        assert sentinel.encode() not in raw_frame
        committed = json.loads(raw_frame)
        params = committed["params"]
        assert params["event"] == "assistant_message"
        message = params["message"]
        assert message["content"] == [
            {"type": "thinking", "text": "summary", "body": "summary"}
        ]
        assert "metadata" not in message
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_unexpected_turn_error_matches_event_schema(tmp_path: Path) -> None:
    duplicate = ToolCall("duplicate-id", "read", {})
    backend = FakeBackend([ScriptedTurn(tool_calls=[duplicate, duplicate])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "trigger failure"})
        error = await _event(reader, "error")
        assert error["error"] == {
            "code": "server_error",
            "message": "duplicate tool call id in one execution batch",
        }
        assert error["data"] == {}
        assert isinstance(error["session_id"], str)
        assert (await _request(reader, writer, 4, "status"))[-1]["result"][
            "state"
        ] == "idle"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_round_trip_and_deny(tmp_path: Path) -> None:
    target = tmp_path / "input.txt"
    target.write_text("approved")
    call = ToolCall("call-1", "read", {"path": str(target)})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read it"})
        approval = await _event(reader, "approval_request")
        assert approval["request_id"] == "call-1"
        frames = await _request(reader, writer, 4, "deny", {"request_id": "call-1"})
        assert frames[-1]["result"]["decision"] == "deny"
        approval_end = _named_events(frames, "approval_end")
        assert len(approval_end) == 1
        assert approval_end[0]["request_id"] == approval["request_id"]
        end = await _event(reader, "tool_end")
        assert end["tool_result"]["is_error"] is True
        await _event(reader, "turn_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_abort_emits_approval_end_before_turn_aborted(tmp_path: Path) -> None:
    call = ToolCall("call-1", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read it"})
        approval = await _event(reader, "approval_request")
        frames = await _request(reader, writer, 4, "abort")
        frames += await _request(reader, writer, 5, "status")
        names = [frame.get("params", {}).get("event") for frame in frames]
        ends = _named_events(frames, "approval_end")
        assert len(ends) == 1
        assert ends[0]["request_id"] == approval["request_id"]
        assert names.index("approval_end") < names.index("turn_aborted")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_client_close_emits_approval_end_once(tmp_path: Path) -> None:
    call = ToolCall("call-1", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read it"})
        approval = await _event(reader, "approval_request")
        assert server._client is not None
        closing = asyncio.create_task(server._client.close())
        frames = await _frames_until_eof(reader)
        await closing
        ends = _named_events(frames, "approval_end")
        assert [event["request_id"] for event in ends] == [approval["request_id"]]
    finally:
        writer.close()
        await server.close()


@pytest.mark.asyncio
async def test_server_shutdown_emits_approval_end_once(tmp_path: Path) -> None:
    call = ToolCall("call-1", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    await _request(reader, writer, 3, "send", {"text": "read it"})
    approval = await _event(reader, "approval_request")
    closing = asyncio.create_task(server.close())
    frames = await _frames_until_eof(reader)
    await closing
    ends = _named_events(frames, "approval_end")
    assert [event["request_id"] for event in ends] == [approval["request_id"]]
    writer.close()


@pytest.mark.asyncio
async def test_foreground_and_delegated_same_raw_id_distinct_request_ids_and_matched_ends(
    tmp_path: Path,
) -> None:
    raw_id = "shared-call"
    target = tmp_path / "input.txt"
    target.write_text("data")
    foreground = ToolCall(raw_id, "read", {"path": str(target)})
    agent = ToolCall(
        "agent-one",
        "agent",
        {"prompt": "read", "description": "child", "background": True},
    )
    delegated = ToolCall(raw_id, "read", {"path": str(target)})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[foreground, agent]),
            ScriptedTurn(tool_calls=[delegated]),
            ScriptedTurn([TextContent("done")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "both"})
        first = await _event(reader, "approval_request")
        second = await _event(reader, "approval_request")
        requests = {event["delegated"]: event for event in (first, second)}
        assert requests[False]["tool_call"]["id"] == raw_id
        assert requests[True]["tool_call"]["id"] == raw_id
        assert requests[False]["request_id"] != requests[True]["request_id"]

        frames = await _request(
            reader, writer, 4, "deny", {"request_id": requests[False]["request_id"]}
        )
        frames += await _request(
            reader, writer, 5, "deny", {"request_id": requests[True]["request_id"]}
        )
        ends = _named_events(frames, "approval_end")
        assert sorted(event["request_id"] for event in ends) == sorted(
            event["request_id"] for event in requests.values()
        ), frames
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_child_cancel_while_approval_pending_emits_one_matching_approval_end(
    tmp_path: Path,
) -> None:
    target = tmp_path / "input.txt"
    target.write_text("data")
    agent = ToolCall(
        "agent-one",
        "agent",
        {"prompt": "read", "description": "child", "background": True},
    )
    child_call = ToolCall("child-call", "read", {"path": str(target)})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[agent]), ScriptedTurn(tool_calls=[child_call])]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "delegate"})
        approval = await _event(reader, "approval_request")
        assert approval["delegated"] is True
        assert server.runtime.loop is not None
        assert server.runtime.loop._background_owner.cancel(
            approval["agent_instance_id"]
        )
        await asyncio.sleep(0.1)
        frames = await _request(reader, writer, 4, "status")
        ends = _named_events(frames, "approval_end")
        assert [event["request_id"] for event in ends] == [approval["request_id"]]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_status_pending_delegated_identity_matches_event(tmp_path: Path) -> None:
    target = tmp_path / "input.txt"
    target.write_text("data")
    agent = ToolCall(
        "agent-one",
        "agent",
        {"prompt": "read", "description": "child", "background": True},
    )
    child_call = ToolCall("child-call", "read", {"path": str(target)})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[agent]), ScriptedTurn(tool_calls=[child_call])]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "delegate"})
        approval = await _event(reader, "approval_request")
        status = (await _request(reader, writer, 4, "status"))[-1]["result"]
        pending = status["pending_approvals"]
        assert len(pending) == 1
        assert pending[0]["request_id"] == approval["request_id"]
        assert pending[0]["delegated"] is True
        assert pending[0]["agent_instance_id"] == approval["agent_instance_id"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_status_pending_foreground_has_delegated_false_and_no_child_id(
    tmp_path: Path,
) -> None:
    call = ToolCall("call-1", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read it"})
        approval = await _event(reader, "approval_request")
        status = (await _request(reader, writer, 4, "status"))[-1]["result"]
        pending = status["pending_approvals"]
        assert len(pending) == 1
        assert pending[0]["request_id"] == approval["request_id"]
        assert pending[0]["delegated"] is False
        assert "agent_instance_id" not in pending[0]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_decision_then_loop_end_emits_single_approval_end(tmp_path: Path) -> None:
    call = ToolCall("call-1", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read it"})
        approval = await _event(reader, "approval_request")
        frames = await _request(
            reader, writer, 4, "deny", {"request_id": approval["request_id"]}
        )
        frames += await _frames_until_event(reader, "turn_end")
        ends = _named_events(frames, "approval_end")
        assert [event["request_id"] for event in ends] == [approval["request_id"]]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_lifecycle_prunes_ended_approvals(tmp_path: Path) -> None:
    turns = 20
    backend = FakeBackend(
        [
            turn
            for index in range(turns)
            for turn in (
                ScriptedTurn(
                    tool_calls=[
                        ToolCall(
                            f"call-{index}",
                            "read",
                            {"path": str(tmp_path / f"input-{index}")},
                        )
                    ]
                ),
                ScriptedTurn([TextContent("done")]),
            )
        ]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        for index in range(turns):
            await _request(reader, writer, 3 + index * 2, "send", {"text": "read it"})
            approval = await _event(reader, "approval_request")
            await _request(
                reader,
                writer,
                4 + index * 2,
                "deny",
                {"request_id": approval["request_id"]},
            )
            await _event(reader, "tool_end")
            await _event(reader, "agent_end")

        assert server._client is not None
        assert len(server._client._approvals._by_key) == 0
        assert len(server._client._approvals._wires_by_key) == 0
        assert len(server._client._approvals._keys_by_wire) == 0
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_and_steer_continue_the_same_turn(tmp_path: Path) -> None:
    target = tmp_path / "input.txt"
    target.write_text("approved")
    call = ToolCall("call-1", "read", {"path": str(target)})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read it"})
        await _event(reader, "approval_request")
        await _request(reader, writer, 4, "steer", {"text": "also check this"})
        await _request(reader, writer, 5, "approve", {"request_id": "call-1"})
        await _event(reader, "assistant_message")
        await _event(reader, "turn_end")
        assert any(
            message.role.value == "user"
            and any(getattr(block, "text", "") == "also check this" for block in message.content)
            for message in backend.calls[1][0]
        )
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_scope_always_tool_records_session_policy_and_skips_next_turn(
    tmp_path: Path,
) -> None:
    """always_tool approval mutates always_allow, so the next turn's same-tool
    call auto-runs without prompting. Other tools still ask, and the effect
    lasts the session only — a legacy client that omits `scope` sees the old
    per-request behavior."""

    first = tmp_path / "one.txt"
    first.write_text("first")
    second = tmp_path / "two.txt"
    second.write_text("second")
    first_call = ToolCall("call-1", "read", {"path": str(first)})
    second_call = ToolCall("call-2", "read", {"path": str(second)})
    bash_call = ToolCall("call-3", "bash", {"command": "true"})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[first_call]),
            ScriptedTurn(tool_calls=[second_call, bash_call]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read them"})
        first_ask = await _event(reader, "approval_request")
        assert first_ask["request_id"] == "call-1"
        allowed = await _request(
            reader,
            writer,
            4,
            "approve",
            {"request_id": "call-1", "scope": "always_tool"},
        )
        assert allowed[-1]["result"]["scope"] == "always_tool"
        assert allowed[-1]["result"]["decision"] == "approve"
        assert server.runtime.policy is not None
        assert any(
            rule.tool == "read" and rule.pattern is None
            for rule in server.runtime.policy.always_allow
        )
        # The next turn's read auto-approves (no fresh approval_request for
        # call-2). The bash call in the same turn still asks. Its request can
        # arrive before or after the approval response; both orders are valid.
        asks = _named_events(allowed, "approval_request")
        next_ask = asks[-1] if asks else await _event(reader, "approval_request")
        assert next_ask["request_id"] == "call-3", (
            "read must auto-approve after always_tool, so the next ask is bash"
        )
        await _request(reader, writer, 5, "deny", {"request_id": "call-3"})
        await _event(reader, "turn_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_without_scope_keeps_legacy_shape(tmp_path: Path) -> None:
    """Omitting `scope` preserves the pre-ZETA-131 wire shape: no `scope` key
    in the result and no policy mutation. Legacy clients keep working."""

    target = tmp_path / "input.txt"
    target.write_text("approved")
    call = ToolCall("call-1", "read", {"path": str(target)})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read it"})
        await _event(reader, "approval_request")
        frames = await _request(
            reader, writer, 4, "approve", {"request_id": "call-1"}
        )
        result = frames[-1]["result"]
        assert result == {
            "accepted": True,
            "request_id": "call-1",
            "decision": "approve",
        }
        assert server.runtime.policy is not None
        assert not server.runtime.policy.always_allow
        await _event(reader, "turn_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_scope_rejects_invalid_values(tmp_path: Path) -> None:
    target = tmp_path / "input.txt"
    target.write_text("approved")
    call = ToolCall("call-1", "read", {"path": str(target)})
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "read it"})
        await _event(reader, "approval_request")
        rejected = await _request(
            reader,
            writer,
            4,
            "approve",
            {"request_id": "call-1", "scope": "session"},
        )
        assert rejected[-1]["error"]["code"] == -32602
        rejected_deny = await _request(
            reader,
            writer,
            5,
            "deny",
            {"request_id": "call-1", "scope": "always_tool"},
        )
        assert rejected_deny[-1]["error"]["code"] == -32602
        # Non-string scope shapes (array, object, null, integer, boolean)
        # must return -32602, not the -32000 the set-membership check raised
        # before the type guard landed.
        next_id = 6
        for bad_scope in ([], {}, None, 1, True):
            rejected_shape = await _request(
                reader,
                writer,
                next_id,
                "approve",
                {"request_id": "call-1", "scope": bad_scope},
            )
            error = rejected_shape[-1]["error"]
            assert error["code"] == -32602, f"scope={bad_scope!r} error {error!r}"
            assert error["message"] == "scope must be 'once' or 'always_tool'"
            next_id += 1
        await _request(reader, writer, next_id, "deny", {"request_id": "call-1"})
        await _event(reader, "turn_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_resumed_approval_finishes_idle_after_terminal_event(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    target = tmp_path / "input.txt"
    target.write_text("approved")
    call = ToolCall("resumed-call", "read", {"path": str(target)})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    opened.store.close()
    backend = FakeBackend([])
    server = ZetaServer(
        home=tmp_path,
        provider="codex",
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "resume", {"session_id": opened.metadata.session_id})
        approved = await _request(
            reader, writer, 3, "approve", {"request_id": call.id}
        )
        assert approved[-1]["result"]["decision"] == "approve"
        await _event(reader, "tool_end")
        status = await _request(reader, writer, 4, "status")
        assert status[-1]["result"]["state"] == "idle"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_abort_mid_stream(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("first"), TextContent("second")], delay=0.2)])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "stop"})
        await _event(reader, "assistant_delta")
        frames = await _request(reader, writer, 4, "abort")
        assert frames[-1]["result"]["aborted"] is True
        assert any(frame.get("params", {}).get("event") == "turn_aborted" for frame in frames)
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["session", "foreground"])
async def test_abort_captures_turn_before_approval_end_write(
    tmp_path: Path,
    scope: str,
) -> None:
    call = ToolCall("approval-race", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "wait for approval"})
        await _event(reader, "approval_request")

        assert server._client is not None
        client = server._client
        original_write = client._write
        approval_end_started = asyncio.Event()
        allow_approval_end = asyncio.Event()

        async def pause_approval_end(payload: bytes) -> None:
            frame = json.loads(payload)
            if frame.get("params", {}).get("event") == "approval_end":
                approval_end_started.set()
                await allow_approval_end.wait()
            await original_write(payload)

        client._write = pause_approval_end  # type: ignore[method-assign]
        abort = asyncio.create_task(
            _request(reader, writer, 4, "abort", {"scope": scope})
        )
        await asyncio.wait_for(approval_end_started.wait(), TIMEOUT)
        while client._turn_task is not None:
            await asyncio.sleep(0)
        allow_approval_end.set()

        frames = await asyncio.wait_for(abort, TIMEOUT)
        assert frames[-1]["result"] == {"aborted": True}
        assert _named_events(frames, "turn_aborted")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_foreground_abort_only_ends_foreground_approval(
    tmp_path: Path,
) -> None:
    call = ToolCall("foreground-approval", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    child_store = ConversationStore(tmp_path / "delegated-child")
    child_task: asyncio.Task[None] | None = None
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "wait for approval"})
        foreground = await _event(reader, "approval_request")

        delegated_call = ToolCall("child-approval", "bash", {"command": "true"})
        delegated_request = ApprovalRequest(
            delegated_call.id,
            delegated_call,
            child_instance_id="background-child",
        )
        child_store.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(delegated_call)]),
            [(delegated_call.id, delegated_call)],
        )
        assert server.runtime.policy is not None
        server.runtime.policy.register_delegated(
            delegated_request,
            child_store,
            child_instance_id="background-child",
        )
        assert server.runtime.loop is not None
        child_task = asyncio.create_task(asyncio.sleep(60))
        server.runtime.loop._background_owner.register(
            "background-child", child_task.cancel, child_task
        )
        assert server._client is not None
        server._client._approvals.observe(delegated_request)

        frames = await _request(
            reader, writer, 4, "abort", {"scope": "foreground"}
        )
        ended = _named_events(frames, "approval_end")
        assert [event["request_id"] for event in ended] == [
            foreground["request_id"]
        ]
        assert delegated_request.key in server._client._approvals.active_keys()
        assert any(
            request.key == delegated_request.key
            for request in server.runtime.policy.pending_requests()
        )
        assert not child_task.done()
    finally:
        await _close(server, writer)
        child_store.close()


@pytest.mark.asyncio
async def test_foreground_abort_mid_stream_keeps_pending_steering(
    tmp_path: Path,
) -> None:
    backend = BlockingThenCaptureBackend()
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        hello = await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        assert hello[-1]["result"]["capabilities"]["features"] == ["abort_scope"]
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "start"})
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        await _request(reader, writer, 4, "steer", {"text": "keep this"})

        frames = await _request(
            reader, writer, 5, "abort", {"scope": "foreground"}
        )
        assert frames[-1]["result"] == {"aborted": True}
        assert _named_events(frames, "turn_aborted")
        assert server.runtime.loop is not None
        assert server.runtime.loop.has_pending_steering

        await _request(reader, writer, 6, "send", {"text": "continue"})
        await _event(reader, "agent_end")
        assert any(
            message.role is MessageRole.USER
            and any(
                isinstance(block, TextContent) and block.text == "keep this"
                for block in message.content
            )
            for message in backend.calls[1]
        )
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_clear_steering_returns_cleared_count(tmp_path: Path) -> None:
    backend = BlockingThenCaptureBackend()
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        hello = await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        assert "clear_steering" in hello[-1]["result"]["capabilities"]["requests"]
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "start"})
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        await _request(reader, writer, 4, "steer", {"text": "discard one"})
        await _request(reader, writer, 5, "steer", {"text": "discard two"})

        cleared = await _request(reader, writer, 6, "clear_steering")
        assert cleared[-1]["result"] == {"cleared": 2}
        assert server.runtime.loop is not None
        assert not server.runtime.loop.has_pending_steering
        assert (await _request(reader, writer, 7, "clear_steering"))[-1][
            "result"
        ] == {"cleared": 0}
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_clear_steering_requires_abort_scope_feature(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex"
    )
    reader, writer = await _ready(server)
    try:
        frames = await _request(reader, writer, 3, "clear_steering")
        assert frames[-1]["error"] == {
            "code": -32602,
            "message": "clear_steering requires the negotiated abort_scope feature",
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_foreground_abort_cancels_tool_but_keeps_background_child(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "foreground-finished"
    command = (
        f"{sys.executable} -c \"import pathlib,time; time.sleep(0.5); "
        f"pathlib.Path({str(marker)!r}).write_text('done')\""
    )
    backend = BackgroundChildAndForegroundToolBackend(command)
    server = ZetaServer(provider="codex",
        home=tmp_path,
        cwd=tmp_path,
        socket_path=_socket_path(tmp_path),
        cli_yolo=True,
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "start child"})
        await _event(reader, "agent_end")
        await asyncio.wait_for(backend.child_started.wait(), TIMEOUT)
        loop = server.runtime.loop
        assert loop is not None
        owner = loop._background_owner
        assert owner.running

        await _request(reader, writer, 4, "send", {"text": "run slowly"})
        await _event(reader, "tool_start")
        frames = await _request(
            reader, writer, 5, "abort", {"scope": "foreground"}
        )
        assert frames[-1]["result"] == {"aborted": True}
        assert _named_events(frames, "turn_aborted")
        assert owner.running

        backend.release_child.set()
        completion = await _event(reader, "tool_end")
        assert completion["tool_call"]["id"] == "background-child"
        assert completion["tool_result"]["content"].startswith("child complete")
        await asyncio.sleep(0.6)
        assert not marker.exists()
    finally:
        backend.release_child.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_abort_scope_requires_negotiation_and_rejects_unknown_scope(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _ready(server)
    try:
        unnegotiated = await _request(
            reader, writer, 3, "abort", {"scope": "foreground"}
        )
        assert unnegotiated[-1]["error"]["code"] == -32602
    finally:
        await _close(server, writer)

    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        unknown = await _request(reader, writer, 3, "abort", {"scope": "turn"})
        assert unknown[-1]["error"]["code"] == -32602
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_resume_second_client_and_malformed_frame(tmp_path: Path) -> None:
    first = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _ready(first)
    session = (await _request(reader, writer, 3, "status"))[-1]["result"]["session"]["session_id"]
    second_reader, second_writer = await asyncio.wait_for(
        asyncio.open_unix_connection(str(first.socket_path)), TIMEOUT
    )
    refused = await _read(second_reader)
    assert refused["error"]["code"] == -32001
    second_writer.close()
    await asyncio.wait_for(second_writer.wait_closed(), TIMEOUT)
    writer.write(b"not json\n")
    await writer.drain()
    malformed = await _read(reader)
    assert malformed["error"]["code"] == -32700
    await _close(first, writer)

    resumed = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _connect(resumed)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        frames = await _request(reader, writer, 2, "resume", {"session_id": session})
        assert frames[-1]["result"]["session"]["session_id"] == session
    finally:
        await _close(resumed, writer)


@pytest.mark.asyncio
async def test_server_can_bind_localhost_port(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        assert server.port != 0
        assert server.address.startswith("127.0.0.1:")
        result = await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        assert result[-1]["result"]["protocol_version"] == "1.0"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_oversized_frame_returns_error_and_keeps_connection_usable(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _ready(server)
    try:
        oversized = (
            b'{"jsonrpc":"2.0","id":"oversized","method":"status","params":{"padding":"'
            + b"x" * MAX_FRAME_BYTES
            + b'"}}\n'
        )
        writer.write(oversized)
        await writer.drain()
        error = await _read(reader)
        assert error["id"] == "oversized"
        assert error["error"]["code"] == -32600
        status = await _request(reader, writer, 3, "status")
        assert status[-1]["result"]["session"] is not None
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_maximum_legal_frame_gets_bounded_error_response(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
    await _request(reader, writer, 2, "new_session", {"provider": "codex"})
    assert server.runtime.opened is not None
    server.runtime.opened.metadata.system_prompt = "x" * MAX_FRAME_BYTES
    request_id = "legal-frame"
    padding_length = MAX_FRAME_BYTES
    while True:
        candidate = (
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": "status",
                    "params": {"padding": "x" * padding_length},
                },
                separators=(",", ":"),
            )
            + "\n"
        ).encode()
        if len(candidate) <= MAX_FRAME_BYTES:
            break
        padding_length -= len(candidate) - MAX_FRAME_BYTES
    padding_length += MAX_FRAME_BYTES - len(candidate)
    candidate = (
        json.dumps(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "status",
                "params": {"padding": "x" * padding_length},
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    assert len(candidate) == MAX_FRAME_BYTES
    try:
        writer.write(candidate)
        await writer.drain()
        raw = await _read_raw(reader)
        response_frame = json.loads(raw)
        assert response_frame["id"] == request_id
        assert response_frame["error"]["code"] == -32007
        assert len(raw) <= MAX_FRAME_BYTES
    finally:
        await _close(server, writer)


def test_request_id_limit_is_inclusive() -> None:
    valid = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "x" * MAX_REQUEST_ID_BYTES,
            "method": "status",
        }
    ).encode()
    invalid = json.dumps(
        {
            "jsonrpc": "2.0",
            "id": "x" * (MAX_REQUEST_ID_BYTES + 1),
            "method": "status",
        }
    ).encode()
    from zeta.server.protocol import ProtocolError

    codec = FrameCodec()
    assert codec.parse_request(valid)["id"] == "x" * MAX_REQUEST_ID_BYTES
    with pytest.raises(ProtocolError, match="request id exceeds"):
        codec.parse_request(invalid)


@pytest.mark.asyncio
async def test_huge_numeric_request_id_returns_error_and_keeps_connection_usable(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _ready(server)
    try:
        writer.write(
            b'{"jsonrpc":"2.0","id":' + b"7" * 5_000 + b',"method":"status","params":{}}\n'
        )
        await writer.drain()
        error = await _read(reader)
        assert error["id"] is None
        assert error["error"]["code"] == -32600
        status = await _request(reader, writer, 3, "status")
        assert status[-1]["result"]["session"] is not None
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_malformed_and_wrong_type_frames_keep_connection_usable(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _ready(server)
    try:
        writer.write(
            b'{"jsonrpc":"2.0","id":"known-id","method":NaN,"params":{}}\n'
        )
        await writer.drain()
        malformed = await _read(reader)
        assert malformed["id"] == "known-id"
        assert malformed["error"]["code"] == -32700

        writer.write(
            b'{"jsonrpc":"2.0","id":["wrong-type"],"method":"status","params":{}}\n'
        )
        await writer.drain()
        wrong_type = await _read(reader)
        assert wrong_type["id"] is None
        assert wrong_type["error"]["code"] == -32600

        status = await _request(reader, writer, 3, "status")
        assert status[-1]["result"]["session"] is not None
    finally:
        await _close(server, writer)


def test_protocol_schema_documents_all_reviewed_event_contracts() -> None:
    protocol = (Path(__file__).parents[1] / "docs" / "serve-protocol.md").read_text(
        encoding="utf-8"
    )
    rows = protocol.splitlines()
    assert "| `-32007` | outbound frame exceeds the size limit |" in rows
    assert (
        "| `turn_start`, `turn_end`, `agent_start`, `agent_end`, `message_start`, "
        "`turn_aborted`, `compaction_start`, `compaction_end` | `event` | "
        "`session_id`, `data: object` |"
    ) in rows
    assert (
        "| `tool_output` | `event`, `tool_call: ToolCall`, `output: string`, "
        "`data: object` | `session_id` |"
    ) in rows
    assert "during tool execution or approval waits" in protocol


@pytest.mark.asyncio
async def test_list_sessions_marks_oversized_result_as_truncated(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    huge = SessionMetadata.new(
        session_id="a" * 32,
        provider="codex",
        model="offline",
        cwd=str(tmp_path),
        retained_tail=8,
        compaction_budget=200_000,
        system_prompt="x" * MAX_FRAME_BYTES,
    )
    server.runtime.list_sessions = lambda: [huge]
    reader, writer = await _ready(server)
    try:
        frames = await _request(reader, writer, 3, "list_sessions")
        result = frames[-1]["result"]
        assert result["truncated"] is True
        assert result["sessions"] == []
        assert len(json.dumps(frames[-1]).encode()) <= MAX_FRAME_BYTES
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_list_sessions_uses_maximum_request_id_for_boundary(
    tmp_path: Path,
) -> None:
    codec = FrameCodec()
    probe = SessionMetadata.new(
        session_id="a" * 32,
        provider="codex",
        model="offline",
        cwd=str(tmp_path),
        retained_tail=8,
        compaction_budget=200_000,
    )
    base_size = len(codec.response(0, {"sessions": [probe.to_dict()]}))
    huge = SessionMetadata.new(
        session_id=probe.session_id,
        provider="codex",
        model="offline",
        cwd=str(tmp_path),
        retained_tail=8,
        compaction_budget=200_000,
        system_prompt="x" * (MAX_FRAME_BYTES - 128 - base_size),
    )
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    server.runtime.list_sessions = lambda: [huge]
    reader, writer = await _connect(server)
    request_id = "x" * MAX_REQUEST_ID_BYTES
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        frames = await _request(reader, writer, request_id, "list_sessions")
        assert frames[-1]["result"] == {
            "sessions": [],
            "truncated": True,
            "next_offset": 0,
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_invalid_utf8_tail_preserves_request_id(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _ready(server)
    try:
        writer.write(
            b'{"jsonrpc":"2.0","id":"tail-id","method":"status","params":{}}'
            b"\xff\n"
        )
        await writer.drain()
        error = await _read(reader)
        assert error["id"] == "tail-id"
        assert error["error"]["code"] == -32700
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_socket_start_rejects_regular_path_and_removes_stale_socket(
    tmp_path: Path,
) -> None:
    socket_path = _socket_path(tmp_path)
    socket_path.write_text("do not delete")
    server = ZetaServer(home=tmp_path, socket_path=socket_path, provider="codex")
    with pytest.raises(RuntimeError, match="non-socket"):
        await server.start()
    socket_path.unlink()

    stale = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    stale.bind(str(socket_path))
    stale.close()
    await server.start()
    try:
        assert stat.S_IMODE(socket_path.stat().st_mode) == 0o600
    finally:
        await server.close()
    assert not socket_path.exists()


@pytest.mark.asyncio
async def test_rejected_pre_hello_request_closes_connection(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _connect(server)
    try:
        writer.write(b'{"jsonrpc":"2.0","id":1,"method":"status","params":{}}\n')
        await writer.drain()
        error = await _read(reader)
        assert error["error"]["code"] == -32002
        assert await asyncio.wait_for(reader.read(), TIMEOUT) == b""
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()


@pytest.mark.asyncio
async def test_disconnect_aborts_pending_approval_without_phantom(
    tmp_path: Path,
) -> None:
    call = ToolCall("call-1", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    await _request(reader, writer, 3, "send", {"text": "read it"})
    await _event(reader, "approval_request")
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    assert server.runtime.policy is not None
    assert server.runtime.policy.pending_requests() == []
    await server.close()


@pytest.mark.asyncio
async def test_late_approval_after_disconnect_abort_is_rejected(tmp_path: Path) -> None:
    call = ToolCall("call-1", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call])])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    await _request(reader, writer, 3, "send", {"text": "read it"})
    await _event(reader, "approval_request")
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    second_reader, second_writer = await asyncio.open_unix_connection(str(server.socket_path))
    try:
        await _request(second_reader, second_writer, 1, "hello", {"protocol_version": "1.0"})
        error = (
            await _request(
                second_reader,
                second_writer,
                2,
                "approve",
                {"request_id": "call-1"},
            )
        )[-1]
        assert error["error"]["code"] == -32006
    finally:
        await _close(server, second_writer)


@pytest.mark.asyncio
async def test_client_close_persists_streamed_data(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("first"), TextContent("second")], delay=0.2)])
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    session_id = server.runtime.session_id
    await _request(reader, writer, 3, "send", {"text": "stream"})
    await _event(reader, "assistant_delta")
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.1)
    messages = server.runtime.manager.open(session_id).store.messages()
    assert any(
        message.role.value == "assistant"
        and "first" in "".join(getattr(block, "text", "") for block in message.content)
        for message in messages
    )
    await server.close()


def test_delegated_approval_wire_keys_are_unique() -> None:
    approvals = ApprovalLifecycle()
    first = approvals.wire_id(("child-a", "same"))
    second = approvals.wire_id(("child-b", "same"))
    assert first != second
    assert approvals.core_key(first) == ("child-a", "same")
    assert approvals.core_key(second) == ("child-b", "same")


def test_reused_core_key_gets_fresh_wire_id_after_end() -> None:
    approvals = ApprovalLifecycle()
    request = ApprovalRequest(
        "same", ToolCall("same", "read", {"path": "input"}), child_instance_id="child"
    )

    first = approvals.observe(request)["request_id"]
    assert approvals.end(request.key) is not None
    approvals.prune_ended()
    second = approvals.observe(request)["request_id"]

    assert first != second


@pytest.mark.asyncio
async def test_negotiated_client_receives_assistant_reset(tmp_path: Path) -> None:
    backend, hello, frames = await _provider_retry_frames(
        tmp_path, features=["assistant_reset"]
    )

    assert hello["capabilities"]["features"] == ["assistant_reset"]
    events = [
        frame["params"]["event"]
        for frame in frames
        if frame.get("method") == "event"
    ]
    assert backend.calls == 2
    assert events.index("retry") < events.index("assistant_reset")
    assert events == [
        "agent_start",
        "turn_start",
        "assistant_delta",
        "retry",
        "assistant_reset",
        "assistant_delta",
        "assistant_message",
        "turn_end",
        "agent_end",
    ]


@pytest.mark.asyncio
async def test_serve_client_without_reset_optin_keeps_post_stream_retry_off(
    tmp_path: Path,
) -> None:
    backend, _hello, frames = await _provider_retry_frames(tmp_path, features=[])

    events = [
        frame["params"]
        for frame in frames
        if frame.get("method") == "event"
    ]
    assert backend.calls == 1
    assert [event["delta"] for event in events if event["event"] == "assistant_delta"] == [
        "discard me"
    ]
    assert not any(event["event"] in {"retry", "assistant_reset"} for event in events)
    assert any(event["event"] == "error" for event in events)


@pytest.mark.asyncio
async def test_serve_emits_assistant_reset_before_retry_deltas() -> None:
    client = object.__new__(_Client)
    client.server = SimpleNamespace(runtime=SimpleNamespace(state=None))
    notifications: list[tuple[str, dict[str, object]]] = []

    async def notify(
        event: str, session_id: str | None, **fields: object
    ) -> None:
        del session_id
        notifications.append((event, fields))

    async def ignore_stale_approvals(_session_id: str | None) -> None:
        return None

    client._notify = notify
    client._end_stale_approvals = ignore_stale_approvals
    await client._event(
        StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="discard me"),
        session_id="session",
    )
    await client._event(
        StreamEvent(StreamEventType.RETRY, data={"text": "retry scheduled"}),
        session_id="session",
    )
    await client._event(
        StreamEvent(StreamEventType.ASSISTANT_RESET),
        session_id="session",
    )
    await client._event(
        StreamEvent(StreamEventType.MESSAGE_UPDATE, delta="keep me"),
        session_id="session",
    )

    assert [event for event, _fields in notifications] == [
        "assistant_delta",
        "retry",
        "assistant_reset",
        "assistant_delta",
    ]


@pytest.mark.asyncio
async def test_live_approval_without_exact_pending_request_is_not_approvable(
    tmp_path: Path,
) -> None:
    policy = ApprovalPolicy(store=ConversationStore(tmp_path))
    client = object.__new__(_Client)
    client.server = SimpleNamespace(
        runtime=SimpleNamespace(policy=policy, state=None)
    )
    notifications: list[tuple[str, dict[str, object]]] = []

    async def notify(
        event: str, session_id: str | None, **fields: object
    ) -> None:
        del session_id
        notifications.append((event, fields))

    client._notify = notify
    call = ToolCall("missing", "write", {"path": "provider/path"})
    await client._event(
        StreamEvent(
            StreamEventType.TOOL_APPROVAL_START,
            tool_call=call,
            data={"agent_instance_id": "child"},
        ),
        session_id="session",
    )

    assert [event for event, _fields in notifications] == ["error"]
    assert notifications[0][1]["error"] == {
        "code": "approval_context_missing",
        "message": "approval request is unavailable; denying live approval",
    }


@pytest.mark.asyncio
async def test_bad_resume_preserves_current_session(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _ready(server)
    try:
        before = (await _request(reader, writer, 3, "status"))[-1]["result"]["session"][
            "session_id"
        ]
        error = (await _request(reader, writer, 4, "resume", {"session_id": "missing"}))[-1]
        assert error["error"]["code"] == -32602
        after = (await _request(reader, writer, 5, "status"))[-1]["result"]["session"]["session_id"]
        assert after == before
        frames = await _request(reader, writer, 6, "send", {"text": "still here"})
        assert frames[-1]["result"]["accepted"] is True
        await _event(reader, "assistant_message")
        await _event(reader, "turn_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_session_switch_clears_delegated_mappings(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
    )
    reader, writer = await _ready(server)
    first_session = server.runtime.session_id
    try:
        assert server._client is not None
        approvals = server._client._approvals
        first_wire = approvals.wire_id(("child-a", "same"))
        await _request(reader, writer, 3, "new_session", {"provider": "codex"})
        assert approvals.core_key(first_wire) == first_wire

        second_wire = approvals.wire_id(("child-b", "same"))
        await _request(
            reader, writer, 4, "resume", {"session_id": first_session}
        )
        assert approvals.core_key(second_wire) == second_wire
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_session_swap_resets_usage_for_create_and_resume(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first")], usage={"input_tokens": 3}),
            ScriptedTurn([TextContent("second")], usage={"input_tokens": 5}),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        provider="codex",
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    first_session = server.runtime.session_id
    try:
        await _request(reader, writer, 3, "send", {"text": "first"})
        await _event(reader, "usage")
        assert (await _request(reader, writer, 4, "status"))[-1]["result"]["usage"]

        await _request(reader, writer, 5, "new_session", {"provider": "codex"})
        assert (await _request(reader, writer, 6, "status"))[-1]["result"]["usage"] == {}

        await _request(reader, writer, 7, "send", {"text": "second"})
        await _event(reader, "usage")
        assert (await _request(reader, writer, 8, "status"))[-1]["result"]["usage"]

        await _request(reader, writer, 9, "resume", {"session_id": first_session})
        assert (await _request(reader, writer, 10, "status"))[-1]["result"]["usage"] == {}
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_parameterless_new_session_uses_server_defaults_after_override(
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, str | None]] = []

    def build_backend(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
        calls.append((provider, model))
        return FakeBackend([]), model or "offline"

    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        model="server-default",
        backend_factory=build_backend,
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(
            reader,
            writer,
            3,
            "new_session",
            {"provider": "codex", "model": "session-override"},
        )
        result = await _request(reader, writer, 4, "new_session")
        assert result[-1]["result"]["session"]["model"] == "server-default"
        assert calls[-1] == ("codex", "server-default")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_session_swap_keeps_old_background_event_identity_until_shutdown(
    tmp_path: Path,
) -> None:
    parent_call = ToolCall(
        "agent-one",
        "agent",
        {"prompt": "wait", "description": "child", "background": True},
    )
    child_call = ToolCall("child-call", "read", {"path": str(tmp_path / "input")})
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[parent_call]),
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn([TextContent("parent done")]),
        ]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    old_session_id = server.runtime.session_id
    assert server.runtime.opened is not None
    for index in range(300):
        server.runtime.opened.store.append_message(
            with_message_origin(Message(MessageRole.USER, [TextContent(f"history-{index}")]), MessageOrigin.USER)
        )
    try:
        await _request(reader, writer, 3, "send", {"text": "start"})
        approval = await _event(reader, "approval_request")
        assert approval["delegated"] is True
        # A child approval and per-provider turn_end do not signal that the
        # parent agent loop is idle. agent_end is the session-swap handoff.
        await _event(reader, "agent_end")
        swap_frames = await _request(reader, writer, 4, "new_session", {"provider": "codex"})
        new_session_id = swap_frames[-1]["result"]["session"]["session_id"]
        assert new_session_id != old_session_id
        assert all(
            frame.get("params", {}).get("session_id") == old_session_id
            for frame in swap_frames
            if frame.get("params") is not None
        )

        terminal = next(
            (
                frame
                for frame in swap_frames
                if frame.get("params", {}).get("event") == "tool_end"
                and frame.get("params", {}).get("data", {}).get("notification_id")
            ),
            None,
        )
        while terminal is None:
            frame = json.loads(await asyncio.wait_for(reader.readline(), TIMEOUT))
            params = frame.get("params", {})
            if params.get("event") == "tool_end" and params.get("data", {}).get("notification_id"):
                terminal = frame
        assert terminal["params"]["session_id"] == old_session_id
        status = await _request(reader, writer, 5, "status")
        assert status[-1]["result"]["session"]["session_id"] == new_session_id
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", ["write", "bash"])
async def test_delegated_approval_stream_carries_captured_execution_facts(
    tmp_path: Path,
    tool_name: str,
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    if tool_name == "write":
        outside = tmp_path / "outside"
        outside.mkdir()
        (repository / "innocent-alias").symlink_to(outside, target_is_directory=True)
        child_call = ToolCall(
            "child-call",
            "write",
            {"path": "innocent-alias/file.txt", "content": "data"},
        )
        expected_field = "resolved_path"
        expected_value = str(outside / "file.txt")
    else:
        shell_cwd = repository / "custom-cwd"
        shell_cwd.mkdir()
        child_call = ToolCall(
            "child-call",
            "bash",
            {"command": "pwd", "cwd": str(shell_cwd)},
        )
        expected_field = "effective_cwd"
        expected_value = str(shell_cwd)

    parent_call = ToolCall(
        "agent-one",
        "agent",
        {"prompt": "run child tool", "description": "child", "background": True},
    )
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[parent_call]),
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn([TextContent("done")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path / "home",
        cwd=repository,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "delegate"})
        approval = await _event(reader, "approval_request")
        assert approval["delegated"] is True
        assert approval["approval_display"][expected_field] == expected_value
        await _request(
            reader,
            writer,
            4,
            "deny",
            {"request_id": approval["request_id"]},
        )
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_parent_mode_resolves_live_delegated_approvals_independently(
    tmp_path: Path,
) -> None:
    target = tmp_path / "input.txt"
    target.write_text("child data", encoding="utf-8")
    child_call = ToolCall("same-provider-id", "read", {"path": str(target)})
    parent_calls = [
        ToolCall(
            "agent-one",
            "agent",
            {"prompt": "read", "description": "child one", "background": True},
        ),
        ToolCall(
            "agent-two",
            "agent",
            {"prompt": "read", "description": "child two", "background": True},
        ),
    ]
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=parent_calls),
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn(tool_calls=[child_call]),
            ScriptedTurn([TextContent("done")]),
            ScriptedTurn([TextContent("done")]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "delegate"})
        first = await _event(reader, "approval_request")
        second = await _event(reader, "approval_request")
        assert first["request_id"] != second["request_id"]
        assert first["tool_call"]["id"] == second["tool_call"]["id"]
        assert first["delegated"] is True
        rejected = await _request(
            reader,
            writer,
            4,
            "approve",
            {"request_id": first["request_id"], "scope": "always_tool"},
        )
        assert rejected[-1]["error"]["code"] == -32602

        first_result = await _request(
            reader, writer, 7, "approve", {"request_id": first["request_id"]}
        )
        second_result = await _request(
            reader, writer, 5, "approve", {"request_id": second["request_id"]}
        )
        assert first_result[-1]["result"]["accepted"] is True
        assert second_result[-1]["result"]["accepted"] is True
        terminal_children: set[str] = set()
        agent_ended = False

        def record_progress(frames: list[dict[str, Any]]) -> None:
            nonlocal agent_ended
            for event in frames:
                params = event.get("params", {})
                if params.get("event") == "agent_end":
                    agent_ended = True
                if params.get("event") != "tool_end":
                    continue
                tool_call = params.get("tool_call") or {}
                result = params.get("tool_result") or {}
                structured = result.get("structured_content") or {}
                if (
                    tool_call.get("id") in {"agent-one", "agent-two"}
                    and params.get("data", {}).get("notification_id")
                    and structured.get("status") != "running"
                ):
                    terminal_children.add(tool_call["id"])

        record_progress(first_result)
        record_progress(second_result)
        while len(terminal_children) < 2:
            frame = await asyncio.wait_for(reader.readline(), TIMEOUT)
            assert frame
            record_progress([json.loads(frame)])
        # Terminal child notifications can precede parent-turn finalization.
        # agent_end is published only after the server marks the turn idle.
        if not agent_ended:
            await _event(reader, "agent_end")
        assert server.runtime.policy is not None
        assert server.runtime.policy.pending_requests() == []
        status = await _request(reader, writer, 6, "status")
        assert status[-1]["result"]["state"] == "idle"
        assert terminal_children == {"agent-one", "agent-two"}
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_and_tui_composition_have_matching_runtime_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "zeta-home"
    home.mkdir()
    (home / "settings.toml").write_text(
        'provider = "codex"\n'
        "token_budget = 12345\n"
        "stream_stall_seconds = 45\n"
        "stream_stall_retries = 4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    tui_calls: list[tuple[object, ...]] = []
    serve_calls: list[tuple[object, ...]] = []
    from tests.support.server_backend import ServerFakeBackend
    from tests.support.tui_backend import FakeInteractiveBackend
    from zeta.cli.main import build_parser
    from zeta.server import runtime as server_runtime
    from zeta.tui import app as tui_app

    def tui_backend(
        provider: str,
        model: str | None,
        *,
        home: Path,
        stall_seconds: float | None = None,
        stall_retries: int | None = None,
        token_budget: int | None = None,
    ) -> tuple[FakeInteractiveBackend, str]:
        tui_calls.append(
            (provider, model, home, stall_seconds, stall_retries, token_budget)
        )
        selected = model or "offline"
        return FakeInteractiveBackend(model=selected), selected

    def serve_backend(
        provider: str,
        model: str | None,
        home: Path,
        *,
        stall_seconds: float | None = None,
        stall_retries: int | None = None,
        require_credentials: bool = False,
        token_budget: int | None = None,
    ) -> tuple[ServerFakeBackend, str]:
        assert not require_credentials
        serve_calls.append(
            (provider, model, home, stall_seconds, stall_retries, token_budget)
        )
        selected = model or "offline"
        return ServerFakeBackend(model=selected), selected

    monkeypatch.setattr(tui_app, "build_backend", tui_backend)
    monkeypatch.setattr(server_runtime, "default_backend", serve_backend)
    monkeypatch.delenv("ZETA_TEST_SCRIPTED_PROVIDER")
    tui = tui_app.create_app(build_parser().parse_args(["--provider", "codex"]))
    server = ZetaServer(home=home, socket_path=_socket_path(tmp_path), provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session")
        tui_loop = tui.loop
        serve_loop = server.runtime.loop
        assert serve_loop is not None
        assert tui_calls == serve_calls
        assert tui_loop.context_assembler.token_budget == serve_loop.context_assembler.token_budget
        assert (
            tui_loop.context_assembler.retained_tail == serve_loop.context_assembler.retained_tail
        )
        assert (tui_loop._background_event_sink is not None) == (
            serve_loop._background_event_sink is not None
        )
    finally:
        await tui.loop.close()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_sigterm_cleans_socket_after_streaming_delta(tmp_path: Path) -> None:
    socket_path = _socket_path(tmp_path)
    script = """
import asyncio
import sys
from zeta.server import ZetaServer, run_server


async def main() -> None:
    server = ZetaServer(
        home=sys.argv[1], socket_path=sys.argv[2], provider="codex"
    )
    await run_server(server)


asyncio.run(main())
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        script,
        str(tmp_path),
        str(socket_path),
        env=environment,
    )
    try:
        for _ in range(100):
            if socket_path.exists():
                break
            await asyncio.sleep(0.01)
        reader, writer = await asyncio.open_unix_connection(str(socket_path))
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "hello"})
        assert (await _event(reader, "assistant_delta"))["delta"]
        process.send_signal(signal.SIGTERM)
        await asyncio.wait_for(process.wait(), TIMEOUT)
        assert process.returncode == 0
        writer.close()
        await writer.wait_closed()
    finally:
        if process.returncode is None:
            process.kill()
            await process.wait()
    assert not socket_path.exists()


async def _ready_extensions(server):
    reader, writer = await _connect(server)
    hello = (await _request(reader, writer, 1, "hello", {"protocol_version": "1.0", "client_version": "1.1"}))[-1]["result"]
    assert hello["protocol_version"] == "1.1"
    assert "send_images" in hello["capabilities"]["requests"]
    session = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]
    return reader, writer, session["session_id"]


@pytest.mark.asyncio
async def test_extensions_negotiate_and_old_clients_remain_unchanged(tmp_path):
    from zeta.server.ergonomics import EXTENSION_REQUESTS
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        hello = (await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"}))[-1]["result"]
        assert hello["protocol_version"] == "1.0"
        assert not set(EXTENSION_REQUESTS) & set(hello["capabilities"]["requests"])
        for method in EXTENSION_REQUESTS:
            assert (await _request(reader, writer, method, method))[-1]["error"]["code"] == -32601
        assert "result" in (await _request(reader, writer, 3, "new_session", {"provider": "codex"}))[-1]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_tree_fork_switch_and_history_persist(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    async def rpc(method, **params):
        return (await _request(reader, writer, method, method, {"session_id": sid, **params}))[-1]
    try:
        assert (await rpc("session_tree"))["result"] == {"branches": []}
        for text in ("first", "second"):
            await _request(reader, writer, text, "send", {"text": text})
            await _event(reader, "agent_end")
        history = (await rpc("session_history"))["result"]
        user = [row for row in history["messages"] if row["role"] == "user"]
        assert [row["content"][0]["text"] for row in user] == ["first", "second"]
        original = (await rpc("session_tree"))["result"]["branches"][0]["id"]
        assert (await rpc("fork_message", message_id="missing"))["error"]["code"] == -32602
        branches = (await rpc("fork_message", message_id=user[0]["id"]))["result"]["branches"]
        assert len(branches) == 2
        assert all(branch["depth"] == 1 for branch in branches)
        assert sum(branch["current"] for branch in branches) == 1
        forked = (await rpc("session_history"))["result"]["messages"]
        assert len(forked) == 1 and forked[0]["id"] == user[0]["id"]
        assert (await rpc("switch_branch", head_id="missing"))["error"]["code"] == -32602
        branches = (await rpc("switch_branch", head_id=original))["result"]["branches"]
        assert sum(branch["current"] for branch in branches) == 1
        restored = (await rpc("session_history"))["result"]["messages"]
        assert restored == history["messages"]
        reopened = SessionManager(tmp_path).open(sid)
        assert [m.to_dict() for m in reopened.store.messages()] == [m.to_dict() for m in server.runtime.loop.store.messages()]
        assert (await rpc("session_history", offset=-1))["error"]["code"] == -32602
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_session_history_hides_empty_turn_nudge(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    try:
        store = server.runtime.opened.store
        expected = []
        for index in range(9):
            if index == 4:
                store.append_message(
                    with_message_origin(Message(
                        MessageRole.USER,
                        [TextContent("hidden recovery prompt")],
                        metadata={"zeta_event": "empty_turn_nudge"},
                    ), MessageOrigin.USER)
                )
            expected.append(
                store.append_message(
                    Message(MessageRole.ASSISTANT, [TextContent(f"visible-{index}")])
                ).id
            )

        first = (
            await _request(
                reader,
                writer,
                "history-1",
                "session_history",
                {"session_id": sid, "offset": 0},
            )
        )[-1]["result"]
        second = (
            await _request(
                reader,
                writer,
                "history-2",
                "session_history",
                {"session_id": sid, "offset": first["next_offset"]},
            )
        )[-1]["result"]

        assert [row["id"] for row in first["messages"]] == expected[:8]
        assert first["next_offset"] == 8
        assert [row["id"] for row in second["messages"]] == expected[8:]
        assert second["next_offset"] is None
        assert "hidden recovery prompt" not in json.dumps([first, second])
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("code", "status_code", "expected_code"),
    [
        ("backend_error", None, "backend_error"),
        ("auth_error", None, "model_access_error"),
        ("model_not_found", None, "model_access_error"),
        ("permission_denied", None, "model_access_error"),
    ],
)
async def test_history_projects_bounded_failed_turn_state(
    tmp_path: Path, code: str, status_code: int | None, expected_code: str
) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    try:
        server.runtime.opened.store.append_message(
            Message(
                MessageRole.ASSISTANT,
                [ThinkingContent("", "signed")],
                metadata={
                    FAILED_TURN_MARKER: True,
                    FAILED_TURN_ERROR: {
                        "code": code,
                        "message": "provider disconnected",
                        "status_code": status_code,
                        "provider_error": True,
                        "secret": "must not cross the protocol",
                    },
                    "fd_diagnostics": {"open_fd_count": 99},
                },
            )
        )
        result = (await _request(
            reader,
            writer,
            "history",
            "session_history",
            {"session_id": sid},
        ))[-1]["result"]
        failed = result["messages"][-1]
        assert failed["failed_turn"] == {
            "code": expected_code,
            "message": "provider disconnected",
            "provider_error": True,
        }
        assert "metadata" not in failed
        assert "secret" not in json.dumps(failed)
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_reconnect_streams_pending_notification_turn(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("notified")])])
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="child done",
    )
    try:
        history = (
            await _request(
                reader,
                writer,
                3,
                "session_history",
                {"session_id": session_id},
            )
        )[-1]["result"]["messages"]
        notification = next(row for row in history if row.get("notification"))
        assert notification["content"][0]["text"] == "child done"

        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        frames = await _request(
            reader,
            writer,
            4,
            "hello",
            {"protocol_version": "1.0"},
        )
        frames += await _frames_until_event(reader, "turn_end")
        receipts = _named_events(frames, "sub_agent_receipt")
        assert receipts[0]["data"]["child_instance_id"] == "child-1"
        assert len(backend.calls) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_session_history_handles_task_legacy_and_unknown_kinds(
    tmp_path: Path,
) -> None:
    # S7: session_history renders task_exited, legacy (no kind), and unknown
    # notification kinds, preserving each kind on the row.
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_task_notification(
        task_id="task-1", command="printf hi", exit_code=0, output_tail="hi"
    )
    store._append_row(
        "notification",
        {
            "child_instance_id": "child-legacy",
            "child_session_path": "agents/1",
            "description": "background child",
            "status": "completed",
            "text": "legacy done",
        },
    )
    store._append_row("notification", {"kind": "monitor_alert", "text": "heads up"})
    try:
        history = (
            await _request(
                reader, writer, 3, "session_history", {"session_id": session_id}
            )
        )[-1]["result"]["messages"]
        rows = [row for row in history if row.get("notification")]
        by_kind = {
            row["notification"].get("kind", "agent_completion"): row for row in rows
        }
        assert set(by_kind) == {"task_exited", "agent_completion", "monitor_alert"}
        assert (
            "background task task-1 exited"
            in by_kind["task_exited"]["content"][0]["text"]
        )
        assert by_kind["agent_completion"]["content"][0]["text"] == "legacy done"
        assert by_kind["monitor_alert"]["content"][0]["text"] == "heads up"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_live_notification_events_dispatch_on_kind(tmp_path: Path) -> None:
    # S7: live notification events dispatch task_exited as task_exit_notification
    # and every other kind (including unknown) as sub_agent_receipt.
    backend = FakeBackend([ScriptedTurn([TextContent("notified")])])
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_task_notification(
        task_id="task-1", command="printf hi", exit_code=0, output_tail="hi"
    )
    store.append_agent_notification(
        "child-1",
        child_session_path="/tmp/child-1",
        description="child",
        status="completed",
        text="child done",
    )
    store._append_row("notification", {"kind": "monitor_alert", "text": "heads up"})
    try:
        writer.close()
        await writer.wait_closed()
        await asyncio.sleep(0.05)
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        frames = await _request(
            reader, writer, 4, "hello", {"protocol_version": "1.0"}
        )
        frames += await _frames_until_event(reader, "turn_end")
        events = [
            frame["params"]
            for frame in frames
            if frame.get("params", {}).get("event")
            in {"task_exit_notification", "sub_agent_receipt"}
        ]
        task_events = [e for e in events if e["event"] == "task_exit_notification"]
        receipt_events = [e for e in events if e["event"] == "sub_agent_receipt"]
        assert len(task_events) == 1
        assert task_events[0]["data"]["task_id"] == "task-1"
        assert task_events[0]["data"]["kind"] == "task_exited"
        assert {
            e["data"].get("kind", "agent_completion") for e in receipt_events
        } == {"agent_completion", "monitor_alert"}
        assert len(backend.calls) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("block_count,text", [(0, ""), (20, "\x00" * 8000), (130, "x" * 8000)],
                         ids=["large-tools", "json-escaping", "near-limit-message"])
async def test_history_pages_large_messages_with_bounded_tool_arguments(tmp_path, block_count, text):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    # The default asyncio reader limit is only 64 KiB.
    reader._limit = MAX_FRAME_BYTES
    try:
        store = server.runtime.opened.store
        entries = []
        for index in range(17):
            message = Message(MessageRole.ASSISTANT, [
                *[TextContent(text) for _ in range(block_count)],
                ToolUseContent(ToolCall(str(index), "write", {"content": "x" * 135_000})),
            ])
            if not block_count:
                assert FrameCodec().response_fits("message", message.to_dict())
            entries.append(store.append_message(message))
        offset = 0
        rows = []
        page_sizes = []
        # Control characters exercise the largest legal encoded request id.
        request_id = "\x00" * MAX_REQUEST_ID_BYTES
        while offset is not None:
            response = (await _request(reader, writer, request_id, "session_history", {
                "session_id": sid, "offset": offset,
            }))[-1]
            assert "error" not in response
            page = response["result"]
            encoded = FrameCodec().response(request_id, page)
            assert len(encoded) <= MAX_FRAME_BYTES
            if block_count == 130:
                assert len(encoded) > MAX_FRAME_BYTES - 10_000
            assert page["messages"]
            page_sizes.append(len(page["messages"]))
            rows.extend(page["messages"])
            next_offset = page["next_offset"]
            assert next_offset is None or next_offset == offset + len(page["messages"])
            offset = next_offset
        assert [row["id"] for row in rows] == [entry.id for entry in entries]
        assert page_sizes == ([8, 8, 1] if not block_count else [1] * 17)
        for index, row in enumerate(rows):
            assert row["content"][:-1] == [{"type": "text", "text": text}] * block_count
            assert row["content"][-1] == {
                "type": "tool_use", "tool_call": {"id": str(index), "name": "write", "arguments": {}},
            }
        assert all(message.content[-1].tool_call.arguments == {"content": "x" * 135_000}
                   for message in store.messages())
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_history_advances_past_single_oversized_persisted_message(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    try:
        store = server.runtime.opened.store
        oversized = store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("x" * 8000)] * 140), MessageOrigin.USER))
        following = store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("after")]), MessageOrigin.USER))
        response = (await _request(reader, writer, "history", "session_history", {"session_id": sid}))[-1]
        assert "error" not in response
        page = response["result"]
        assert page["next_offset"] is None
        assert [row["id"] for row in page["messages"]] == [oversized.id, following.id]
        assert "truncated" in page["messages"][0]["content"][0]["text"]
        assert page["messages"][1]["content"] == [{"type": "text", "text": "after"}]
        assert len(store.messages()[0].content) == 140
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_settings_apply_to_active_session_and_resume(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    async def rpc(method, **params):
        return (await _request(reader, writer, method, method, {"session_id": sid, **params}))[-1]
    try:
        catalog = (await rpc("model_catalog"))["result"]
        assert catalog["providers"]["gpt-5.6-luna"] == "codex"
        assert (await rpc("session_settings"))["result"] == {
            "model": "gpt-5.6-luna",
            "approval_mode": "ask",
        }
        for model, mode in (("invalid", "ask"), ("gpt-5.6-luna", "invalid")):
            assert (await rpc("set_settings", model=model, approval_mode=mode))["error"]["code"] == -32602
        settings = {"model": "gpt-5.6-sol", "approval_mode": "deny"}
        assert (await rpc("set_settings", **settings))["result"] == settings
        assert server.runtime.metadata.model_fallback == (
            "codex", "gpt-5.6-luna", 1_050_000
        )
        assert server.runtime.loop.backend.model == "gpt-5.6-sol"
        assert server.runtime.policy.default.value == "deny"
        await _request(reader, writer, "new", "new_session", {"provider": "codex"})
        assert (await rpc("set_settings", **settings))["error"]["code"] == -32003
        assert server.runtime.policy.default.value == "ask"
        await _request(reader, writer, "resume", "resume", {"session_id": sid})
        assert (await rpc("session_settings"))["result"] == settings
        assert not (tmp_path / "settings.json").exists()
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_new_session_resume_and_status_report_effective_yolo_mode(tmp_path):
    # ZETA-131 round 3: with settings.toml yolo=true, composition wires
    # the live policy default to `allow`, but the stored session metadata
    # keeps `approval_mode = None` until an explicit `set_settings`
    # writes it. The wire responses for `new_session`, `resume`, and
    # `status` must project the RUNNING policy default so the frontend client header
    # indicator can paint the auto-approve state — a raw `to_dict()`
    # emits `approval_mode: null`, which the frontend client treats as "no update"
    # and the indicator stays hidden.
    (tmp_path / "settings.toml").write_text("yolo = true\n", encoding="utf-8")
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        first = (await _request(reader, writer, 2, "new_session", {"provider": "codex"}))[-1]["result"]["session"]
        sid_a = first["session_id"]
        assert first["approval_mode"] == "allow"
        # A first `status` right after new_session — the production order.
        status_a = (await _request(reader, writer, 3, "status"))[-1]["result"]
        assert status_a["session"]["session_id"] == sid_a
        assert status_a["session"]["approval_mode"] == "allow"
        # Switch: a second `new_session` returns a distinct id, same mode.
        second = (await _request(reader, writer, 4, "new_session", {"provider": "codex"}))[-1]["result"]["session"]
        sid_b = second["session_id"]
        assert sid_b != sid_a
        assert second["approval_mode"] == "allow"
        # Resume the ORIGINAL session — its stored metadata still has
        # `approval_mode: null` on disk (we never called `set_settings`).
        # The resume response must surface the effective policy anyway.
        stored = SessionManager(tmp_path).read_metadata(sid_a)
        assert stored.approval_mode is None
        resumed = (await _request(reader, writer, 5, "resume", {"session_id": sid_a}))[-1]["result"]["session"]
        assert resumed["session_id"] == sid_a
        assert resumed["approval_mode"] == "allow"
        # A trailing `status` (the periodic poll after switch/reconnect)
        # also carries the effective mode.
        status_final = (await _request(reader, writer, 6, "status"))[-1]["result"]
        assert status_final["session"]["session_id"] == sid_a
        assert status_final["session"]["approval_mode"] == "allow"
        # And the disk metadata is UNCHANGED — the projection is a
        # read-side transform, not a write-through.
        assert SessionManager(tmp_path).read_metadata(sid_a).approval_mode is None
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_images_persist_forward_and_reject_invalid_input(tmp_path):
    import base64

    from zeta.protocol.types import ImageContent
    from zeta.server.ergonomics import MAX_IMAGE_BYTES
    backend = FakeBackend([ScriptedTurn(content=[TextContent("seen")])])
    server = ZetaServer(provider="codex", home=tmp_path, port=0, backend_factory=lambda provider, model, home: (backend, model or "offline"))
    reader, writer, sid = await _ready_extensions(server)
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    image = {"name": "test.png", "mime_type": "image/png", "data": base64.b64encode(png).decode()}
    async def send(images, session_id=sid):
        return (await _request(reader, writer, "images", "send_images", {"session_id": session_id, "text": "inspect", "images": images}))[-1]
    try:
        assert (await send([image], "missing"))["error"]["code"] == -32003
        for invalid in ({**image, "name": "../escape.png"}, {**image, "data": "!"}, {**image, "mime_type": "text/plain"}, {**image, "data": base64.b64encode(b'x').decode()}, {**image, "data": base64.b64encode(png + b'x' * MAX_IMAGE_BYTES).decode()}):
            assert (await send([invalid]))["error"]["code"] == -32602
        assert not (server.runtime.opened.store.session_dir / "attachments").exists()
        assert (await send([image]))["result"]["accepted"]
        await _event(reader, "agent_end")
        message = server.runtime.loop.store.messages()[0]
        attachment = next(block for block in message.content if isinstance(block, ImageContent))
        assert attachment.size == len(png)
        assert Path(attachment.path).read_bytes() == png
        assert Path(attachment.path).is_relative_to(server.runtime.opened.store.session_dir)
        assert attachment in backend.calls[0][0][-1].content
        history = (await _request(reader, writer, "history", "session_history", {"session_id": sid}))[-1]["result"]["messages"]
        assert history[0]["content"][1] == {"type": "attachment", "name": "test.png", "size": len(png)}
        assert "data" not in history[0]["content"][1]
        assert SessionManager(tmp_path).open(sid).store.messages()[0] == message
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("failure", [None, "backend", "metadata"])
async def test_settings_retune_budget_atomically(tmp_path, monkeypatch, pinned, failure):
    from zeta.core.slash import MODEL_CONTEXT_WINDOWS

    monkeypatch.setitem(
        MODEL_CONTEXT_WINDOWS,
        "codex",
        {"gpt-5.6-luna": 1_050_000, "gpt-5.6-sol": 400_000},
    )
    if pinned:
        (tmp_path / "settings.toml").write_text("token_budget = 123456\n")
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    runtime = server.runtime
    metadata_before = runtime.metadata.to_dict()
    old_budget = 123456 if pinned else 1_050_000
    assert runtime.loop.context_assembler.token_budget == old_budget
    try:
        with monkeypatch.context() as patch:
            if failure == "backend":
                original = runtime.loop.set_model

                def fail_model(model):
                    original(model)
                    if model == "gpt-5.6-sol":
                        raise OSError("partial backend change")

                patch.setattr(runtime.loop, "set_model", fail_model)
            elif failure == "metadata":
                import zeta.core.session as session_module
                original = session_module.os.replace

                def fail_metadata(source, destination, **kwargs):
                    if Path(destination).name == "meta.json":
                        raise OSError("metadata persistence failed")
                    return original(source, destination, **kwargs)

                patch.setattr(session_module.os, "replace", fail_metadata)
            response = (await _request(reader, writer, "settings", "set_settings", {
                "session_id": sid, "model": "gpt-5.6-sol", "approval_mode": "deny",
            }))[-1]
        if failure:
            assert response["error"]["code"] == -32000
            assert runtime.metadata.to_dict() == metadata_before
            assert runtime.loop.backend.model == "gpt-5.6-luna"
            assert runtime.loop.context_assembler.token_budget == old_budget
            assert runtime.policy.default.value == "ask"
            assert SessionManager(tmp_path).open(sid).metadata.to_dict() == metadata_before
        else:
            assert response["result"] == {"model": "gpt-5.6-sol", "approval_mode": "deny"}
            expected_budget = 123456 if pinned else 400_000
            assert runtime.loop.context_assembler.token_budget == expected_budget
            assert runtime.metadata.compaction_budget == expected_budget
            assert runtime.metadata.budget_pinned == pinned
            assert SessionManager(tmp_path).open(sid).metadata.compaction_budget == expected_budget
            await _request(reader, writer, "resume", "resume", {"session_id": sid})
            assert runtime.loop.context_assembler.token_budget == expected_budget
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_server_new_session_defaults_to_evict(tmp_path: Path) -> None:
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(tmp_path, provider="codex")
    try:
        metadata = await runtime.create_session()

        assert not hasattr(metadata, "compaction")
        assert runtime.loop is not None
        assert "recall_history" in runtime.loop.tool_registry.registered_names
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_image_names_match_verified_types(tmp_path):
    import base64

    from zeta.server.ergonomics import DIRECTION_CONTROLS, image_message
    from zeta.server.protocol import ProtocolError
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(tmp_path, provider="codex")
    await runtime.create_session()
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    item = {"name": "safe.png", "mime_type": "image/png", "data": base64.b64encode(png).decode()}
    invalid_names = ["../bad.png", "..\\bad.png", "bad/path.png", "bad\\path.png", "bad.jpg", "bad", "bad.png.exe", "bad\x7f.png"]
    invalid_names += [f"bad{control}.png" for control in DIRECTION_CONTROLS]
    try:
        for name in invalid_names:
            with pytest.raises(ProtocolError):
                image_message(runtime, {"images": [{**item, "name": name}]})
        attachments = runtime.opened.store.session_dir / "attachments"
        assert not attachments.exists()
        for mime, raw, names in [
            ("image/png", png, ["safe.PNG", "写真.png"]),
            ("image/jpeg", b"\xff\xd8\xff\xe0", ["safe.jpg", "safe.JPEG"]),
            ("image/gif", b"GIF89a\x01\x00\x01\x00", ["safe.gif"]),
            ("image/webp", b"RIFF" + (22).to_bytes(4, "little") + b"WEBPVP8X" + (10).to_bytes(4, "little") + bytes(10), ["safe.webp"]),
        ]:
            for name in names:
                message = image_message(runtime, {"images": [{"name": name, "mime_type": mime, "data": base64.b64encode(raw).decode()}]})
                assert Path(message.content[1].path).name == name
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["write", "flush", "replace"])
@pytest.mark.parametrize("fail_at", [1, 2])
@pytest.mark.parametrize("existing", [False, True])
async def test_attachment_failure_removes_entire_batch(tmp_path, monkeypatch, failure, fail_at, existing):
    import base64
    from contextlib import contextmanager

    from zeta.server import ergonomics
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(tmp_path, provider="codex")
    await runtime.create_session()
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    item = {"name": "same.png", "mime_type": "image/png", "data": base64.b64encode(png).decode()}
    attachments = runtime.opened.store.session_dir / "attachments"
    if existing:
        ergonomics.image_message(runtime, {"images": [item]})
    before = set(attachments.rglob("*"))
    original_open = os.fdopen
    original_replace = ergonomics.os.replace
    writes = 0
    replacements = 0

    @contextmanager
    def failing_open(fd, *args, **kwargs):
        nonlocal writes
        with original_open(fd, *args, **kwargs) as handle:
            if args == ("wb",):
                writes += 1
                if writes == fail_at:
                    from unittest.mock import Mock
                    wrapper = Mock(wraps=handle)
                    if failure == "write":
                        def fail_write(raw):
                            handle.write(raw[:4])
                            handle.flush()
                            raise OSError("mid-write failure")
                        wrapper.write.side_effect = fail_write
                        yield wrapper
                        return
                    if failure == "flush":
                        wrapper.flush.side_effect = OSError("flush failure")
                        yield wrapper
                        return
            yield handle

    def failing_replace(source, destination, **kwargs):
        nonlocal replacements
        from zeta.core.session_files import read_session_file
        assert read_session_file(kwargs["src_dir_fd"], source) == png
        assert destination not in os.listdir(kwargs["dst_dir_fd"])
        replacements += 1
        if failure == "replace" and replacements == fail_at:
            raise OSError("replace failure")
        return original_replace(source, destination, **kwargs)

    try:
        with monkeypatch.context() as patch:
            patch.setattr(os, "fdopen", failing_open)
            patch.setattr(ergonomics.os, "replace", failing_replace)
            with pytest.raises(OSError):
                ergonomics.image_message(runtime, {"images": [item, item]})
        assert attachments.exists() == existing
        assert set(attachments.rglob("*")) == before
        assert all(path.read_bytes() == png for path in before if path.is_file())
        assert runtime.opened.store.messages() == []
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["claude", "codex"])
async def test_real_catalog_is_complete_for_real_launch(tmp_path, provider):
    from zeta.models.catalog import PROVIDER_MODELS, known_model_names

    server = ZetaServer(home=tmp_path, port=0, provider=provider,
                        backend_factory=lambda p, m, h: (FakeBackend([]), m or "offline"))
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        sid = server.runtime.session_id
        result = (await _request(reader, writer, 3, "model_catalog", {"session_id": sid}))[-1]["result"]
        assert result["models"] == known_model_names()
        assert result["providers"] == {m: p for p, models in PROVIDER_MODELS.items() for m in models}
        assert not {"offline", "faster"} & set(result["models"])
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("pinned", [False, True])
@pytest.mark.parametrize("failure", [None, "backend", "metadata"])
async def test_cross_provider_settings_preserve_session_and_budget(tmp_path, monkeypatch, pinned, failure):
    from zeta.core.slash import budget_for_model

    if pinned:
        (tmp_path / "settings.toml").write_text("token_budget = 123456\n")
    builds = []

    def build(provider, model, home):
        builds.append((provider, model, home))
        if provider == "codex" and failure == "backend":
            raise RuntimeError("target backend failed")
        return FakeBackend([ScriptedTurn([TextContent(provider)])]), model

    server = ZetaServer(home=tmp_path, port=0, provider="claude", model="claude-sonnet-4-6", backend_factory=build)
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        runtime = server.runtime
        sid = runtime.session_id
        state, loop, opened = runtime.state, runtime.loop, runtime.opened
        backend = loop.backend
        metadata_before = runtime.metadata.to_dict()
        budget_before = loop.context_assembler.token_budget
        opened.store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent("keep history")]), MessageOrigin.USER))
        history = opened.store.messages()
        runtime.usage["input_tokens"] = 123
        with monkeypatch.context() as patch:
            if failure == "metadata":
                def fail_metadata(*args, **kwargs):
                    raise OSError("metadata persistence failed")
                patch.setattr(runtime.manager, "record_session_settings", fail_metadata)
            response = (await _request(reader, writer, 3, "set_settings", {
                "session_id": sid, "model": "gpt-5.4", "approval_mode": "deny",
            }))[-1]
        assert builds[-1] == ("codex", "gpt-5.4", tmp_path)
        assert runtime.state is state and runtime.loop is loop and runtime.opened is opened
        assert opened.store.messages() == history
        assert runtime.usage == {"input_tokens": 123}
        if failure:
            assert response["error"]["code"] == -32000
            assert runtime.metadata.to_dict() == metadata_before
            assert SessionManager(tmp_path).open(sid).metadata.to_dict() == metadata_before
            assert loop.backend is backend
            assert loop.context_assembler.backend is backend
            assert loop.context_assembler.token_budget == budget_before
            assert runtime.policy.default.value == "ask"
        else:
            assert response["result"] == {"model": "gpt-5.4", "approval_mode": "deny"}
            assert loop.backend is not backend
            assert loop.context_assembler.backend is loop.backend
            assert loop.context_assembler.compaction_policy.backend is loop.backend
            expected = 123456 if pinned else budget_for_model("codex", "gpt-5.4")
            assert loop.context_assembler.token_budget == expected
            assert runtime.metadata.compaction_budget == expected
            assert runtime.metadata.budget_pinned == pinned
            assert runtime.metadata.provider == "codex"
            assert runtime.metadata.override_audit[-1]["provider"] == {"from": "claude", "to": "codex"}
            await _request(reader, writer, 4, "send", {"text": "use new backend"})
            await _event(reader, "agent_end")
            assert runtime.opened.store.messages()[-1].content == [TextContent("codex")]
            await _request(reader, writer, 5, "resume", {"session_id": sid})
            assert runtime.provider == "codex" and runtime.model == "gpt-5.4"
            assert runtime.loop.context_assembler.token_budget == expected
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_cross_provider_settings_reject_mid_turn_then_apply_when_idle(tmp_path):
    from tests.support.server_backend import ServerFakeBackend

    server = ZetaServer(home=tmp_path, port=0, provider="claude", model="claude-sonnet-4-6",
                        backend_factory=lambda p, m, h: (ServerFakeBackend(delay=0.2, model=m), m))
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        params = {"session_id": server.runtime.session_id, "model": "gpt-5.4", "approval_mode": "ask"}
        backend = server.runtime.loop.backend
        await _request(reader, writer, 3, "send", {"text": "hello"})
        response = (await _request(reader, writer, 4, "set_settings", params))[-1]
        assert response["error"]["code"] == -32004
        assert server.runtime.loop.backend is backend
        await _request(reader, writer, 5, "abort")
        assert "result" in (await _request(reader, writer, 6, "set_settings", params))[-1]
        assert server.runtime.provider == "codex"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ["claude", "codex"])
async def test_cross_provider_missing_credentials_returns_rpc_error(tmp_path, monkeypatch, target):
    from zeta.providers.anthropic import AnthropicCredentialStore
    from zeta.providers.auth import OAuthTokens
    from zeta.providers.codex import CodexCredentialStore

    monkeypatch.delenv("ZETA_ALLOW_API_KEY", raising=False)
    monkeypatch.delenv("ZETA_TEST_SCRIPTED_PROVIDER")
    monkeypatch.setattr(AnthropicCredentialStore, "bootstrap", lambda self: None)
    monkeypatch.setattr(CodexCredentialStore, "bootstrap", lambda self: None)
    source, source_model, model = ("claude", "claude-sonnet-4-6", "gpt-5.4") if target == "codex" else ("codex", "gpt-5.4", "claude-sonnet-4-6")
    server = ZetaServer(home=tmp_path, port=0, provider=source, model=source_model)
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        runtime = server.runtime
        before = runtime.metadata.to_dict()
        backend = runtime.loop.backend
        response = (await _request(reader, writer, 3, "set_settings", {
            "session_id": runtime.session_id, "model": model, "approval_mode": "deny",
        }))[-1]
        assert response["error"]["code"] == -32000
        assert "login" in response["error"]["message"]
        assert runtime.metadata.to_dict() == before
        assert runtime.loop.backend is backend
        assert "result" in (await _request(reader, writer, 4, "status"))[-1]
        credential_type = AnthropicCredentialStore if target == "claude" else CodexCredentialStore
        credential_path = "anthropic-oauth.json" if target == "claude" else "codex-oauth.json"
        credential_type(tmp_path / credential_path).save(OAuthTokens("test-access", "test-refresh", 4_000_000_000))
        response = (await _request(reader, writer, 5, "set_settings", {
            "session_id": runtime.session_id, "model": model, "approval_mode": "deny",
        }))[-1]
        assert response["result"] == {"model": model, "approval_mode": "deny"}
        assert runtime.provider == target
        assert runtime.loop.backend is not backend
        assert isinstance(runtime.loop.backend.token_store, credential_type)
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("server_provider", ["claude", "codex"])
async def test_removed_provider_resume_is_rejected_without_session_mutation(
    tmp_path: Path, server_provider: str
) -> None:
    opened = SessionManager(tmp_path).create(
        provider="fake", model="offline", cwd=tmp_path
    )
    with opened.store.path.open("ab") as stream:
        stream.write(b'{"type":')
    before_files = {
        path: path.read_bytes()
        for path in opened.store.session_dir.rglob("*")
        if path.is_file()
    }
    builds: list[str] = []

    def build(provider: str, model: str | None, _home: Path):
        builds.append(provider)
        return FakeBackend([ScriptedTurn([TextContent("still here")])]), (
            model or "gpt-5.6-luna"
        )

    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider=server_provider,
        backend_factory=build,
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        for request_id in (2, 4):
            state = server.runtime.state
            before_builds = list(builds)
            response = (
                await _request(
                    reader,
                    writer,
                    request_id,
                    "resume",
                    {"session_id": opened.metadata.session_id},
                )
            )[-1]
            assert response["error"] == {
                "code": -32602,
                "message": "the fake provider was removed; choose claude, codex or ollama",
            }
            assert server.runtime.state is state
            assert builds == before_builds
            assert {
                path: path.read_bytes() for path in before_files
            } == before_files
            if state is None:
                await _request(reader, writer, 3, "new_session")
        assert (await _request(reader, writer, 5, "send", {"text": "still here"}))[-1][
            "result"
        ]["accepted"]
        await _event(reader, "turn_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("server_provider", ["claude", "codex"])
async def test_session_listing_includes_real_and_removed_provider_sessions(
    tmp_path: Path, server_provider: str
) -> None:
    manager = SessionManager(tmp_path)
    sessions = {
        provider: manager.create(provider=provider, model=model, cwd=tmp_path).metadata.session_id
        for provider, model in (
            ("fake", "offline"),
            ("claude", "claude-sonnet-4-6"),
            ("codex", "gpt-5.4"),
        )
    }
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider=server_provider,
        backend_factory=lambda _provider, model, _home: (
            FakeBackend([]),
            model or "gpt-5.6-luna",
        ),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        response = (await _request(reader, writer, 2, "list_sessions"))[-1]["result"]
        assert {row["session_id"] for row in response["sessions"]} == set(
            sessions.values()
        )
        rejected = (
            await _request(
                reader,
                writer,
                3,
                "resume",
                {"session_id": sessions["fake"]},
            )
        )[-1]
        assert rejected["error"]["message"] == (
            "the fake provider was removed; choose claude, codex or ollama"
        )
        for request_id, provider in enumerate(("claude", "codex"), start=4):
            resumed = (
                await _request(
                    reader,
                    writer,
                    request_id,
                    "resume",
                    {"session_id": sessions[provider]},
                )
            )[-1]
            assert resumed["result"]["session"]["session_id"] == sessions[provider]
    finally:
        await _close(server, writer)


@pytest.mark.parametrize(
    ("settings_text", "expected_provider"),
    [
        ('model = "gpt-5.6-luna"\n', "codex"),
        ('provider = "claude"\n', "claude"),
    ],
)
def test_plain_serve_resolves_provider_from_settings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    settings_text: str,
    expected_provider: str,
) -> None:
    from zeta import server as server_module
    from zeta.cli.main import main as cli_main
    from zeta.models.catalog import PROVIDER_MODELS, known_model_names

    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text(settings_text)
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)
    legacy = SessionManager(home).create(
        provider="fake", model="offline", cwd=tmp_path
    ).metadata.session_id

    async def check_server(server: ZetaServer) -> None:
        server.socket_path = _socket_path(tmp_path)
        server.runtime.backend_factory = lambda _provider, model, _home: (
            FakeBackend([]),
            model or "claude-sonnet-4-6",
        )
        reader, writer = await _connect(server)
        try:
            await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
            created = (await _request(reader, writer, 2, "new_session"))[-1]["result"]["session"]
            assert created["provider"] == expected_provider
            sid = created["session_id"]
            catalog = (
                await _request(reader, writer, 3, "model_catalog", {"session_id": sid})
            )[-1]["result"]
            assert catalog == {
                "models": known_model_names(),
                "providers": {
                    model: provider
                    for provider, models in PROVIDER_MODELS.items()
                    for model in models
                },
            }
            listed = (await _request(reader, writer, 4, "list_sessions"))[-1]["result"]
            assert {row["session_id"] for row in listed["sessions"]} == {sid, legacy}
            rejected = (
                await _request(reader, writer, 5, "resume", {"session_id": legacy})
            )[-1]
            assert rejected["error"]["message"] == (
                "the fake provider was removed; choose claude, codex or ollama"
            )
            assert server.runtime.session_id == sid
        finally:
            await _close(server, writer)

    monkeypatch.setattr(server_module, "run_server", check_server)
    assert cli_main(["serve"]) == 0


@pytest.mark.asyncio
async def test_session_list_includes_single_line_first_message_preview(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "first\nmessage\tpreview"})
        await _event(reader, "turn_end")
        await _request(reader, writer, 4, "send", {"text": "later message"})
        await _event(reader, "turn_end")
        result = (await _request(reader, writer, 5, "list_sessions"))[-1]["result"]
        assert result["sessions"][0]["first_message_preview"] == "first message preview"
        assert "name" not in result["sessions"][0]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("start_provider,start_model", [
    ("codex", "gpt-5.4"), ("claude", "claude-sonnet-4-6"),
])
@pytest.mark.parametrize("restart", [False, True])
@pytest.mark.parametrize("raise_error", [False, True])
@pytest.mark.parametrize("status_code,detail", [
    (400, "Unsupported subscription"),
    (404, "Resource does not exist"),
    (403, "Access denied for this account"),
])
async def test_unusable_model_reverts_durably_and_next_send_works(
    tmp_path, start_provider, start_model, restart, raise_error, status_code, detail,
):
    from zeta.protocol.types import ErrorInfo, StreamEvent, StreamEventType

    completions = []

    class AccountBackend(FakeBackend):
        async def complete(self, messages, tool_schemas):
            completions.append(self.model)
            if self.model == "gpt-5.4-mini":
                if raise_error:
                    from zeta.providers.codex_errors import CodexHTTPError
                    raise CodexHTTPError(detail, status_code=status_code)
                yield StreamEvent(StreamEventType.ERROR, error=ErrorInfo(
                    "http_error", detail, status_code=status_code,
                ))
            else:
                async for event in super().complete(messages, tool_schemas):
                    yield event

    def build(provider, model, home):
        backend = AccountBackend([ScriptedTurn([TextContent("usable model")])])
        backend.model = model
        return backend, model

    def make_server():
        return ZetaServer(home=tmp_path, port=0, provider=start_provider,
                          model=start_model, backend_factory=build)

    server = make_server()
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        runtime = server.runtime
        sid = runtime.session_id
        old_budget = runtime.metadata.compaction_budget
        for model in ["gpt-5.4-mini", "gpt-5.4", "gpt-5.4-mini"]:
            applied = (await _request(reader, writer, 3, "set_settings", {
                "session_id": sid, "model": model, "approval_mode": "deny",
            }))[-1]
            assert "result" in applied
            assert completions == [], "Apply must never run a completion"
            expected = None if model == start_model else (start_provider, start_model, old_budget)
            assert runtime.metadata.model_fallback == expected
        if restart:
            await _close(server, writer)
            server = make_server()
            reader, writer = await _connect(server)
            await _request(reader, writer, 4, "hello", {"protocol_version": "1.1"})
            await _request(reader, writer, 5, "resume", {"session_id": sid})
            runtime = server.runtime
        await _request(reader, writer, 6, "send", {"text": "first request"})
        failure = await _event(reader, "error")
        assert failure["error"]["code"] == "model_reverted"
        assert f"Model reverted to {start_model} ({start_provider})" in failure["error"]["message"]
        assert detail in failure["error"]["message"]
        await _event(reader, "agent_end")
        metadata = SessionManager(tmp_path).open(sid).metadata
        assert (metadata.provider, metadata.model) == (start_provider, start_model)
        assert metadata.model_fallback is None
        assert metadata.compaction_budget == old_budget
        assert runtime.loop.context_assembler.token_budget == old_budget
        assert runtime.loop.backend is runtime.loop.context_assembler.backend
        assert runtime.loop.backend is runtime.loop.context_assembler.compaction_policy.backend
        assert runtime.policy.default.value == "deny"
        await _request(reader, writer, 7, "send", {"text": "next request"})
        message = await _event(reader, "assistant_message")
        assert message["message"]["content"][0]["text"] == "usable model"
        await _event(reader, "agent_end")
        assert completions == ["gpt-5.4-mini", start_model]
    finally:
        await _close(server, writer)


@pytest.mark.parametrize("code,status_code,provider_error,expected", [
    ("auth_error", None, True, True),
    ("http_error", 400, True, True),
    ("http_error", 403, True, True),
    ("http_error", 404, True, True),
    ("permission_denied", None, True, True),
    ("model_not_found", None, True, True),
    ("http_error", 429, True, False),
    ("http_error", 500, True, False),
    ("transport_error", None, True, False),
    ("auth_error", 403, False, False),
    ("http_error", None, True, False),
])
def test_model_entitlement_error_classification(code, status_code, provider_error, expected):
    from zeta.protocol.types import ErrorInfo
    from zeta.server.model_selection import entitlement_error

    # Wording must never determine classification, even when it suggests recovery.
    error = ErrorInfo(code, "Model unavailable: credentials required", status_code, provider_error)
    assert entitlement_error(error) is expected


@pytest.mark.asyncio
async def test_successful_selection_clears_fallback_and_background_errors_do_not_revert(tmp_path):
    from zeta.protocol.types import ErrorInfo, StreamEvent, StreamEventType

    server = ZetaServer(
        home=tmp_path, port=0, provider="claude", model="claude-sonnet-4-6",
        backend_factory=lambda provider, model, home: (
            FakeBackend([ScriptedTurn([TextContent("success")])]), model,
        ),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        runtime = server.runtime
        sid = runtime.session_id
        await _request(reader, writer, 3, "set_settings", {
            "session_id": sid, "model": "gpt-5.4", "approval_mode": "ask",
        })
        pending = runtime.metadata.model_fallback
        assert pending
        client = server._client
        await client._event(StreamEvent(StreamEventType.ERROR, error=ErrorInfo("auth_error", "child login", provider_error=True)),
                            session_id=sid, background=True)
        await _event(reader, "error")
        assert runtime.metadata.model_fallback == pending
        await client._event(StreamEvent(StreamEventType.ERROR, error=ErrorInfo("transport_error", "connection reset")),
                            session_id=sid)
        await _event(reader, "error")
        assert runtime.metadata.model_fallback == pending
        await _request(reader, writer, 4, "send", {"text": "confirm selection"})
        await _event(reader, "agent_end")
        assert runtime.metadata.model_fallback is None
        assert SessionManager(tmp_path).open(sid).metadata.model_fallback is None
        await client._event(StreamEvent(StreamEventType.ERROR, error=ErrorInfo("auth_error", "expired later", provider_error=True)),
                            session_id=sid)
        await _event(reader, "error")
        assert runtime.model == "gpt-5.4"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("version", ["1.0", "1.1"])
async def test_recovery_storage_and_error_codes_respect_wire_version(tmp_path, version):
    from zeta.protocol.types import ErrorInfo, StreamEvent, StreamEventType
    from zeta.server import model_selection

    server = ZetaServer(
        home=tmp_path, port=0, provider="claude", model="claude-sonnet-4-6",
        backend_factory=lambda provider, model, home: (FakeBackend([]), model),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": version})
        created = (await _request(reader, writer, 2, "new_session"))[-1]["result"]
        assert "model_fallback" not in created["session"]
        runtime = server.runtime
        sid = runtime.session_id
        model_selection.apply(runtime, "gpt-5.4", "ask")
        assert runtime.metadata.model_fallback is not None
        resumed = (await _request(reader, writer, 3, "resume", {"session_id": sid}))[-1]["result"]
        status = (await _request(reader, writer, 4, "status"))[-1]["result"]
        listed = (await _request(reader, writer, 5, "list_sessions"))[-1]["result"]
        for session in [resumed["session"], status["session"], *listed["sessions"]]:
            assert "model_fallback" not in session
        assert runtime.metadata.model_fallback is not None
        info = ErrorInfo("http_error", "Resource absent", status_code=404, provider_error=True)
        # Event persistence must retain structured information as well.
        event = StreamEvent.from_dict(StreamEvent(StreamEventType.ERROR, error=info).to_dict())
        assert event.error == info
        await server._client._event(event, session_id=sid)
        failure = (await _event(reader, "error"))["error"]
        assert set(failure) == {"code", "message"}
        if version == "1.0":
            assert failure == {"code": "http_error", "message": "Resource absent"}
        else:
            assert failure["code"] == "model_reverted"
        assert runtime.model == "claude-sonnet-4-6"
        assert runtime.metadata.model_fallback is None
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_return_to_fallback_does_not_restore_itself(tmp_path):
    from zeta.providers.codex_errors import CodexHTTPError

    class DeniedBackend(FakeBackend):
        async def complete(self, messages, tool_schemas):
            raise CodexHTTPError("Access denied", status_code=403)
            yield  # pragma: no cover - keep the provider's async iterator contract

    server = ZetaServer(
        home=tmp_path, port=0, provider="codex", model="gpt-5.4",
        backend_factory=lambda provider, model, home: (DeniedBackend([]), model),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        sid = server.runtime.session_id
        for model in ["gpt-5.4-mini", "gpt-5.4"]:
            await _request(reader, writer, 3, "set_settings", {
                "session_id": sid, "model": model, "approval_mode": "ask",
            })
        assert server.runtime.metadata.model_fallback is None
        assert SessionManager(tmp_path).open(sid).metadata.model_fallback is None
        await _request(reader, writer, 4, "send", {"text": "try returned selection"})
        failure = (await _event(reader, "error"))["error"]
        assert failure == {"code": "model_access_error", "message": "Access denied"}
        await _event(reader, "agent_end")
        assert server.runtime.model == "gpt-5.4"
        assert server.runtime.metadata.model_fallback is None
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_mcp_auth_failure_does_not_revert_model(tmp_path, monkeypatch):
    from zeta.providers.codex_errors import CodexAuthError
    from zeta.server import model_selection

    server = ZetaServer(
        home=tmp_path, port=0, provider="claude", model="claude-sonnet-4-6",
        backend_factory=lambda provider, model, home: (FakeBackend([]), model),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        runtime = server.runtime
        model_selection.apply(runtime, "gpt-5.4", "ask")
        fallback = runtime.metadata.model_fallback

        async def fail_mcp_setup():
            raise CodexAuthError("MCP OAuth credentials expired", status_code=403)

        monkeypatch.setattr(runtime.loop, "_ensure_mcp_servers", fail_mcp_setup)
        await _request(reader, writer, 3, "send", {"text": "start MCP"})
        failure = (await _event(reader, "error"))["error"]
        assert failure == {"code": "auth_error", "message": "MCP OAuth credentials expired"}
        await _event(reader, "agent_end")
        assert runtime.model == "gpt-5.4"
        assert runtime.metadata.model_fallback == fallback
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,stream_events", [
    ("claude", [{"type": "error", "error": {"type": "permission_error", "message": "Denied"}}]),
    ("claude", [{"type": "error", "error": {"type": "not_found_error", "message": "Denied"}}]),
    ("claude", [
        {"type": "message_start", "message": {"id": "test-message"}},
        {"type": "error", "error": {"type": "permission_error", "message": "Denied"}},
    ]),
    ("claude", [{"type": "error", "error": {"status_code": 403, "message": "Denied"}}]),
    ("codex", [{"type": "error", "status_code": 403, "message": "Denied"}]),
    ("codex", [{"type": "error", "code": "model_not_found", "message": "Denied"}]),
    ("codex", [{"type": "error", "error": {"type": "permission_denied", "message": "Denied"}}]),
    ("codex", [
        {"type": "response.created", "response": {"id": "test-response"}},
        {"type": "response.failed", "response": {
            "error": {"code": "model_not_found", "message": "Denied"},
        }},
    ]),
])
async def test_stream_access_error_restores_model_over_wire(tmp_path, provider, stream_events):
    import httpx

    from zeta.providers.anthropic import _decode_response as decode_anthropic
    from zeta.providers.codex import _decode_response as decode_codex

    decode = decode_anthropic if provider == "claude" else decode_codex
    original = "claude-sonnet-4-6" if provider == "claude" else "gpt-5.4"
    selected = "claude-opus-4-6" if provider == "claude" else "gpt-5.4-mini"
    stream = "".join(f"data: {json.dumps(event)}\n\n" for event in stream_events)

    class StreamBackend(FakeBackend):
        async def complete(self, messages, tool_schemas):
            if self.model == selected:
                response = httpx.Response(200, text=stream)
                try:
                    async for event in decode(response):
                        yield event
                finally:
                    await response.aclose()
            else:
                async for event in super().complete(messages, tool_schemas):
                    yield event

    def build(provider, model, home):
        backend = StreamBackend([ScriptedTurn([TextContent("restored")])])
        backend.model = model
        return backend, model

    server = ZetaServer(home=tmp_path, port=0, provider=provider, model=original,
                        backend_factory=build)
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session")
        sid = server.runtime.session_id
        result = (await _request(reader, writer, 3, "set_settings", {
            "session_id": sid, "model": selected, "approval_mode": "ask",
        }))[-1]
        assert "result" in result
        assert server.runtime.metadata.model_fallback is not None
        await _request(reader, writer, 4, "send", {"text": "try selection"})
        failure = (await _event(reader, "error"))["error"]
        assert failure["code"] == "model_reverted"
        await _event(reader, "agent_end")
        assert server.runtime.model == original
        metadata = SessionManager(tmp_path).open(sid).metadata
        assert (metadata.provider, metadata.model) == (provider, original)
        assert metadata.model_fallback is None
        await _request(reader, writer, 5, "send", {"text": "try fallback"})
        message = await _event(reader, "assistant_message")
        assert message["message"]["content"][0]["text"] == "restored"
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["claude", "codex-flat", "codex-nested", "codex-failed"])
@pytest.mark.parametrize("status", [403, 404])
@pytest.mark.parametrize("message_fields", [{}, {"message": None}, {"message": {"text": "Denied"}}])
@pytest.mark.parametrize("partial", [False, True])
async def test_stream_access_recovery_does_not_depend_on_message(
    tmp_path, shape, status, message_fields, partial,
):
    import httpx

    from zeta.providers.anthropic import AnthropicStreamError
    from zeta.providers.anthropic import _decode_response as decode_anthropic
    from zeta.providers.codex import CodexStreamError
    from zeta.providers.codex import _decode_response as decode_codex
    from zeta.runtime.loop.agent import _error_info
    from zeta.server.model_selection import entitlement_error

    provider = "claude" if shape == "claude" else "codex"
    code = "permission_denied" if status == 403 else "model_not_found"
    detail = {"status_code": status, **message_fields}
    events = []
    if provider == "claude":
        detail["type"] = "permission_error" if status == 403 else "not_found_error"
        if partial:
            events = [
                {"type": "message_start", "message": {"id": "test-message"}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text"}},
                {"type": "content_block_delta", "index": 0,
                 "delta": {"type": "text_delta", "text": "partial"}},
            ]
        events.append({"type": "error", "error": detail})
    else:
        detail["code"] = code
        if partial or shape == "codex-failed":
            events = [{"type": "response.created", "response": {"id": "test-response"}}]
        if partial:
            events.extend([
                {"type": "response.output_item.added", "output_index": 0,
                 "item": {"type": "message", "id": "test-message"}},
                {"type": "response.content_part.added", "output_index": 0, "content_index": 0,
                 "part": {"type": "output_text"}},
                {"type": "response.output_text.delta", "output_index": 0, "content_index": 0,
                 "delta": "partial"},
            ])
        if shape == "codex-flat":
            events.append({"type": "error", **detail})
        elif shape == "codex-nested":
            events.append({"type": "error", "error": detail})
        else:
            events.append({"type": "response.failed", "response": {"error": detail}})

    decode = decode_anthropic if provider == "claude" else decode_codex
    error_type = AnthropicStreamError if provider == "claude" else CodexStreamError
    response = httpx.Response(200, text="".join(f"data: {json.dumps(e)}\n\n" for e in events))
    decoded = []
    try:
        with pytest.raises(error_type) as raised:
            async for item in decode(response):
                decoded.append(item)
        info = _error_info(raised.value, provider_error=True)
        assert (info.code, info.status_code) == (code, status)
        assert entitlement_error(info)
        assert any(item.delta == "partial" for item in decoded) is partial
    finally:
        await response.aclose()

    # Reuse the round-3 wire scenario without changing its assertions:
    # dedicated code, durable restoration, and a successful next send.
    await test_stream_access_error_restores_model_over_wire(tmp_path, provider, events)


@pytest.mark.asyncio
async def test_session_management_lifecycle_and_active_delete(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        hello = (await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"}))[-1]["result"]
        assert {"rename_session", "delete_session"} <= set(hello["capabilities"]["requests"])
        created = (await _request(reader, writer, 2, "new_session"))[-1]["result"]["session"]
        sid = created["session_id"]
        await _request(reader, writer, 3, "send", {"text": "original preview"})
        await _event(reader, "turn_end")
        renamed = (await _request(reader, writer, 4, "rename_session", {"session_id": sid, "name": "  Project   notes  "}))[-1]["result"]["session"]
        assert renamed["name"] == "Project notes"
        assert server.runtime.metadata.name == "Project notes"
        row = (await _request(reader, writer, 5, "list_sessions"))[-1]["result"]["sessions"][0]
        assert row["name"] == "Project notes"
        assert row["first_message_preview"] == "original preview"
        error = (await _request(reader, writer, 6, "delete_session", {"session_id": sid[:8]}))[-1]["error"]
        assert error["data"]["code"] == "active_session"
        assert server.runtime.session_id == sid
        assert (tmp_path / "sessions" / sid).exists()
        second = (await _request(reader, writer, 7, "new_session"))[-1]["result"]["session"]
        resumed = (await _request(reader, writer, 8, "resume", {"session_id": sid}))[-1]["result"]["session"]
        assert resumed["name"] == "Project notes"
        # Switching away closes the prior store and releases its lifetime lease.
        assert "result" in (await _request(reader, writer, 9, "delete_session", {"session_id": second["session_id"]}))[-1]
    finally:
        await _close(server, writer)

    # A fresh server reads the saved name; management needs no active session.
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        assert SessionManager(tmp_path).read_metadata(sid).name == "Project notes"
        for invalid in (None, 42, [], "x" * 61):
            result = (await _request(reader, writer, 2, "rename_session", {"session_id": sid, "name": invalid}))[-1]
            assert result["error"]["code"] == -32602
        await _request(reader, writer, 3, "rename_session", {"session_id": sid, "name": " \t\n "})
        row = next(row for row in (await _request(reader, writer, 4, "list_sessions"))[-1]["result"]["sessions"] if row["session_id"] == sid)
        assert row["name"] == ""
        assert row["first_message_preview"] == "original preview"
        # An external open store blocks RPC deletion even with no active server session.
        with SessionManager(tmp_path).open(sid).store:
            error = (await _request(reader, writer, 5, "delete_session", {"session_id": sid}))[-1]["error"]
            assert "in use" in error["message"]
            assert (tmp_path / "sessions" / sid).exists()
        assert (await _request(reader, writer, 5, "delete_session", {"session_id": sid}))[-1]["result"] == {"session_id": sid}
        assert not (tmp_path / "sessions" / sid).exists()
        assert all(row["session_id"] != sid for row in (await _request(reader, writer, 6, "list_sessions"))[-1]["result"]["sessions"])
        corrupt = tmp_path / "sessions" / "corrupt"
        corrupt.mkdir()
        (corrupt / "meta.json").write_bytes(b"\xff")
        assert "result" in (await _request(reader, writer, 7, "delete_session", {"session_id": "corrupt"}))[-1]
        assert not corrupt.exists()
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_session_management_legacy_gate_and_preview(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _ready(server)
    try:
        sid = server.runtime.session_id
        server.runtime.manager.rename(sid, "private display name")
        for method, params in (("rename_session", {"name": "changed"}), ("delete_session", {})):
            result = (await _request(reader, writer, 3, method, {"session_id": sid, **params}))[-1]
            assert result["error"]["code"] == -32601
        row = (await _request(reader, writer, 4, "list_sessions"))[-1]["result"]["sessions"][0]
        assert "name" not in row
        expected = server.runtime.metadata.to_dict()
        expected.pop("name")
        # The only gated field is name; all original metadata and preview remain.
        row.pop("first_message_preview")
        expected["updated_at"] = row["updated_at"]
        assert row == expected
        assert SessionManager(tmp_path).read_metadata(sid).name == "private display name"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("session_id", ["../outside", "/tmp/outside", ".", "..", "", "bad\x00id"])
async def test_session_delete_rpc_rejects_unsafe_ids(tmp_path, session_id):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        result = (await _request(reader, writer, 2, "delete_session", {"session_id": session_id}))[-1]
        assert result["error"]["code"] == -32602
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_resume_does_not_bump_updated_at(tmp_path):
    """ZETA-134 A5: selecting a session must not rewrite its updated_at.

    A read-only session select would otherwise reorder the sidebar with the
    selected row jumping to "now" even though nothing user-visible changed.
    """
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(tmp_path, cwd=tmp_path, provider="codex")
    try:
        first = await runtime.create_session()
        first_id = first.session_id
        await runtime.create_session()
        before = runtime.manager.read_metadata(first_id).updated_at
        await runtime.resume_session(first_id)
        after = runtime.manager.read_metadata(first_id).updated_at
        assert before == after
    finally:
        await runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["close", "replace-close", "activate", "resume-compose"])
async def test_runtime_failure_releases_all_session_leases(tmp_path, monkeypatch, failure):
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(tmp_path, cwd=tmp_path, provider="codex")
    await runtime.create_session()
    old_id = runtime.session_id
    old_store = runtime.opened.store
    incoming = []
    compose = runtime._compose

    def retain_composition(**kwargs):
        composition = compose(**kwargs)
        incoming.append(composition)
        return composition

    monkeypatch.setattr(runtime, "_compose", retain_composition)

    async def fail_close():
        raise RuntimeError("close failed")

    async def fail_activate(_self):
        raise RuntimeError("activation failed")

    try:
        if failure in {"close", "replace-close"}:
            monkeypatch.setattr(runtime.loop, "close", fail_close)
        if failure == "activate":
            monkeypatch.setattr(type(runtime.loop), "activate", fail_activate)
        if failure == "resume-compose":
            await runtime.close()
            def fail_compose(**_kwargs):
                raise RuntimeError("composition failed")
            monkeypatch.setattr(runtime, "_compose", fail_compose)
            operation = runtime.resume_session(old_id)
        elif failure == "close":
            operation = runtime.close()
        else:
            operation = runtime.create_session()
        with pytest.raises(RuntimeError, match="failed"):
            await operation
        assert runtime.state is None
        assert old_store._closed
        assert all(composition.opened.store._closed for composition in incoming)
        # Keep the runtime and exception-producing store alive while deleting.
        sessions = runtime.manager.list_sessions()
        assert len(sessions) == (2 if failure in {"replace-close", "activate"} else 1)
        for metadata in sessions:
            runtime.manager.delete(metadata.session_id)
        assert runtime.manager.list_sessions() == []
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_rename_succeeds_during_slow_stream_and_delete_declines(tmp_path):
    from tests.support.server_backend import ServerFakeBackend
    from zeta.protocol.types import StreamEventType

    started = asyncio.Event()
    release = asyncio.Event()

    class PausedBackend(ServerFakeBackend):
        async def complete(self, messages, tool_schemas):
            async for event in super().complete(messages, tool_schemas):
                yield event
                if event.type == StreamEventType.MESSAGE_UPDATE:
                    started.set()
                    await release.wait()

    server = ZetaServer(
        home=tmp_path, cwd=tmp_path, port=0, provider="codex",
        backend_factory=lambda *_args: (PausedBackend(delay=0), "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        sid = (await _request(reader, writer, 2, "new_session"))[-1]["result"]["session"]["session_id"]
        # Also check an inactive target so the streaming guard, not the active guard, decides deletion.
        other = server.runtime.manager.create(provider="codex", model="offline", cwd=tmp_path)
        other.store.close()
        await _request(reader, writer, 3, "send", {"text": "slow response"})
        await asyncio.wait_for(started.wait(), TIMEOUT)
        result = (await _request(reader, writer, 4, "rename_session", {"session_id": sid, "name": "mid-stream"}))[-1]
        assert result["result"]["session"]["name"] == "mid-stream"
        assert server.runtime.state.status == "running"
        error = (await _request(reader, writer, 5, "delete_session", {"session_id": other.metadata.session_id}))[-1]["error"]
        assert error["code"] == -32004
        release.set()
        await _event(reader, "turn_end")
        assert server.runtime.manager.read_metadata(sid).name == "mid-stream"
        assert "result" in (await _request(reader, writer, 6, "delete_session", {"session_id": other.metadata.session_id}))[-1]
    finally:
        release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_attachment_symlink_cannot_write_outside_session(tmp_path):
    import base64

    from zeta.server.ergonomics import image_message
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(tmp_path / "home", cwd=tmp_path, provider="codex")
    await runtime.create_session()
    outside = tmp_path / "outside"
    outside.mkdir()
    (runtime.opened.store.session_dir / "attachments").symlink_to(outside, target_is_directory=True)
    png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    try:
        with pytest.raises(OSError):
            image_message(runtime, {"images": [{"name": "x.png", "mime_type": "image/png", "data": base64.b64encode(png).decode()}]})
        assert list(outside.iterdir()) == []
    finally:
        await runtime.close()


def _seed_slash_fixtures(tmp_path: Path) -> Path:
    """Seed a home + project layout with one macro and one skill.

    Home holds a skill (`/greet`) so ``slash_list`` and ``slash_run`` observe
    the full builtin + macro + skill triple that the shared registry stitches
    together. Every scope-floor test uses this so the tests fail loudly if
    any tier stops loading. The project is git-initialised so ``/init``
    resolves a project root and returns its model prompt.
    """

    project = tmp_path / "project"
    (project / ".zeta" / "commands").mkdir(parents=True)
    subprocess.run(
        ["git", "init", "--quiet", str(project)],
        check=True,
        capture_output=True,
    )
    (project / ".zeta" / "commands" / "review.md").write_text(
        "---\ndescription: skim the current diff\n---\nDo a review: $ARGUMENTS\n",
        encoding="utf-8",
    )
    (project / ".zeta" / "commands" / "hi.md").write_text(
        "---\ndescription: greet\n---\nSay hi to $ARGUMENTS\n",
        encoding="utf-8",
    )
    home = tmp_path / "home"
    skill_dir = home / "skills" / "greet"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: greet\ndescription: greet the user\nkeywords: [greet]\n---\n\n"
        "Say hi like you mean it.\n",
        encoding="utf-8",
    )
    return project


@pytest.mark.asyncio
async def test_slash_list_reports_builtins_macros_and_named_skill(tmp_path: Path) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    try:
        result = (
            await _request(
                reader, writer, 5, "slash_list", {"session_id": sid}
            )
        )[-1]["result"]
        commands = {entry["name"]: entry for entry in result["commands"]}
        # Runnable built-ins the server drives directly. `/init` and `/help`
        # live at the scope floor — the menu depends on them appearing here
        # or the ZETA-130 lane loses two anchor commands.
        for name in ("status", "compact", "model", "help", "init"):
            assert name in commands
            assert commands[name]["client_only"] is False
        # Client-only built-ins still appear so the menu can render them.
        for name in ("theme", "fork"):
            assert commands[name]["client_only"] is True
        assert commands["vim"]["client_only"] is False
        # Prompt macros advertise their source directory.
        assert commands["review"]["kind"] == "macro-prompt"
        assert commands["review"]["source"] == "project"
        # The named skill from `~/.zeta/skills/greet/` must be present and
        # tagged as a skill so a frontend client can source-badge it.
        assert commands["greet"]["kind"] == "skill"
        assert commands["greet"]["client_only"] is False
        # session_id must match the active session.
        error = (
            await _request(
                reader, writer, 6, "slash_list", {"session_id": "other"}
            )
        )[-1]["error"]
        assert error["code"] == -32003
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_does_not_advertise_or_run_tui_approval_commands(
    tmp_path: Path,
) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    try:
        run_result = (
            await _request(
                reader, writer, "run", "slash_run",
                {"session_id": sid, "text": "/approve"},
            )
        )[-1]["result"]
        list_result = (
            await _request(
                reader, writer, "list", "slash_list", {"session_id": sid}
            )
        )[-1]["result"]

        assert run_result == {"kind": "unknown", "name": "approve"}
        assert {entry["name"] for entry in list_result["commands"]}.isdisjoint(
            {"approve", "deny"}
        )
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_slash_memory_accept(tmp_path: Path) -> None:
    project = _seed_slash_fixtures(tmp_path)
    home = tmp_path / "home"
    registry = __import__("zeta.project_registry", fromlist=["ProjectRegistry"]).ProjectRegistry(home / "projects")
    seeded = registry.create_project("demo", "scope", project)
    registry.initialize_memory(seeded.project_id)
    snapshot = registry.memory_snapshot(seeded.project_id)
    registry.compare_and_swap_memory(
        seeded.project_id,
        expected_digest=snapshot.digest,
        updates={"state.md": "automatic\n"},
        provenance={"session_id": "s", "seq_start": 1, "seq_end": 1},
    )
    server = ZetaServer(home=home, cwd=project, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    try:
        result = (await _request(
            reader, writer, "memory-accept", "slash_run",
            {"session_id": sid, "text": "/memory accept state.md"},
        ))[-1]["result"]
        assert result == {"kind": "output", "text": "memory accepted: state.md"}
        assert registry.memory_log(seeded.project_id)[-1]["provenance"] == {"accepted_by": "user"}
    finally:
        await _close(server, writer)


def test_memory_command_parity_tui_and_serve() -> None:
    from types import SimpleNamespace

    from zeta.tui.slash_handlers import SlashHandlerMixin

    class Registry:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str]] = []

        def ensure_memory_supported(self, project_id: str) -> None:
            self.calls.append(("format", project_id))

        def memory_format(self, project_id: str) -> int:
            self.calls.append(("format-version", project_id))
            return 1

        def memory_log(self, project_id: str, *, limit: int = 100) -> list[dict[str, object]]:
            self.calls.append(("log", project_id))
            return [{"version": "v1", "kind": "update"}]

        def undo_memory(self, project_id: str) -> list[tuple[str, str]]:
            self.calls.append(("undo", project_id))
            return [("state.md", "old")]

        def accept_memory(self, project_id: str, name: str) -> list[tuple[str, str]]:
            self.calls.append(("accept", f"{project_id}:{name}"))
            return [(name, "accepted")]

    tui_registry = Registry()
    tui = SimpleNamespace(
        loop=SimpleNamespace(
            project_registry=tui_registry,
            session_metadata=SimpleNamespace(project_id="p"),
        )
    )
    runtime = SimpleNamespace(
        metadata=SimpleNamespace(project_id="p"),
        manager=SimpleNamespace(project_registry=Registry()),
    )
    tui_outputs = [SlashHandlerMixin.slash_memory(tui, command) for command in ("log", "undo", "accept state.md")]
    serve_outputs = [ServerSlashSession(runtime).slash_memory(command) for command in ("log", "undo", "accept state.md")]
    assert tui_outputs == serve_outputs
    assert tui_registry.calls == runtime.manager.project_registry.calls


async def test_slash_run_dispatches_scope_floor(tmp_path: Path) -> None:
    project = _seed_slash_fixtures(tmp_path)
    server = ZetaServer(home=tmp_path / "home", cwd=project, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)

    async def run(text: str) -> dict:
        return (
            await _request(
                reader,
                writer,
                text,
                "slash_run",
                {"session_id": sid, "text": text},
            )
        )[-1]

    try:
        # /status returns a composed notice.
        result = (await run("/status"))["result"]
        assert result["kind"] == "output"
        assert "session_id:" in result["text"]

        # /help enumerates commands so a frontend client can show the same catalog inline.
        result = (await run("/help"))["result"]
        assert result["kind"] == "output"
        assert "/status" in result["text"]

        # /init inside a project returns a model prompt (no args allowed).
        result = (await run("/init"))["result"]
        assert result["kind"] == "model_input"
        assert result["text"]

        # /model without arguments hands off to the client's picker
        # surface (Settings on the frontend client); it never returns a bare text
        # notice a user cannot act on.
        result = (await run("/model"))["result"]
        assert result == {"kind": "client_only", "name": "model"}

        # /model switching applies through the shared settings path.
        result = (await run("/model gpt-5.6-sol"))["result"]
        assert result == {"kind": "output", "text": "model: gpt-5.6-sol"}
        assert server.runtime.model == "gpt-5.6-sol"

        result = (await run("/vim off"))["result"]
        assert result == {"kind": "output", "text": "vim mode: off"}
        assert server.runtime.metadata.vim_mode is False
        result = (await run("/vim toggle"))["result"]
        assert result == {"kind": "output", "text": "vim mode: on"}
        assert server.runtime.metadata.vim_mode is True

        # /compact is safe on an empty conversation; it reports nothing to compact.
        result = (await run("/compact"))["result"]
        assert result["kind"] == "output"

        # Prompt macros return model input, not chat text.
        result = (await run("/hi Henry"))["result"]
        assert result == {
            "kind": "model_input",
            "text": "Say hi to Henry",
            "display_text": "/hi Henry",
            "origin": "slash_expansion",
        }

        # A named skill loads and returns its prompt body as model input so a
        # frontend client can send it up the shared send path. The test seeds a
        # `~/.zeta/skills/greet` skill in `_seed_slash_fixtures` — this is
        # the real skill seam the earlier version of the test never touched.
        result = (await run("/greet"))["result"]
        assert result["kind"] == "model_input"
        assert "Say hi like you mean it" in result["text"]

        # Client-only commands report themselves so the frontend client can dispatch locally.
        result = (await run("/theme"))["result"]
        assert result == {"kind": "client_only", "name": "theme"}

        # Client-only notices are frontend-neutral.
        result = (await run("/fork"))["result"]
        assert result == {"kind": "client_only", "name": "fork"}
        assert ServerSlashSession(server.runtime)._client_only("fork") == (
            "/fork: runs client-side; open the command in your frontend client"
        )

        # Unknown commands are rejected structurally, never posted to the model.
        result = (await run("/nosuchcommand"))["result"]
        assert result == {"kind": "unknown", "name": "nosuchcommand"}

        # Text that does not start with `/` is a shape error, not a passthrough.
        error = (await run("hello"))["error"]
        assert error["code"] == -32602
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_slash_extensions_reject_legacy_clients(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        for method in ("slash_list", "slash_run"):
            error = (
                await _request(
                    reader,
                    writer,
                    method,
                    method,
                    {"session_id": "any", "text": "/status"},
                )
            )[-1]["error"]
            assert error["code"] == -32601
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_slash_run_guards_mutations_while_approvals_pending(tmp_path: Path) -> None:
    """`/compact` and `/model <name>` must fire ``require_mutable`` before
    dispatching. Otherwise a mutation lands while the loop is idle but the
    store still holds an outstanding tool call — the same seam every other
    mutation RPC guards with the exact same check."""

    manager = SessionManager(tmp_path)
    opened = manager.create(provider="codex", model="offline", cwd=tmp_path)
    # Seed one outstanding tool call so ``pending_requests()`` returns it
    # even though no turn task is running — this is the scenario the fix
    # closes: _require_idle passes, require_mutable must not.
    call = ToolCall("guarded-call", "read", {"path": str(tmp_path / "input")})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    opened.store.close()
    server = ZetaServer(
        home=tmp_path,
        provider="codex",
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "resume", {"session_id": opened.metadata.session_id})
        sid = opened.metadata.session_id

        async def run(rid: object, text: str) -> dict:
            return (
                await _request(
                    reader, writer, rid, "slash_run", {"session_id": sid, "text": text}
                )
            )[-1]

        # Read-only commands remain dispatchable — the guard only trips
        # mutating built-ins so `/status` still returns while an approval
        # is pending.
        assert (await run(3, "/status"))["result"]["kind"] == "output"
        vim_status = (await run("vim-read", "/vim"))["result"]
        assert vim_status == {
            "kind": "output",
            "text": f"vim mode: {'on' if server.runtime.metadata.vim_mode else 'off'}",
        }
        # An unknown `/vim` argument is read-only, even while approval is pending.
        assert (await run("vim-invalid", "/vim bogus"))["result"] == {
            "kind": "output",
            "text": "vim mode unchanged: use /vim on, /vim off, or /vim toggle",
        }
        assert server.runtime.metadata.vim_mode is True
        # `/compact` mutates the context store — must reject.
        assert (await run(4, "/compact"))["error"]["code"] == -32004
        # `/model` with args mutates settings — must reject.
        assert (await run(5, "/model gpt-5.6-sol"))["error"]["code"] == -32004
        # `/model` with no args is read-only — must still hand off to the
        # client's picker (never the mutation guard).
        assert (await run(6, "/model"))["result"] == {
            "kind": "client_only",
            "name": "model",
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_slash_run_rejects_during_running_turn(tmp_path: Path) -> None:
    from tests.support.server_backend import ServerFakeBackend
    from zeta.protocol.types import StreamEventType

    started = asyncio.Event()
    release = asyncio.Event()

    class PausedBackend(ServerFakeBackend):
        async def complete(self, messages, tool_schemas):
            async for event in super().complete(messages, tool_schemas):
                yield event
                if event.type == StreamEventType.MESSAGE_UPDATE:
                    started.set()
                    await release.wait()

    server = ZetaServer(
        home=tmp_path,
        cwd=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda *_args: (PausedBackend(delay=0), "offline"),
    )
    reader, writer, sid = await _ready_extensions(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "slow"})
        await asyncio.wait_for(started.wait(), TIMEOUT)
        error = (
            await _request(
                reader,
                writer,
                4,
                "slash_run",
                {"session_id": sid, "text": "/status"},
            )
        )[-1]["error"]
        assert error["code"] == -32004
    finally:
        release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_server_approval_carries_trusted_project_display(tmp_path: Path) -> None:
    # The session is bound to a real project; the approval wire must carry the
    # harness-owned project facts (id/name/filename/size/preview) rather than
    # any provider-derived rendering.
    from zeta.project_registry import ProjectRegistry

    home = tmp_path / "home"
    repository = tmp_path / "repo"
    repository.mkdir()
    project = ProjectRegistry(home / "projects").create_project(
        "demo", "scope", repository
    )
    call = ToolCall(
        "call-1",
        "project_update",
        {"name": "state.md", "content": "real body"},
    )
    backend = FakeBackend(
        [ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])]
    )
    server = ZetaServer(provider="codex",
        home=home,
        cwd=repository,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    try:
        await _request(reader, writer, 3, "send", {"text": "update memory"})
        approval = await _event(reader, "approval_request")
        display = approval["approval_display"]
        assert display["project_id"] == project.project_id
        assert display["project_name"] == "demo"
        assert display["filename"] == "state.md"
        assert display["utf8_bytes"] == len(b"real body")
        assert display["preview"] == "real body"
        # The same trusted facts appear in the status snapshot.
        status = (await _request(reader, writer, 4, "status"))[-1]["result"]
        pending = status["pending_approvals"][0]
        assert pending["approval_display"]["project_id"] == project.project_id
        assert pending["approval_display"]["filename"] == "state.md"
        await _request(reader, writer, 5, "deny", {"request_id": "call-1"})
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("registered", [False, True])
async def test_server_uses_only_existing_project_without_new_git_discovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, registered: bool
) -> None:
    from zeta.core import project_context
    from zeta.server.runtime import ServerRuntime

    cwd = tmp_path / "repo"
    cwd.mkdir()
    await asyncio.to_thread(
        subprocess.run, ["git", "init"], cwd=cwd, check=True, capture_output=True
    )
    home = tmp_path / "zeta-home"
    manager = SessionManager(home)
    existing = (
        manager.project_registry.find_or_create_for_directory(cwd)
        if registered
        else None
    )

    def fail_new_discovery(*args: object, **kwargs: object) -> object:
        raise AssertionError("server must not run automatic project discovery")

    monkeypatch.setattr(project_context, "_run_discovery_git", fail_new_discovery)
    runtime = ServerRuntime(home, cwd=cwd, provider="codex")
    try:
        metadata = await runtime.create_session()
        assert metadata.project_id == (
            existing.project_id if existing is not None else None
        )
        assert len(manager.project_registry.list_projects()) == int(registered)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_set_compaction_rejects_removed_summary_mode(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, sid = await _ready_extensions(server)
    try:
        response = (await _request(reader, writer, "c1", "set_compaction", {
            "session_id": sid, "mode": "summary",
        }))[-1]
        assert response["error"]["code"] == -32602
        assert response["error"]["message"] == (
            "compaction mode selection was removed; eviction is always used"
        )
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_set_compaction_requires_protocol_1_1(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _ready(server)
    try:
        response = (await _request(reader, writer, 3, "set_compaction", {"mode": "summary"}))[-1]
        assert response["error"]["code"] == -32601
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_same_session_resume_preserves_background_child(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    await server.runtime.create_session(provider="codex")
    loop = server.runtime.loop
    assert loop is not None
    owner = loop._background_owner
    child = asyncio.create_task(asyncio.sleep(60))
    canceled = False

    def cancel() -> None:
        nonlocal canceled
        canceled = True
        child.cancel()

    owner.register("child", cancel, child)
    await server.runtime.resume_session(server.runtime.session_id)
    try:
        assert server.runtime.loop is loop
        assert owner.owns_running("child")
        assert not canceled
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_resume_other_session_keeps_existing_semantics(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    await server.runtime.create_session(provider="codex")
    first_session = server.runtime.session_id
    await server.runtime.create_session(provider="codex")
    second_session = server.runtime.session_id
    second_loop = server.runtime.loop
    assert second_loop is not None
    owner = second_loop._background_owner
    child = asyncio.create_task(asyncio.sleep(60))
    owner.register("child", child.cancel, child)
    try:
        await server.runtime.resume_session(first_session)
        assert second_session != first_session
        assert server.runtime.loop is not second_loop
        assert not owner.owns_running("child")
    finally:
        await server.close()


def test_serve_yolo_sets_allow_default(tmp_path: Path, monkeypatch) -> None:
    from zeta import server as server_module
    from zeta.cli.main import main as cli_main

    monkeypatch.setenv("ZETA_HOME", str(tmp_path))

    async def check_server(server):
        assert server.runtime._config(None, None).yolo is True

    monkeypatch.setattr(server_module, "run_server", check_server)
    assert cli_main(["--provider", "codex", "--yolo", "serve"]) == 0


def test_serve_no_yolo_overrides_settings(tmp_path: Path, monkeypatch) -> None:
    from zeta import server as server_module
    from zeta.cli.main import main as cli_main

    (tmp_path / "settings.toml").write_text("yolo = true\n")
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))

    async def check_server(server):
        assert server.runtime._config(None, None).yolo is False

    monkeypatch.setattr(server_module, "run_server", check_server)
    assert cli_main(["--provider", "codex", "--no-yolo", "serve"]) == 0


@pytest.mark.asyncio
async def test_failed_wake_turn_keeps_durable_notification_input(tmp_path: Path) -> None:
    backend = FailingWakeBackend()
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child", child_session_path="/tmp/child", description="child", status="completed", text="done"
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(reader, writer, 4, "hello", {"protocol_version": "1.0"})
        await _frames_until_event(reader, "agent_end")
        while server.runtime.loop.notification_turn_state != "idle":
            await asyncio.sleep(0)
        assert backend.calls == 1
        assert store.agent_notifications() == []
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in store.messages()
        ) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_foreground_abort_retries_interrupted_notification_turn(
    tmp_path: Path,
) -> None:
    backend = BlockingThenCaptureBackend()
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child", child_session_path="/tmp/child", description="child", status="completed", text="done"
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(
            reader,
            writer,
            4,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        frames = await _request(
            reader, writer, 5, "abort", {"scope": "foreground"}
        )
        assert frames[-1]["result"]["aborted"] is True
        await _frames_until_event(reader, "agent_end")
        while store.agent_notifications():
            await asyncio.sleep(0)
        assert len(backend.calls) == 2
        assert store.agent_notifications() == []
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in store.messages()
        ) == 1
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in backend.calls[1]
        ) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_foreground_abort_before_notification_persistence_delivers_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("delivered")])])
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    original_append = store.append_message_async
    append_started = asyncio.Event()
    append_calls = 0

    async def block_first_append(message, *, on_persisted=None):
        nonlocal append_calls
        append_calls += 1
        if append_calls == 1:
            append_started.set()
            await asyncio.Event().wait()
        await original_append(message, on_persisted=on_persisted)

    monkeypatch.setattr(store, "append_message_async", block_first_append)
    store.append_agent_notification(
        "child",
        child_session_path="/tmp/child",
        description="child",
        status="completed",
        text="done",
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(
            reader,
            writer,
            4,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        await asyncio.wait_for(append_started.wait(), TIMEOUT)
        frames = await _request(
            reader, writer, 5, "abort", {"scope": "foreground"}
        )
        assert frames[-1]["result"]["aborted"] is True
        await _frames_until_event(reader, "agent_end")
        assert len(backend.calls) == 1
        assert append_calls == 2
        assert store.agent_notifications() == []
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in store.messages()
        ) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "session"])
async def test_session_abort_does_not_retry_notification_turn(
    tmp_path: Path,
    scope: str | None,
) -> None:
    backend = DisconnectThenSucceedBackend()
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child",
        child_session_path="/tmp/child",
        description="child",
        status="completed",
        text="done",
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        features = ["abort_scope"] if scope is not None else []
        await _request(
            reader,
            writer,
            4,
            "hello",
            {"protocol_version": "1.1", "features": features},
        )
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        params = {} if scope is None else {"scope": scope}
        frames = await _request(reader, writer, 5, "abort", params)
        assert frames[-1]["result"]["aborted"] is True
        await asyncio.sleep(0.05)
        assert backend.calls == 1
        assert store.agent_notifications() == []
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in store.messages()
        ) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_repeated_foreground_notification_aborts_schedule_one_retry_each(
    tmp_path: Path,
) -> None:
    backend = AbortTwiceThenSucceedBackend()
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child",
        child_session_path="/tmp/child",
        description="child",
        status="completed",
        text="done",
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(
            reader,
            writer,
            4,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        await asyncio.wait_for(backend.started[0].wait(), TIMEOUT)
        first = await _request(
            reader, writer, 5, "abort", {"scope": "foreground"}
        )
        assert first[-1]["result"]["aborted"] is True
        await asyncio.wait_for(backend.started[1].wait(), TIMEOUT)
        await asyncio.sleep(0.05)
        assert len(backend.calls) == 2

        second = await _request(
            reader, writer, 6, "abort", {"scope": "foreground"}
        )
        assert second[-1]["result"]["aborted"] is True
        await _frames_until_event(reader, "agent_end")
        assert len(backend.calls) == 3
        assert store.agent_notifications() == []
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in store.messages()
        ) == 1
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in backend.calls[2]
        ) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_count", [2, 5])
async def test_immediate_foreground_aborts_keep_one_notification_retry(
    tmp_path: Path,
    abort_count: int,
) -> None:
    backend = BlockingThenCaptureBackend()
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child",
        child_session_path="/tmp/child",
        description="child",
        status="completed",
        text="done",
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(
            reader,
            writer,
            4,
            "hello",
            {"protocol_version": "1.1", "features": ["abort_scope"]},
        )
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        for offset in range(abort_count):
            frames = await _request(
                reader,
                writer,
                5 + offset,
                "abort",
                {"scope": "foreground"},
            )
            assert frames[-1]["result"]["aborted"] is True

        await _frames_until_event(reader, "agent_end")
        while server.runtime.loop.notification_turn_state != "idle":
            await asyncio.sleep(0)
        await asyncio.sleep(0.05)
        assert len(backend.calls) == 2
        assert server.runtime.loop.schedule_notification_turn() is False
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_disconnected_wake_turn_reuses_durable_notification_input(
    tmp_path: Path,
) -> None:
    backend = DisconnectThenSucceedBackend()
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child", child_session_path="/tmp/child", description="child", status="completed", text="done"
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(reader, writer, 4, "hello", {"protocol_version": "1.0"})
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        writer.close()
        await writer.wait_closed()
        while server._client_active:
            await asyncio.sleep(0)
        assert store.agent_notifications() == []

        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(reader, writer, 5, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 6, "send", {"text": "retry"})
        await _frames_until_event(reader, "agent_end")
        assert backend.calls == 2
        assert sum(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in store.messages()
        ) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_send_immediately_after_reconnect_does_not_steal_wake_turn(
    tmp_path: Path,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("notification handled")]),
            ScriptedTurn([TextContent("user handled")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    server.runtime.opened.store.append_agent_notification(
        "child", child_session_path="/tmp/child", description="child", status="completed", text="done"
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(reader, writer, 4, "hello", {"protocol_version": "1.0"})
        rejected = await _request(reader, writer, 5, "send", {"text": "user now"})
        assert rejected[-1]["error"]["code"] == -32004
        await _frames_until_event(reader, "agent_end")
        while server._client is not None and server._client._turn_busy():
            await asyncio.sleep(0)

        user_frames = await _request(
            reader, writer, 6, "send", {"text": "user now"}
        )
        if not any(
            frame.get("params", {}).get("event") == "agent_end"
            for frame in user_frames
        ):
            await _frames_until_event(reader, "agent_end")
        assert len(backend.calls) == 2
        assert any(
            message.metadata.get("zeta_event") == "agent_notifications"
            for message in backend.calls[0][0]
        )
        assert any(
            message.role is MessageRole.USER
            and any(
                isinstance(block, TextContent) and block.text == "user now"
                for block in message.content
            )
            for message in backend.calls[1][0]
        )
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_disconnected_child_survives_client_reconnect(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer, session_id = await _ready_extensions(server)
    loop = server.runtime.loop
    assert loop is not None
    owner = loop._background_owner
    child = asyncio.create_task(asyncio.sleep(60))
    owner.register("child", child.cancel, child)
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        await _request(reader, writer, 4, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 5, "resume", {"session_id": session_id})
        assert server.runtime.loop is loop
        assert owner.owns_running("child")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_reconnect_runs_pending_notification_turn_once(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("notified")])])
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, _session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child",
        child_session_path="/tmp/child",
        description="child",
        status="completed",
        text="done",
    )
    writer.close()
    await writer.wait_closed()
    await asyncio.sleep(0.05)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", server.port)
        frames = await _request(
            reader, writer, 4, "hello", {"protocol_version": "1.0"}
        )
        frames += await _frames_until_event(reader, "agent_end")
        assert any(
            frame.get("params", {}).get("event") == "assistant_message"
            for frame in frames
        )
        assert len(backend.calls) == 1
        while store.agent_notifications():
            await asyncio.sleep(0)
        assert store.agent_notifications() == []
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_repeated_resume_does_not_duplicate_notification_turn(tmp_path: Path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("notified")])])
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer, session_id = await _ready_extensions(server)
    store = server.runtime.opened.store
    store.append_agent_notification(
        "child",
        child_session_path="/tmp/child",
        description="child",
        status="completed",
        text="done",
    )
    try:
        await _request(reader, writer, 3, "resume", {"session_id": session_id})
        await _request(reader, writer, 4, "resume", {"session_id": session_id})
        await _frames_until_event(reader, "turn_end")
        await _request(reader, writer, 5, "resume", {"session_id": session_id})
        await asyncio.sleep(0.05)
        assert len(backend.calls) == 1
        assert store.agent_notifications() == []
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_resume_missing_session_structured_error(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _ready(server)
    try:
        response = (
            await _request(reader, writer, 3, "resume", {"session_id": "missing"})
        )[-1]
        assert response["error"] == {
            "code": -32602,
            "message": "session missing was not found",
            "data": {"code": "session_not_found", "session_id": "missing"},
        }
    finally:
        await _close(server, writer)
