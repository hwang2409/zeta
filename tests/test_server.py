"""Real-socket tests for the native frontend server."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import socket
import stat
import sys
from pathlib import Path
from typing import Any

import pytest

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.session import SessionManager, SessionMetadata
from zeta.server import ZetaServer
from zeta.server.protocol import MAX_FRAME_BYTES, MAX_REQUEST_ID_BYTES, FrameCodec
from zeta.server.server import _Client
from zeta.types import Message, MessageRole, TextContent, ToolCall, ToolUseContent

TIMEOUT = 3


@pytest.fixture(autouse=True)
def _controlled_terminal_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("TERM", raising=False)
    monkeypatch.delenv("COLORTERM", raising=False)


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


async def _event(reader: asyncio.StreamReader, name: str) -> dict[str, Any]:
    while True:
        frame = await _read(reader)
        if frame.get("params", {}).get("event") == name:
            return frame["params"]


async def _close(server: ZetaServer, writer: asyncio.StreamWriter) -> None:
    writer.close()
    await asyncio.wait_for(writer.wait_closed(), TIMEOUT)
    await asyncio.wait_for(server.close(), TIMEOUT)


async def _ready(server: ZetaServer):
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
    await _request(reader, writer, 2, "new_session", {"provider": "fake"})
    return reader, writer


@pytest.mark.asyncio
async def test_server_streams_fake_turn_over_real_socket(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
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
async def test_unexpected_turn_error_matches_event_schema(tmp_path: Path) -> None:
    duplicate = ToolCall("duplicate-id", "read", {})
    backend = FakeBackend([ScriptedTurn(tool_calls=[duplicate, duplicate])])
    server = ZetaServer(
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
    server = ZetaServer(
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
        end = await _event(reader, "tool_end")
        assert end["tool_result"]["is_error"] is True
        await _event(reader, "turn_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_and_steer_continue_the_same_turn(tmp_path: Path) -> None:
    target = tmp_path / "input.txt"
    target.write_text("approved")
    call = ToolCall("call-1", "read", {"path": str(target)})
    backend = FakeBackend([ScriptedTurn(tool_calls=[call]), ScriptedTurn([TextContent("done")])])
    server = ZetaServer(
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
async def test_resumed_approval_finishes_idle_after_terminal_event(tmp_path: Path) -> None:
    manager = SessionManager(tmp_path)
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    target = tmp_path / "input.txt"
    target.write_text("approved")
    call = ToolCall("resumed-call", "read", {"path": str(target)})
    opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
        [(call.id, call)],
    )
    backend = FakeBackend([])
    server = ZetaServer(
        home=tmp_path,
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
    server = ZetaServer(
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
async def test_resume_second_client_and_malformed_frame(tmp_path: Path) -> None:
    first = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
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

    resumed = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
    reader, writer = await _connect(resumed)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        frames = await _request(reader, writer, 2, "resume", {"session_id": session})
        assert frames[-1]["result"]["session"]["session_id"] == session
    finally:
        await _close(resumed, writer)


@pytest.mark.asyncio
async def test_server_can_bind_localhost_port(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="fake")
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
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
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
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
    await _request(reader, writer, 2, "new_session", {"provider": "fake"})
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
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
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
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
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
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
    huge = SessionMetadata.new(
        session_id="a" * 32,
        provider="fake",
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
        provider="fake",
        model="offline",
        cwd=str(tmp_path),
        retained_tail=8,
        compaction_budget=200_000,
    )
    base_size = len(codec.response(0, {"sessions": [probe.to_dict()]}))
    huge = SessionMetadata.new(
        session_id=probe.session_id,
        provider="fake",
        model="offline",
        cwd=str(tmp_path),
        retained_tail=8,
        compaction_budget=200_000,
        system_prompt="x" * (MAX_FRAME_BYTES - 128 - base_size),
    )
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
    server.runtime.list_sessions = lambda: [huge]
    reader, writer = await _connect(server)
    request_id = "x" * MAX_REQUEST_ID_BYTES
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "fake"})
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
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
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
    server = ZetaServer(home=tmp_path, socket_path=socket_path, provider="fake")
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
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
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
    server = ZetaServer(
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
    server = ZetaServer(
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
    server = ZetaServer(
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
    client = object.__new__(_Client)
    client._approval_wires = {}
    client._approval_keys = {}
    first = client._wire_approval_key(("child-a", "same"))
    second = client._wire_approval_key(("child-b", "same"))
    assert first != second
    assert client._approval_keys[first] == ("child-a", "same")
    assert client._approval_keys[second] == ("child-b", "same")


@pytest.mark.asyncio
async def test_bad_resume_preserves_current_session(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path), provider="fake")
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
async def test_session_swap_resets_usage_for_create_and_resume(tmp_path: Path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first")], usage={"input_tokens": 3}),
            ScriptedTurn([TextContent("second")], usage={"input_tokens": 5}),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    first_session = server.runtime.session_id
    try:
        await _request(reader, writer, 3, "send", {"text": "first"})
        await _event(reader, "usage")
        assert (await _request(reader, writer, 4, "status"))[-1]["result"]["usage"]

        await _request(reader, writer, 5, "new_session", {"provider": "fake"})
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
        provider="fake",
        model="server-default",
        backend_factory=build_backend,
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "fake"})
        await _request(
            reader,
            writer,
            3,
            "new_session",
            {"provider": "fake", "model": "session-override"},
        )
        result = await _request(reader, writer, 4, "new_session")
        assert result[-1]["result"]["session"]["model"] == "server-default"
        assert calls[-1] == ("fake", "server-default")
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
        [ScriptedTurn(tool_calls=[parent_call]), ScriptedTurn(tool_calls=[child_call])]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _ready(server)
    old_session_id = server.runtime.session_id
    try:
        await _request(reader, writer, 3, "send", {"text": "start"})
        await _event(reader, "approval_request")
        swap_frames = await _request(reader, writer, 4, "new_session", {"provider": "fake"})
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
    server = ZetaServer(
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

        first_result = await _request(
            reader, writer, 4, "approve", {"request_id": first["request_id"]}
        )
        second_result = await _request(
            reader, writer, 5, "approve", {"request_id": second["request_id"]}
        )
        assert first_result[-1]["result"]["accepted"] is True
        assert second_result[-1]["result"]["accepted"] is True
        terminal_children: set[str] = set()

        def record_terminal_children(frames: list[dict[str, Any]]) -> None:
            for event in frames:
                params = event.get("params", {})
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

        record_terminal_children(first_result)
        record_terminal_children(second_result)
        while len(terminal_children) < 2:
            frame = await asyncio.wait_for(reader.readline(), TIMEOUT)
            assert frame
            record_terminal_children([json.loads(frame)])
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
        'provider = "fake"\n'
        "token_budget = 12345\n"
        "stream_stall_seconds = 45\n"
        "stream_stall_retries = 4\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(tmp_path)

    tui_calls: list[tuple[object, ...]] = []
    serve_calls: list[tuple[object, ...]] = []
    from zeta.cli import build_parser
    from zeta.server import runtime as server_runtime
    from zeta.server.fake_backend import ServerFakeBackend
    from zeta.tui import app as tui_app
    from zeta.tui.fake_backend import FakeInteractiveBackend

    def tui_backend(
        provider: str,
        model: str | None,
        *,
        home: Path,
        stall_seconds: float | None = None,
        stall_retries: int | None = None,
    ) -> tuple[FakeInteractiveBackend, str]:
        tui_calls.append((provider, model, home, stall_seconds, stall_retries))
        selected = model or "offline"
        return FakeInteractiveBackend(model=selected), selected

    def serve_backend(
        provider: str,
        model: str | None,
        home: Path,
        *,
        stall_seconds: float | None = None,
        stall_retries: int | None = None,
    ) -> tuple[ServerFakeBackend, str]:
        serve_calls.append((provider, model, home, stall_seconds, stall_retries))
        selected = model or "offline"
        return ServerFakeBackend(model=selected), selected

    monkeypatch.setattr(tui_app, "build_backend", tui_backend)
    monkeypatch.setattr(server_runtime, "default_backend", serve_backend)
    tui = tui_app.create_app(build_parser().parse_args(["--provider", "fake"]))
    server = ZetaServer(home=home, socket_path=_socket_path(tmp_path), provider="fake")
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
        home=sys.argv[1], socket_path=sys.argv[2], provider="fake"
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
        await _request(reader, writer, 2, "new_session", {"provider": "fake"})
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
    session = (await _request(reader, writer, 2, "new_session", {"provider": "fake"}))[-1]["result"]["session"]
    return reader, writer, session["session_id"]


@pytest.mark.asyncio
async def test_extensions_negotiate_and_old_clients_remain_unchanged(tmp_path):
    from zeta.server.ergonomics import EXTENSION_REQUESTS
    server = ZetaServer(home=tmp_path, port=0, provider="fake")
    reader, writer = await _connect(server)
    try:
        hello = (await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"}))[-1]["result"]
        assert hello["protocol_version"] == "1.0"
        assert not set(EXTENSION_REQUESTS) & set(hello["capabilities"]["requests"])
        for method in EXTENSION_REQUESTS:
            assert (await _request(reader, writer, method, method))[-1]["error"]["code"] == -32601
        assert "result" in (await _request(reader, writer, 3, "new_session", {"provider": "fake"}))[-1]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_tree_fork_switch_and_history_persist(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="fake")
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
async def test_settings_apply_to_active_session_and_resume(tmp_path):
    server = ZetaServer(home=tmp_path, port=0, provider="fake")
    reader, writer, sid = await _ready_extensions(server)
    async def rpc(method, **params):
        return (await _request(reader, writer, method, method, {"session_id": sid, **params}))[-1]
    try:
        assert (await rpc("model_catalog"))["result"] == {"models": ["faster", "offline"]}
        assert (await rpc("session_settings"))["result"] == {"model": "offline", "approval_mode": "ask"}
        for model, mode in (("invalid", "ask"), ("offline", "invalid")):
            assert (await rpc("set_settings", model=model, approval_mode=mode))["error"]["code"] == -32602
        settings = {"model": "faster", "approval_mode": "deny"}
        assert (await rpc("set_settings", **settings))["result"] == settings
        assert server.runtime.loop.backend.model == "faster"
        assert server.runtime.policy.default.value == "deny"
        await _request(reader, writer, "new", "new_session", {"provider": "fake"})
        assert (await rpc("set_settings", **settings))["error"]["code"] == -32003
        assert server.runtime.policy.default.value == "ask"
        await _request(reader, writer, "resume", "resume", {"session_id": sid})
        assert (await rpc("session_settings"))["result"] == settings
        assert not (tmp_path / "settings.json").exists()
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_images_persist_forward_and_reject_invalid_input(tmp_path):
    import base64

    from zeta.server.ergonomics import MAX_IMAGE_BYTES
    from zeta.types import ImageContent
    backend = FakeBackend([ScriptedTurn(content=[TextContent("seen")])])
    server = ZetaServer(home=tmp_path, port=0, backend_factory=lambda provider, model, home: (backend, model or "offline"))
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
