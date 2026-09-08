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
from zeta.core.session import SessionMetadata
from zeta.server import ZetaServer
from zeta.server.protocol import MAX_FRAME_BYTES
from zeta.server.server import _Client
from zeta.types import TextContent, ToolCall

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


async def _request(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    request_id: int,
    method: str,
    params: dict[str, object] | None = None,
) -> list[dict[str, Any]]:
    writer.write(
        (json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params or {}}) + "\n").encode()
    )
    await asyncio.wait_for(writer.drain(), TIMEOUT)
    frames: list[dict[str, Any]] = []
    while True:
        frame = await _read(reader)
        frames.append(frame)
        if frame.get("id") == request_id:
            return frames


async def _event(
    reader: asyncio.StreamReader, name: str
) -> dict[str, Any]:
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
        assert any(
            frame.get("params", {}).get("event") == "turn_aborted" for frame in frames
        )
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
        result = await _request(
            reader, writer, 1, "hello", {"protocol_version": "1.0"}
        )
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
        writer.write(b"x" * (MAX_FRAME_BYTES + 32) + b"\n")
        await writer.drain()
        error = await _read(reader)
        assert error["error"]["code"] == -32600
        status = await _request(reader, writer, 3, "status")
        assert status[-1]["result"]["session"] is not None
    finally:
        await _close(server, writer)


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
        writer.write(
            b'{"jsonrpc":"2.0","id":1,"method":"status","params":{}}\n'
        )
        await writer.drain()
        error = await _read(reader)
        assert error["error"]["code"] == -32002
        assert await asyncio.wait_for(reader.read(), TIMEOUT) == b""
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()


@pytest.mark.asyncio
async def test_disconnect_aborts_pending_approval_without_phantom(tmp_path: Path) -> None:
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
        after = (await _request(reader, writer, 5, "status"))[-1]["result"]["session"][
            "session_id"
        ]
        assert after == before
    finally:
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
