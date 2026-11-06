"""Ownership regressions for the serve control lifetime."""

from __future__ import annotations

import asyncio
import gc
import json
import uuid
import weakref
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import zeta.core.session as session_module
from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.test_server import (
    TIMEOUT,
    _close,
    _connect,
    _read_raw,
    _request,
    _socket_path,
)
from tests.test_server_observers import _controller_event_stream
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolSchema,
)
from zeta.server import ZetaServer
from zeta.server.connections import ConnectionRegistry
from zeta.server.control import ServeControl
from zeta.server.protocol import FrameCodec
from zeta.server.server import _Client


class _GatedBackend(CompletionBackend):
    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        self.entered.set()
        await self.release.wait()
        yield StreamEvent(StreamEventType.MESSAGE_START)
        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, content=TextContent("done"))
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
        )


class _PostTurnBackend(CompletionBackend):
    def __init__(self) -> None:
        self.calls = 0
        self.finalization_entered = asyncio.Event()
        self.release_finalization = asyncio.Event()
        self.woke = asyncio.Event()
        self.later_woke = asyncio.Event()

    async def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        self.calls += 1
        if self.calls == 1:
            yield StreamEvent(StreamEventType.MESSAGE_END)
            self.finalization_entered.set()
            await self.release_finalization.wait()
            return
        if self.calls == 2:
            self.woke.set()
        elif self.calls == 3:
            self.later_woke.set()
        yield StreamEvent(StreamEventType.MESSAGE_END)


class _ControllerConnection:
    def __init__(self) -> None:
        self.codec = FrameCodec()
        self.frames: list[bytes] = []
        self._write_lock = asyncio.Lock()
        self.server = type(
            "Server",
            (),
            {
                "runtime": type(
                    "Runtime", (), {"set_post_stream_provider_retry": lambda *_: None}
                )()
            },
        )()

    async def _write_physical(self, frame: bytes) -> bool:
        async with self._write_lock:
            self.frames.append(frame)
            return True

    async def close(self) -> None:
        pass


@pytest.mark.asyncio
async def test_memory_updated_during_pre_hello_claim_is_delivered_after_hello() -> None:
    registry = ConnectionRegistry(asyncio.get_running_loop())
    connection = _ControllerConnection()
    registry.register(connection)
    registry.negotiate_role(connection, "controller", "1.2", frozenset({"memory_updated"}))
    control = object.__new__(ServeControl)
    control.closing = False
    control.server = SimpleNamespace(connections=registry)
    control._notify = lambda event, session_id=None, **fields: registry.publish_controller(event, session_id, **fields)
    control._publish_memory_notice("session", "updated")
    await asyncio.sleep(0)
    assert await registry.send(connection, b"hello\n")
    assert json.loads(connection.frames[1])["params"]["event"] == "memory_updated"
    registry.release(connection)
    await registry.aclose()


@pytest.mark.asyncio
async def test_protocol_1_2_projection_keeps_negotiated_version_before_hello() -> None:
    registry = ConnectionRegistry(asyncio.get_running_loop())
    connection = _ControllerConnection()
    registry.register(connection)
    registry.negotiate_role(connection, "controller", "1.2", frozenset())
    control = object.__new__(ServeControl)
    control.server = SimpleNamespace(connections=registry)
    assert control._protocol_version() == "1.2"
    registry.release(connection)
    await registry.aclose()


def test_no_server_caller_branches_on_ready_controller() -> None:
    server_sources = "".join(
        path.read_text()
        for path in Path("src/zeta/server").glob("*.py")
        if path.name != "connections.py"
    )
    assert "current_controller" not in server_sources


@pytest.mark.asyncio
async def test_controller_event_emitted_before_hello_is_delivered_after_hello() -> None:
    registry = ConnectionRegistry(asyncio.get_running_loop())
    connection = _ControllerConnection()
    registry.register(connection)
    registry.negotiate_role(connection, "controller", "1.2", frozenset())
    await registry.publish_controller("assistant_delta", "session", delta="x")
    assert await registry.send(connection, b"hello\n")
    assert json.loads(connection.frames[1])["params"]["event"] == "assistant_delta"
    registry.release(connection)


@pytest.mark.asyncio
async def test_controller_pre_hello_events_keep_emission_order() -> None:
    registry = ConnectionRegistry(asyncio.get_running_loop())
    connection = _ControllerConnection()
    registry.register(connection)
    registry.negotiate_role(connection, "controller", "1.2", frozenset())
    for value in ("a", "b", "c"):
        await registry.publish_controller("assistant_delta", "session", delta=value)
    assert await registry.send(connection, b"hello\n")
    assert [json.loads(frame)["params"]["delta"] for frame in connection.frames[1:]] == [
        "a",
        "b",
        "c",
    ]
    registry.release(connection)


@pytest.mark.asyncio
async def test_controller_concurrent_outputs_keep_admission_order() -> None:
    registry = ConnectionRegistry(asyncio.get_running_loop())
    connection = _ControllerConnection()
    registry.register(connection)
    registry.negotiate_role(connection, "controller", "1.2", frozenset())
    assert await registry.send(connection, b"hello\n")
    connection.frames.clear()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    async def gated_write(frame: bytes) -> bool:
        async with connection._write_lock:
            if not first_started.is_set():
                first_started.set()
                await release_first.wait()
            connection.frames.append(frame)
            return True

    connection._write_physical = gated_write  # type: ignore[method-assign]
    first = asyncio.create_task(
        registry.publish_controller("assistant_delta", "session", delta="first")
    )
    await first_started.wait()
    second = asyncio.create_task(
        registry.publish_controller("assistant_delta", "session", delta="second")
    )
    await asyncio.sleep(0)
    assert connection.frames == []
    release_first.set()
    assert await asyncio.gather(first, second) == [True, True]
    assert [json.loads(frame)["params"]["delta"] for frame in connection.frames] == [
        "first",
        "second",
    ]
    registry.release(connection)


@pytest.mark.asyncio
async def test_observer_concurrent_outputs_keep_admission_order() -> None:
    registry = ConnectionRegistry(asyncio.get_running_loop())
    connection = _ControllerConnection()
    registry.register(connection)
    registry.negotiate_role(
        connection, "observer", "1.2", frozenset({"serve_observers"})
    )
    assert await registry.send(connection, b"hello\n")
    connection.frames.clear()
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    original_write = connection._write_physical

    async def gated_write(frame: bytes) -> bool:
        if not first_started.is_set():
            first_started.set()
            await release_first.wait()
        return await original_write(frame)

    connection._write_physical = gated_write  # type: ignore[method-assign]
    first = asyncio.create_task(registry.send(connection, b"first\n"))
    await first_started.wait()
    second = asyncio.create_task(registry.send(connection, b"second\n"))
    await asyncio.sleep(0)
    assert connection.frames == []
    release_first.set()
    assert await first
    assert await second
    assert connection.frames == [b"first\n", b"second\n"]
    registry.release(connection)
    await registry.aclose()


@pytest.mark.asyncio
async def test_controller_claim_failure_drops_pre_hello_events() -> None:
    registry = ConnectionRegistry(asyncio.get_running_loop())
    first = _ControllerConnection()
    registry.register(first)
    registry.negotiate_role(first, "controller", "1.2", frozenset())
    await registry.publish_controller("assistant_delta", "session", delta="old")
    registry.release(first)
    second = _ControllerConnection()
    registry.register(second)
    registry.negotiate_role(second, "controller", "1.2", frozenset())
    assert await registry.send(second, b"hello\n")
    assert second.frames == [b"hello\n"]
    registry.release(second)


@pytest.mark.asyncio
async def test_controller_pre_hello_buffer_bound_uses_overflow_contract() -> None:
    registry = ConnectionRegistry(asyncio.get_running_loop())
    connection = _ControllerConnection()
    registry.register(connection)
    registry.negotiate_role(connection, "controller", "1.2", frozenset())
    for _ in range(257):
        await registry.publish_controller("assistant_delta", "session", delta="x")
    assert not await registry.send(connection, b"hello\n")
    assert json.loads(connection.frames[1])["error"]["code"] == -32011
    registry.release(connection)


class _RecordingWriter:
    def __init__(self) -> None:
        self.frames: list[bytes] = []
        self.written = asyncio.Event()
        self.queue: asyncio.Queue[bytes] = asyncio.Queue()

    def write(self, frame: bytes) -> None:
        self.frames.append(frame)
        self.queue.put_nowait(frame)
        self.written.set()

    async def drain(self) -> None:
        pass

    def close(self) -> None:
        pass

    async def wait_closed(self) -> None:
        pass


async def _capture_direct_raw_stream(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> bytes:
    counter = iter(range(1000, 2000))
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(counter)))
    monkeypatch.setattr(session_module, "_now", lambda: "2026-01-01T00:00:00+00:00")
    backend = _GatedBackend()
    server = ZetaServer(
        home=home,
        cwd="/tmp",
        socket_path=_socket_path(home),
        provider="codex",
        backend_factory=lambda provider, model, backend_home: (
            backend,
            model or "offline",
        ),
    )
    reader, writer = await _connect(server)
    frames: list[bytes] = []

    async def send(request_id: int, method: str, params: dict[str, object]) -> None:
        writer.write(
            (
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "method": method,
                        "params": params,
                    },
                    separators=(",", ":"),
                )
                + "\n"
            ).encode()
        )
        await writer.drain()

    async def read_until(request_id: int, *, record: bool = True) -> None:
        while True:
            frame = await _read_raw(reader)
            if record:
                frames.append(frame)
            if json.loads(frame).get("id") == request_id:
                return

    try:
        await send(1, "hello", {"protocol_version": "1.2"})
        await read_until(1)
        await send(2, "status", {})
        await read_until(2)
        await send(3, "missing_method", {})
        await read_until(3)
        await send(4, "new_session", {"provider": "codex"})
        await read_until(4, record=False)
        await send(5, "send", {"text": "wait"})
        await read_until(5)
        await backend.entered.wait()
        await send(6, "abort", {})
        await read_until(6)
        writer.close()
        await writer.wait_closed()
        assert await reader.read() == b""
        return b"".join(frames)
    finally:
        backend.release.set()
        await server.close()


async def _replace_registry_controller(
    server: ZetaServer,
) -> tuple[_Client, _RecordingWriter]:
    assert server.connections is not None
    assert server._client is not None
    old_client = server._client
    server.connections.release(old_client)
    writer = _RecordingWriter()
    replacement = _Client(server, asyncio.StreamReader(), writer)
    server.connections.register(replacement)
    replacement.protocol_version = "1.2"
    replacement.features = frozenset({"assistant_message"})
    server.connections.negotiate_role(
        replacement,
        "controller",
        replacement.protocol_version,
        replacement.features,
    )
    replacement.handshaken = True
    assert await server.connections.send(replacement, b"hello\n")
    writer.frames.clear()
    writer.queue.get_nowait()
    return replacement, writer


@pytest.mark.asyncio
async def test_serve_control_detach_preserves_controller_raw_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = (
        Path(__file__).parent / "fixtures" / "serve_control_detach_origin_main.jsonl"
    ).read_bytes()
    actual = await _controller_event_stream(tmp_path, monkeypatch, observer=False)
    actual += await _capture_direct_raw_stream(tmp_path / "direct", monkeypatch)
    assert actual == expected

    server = ZetaServer(home=tmp_path / "owner", port=0, provider="codex")
    await server.start()
    try:
        # Runtime sinks belong to the server lifetime, before any socket claims it.
        assert server.runtime._background_event_sink is not None
        assert server.runtime._memory_notice_sink is not None
        assert server.runtime._background_wake_sink is not None
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_detached_turn_uses_current_controller_projection(tmp_path: Path) -> None:
    backend = _GatedBackend()
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    replacement: _Client | None = None
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "wait"})
        await asyncio.wait_for(backend.entered.wait(), TIMEOUT)
        replacement, replacement_writer = await _replace_registry_controller(server)
        backend.release.set()
        while True:
            frame = json.loads(
                await asyncio.wait_for(replacement_writer.queue.get(), TIMEOUT)
            )
            if frame.get("params", {}).get("event") == "assistant_message":
                break
    finally:
        backend.release.set()
        if replacement is not None:
            await replacement.close()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_post_turn_wake_runs_without_controller(tmp_path: Path) -> None:
    backend = _PostTurnBackend()
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        assert server.connections is not None and server._client is not None
        await _request(reader, writer, 3, "send", {"text": "finish"})
        await asyncio.wait_for(backend.finalization_entered.wait(), TIMEOUT)
        server.connections.release(server._client)
        await server._client.close()
        server.runtime.opened.store.append_agent_notification(
            "first",
            child_session_path="/tmp/first",
            description="first",
            status="completed",
            text="first done",
        )
        backend.release_finalization.set()
        await asyncio.wait_for(backend.woke.wait(), TIMEOUT)

        server.runtime.opened.store.append_agent_notification(
            "later",
            child_session_path="/tmp/later",
            description="later",
            status="completed",
            text="later done",
        )
        sink = server.runtime._background_wake_sink
        assert sink is not None
        sink(server.runtime.session_id)
        await asyncio.wait_for(backend.later_woke.wait(), TIMEOUT)
    finally:
        backend.release_finalization.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_closed_client_cannot_clear_serve_control_state(tmp_path: Path) -> None:
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        backend_factory=lambda provider, model, home: (
            FakeBackend([ScriptedTurn([TextContent("done")])]),
            model or "offline",
        ),
    )
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
    client = server._client
    assert client is not None and server.connections is not None
    released = asyncio.Event()
    original_release = server.connections.release
    client_id = id(client)

    def release(connection: Any) -> None:
        original_release(connection)
        if id(connection) == client_id:
            released.set()

    server.connections.release = release  # type: ignore[method-assign]
    reference = weakref.ref(client)
    writer.close()
    await writer.wait_closed()
    await asyncio.wait_for(released.wait(), TIMEOUT)
    del client
    gc.collect()

    try:
        assert server.runtime._background_event_sink is not None
        assert server.runtime._memory_notice_sink is not None
        assert server.runtime._background_wake_sink is not None
        assert reference() is None
    finally:
        await server.close()
