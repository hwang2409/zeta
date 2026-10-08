"""Idempotent client-delivery tests for the native serve protocol."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    ErrorInfo,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    with_message_origin,
)
from zeta.server import ZetaServer

TIMEOUT = 3


def _socket_path(tmp_path: Path, suffix: str = "") -> Path:
    return Path("/tmp") / f"zeta-delivery-{tmp_path.name}{suffix}.sock"


async def _connect(server: ZetaServer):
    await asyncio.wait_for(server.start(), TIMEOUT)
    return await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT
    )


async def _read(reader: asyncio.StreamReader) -> dict[str, Any]:
    line = await asyncio.wait_for(reader.readline(), TIMEOUT)
    assert line
    return json.loads(line)


async def _request(reader, writer, request_id, method, params=None):
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
    await writer.drain()
    frames = []
    while True:
        frame = await _read(reader)
        frames.append(frame)
        if frame.get("id") == request_id:
            return frames


async def _event(reader, name):
    while True:
        frame = await _read(reader)
        if frame.get("params", {}).get("event") == name:
            return frame["params"]


async def _close(server: ZetaServer, writer: asyncio.StreamWriter) -> None:
    writer.close()
    await writer.wait_closed()
    await server.close()


class BlockingBackend:
    def __init__(self) -> None:
        self.calls = 0
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, messages, tool_schemas):
        del messages, tool_schemas
        self.calls += 1
        self.started.set()
        await self.release.wait()
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
        )


@pytest.mark.asyncio
async def test_duplicate_send_starts_one_turn(tmp_path: Path) -> None:
    backend = BlockingBackend()
    server = ZetaServer(
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
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        assert hello[-1]["result"]["capabilities"]["features"] == ["delivery_id"]
        await _request(reader, writer, 2, "new_session", {"provider": "fake"})
        first = await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "hello", "delivery_id": "batch-1"},
        )
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        duplicate = await _request(
            reader,
            writer,
            4,
            "send",
            {"text": "hello", "delivery_id": "batch-1"},
        )
        assert first[-1]["result"]["accepted"] is True
        assert duplicate[-1]["result"] == {
            **first[-1]["result"],
            "duplicate": True,
        }
        assert backend.calls == 1
        assert len(server.runtime.opened.store.messages()) == 1
        backend.release.set()
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_duplicate_steer_is_queued_once_and_status_becomes_delivered(
    tmp_path: Path,
) -> None:
    target = tmp_path / "input.txt"
    target.write_text("data")
    backend = FakeBackend(
        [
            ScriptedTurn(tool_calls=[ToolCall("read-1", "read", {"path": str(target)})]),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
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
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "fake"})
        await _request(reader, writer, 3, "send", {"text": "read"})
        await _event(reader, "approval_request")
        accepted = await _request(
            reader,
            writer,
            4,
            "steer",
            {"text": "also this", "delivery_id": "batch-steer"},
        )
        duplicate = await _request(
            reader,
            writer,
            5,
            "steer",
            {"text": "also this", "delivery_id": "batch-steer"},
        )
        queued = await _request(
            reader,
            writer,
            6,
            "delivery_status",
            {"delivery_id": "batch-steer"},
        )
        assert accepted[-1]["result"] == {"accepted": True}
        assert duplicate[-1]["result"] == {"accepted": True, "duplicate": True}
        assert queued[-1]["result"]["status"] == "queued"
        assert server.runtime.loop.has_pending_steering

        await _request(reader, writer, 7, "deny", {"request_id": "read-1"})
        await _event(reader, "agent_end")
        delivered = await _request(
            reader,
            writer,
            8,
            "delivery_status",
            {"delivery_id": "batch-steer"},
        )
        assert delivered[-1]["result"]["status"] == "delivered"
        texts = [
            block.text
            for message in server.runtime.opened.store.messages()
            if message.role is MessageRole.USER
            for block in message.content
            if isinstance(block, TextContent)
        ]
        assert texts.count("also this") == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_server_restart_keeps_delivery_deduplication(tmp_path: Path) -> None:
    first_backend = FakeBackend([ScriptedTurn([TextContent("done")])])
    first_server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-1"),
        backend_factory=lambda provider, model, home: (
            first_backend,
            model or "offline",
        ),
    )
    reader, writer = await _connect(first_server)
    await _request(
        reader,
        writer,
        1,
        "hello",
        {"protocol_version": "1.1", "features": ["delivery_id"]},
    )
    created = await _request(reader, writer, 2, "new_session", {"provider": "fake"})
    session_id = created[-1]["result"]["session"]["session_id"]
    await _request(
        reader,
        writer,
        3,
        "send",
        {"text": "once", "delivery_id": "restart-1"},
    )
    await _event(reader, "agent_end")
    await _close(first_server, writer)

    second_backend = FakeBackend([])
    second_server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-2"),
        backend_factory=lambda provider, model, home: (
            second_backend,
            model or "offline",
        ),
    )
    reader, writer = await _connect(second_server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "resume", {"session_id": session_id})
        duplicate = await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "once", "delivery_id": "restart-1"},
        )
        assert duplicate[-1]["result"] == {
            "accepted": True,
            "session_id": session_id,
            "duplicate": True,
        }
        assert second_backend.calls == []
    finally:
        await _close(second_server, writer)


@pytest.mark.asyncio
async def test_delivery_status_unknown_and_feature_gate(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session", {"provider": "fake"})
        rejected = await _request(
            reader,
            writer,
            3,
            "send",
            {"text": "hello", "delivery_id": "not-negotiated"},
        )
        assert rejected[-1]["error"]["code"] == -32602
    finally:
        await _close(server, writer)

    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path, "-status"))
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        sessions = await _request(reader, writer, 2, "list_sessions")
        session_id = sessions[-1]["result"]["sessions"][0]["session_id"]
        await _request(reader, writer, 3, "resume", {"session_id": session_id})
        unknown = await _request(
            reader,
            writer,
            4,
            "delivery_status",
            {"delivery_id": "missing"},
        )
        invalid = await _request(
            reader,
            writer,
            5,
            "delivery_status",
            {"delivery_id": "bad id"},
        )
        assert unknown[-1]["result"] == {
            "delivery_id": "missing",
            "status": "unknown",
        }
        assert invalid[-1]["error"]["code"] == -32602
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_delivery_effect_callback_only_runs_after_durable_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = ConversationStore(tmp_path / "sessions")
    callback_observations: list[bool] = []
    original_write_bytes = ConversationStore._write_bytes

    def fail_write(self, data):
        del self, data
        raise OSError("injected durable-write failure")

    monkeypatch.setattr(ConversationStore, "_write_bytes", fail_write)
    with pytest.raises(OSError, match="injected"):
        await store.append_client_delivery_async(
            "fault-1",
            "steer",
            "queued",
            {"accepted": True},
            on_persisted=lambda: callback_observations.append(True),
        )
    assert callback_observations == []
    assert store.client_delivery("fault-1") is None

    monkeypatch.setattr(ConversationStore, "_write_bytes", original_write_bytes)
    await store.append_client_delivery_async(
        "fault-1",
        "steer",
        "queued",
        {"accepted": True},
        on_persisted=lambda: callback_observations.append(
            store.client_delivery("fault-1") is not None
        ),
    )
    assert callback_observations == [True]
    store.close()


def test_delivery_lookup_evicts_ids_outside_recent_bound(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_many(
        (
            "client_delivery",
            {
                "delivery_id": f"batch-{index}",
                "method": "steer",
                "status": "queued",
                "outcome": {"accepted": True},
            },
        )
        for index in range(1_001)
    )
    assert store.client_delivery("batch-0") is None
    assert store.client_delivery("batch-1").status == "queued"
    assert store.client_delivery("batch-1000").status == "queued"
    store.close()


def test_delivery_record_accepts_tagged_user_message_atomically(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    message = with_message_origin(
        Message(MessageRole.USER, [TextContent("hello")]), MessageOrigin.USER
    )
    store.append_client_delivery(
        "send-atomic",
        "send",
        "delivered",
        {"accepted": True, "session_id": store.session_id},
        message=message,
    )
    assert store.client_delivery("send-atomic").status == "delivered"
    assert [item.content[0].text for item in store.messages()] == ["hello"]
    store.close()


async def _queued_steer_server(tmp_path: Path, suffix: str):
    backend = BlockingBackend()
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path, suffix),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    await _request(
        reader,
        writer,
        1,
        "hello",
        {
            "protocol_version": "1.1",
            "features": ["delivery_id", "abort_scope"],
        },
    )
    await _request(reader, writer, 2, "new_session", {"provider": "fake"})
    await _request(reader, writer, 3, "send", {"text": "wait"})
    await asyncio.wait_for(backend.started.wait(), TIMEOUT)
    accepted = await _request(
        reader,
        writer,
        4,
        "steer",
        {"text": "later", "delivery_id": "drop-me"},
    )
    assert accepted[-1]["result"] == {"accepted": True}
    return server, backend, reader, writer


@pytest.mark.asyncio
async def test_session_abort_marks_accepted_steering_dropped(tmp_path: Path) -> None:
    server, _backend, reader, writer = await _queued_steer_server(tmp_path, "-abort")
    try:
        await _request(reader, writer, 5, "abort", {"scope": "session"})
        status = await _request(
            reader, writer, 6, "delivery_status", {"delivery_id": "drop-me"}
        )
        assert status[-1]["result"]["status"] == "dropped"
        assert server.runtime.opened.store.client_delivery("drop-me").reason == "abort"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_clear_steering_marks_delivery_dropped_and_duplicate_rejected(
    tmp_path: Path,
) -> None:
    server, backend, reader, writer = await _queued_steer_server(tmp_path, "-clear")
    try:
        cleared = await _request(reader, writer, 5, "clear_steering")
        duplicate = await _request(
            reader,
            writer,
            6,
            "steer",
            {"text": "later", "delivery_id": "drop-me"},
        )
        assert cleared[-1]["result"] == {"cleared": 1}
        assert duplicate[-1]["result"] == {
            "accepted": False,
            "duplicate": True,
            "status": "dropped",
        }
        assert server.runtime.opened.store.client_delivery("drop-me").reason == "clear"
        backend.release.set()
        await _event(reader, "agent_end")
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_disconnect_marks_accepted_steering_dropped(tmp_path: Path) -> None:
    server, _backend, _reader, writer = await _queued_steer_server(tmp_path, "-disconnect")
    writer.close()
    await writer.wait_closed()
    for _ in range(100):
        if not server._client_active:
            break
        await asyncio.sleep(0.01)
    assert not server._client_active

    reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    try:
        await _request(
            reader,
            writer,
            7,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        status = await _request(
            reader, writer, 8, "delivery_status", {"delivery_id": "drop-me"}
        )
        assert status[-1]["result"]["status"] == "dropped"
        assert server.runtime.opened.store.client_delivery("drop-me").reason == "disconnect"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_tool_less_turn_end_marks_late_steering_dropped(tmp_path: Path) -> None:
    server, backend, reader, writer = await _queued_steer_server(tmp_path, "-turn-end")
    try:
        backend.release.set()
        await _event(reader, "agent_end")
        status = await _request(
            reader, writer, 5, "delivery_status", {"delivery_id": "drop-me"}
        )
        assert status[-1]["result"]["status"] == "dropped"
        assert server.runtime.opened.store.client_delivery("drop-me").reason == "turn_end"
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_restart_marks_orphaned_steering_dropped(tmp_path: Path) -> None:
    first = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path, "-restart-1"))
    reader, writer = await _connect(first)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
    created = await _request(reader, writer, 2, "new_session", {"provider": "fake"})
    session_id = created[-1]["result"]["session"]["session_id"]
    await _close(first, writer)

    store = ConversationStore(tmp_path / "sessions", session_id=session_id)
    store.append_client_delivery(
        "restart-steer", "steer", "queued", {"accepted": True}
    )
    store.close()

    resumed = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path, "-restart-2"))
    reader, writer = await _connect(resumed)
    try:
        await _request(
            reader,
            writer,
            3,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 4, "resume", {"session_id": session_id})
        status = await _request(
            reader,
            writer,
            5,
            "delivery_status",
            {"delivery_id": "restart-steer"},
        )
        assert status[-1]["result"]["status"] == "dropped"
        delivery = resumed.runtime.opened.store.client_delivery("restart-steer")
        assert delivery.reason == "restart"
    finally:
        await _close(resumed, writer)


@pytest.mark.asyncio
async def test_orphan_steer_is_rejected_without_acceptance(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, socket_path=_socket_path(tmp_path, "-orphan"))
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "fake"})
        rejected = await _request(
            reader,
            writer,
            3,
            "steer",
            {"text": "orphan", "delivery_id": "orphan-steer"},
        )
        status = await _request(
            reader, writer, 4, "delivery_status", {"delivery_id": "orphan-steer"}
        )
        assert rejected[-1]["error"]["code"] == -32005
        assert status[-1]["result"]["status"] == "unknown"
    finally:
        await _close(server, writer)


def test_delivery_lookup_does_not_read_log_after_load(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    store.append_client_delivery(
        "indexed", "steer", "queued", {"accepted": True}
    )

    class UnreadableEntries(list):
        def __iter__(self):
            raise AssertionError("delivery lookup read the conversation log")

        def __reversed__(self):
            raise AssertionError("delivery lookup read the conversation log")

    store._entries = UnreadableEntries(store._entries)
    assert store.client_delivery("indexed").status == "queued"
    store.close()


class _FailingActiveTurnBackend:
    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def complete(self, messages, tool_schemas):
        del tool_schemas
        self.calls.append(list(messages))
        if len(self.calls) == 1:
            self.started.set()
            await self.release.wait()
            yield StreamEvent(
                StreamEventType.ERROR,
                error=ErrorInfo("backend_error", "failed"),
            )
            return
        yield StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, [TextContent("done")]),
        )


@pytest.mark.asyncio
async def test_failed_turn_drops_queued_steering_before_next_turn(tmp_path: Path) -> None:
    backend = _FailingActiveTurnBackend()
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-failed"),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(
            reader, writer, 1, "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "fake"})
        await _request(reader, writer, 3, "send", {"text": "first"})
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        await _request(
            reader, writer, 4, "steer",
            {"text": "stale", "delivery_id": "failed-steer"},
        )
        backend.release.set()
        await _event(reader, "agent_end")
        delivery = server.runtime.opened.store.client_delivery("failed-steer")
        assert delivery.status == "dropped"
        assert delivery.reason == "failed"
        assert not server.runtime.loop.has_pending_steering

        await _request(reader, writer, 5, "send", {"text": "next"})
        await _event(reader, "agent_end")
        assert not any(
            isinstance(block, TextContent) and block.text == "stale"
            for message in backend.calls[-1]
            for block in message.content
        )
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_steering_is_queued_until_dispatch_and_dropped_after_restart(
    tmp_path: Path,
) -> None:
    target = tmp_path / "input.txt"
    target.write_text("data")
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[ToolCall("read-1", "read", {"path": str(target)})]
            ),
            ScriptedTurn([TextContent("done")]),
        ]
    )
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-dispatch"),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    release = asyncio.Event()
    crash_home = tmp_path.parent / f"{tmp_path.name}-crash"
    session_id = ""
    try:
        await _request(
            reader, writer, 1, "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        created = await _request(
            reader, writer, 2, "new_session", {"provider": "fake"}
        )
        session_id = created[-1]["result"]["session"]["session_id"]
        original = server.runtime.loop.context_assembler.assemble
        entered = asyncio.Event()
        calls = 0

        async def blocked_assemble(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                entered.set()
                await release.wait()
            return await original(*args, **kwargs)

        server.runtime.loop.context_assembler.assemble = blocked_assemble
        await _request(reader, writer, 3, "send", {"text": "read"})
        await _event(reader, "approval_request")
        await _request(
            reader, writer, 4, "steer",
            {"text": "later", "delivery_id": "dispatch-steer"},
        )
        await _request(reader, writer, 5, "deny", {"request_id": "read-1"})
        await asyncio.wait_for(entered.wait(), TIMEOUT)

        delivery = server.runtime.opened.store.client_delivery("dispatch-steer")
        assert delivery.status == "queued"
        assert len(backend.calls) == 1
        shutil.copytree(tmp_path, crash_home)

        release.set()
        async with asyncio.timeout(TIMEOUT):
            while True:
                delivery = server.runtime.opened.store.client_delivery("dispatch-steer")
                if len(backend.calls) == 2 and delivery.status == "delivered":
                    break
                await asyncio.sleep(0.01)
    finally:
        release.set()
        await _close(server, writer)

    resumed = ZetaServer(
        home=crash_home,
        socket_path=_socket_path(tmp_path, "-restarted"),
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )
    reader, writer = await _connect(resumed)
    try:
        await _request(
            reader, writer, 6, "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 7, "resume", {"session_id": session_id})
        status = await _request(
            reader, writer, 8, "delivery_status",
            {"delivery_id": "dispatch-steer"},
        )
        assert status[-1]["result"]["status"] == "dropped"
        delivery = resumed.runtime.opened.store.client_delivery("dispatch-steer")
        assert delivery.reason == "restart"
        assert all(
            not any(
                isinstance(block, TextContent) and block.text == "later"
                for block in message.content
            )
            for message in resumed.runtime.opened.store.messages()
        )
    finally:
        await _close(resumed, writer)
