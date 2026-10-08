"""Idempotent client-delivery tests for the native serve protocol."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
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
