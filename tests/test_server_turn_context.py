from __future__ import annotations

import asyncio

import pytest

from tests.test_server import (
    _close,
    _connect,
    _frames_until_event,
    _request,
)
from zeta.core.fake import FakeBackend, ScriptedTurn
from zeta.protocol.types import MessageRole, TextContent
from zeta.server import ZetaServer

FEATURE = "turn_context"
CONTEXT = "Current Eastern time: 2026-10-07 09:30 EDT."


async def _ready(
    tmp_path,
    backend: FakeBackend,
    *,
    features: list[str] | None = None,
) -> tuple[ZetaServer, asyncio.StreamReader, asyncio.StreamWriter, str]:
    server = ZetaServer(
        home=tmp_path,
        port=0,
        provider="fake",
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    hello = await _request(
        reader,
        writer,
        1,
        "hello",
        {
            "protocol_version": "1.0",
            "client_version": "1.1",
            **({"features": features} if features is not None else {}),
        },
    )
    if features is not None:
        assert hello[-1]["result"]["capabilities"]["features"] == features
    created = await _request(reader, writer, 2, "new_session")
    return server, reader, writer, created[-1]["result"]["session"]["session_id"]


async def _notify(
    server: ZetaServer,
    reader: asyncio.StreamReader,
    notification_id: str,
) -> None:
    server.runtime.opened.store.append_agent_notification(
        notification_id,
        child_session_path=f"/tmp/{notification_id}",
        description="child",
        status="completed",
        text=f"{notification_id} done",
    )
    server._client._schedule_background_wake(server.runtime.session_id)
    await _frames_until_event(reader, "agent_end")
    while server.runtime.loop.notification_turn_state != "idle":
        await asyncio.sleep(0)


def _context_messages(backend: FakeBackend, call: int) -> list:
    return [
        message
        for message in backend.calls[call][0]
        if message.metadata.get("zeta_event") == "agent_notifications"
    ]


@pytest.mark.asyncio
async def test_notification_turn_consumes_context_once_before_provider_call(
    tmp_path,
) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("first handled")]),
            ScriptedTurn([TextContent("second handled")]),
        ]
    )
    server, reader, writer, _ = await _ready(tmp_path, backend, features=[FEATURE])
    try:
        result = await _request(
            reader, writer, 3, "set_turn_context", {"text": CONTEXT}
        )
        assert result[-1]["result"] == {
            "accepted": True,
            "session_id": server.runtime.session_id,
            "pending": True,
        }

        await _notify(server, reader, "child-1")
        await _notify(server, reader, "child-2")

        first = _context_messages(backend, 0)
        second = _context_messages(backend, 1)
        assert len(first) == 1
        assert len(second) == 2
        assert first[-1].role is MessageRole.SYSTEM
        assert first[-1].content[0].text.startswith(
            "client-supplied host context (untrusted data):\n"
        )
        assert CONTEXT in first[-1].content[0].text
        assert CONTEXT not in second[-1].content[0].text
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_send_turn_does_not_consume_pending_context(tmp_path) -> None:
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("user handled")]),
            ScriptedTurn([TextContent("notification handled")]),
        ]
    )
    server, reader, writer, _ = await _ready(tmp_path, backend, features=[FEATURE])
    try:
        await _request(reader, writer, 3, "set_turn_context", {"text": CONTEXT})
        await _request(reader, writer, 4, "send", {"text": "hello"})
        await _frames_until_event(reader, "agent_end")
        await _notify(server, reader, "child-1")

        assert all(CONTEXT not in str(message.to_dict()) for message in backend.calls[0][0])
        assert CONTEXT in _context_messages(backend, 1)[0].content[0].text
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_turn_context_replacement_and_clearing(tmp_path) -> None:
    replacement = "Current Eastern time: 2026-10-07 10:00 EDT."
    backend = FakeBackend(
        [
            ScriptedTurn([TextContent("replacement handled")]),
            ScriptedTurn([TextContent("cleared handled")]),
        ]
    )
    server, reader, writer, _ = await _ready(tmp_path, backend, features=[FEATURE])
    try:
        await _request(reader, writer, 3, "set_turn_context", {"text": CONTEXT})
        await _request(reader, writer, 4, "set_turn_context", {"text": replacement})
        await _notify(server, reader, "child-1")
        assert replacement in _context_messages(backend, 0)[0].content[0].text
        assert CONTEXT not in _context_messages(backend, 0)[0].content[0].text

        await _request(reader, writer, 5, "set_turn_context", {"text": CONTEXT})
        cleared = await _request(
            reader, writer, 6, "set_turn_context", {"text": None}
        )
        assert cleared[-1]["result"]["pending"] is False
        await _notify(server, reader, "child-2")
        assert CONTEXT not in _context_messages(backend, 1)[0].content[0].text
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_turn_context_size_is_bounded_by_utf8_bytes(tmp_path) -> None:
    backend = FakeBackend([])
    server, reader, writer, _ = await _ready(tmp_path, backend, features=[FEATURE])
    try:
        accepted = await _request(
            reader, writer, 3, "set_turn_context", {"text": "a" * 4096}
        )
        assert accepted[-1]["result"]["pending"] is True
        rejected = await _request(
            reader, writer, 4, "set_turn_context", {"text": "é" * 2049}
        )
        assert rejected[-1]["error"]["code"] == -32602
        assert "4096 UTF-8 bytes" in rejected[-1]["error"]["message"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_turn_context_is_persisted_as_non_user_notification_input(tmp_path) -> None:
    backend = FakeBackend([ScriptedTurn([TextContent("handled")])])
    server, reader, writer, _ = await _ready(tmp_path, backend, features=[FEATURE])
    try:
        await _request(reader, writer, 3, "set_turn_context", {"text": CONTEXT})
        await _notify(server, reader, "child-1")

        persisted = [
            message
            for message in server.runtime.opened.store.messages()
            if CONTEXT in str(message.to_dict())
        ]
        assert len(persisted) == 1
        assert persisted[0].role is MessageRole.SYSTEM
        assert persisted[0].metadata["zeta_event"] == "agent_notifications"
        assert persisted[0].metadata["turn_context"] is True
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_set_turn_context_requires_negotiated_feature(tmp_path) -> None:
    backend = FakeBackend([])
    server, reader, writer, _ = await _ready(tmp_path, backend)
    try:
        rejected = await _request(
            reader, writer, 3, "set_turn_context", {"text": CONTEXT}
        )
        assert rejected[-1]["error"]["code"] == -32601
        assert "negotiated turn_context feature" in rejected[-1]["error"]["message"]
    finally:
        await _close(server, writer)
