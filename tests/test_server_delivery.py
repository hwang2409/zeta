"""Idempotent client-delivery tests for the native serve protocol."""

from __future__ import annotations

import asyncio
import os
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from tests.support.fake_backend import FakeBackend, ScriptedTurn
from tests.support.server_client import event as _event
from tests.support.server_client import frames_until_event as _frames_until_event
from tests.support.server_client import request as _request
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
    ToolResult,
    ToolUseContent,
    with_message_origin,
)
from zeta.server import ZetaServer
from zeta.server.delivery import (
    DeliveryRequests,
    ResolvedModelInput,
    _deliver_after_commit,
)

TIMEOUT = 3


def _socket_path(tmp_path: Path, suffix: str = "") -> Path:
    return Path("/tmp") / f"zeta-delivery-{tmp_path.name}{suffix}.sock"


async def _connect(server: ZetaServer):
    await asyncio.wait_for(server.start(), TIMEOUT)
    return await asyncio.wait_for(
        asyncio.open_unix_connection(str(server.socket_path)), TIMEOUT
    )


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
async def test_max_turns_emits_terminal_error(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("data")
    backend = FakeBackend(
        [
            ScriptedTurn(
                tool_calls=[ToolCall("read-1", "read", {"path": str(target)})]
            ),
            ScriptedTurn(tool_calls=[ToolCall("read-2", "read", {"path": str(target)})]),
        ]
    )
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-max-turns"),
        max_turns=1,
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.2"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        await _request(reader, writer, 3, "send", {"text": "read"})
        await _event(reader, "approval_request")
        await _request(reader, writer, 4, "deny", {"request_id": "read-1"})
        error = await _event(reader, "error")
        assert error["error"] == {
            "code": "max_turns",
            "message": "maximum turns reached: 1",
        }
        assert (await _event(reader, "agent_end"))["event"] == "agent_end"
        assert len(backend.calls) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send", "steer"])
async def test_delivery_id_cancel_after_append_finishes_effects(
    tmp_path: Path, method: str
) -> None:
    store = ConversationStore(tmp_path / method, session_id="session-1")
    effects: list[str] = []
    append_started = asyncio.Event()
    release_append = asyncio.Event()
    request_task: asyncio.Task[dict[str, object]] | None = None
    original_append = store.append_client_delivery_async

    async def append(*args: Any, **kwargs: Any):
        append_started.set()
        await release_append.wait()
        result = await original_append(*args, **kwargs)
        assert request_task is not None
        request_task.cancel()
        await asyncio.sleep(0)
        return result

    store.append_client_delivery_async = append
    loop = type("Loop", (), {"steer": lambda self, message: effects.append("steer")})()
    control = type("Control", (), {})()
    control.server = type("Server", (), {"runtime": type("Runtime", (), {
        "loop": loop, "opened": type("Opened", (), {"store": store})(),
        "session_id": "session-1",
    })()})()
    control._model_inputs = type("Inputs", (), {
        "resolve": lambda self, session_id, params, enabled: ResolvedModelInput(
            params["text"], params["text"], MessageOrigin.USER, None
        )
    })()
    control.turn_busy = lambda: method == "steer"
    control.reject_client_turn_if_detached = lambda: None
    control._reject_if_unhealthy = lambda: None

    async def user_message(*args: Any, **kwargs: Any) -> None:
        effects.append("echo")

    control.user_message = user_message
    control.start_turn = lambda *args, **kwargs: effects.append("start")

    requests = DeliveryRequests(control)
    features = frozenset({"delivery_id"})
    request_task = asyncio.create_task(
        getattr(requests, method)({"text": "hello", "delivery_id": "cancel-1"}, features)
    )
    await append_started.wait()
    release_append.set()
    with pytest.raises(asyncio.CancelledError):
        await request_task

    assert store.client_delivery("cancel-1") is not None
    # canceled at commit: no echo, the work still runs once
    assert effects == (["start"] if method == "send" else ["steer"])
    duplicate = await getattr(
        requests, method
    )({"text": "hello", "delivery_id": "cancel-1"}, features)
    assert duplicate["duplicate"] is True
    assert effects == (["start"] if method == "send" else ["steer"])
    store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send", "steer"])
async def test_delivery_id_cancel_before_append_commits_does_nothing(
    tmp_path: Path, method: str
) -> None:
    store = ConversationStore(tmp_path / method, session_id="session-1")
    append_started = asyncio.Event()
    original_append = store.append_client_delivery_async

    async def append(*args: Any, **kwargs: Any):
        append_started.set()
        await asyncio.Event().wait()
        return await original_append(*args, **kwargs)

    store.append_client_delivery_async = append
    loop = type("Loop", (), {"steer": lambda self, message: None})()
    control = type("Control", (), {})()
    control.server = type("Server", (), {"runtime": type("Runtime", (), {
        "loop": loop, "opened": type("Opened", (), {"store": store})(),
        "session_id": "session-1",
    })()})()
    control._model_inputs = type("Inputs", (), {
        "resolve": lambda self, session_id, params, enabled: ResolvedModelInput(
            params["text"], params["text"], MessageOrigin.USER, None
        )
    })()
    control.turn_busy = lambda: method == "steer"
    control.reject_client_turn_if_detached = lambda: None
    control._reject_if_unhealthy = lambda: None
    effects: list[str] = []

    async def user_message(*args: Any, **kwargs: Any) -> None:
        effects.append("echo")

    control.user_message = user_message
    control.start_turn = lambda *args, **kwargs: effects.append("start")
    requests = DeliveryRequests(control)
    task = asyncio.create_task(
        getattr(requests, method)({"text": "hello", "delivery_id": "cancel-2"}, frozenset({"delivery_id"}))
    )
    await append_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert store.client_delivery("cancel-2") is None
    assert effects == []
    store.close()


async def test_delivery_id_echo_waits_for_durable_append(tmp_path: Path) -> None:
    backend = BlockingBackend()
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-echo-order"),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": [
                    "delivery_id",
                    "user_message_event",
                    "user_message_delivery_id",
                ],
            },
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        original_append = server.runtime.opened.store.append_client_delivery_async

        async def fail_append(*args, **kwargs):
            raise OSError("injected durable-write failure")

        server.runtime.opened.store.append_client_delivery_async = fail_append
        failed_send = await _request(
            reader, writer, 3, "send", {"text": "ghost", "delivery_id": "ghost-1"}
        )
        assert failed_send[-1]["error"]["code"] == -32000
        assert server.runtime.opened.store.client_delivery("ghost-1") is None

        await _request(reader, writer, 4, "send", {"text": "active"})
        await asyncio.wait_for(backend.started.wait(), TIMEOUT)
        failed_steer = await _request(
            reader,
            writer,
            5,
            "steer",
            {"text": "ghost steer", "delivery_id": "ghost-steer"},
        )
        assert failed_steer[-1]["error"]["code"] == -32000
        assert server.runtime.opened.store.client_delivery("ghost-steer") is None
        backend.release.set()
        await _event(reader, "agent_end")

        server.runtime.opened.store.append_client_delivery_async = original_append
        await _request(
            reader, writer, 6, "send", {"text": "real", "delivery_id": "real-1"}
        )
        frames = await _frames_until_event(reader, "agent_end")
        echoes = [
            frame["params"]
            for frame in frames
            if frame.get("params", {}).get("event") == "user_message"
        ]
        assert len(echoes) == 1
        assert echoes == [
            {
                "event": "user_message",
                "session_id": server.runtime.session_id,
                "text": "real",
                "mode": "send",
                "attachments": [],
                "delivery_id": "real-1",
            }
        ]
        assert server.runtime.opened.store.client_delivery("real-1") is not None
        backend.release.set()
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_duplicate_send_starts_one_turn(tmp_path: Path) -> None:
    backend = BlockingBackend()
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
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        assert hello[-1]["result"]["capabilities"]["features"] == ["delivery_id"]
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
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
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
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
    first_server = ZetaServer(provider="codex",
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
    created = await _request(reader, writer, 2, "new_session", {"provider": "codex"})
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
    second_server = ZetaServer(provider="codex",
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
    server = ZetaServer(provider="codex", home=tmp_path, socket_path=_socket_path(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
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

    server = ZetaServer(provider="codex", home=tmp_path, socket_path=_socket_path(tmp_path, "-status"))
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


def _evict_delivery(
    store: ConversationStore,
    delivery_id: str,
    *,
    method: str = "send",
    status: str = "delivered",
) -> None:
    acceptance_status = "delivered" if method == "send" else "queued"
    outcome = (
        {"accepted": True, "session_id": store.session_id}
        if method == "send"
        else {"accepted": True}
    )
    rows: list[tuple[str, dict[str, object]]] = [
        (
            "client_delivery",
            {
                "delivery_id": delivery_id,
                "method": method,
                "status": acceptance_status,
                "outcome": outcome,
            },
        )
    ]
    rows.extend(
        (
            "client_delivery",
            {
                "delivery_id": f"filler-{index}",
                "method": "steer",
                "status": "queued",
                "outcome": {"accepted": True},
            },
        )
        for index in range(1_000)
    )
    if status == "dropped":
        rows.append(
            (
                "client_delivery",
                {
                    "delivery_id": delivery_id,
                    "method": "steer",
                    "status": "dropped",
                    "reason": "clear",
                },
            )
        )
    store.append_many(rows)


def test_delivery_lookup_retains_evicted_outcomes_without_ids(tmp_path: Path) -> None:
    store = ConversationStore(tmp_path / "sessions")
    _evict_delivery(store, "old-delivered")
    _evict_delivery(store, "old-dropped", method="steer", status="dropped")

    assert store.client_delivery("old-delivered") is None
    assert store.client_delivery_outcome("old-delivered") == "delivered"
    assert store.client_delivery("old-dropped") is None
    assert store.client_delivery_outcome("old-dropped") == "dropped"
    assert store.client_delivery_outcome("never-seen") is None
    assert store.client_delivery("filler-999").status == "queued"
    store.close()


@pytest.mark.asyncio
async def test_evicted_delivery_status_and_send_reuse_are_distinct(
    tmp_path: Path,
) -> None:
    backend = BlockingBackend()
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
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        _evict_delivery(server.runtime.opened.store, "old-send")

        status = await _request(
            reader, writer, 3, "delivery_status", {"delivery_id": "old-send"}
        )
        rejected = await _request(
            reader,
            writer,
            4,
            "send",
            {"text": "must not run", "delivery_id": "old-send"},
        )
        unknown = await _request(
            reader, writer, 5, "delivery_status", {"delivery_id": "never-seen"}
        )

        assert status[-1]["result"] == {
            "delivery_id": "old-send",
            "status": "evicted",
            "outcome": "delivered",
        }
        assert rejected[-1]["error"] == {
            "code": -32010,
            "message": "delivery id already used",
            "data": {"outcome": "delivered"},
        }
        assert unknown[-1]["result"] == {
            "delivery_id": "never-seen",
            "status": "unknown",
        }
        assert backend.calls == 0
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_evicted_dropped_steer_status_and_reuse_are_distinct(
    tmp_path: Path,
) -> None:
    server = ZetaServer(provider="codex", home=tmp_path, socket_path=_socket_path(tmp_path))
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
        _evict_delivery(
            server.runtime.opened.store,
            "old-steer",
            method="steer",
            status="dropped",
        )

        status = await _request(
            reader, writer, 3, "delivery_status", {"delivery_id": "old-steer"}
        )
        rejected = await _request(
            reader,
            writer,
            4,
            "steer",
            {"text": "must not queue", "delivery_id": "old-steer"},
        )

        assert status[-1]["result"] == {
            "delivery_id": "old-steer",
            "status": "evicted",
            "outcome": "dropped",
        }
        assert rejected[-1]["error"] == {
            "code": -32010,
            "message": "delivery id already used",
            "data": {"outcome": "dropped"},
        }
        assert not server.runtime.loop.has_pending_steering
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_restart_rebuilds_evicted_delivery_outcome(tmp_path: Path) -> None:
    first = ZetaServer(provider="codex", home=tmp_path, socket_path=_socket_path(tmp_path, "-first"))
    reader, writer = await _connect(first)
    await _request(
        reader,
        writer,
        1,
        "hello",
        {"protocol_version": "1.1", "features": ["delivery_id"]},
    )
    created = await _request(reader, writer, 2, "new_session", {"provider": "codex"})
    session_id = created[-1]["result"]["session"]["session_id"]
    _evict_delivery(first.runtime.opened.store, "restart-old")
    await _close(first, writer)

    resumed = ZetaServer(provider="codex", home=tmp_path, socket_path=_socket_path(tmp_path, "-resumed"))
    reader, writer = await _connect(resumed)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "resume", {"session_id": session_id})
        status = await _request(
            reader, writer, 3, "delivery_status", {"delivery_id": "restart-old"}
        )
        assert status[-1]["result"] == {
            "delivery_id": "restart-old",
            "status": "evicted",
            "outcome": "delivered",
        }
    finally:
        await _close(resumed, writer)


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
    server = ZetaServer(provider="codex",
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
    await _request(reader, writer, 2, "new_session", {"provider": "codex"})
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
    first = ZetaServer(provider="codex", home=tmp_path, socket_path=_socket_path(tmp_path, "-restart-1"))
    reader, writer = await _connect(first)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.1"})
    created = await _request(reader, writer, 2, "new_session", {"provider": "codex"})
    session_id = created[-1]["result"]["session"]["session_id"]
    await _close(first, writer)

    store = ConversationStore(tmp_path / "sessions", session_id=session_id)
    store.append_client_delivery(
        "restart-steer", "steer", "queued", {"accepted": True}
    )
    store.close()

    resumed = ZetaServer(provider="codex", home=tmp_path, socket_path=_socket_path(tmp_path, "-restart-2"))
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
    server = ZetaServer(provider="codex", home=tmp_path, socket_path=_socket_path(tmp_path, "-orphan"))
    reader, writer = await _connect(server)
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {"protocol_version": "1.1", "features": ["delivery_id"]},
        )
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
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


def test_steer_acceptance_keeps_legacy_delivery_shape_without_text(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "sessions", session_id="compat-session")
    store.append_client_delivery(
        "compat-steer",
        "steer",
        "queued",
        {"accepted": True},
    )

    (delivery,) = store.replay()
    assert delivery.type == "client_delivery"
    assert delivery.data == {
        "delivery_id": "compat-steer",
        "method": "steer",
        "status": "queued",
        "outcome": {"accepted": True},
    }
    assert all(entry.type not in {"message", "warning"} for entry in store.replay())
    store.close()


@pytest.mark.asyncio
async def test_all_history_fidelity_rows_open_and_replay_with_origin_main_reader(
    tmp_path: Path,
) -> None:
    old_reader_src = os.environ.get("HISTORY_FIDELITY_ORIGIN_MAIN_SRC")
    if old_reader_src is None:
        pytest.skip(
            "set HISTORY_FIDELITY_ORIGIN_MAIN_SRC to an origin/main worktree src directory"
        )

    class StopThenFailBackend:
        def __init__(self) -> None:
            self.calls = 0
            self.started = asyncio.Event()

        async def complete(self, messages, tool_schemas):
            del messages, tool_schemas
            self.calls += 1
            if self.calls == 1:
                self.started.set()
                await asyncio.Event().wait()
            raise RuntimeError("compatibility error turn")
            yield  # pragma: no cover - make this an async generator

    backend = StopThenFailBackend()
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path, "-origin-main-compat"),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
    )
    reader, writer = await _connect(server)
    session_id = ""
    try:
        await _request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": [
                    "abort_scope",
                    "delivery_id",
                    "session_history_fidelity",
                ],
            },
        )
        created = await _request(
            reader, writer, 2, "new_session", {"provider": "codex"}
        )
        session_id = created[-1]["result"]["session"]["session_id"]
        store = server.runtime.opened.store
        store.append_message(
            Message(
                MessageRole.ASSISTANT,
                [ToolUseContent(ToolCall("bash-compat", "bash", {"command": "exit 7"}))],
            )
        )
        store.append_message(
            with_message_origin(
                Message(
                    MessageRole.USER,
                    tool_result=ToolResult(
                        "bash-compat",
                        "failed",
                        is_error=True,
                        structured_content={
                            "exit_code": 7,
                            "stdout": "",
                            "stderr": "failed",
                            "stdout_truncated": False,
                            "stderr_truncated": False,
                            "cwd_after": str(tmp_path),
                        },
                        content_blocks=[
                            {
                                "type": "image",
                                "data": "YWJj",
                                "mimeType": "image/png",
                            }
                        ],
                    ),
                ),
                MessageOrigin.USER,
            )
        )
        for delivery_id, terminal in (
            ("compat-delivered", "delivered"),
            ("compat-dropped", "dropped"),
        ):
            steering = with_message_origin(
                Message(MessageRole.USER, [TextContent(delivery_id)]),
                MessageOrigin.USER,
            )
            store.append_client_delivery(
                delivery_id, "steer", "queued", {"accepted": True}
            )
            store.queue_client_steering(steering, delivery_id)
            if terminal == "delivered":
                batch = store.pending_client_steering()
                assert batch is not None
                await store.deliver_client_steering(batch)
            else:
                store.drop_client_steering("abort")

        history = await _request(
            reader,
            writer,
            30,
            "session_history",
            {"session_id": session_id},
        )
        rows = history[-1]["result"]["messages"]
        assert rows[0]["content"][0]["tool_call"]["arguments"] == {
            "command": "exit 7"
        }
        assert rows[1]["tool_result"]["structured_content"]["exit_code"] == 7
        assert rows[1]["tool_result"]["content_blocks"] == [
            {
                "type": "image",
                "mimeType": "image/png",
                "size": 3,
                "data_omitted": True,
            }
        ]
        delivery_rows = [row for row in rows if row.get("event") == "delivery_status"]
        assert {row["status"] for row in delivery_rows} == {"delivered", "dropped"}
        assert next(row for row in delivery_rows if row["status"] == "dropped")[
            "content"
        ] == [{"type": "text", "text": "steer dropped: abort"}]

        await _request(reader, writer, 3, "send", {"text": "stop this turn"})
        await backend.started.wait()
        await _request(reader, writer, 4, "abort")
        await _event(reader, "turn_aborted")
        await _request(reader, writer, 5, "send", {"text": "fail this turn"})
        await _event(reader, "error")
    finally:
        await _close(server, writer)

    session_root = tmp_path / "sessions"
    result = await asyncio.to_thread(
        subprocess.run,
        [
            sys.executable,
            "-c",
            (
                "import zeta; from pathlib import Path; "
                "from zeta.core.store import ConversationStore; "
                "print(zeta.__file__); "
                "store = ConversationStore(Path(__import__('sys').argv[1]), "
                "session_id=__import__('sys').argv[2]); "
                "rows = store.replay(); store.close(); "
                "print('old reader replayed', len(rows), 'rows')"
            ),
            str(session_root),
            session_id,
        ],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": old_reader_src},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "old reader replayed" in result.stdout
    assert str(Path(old_reader_src) / "zeta" / "__init__.py") in result.stdout


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
    server = ZetaServer(provider="codex",
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
        await _request(reader, writer, 2, "new_session", {"provider": "codex"})
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
    server = ZetaServer(provider="codex",
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
            reader, writer, 2, "new_session", {"provider": "codex"}
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

    resumed = ZetaServer(provider="codex",
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
class _Store:
    def __init__(self) -> None:
        self.rows: dict[str, SimpleNamespace] = {}

    async def append_client_delivery_async(
        self, delivery_id, operation, status, outcome, **kwargs
    ) -> None:
        del kwargs
        self.rows[delivery_id] = SimpleNamespace(
            status=status, outcome=outcome, operation=operation
        )

    def client_delivery(self, delivery_id):
        return self.rows.get(delivery_id)

    def client_delivery_outcome(self, delivery_id):
        del delivery_id


class _Control:
    def __init__(self, *, steer: bool = False) -> None:
        self.store = _Store()
        self.steer_calls = 0
        self.start_calls = 0
        self.echo_started = asyncio.Event()
        self.echo_error = False
        self.echo_stalls = False
        self.release_echo = asyncio.Event()
        self.loop = SimpleNamespace(steer=self._steer)
        self.server = SimpleNamespace(
            runtime=SimpleNamespace(
                loop=self.loop,
                opened=SimpleNamespace(store=self.store),
                session_id="session",
            )
        )
        self._model_inputs = SimpleNamespace(
            resolve=lambda *args, **kwargs: SimpleNamespace(
                text="hello",
                display_text="hello",
                origin=MessageOrigin.USER,
                message=None,
            )
        )
        self._steer = steer

    def turn_busy(self) -> bool:
        return self._steer

    def reject_client_turn_if_detached(self) -> None:
        pass

    def _reject_if_unhealthy(self) -> None:
        pass

    async def user_message(self, *args, **kwargs) -> None:
        del args, kwargs
        self.echo_started.set()
        if self.echo_stalls:
            await self.release_echo.wait()
        if self.echo_error:
            raise ConnectionError("echo failed")

    def start_turn(self, *args, **kwargs) -> None:
        del args, kwargs
        self.start_calls += 1

    def _steer(self, message) -> None:
        del message
        self.steer_calls += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send", "steer"])
async def test_committed_delivery_runs_work_when_echo_fails(method: str) -> None:
    control = _Control(steer=method == "steer")
    control.echo_error = True
    requests = DeliveryRequests(control)
    params = {"text": "hello", "delivery_id": f"{method}-error"}

    with pytest.raises(ConnectionError):
        await getattr(requests, method)(params, frozenset({"delivery_id"}))

    assert control.start_calls + control.steer_calls == 1
    retry = await getattr(requests, method)(params, frozenset({"delivery_id"}))
    assert retry["duplicate"] is True
    assert control.start_calls + control.steer_calls == 1


@pytest.mark.asyncio
async def test_repeat_cancel_does_not_interrupt_echo_cancellation_cleanup() -> None:
    echo_started = asyncio.Event()
    cancel_started = asyncio.Event()
    release_cancel = asyncio.Event()
    cleanup_interrupted = False
    effects: list[str] = []

    async def echo() -> None:
        nonlocal cleanup_interrupted
        echo_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancel_started.set()
            try:
                await release_cancel.wait()
            except asyncio.CancelledError:
                cleanup_interrupted = True

    request = asyncio.create_task(
        _deliver_after_commit(echo, lambda: effects.append("work"), canceled=False)
    )
    await echo_started.wait()
    request.cancel()
    await cancel_started.wait()
    await asyncio.sleep(0)
    request.cancel()
    await asyncio.sleep(0)
    release_cancel.set()

    with pytest.raises(asyncio.CancelledError):
        await request
    assert not cleanup_interrupted
    assert effects == ["work"]


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send", "steer"])
async def test_cancel_cleans_up_echo_and_runs_committed_work(method: str) -> None:
    control = _Control(steer=method == "steer")
    control.echo_stalls = True
    requests = DeliveryRequests(control)
    request = asyncio.create_task(
        getattr(requests, method)(
            {"text": "hello", "delivery_id": f"{method}-cancel"},
            frozenset({"delivery_id", "user_message_event"}),
        )
    )
    await control.echo_started.wait()
    request.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(request, 1)
    assert control.start_calls + control.steer_calls == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["send", "steer"])
async def test_cancel_during_commit_runs_work_without_waiting_for_echo(method: str) -> None:
    control = _Control(steer=method == "steer")
    control.echo_stalls = True
    requests = DeliveryRequests(control)
    request: asyncio.Task[dict[str, object]] | None = None
    commit = control.store.append_client_delivery_async

    async def append_then_cancel(*args, **kwargs) -> None:
        await commit(*args, **kwargs)
        assert request is not None
        request.cancel()  # the first cancel lands after the row is durable
        await asyncio.sleep(0)

    control.store.append_client_delivery_async = append_then_cancel
    request = asyncio.create_task(
        getattr(requests, method)(
            {"text": "hello", "delivery_id": f"{method}-commit-cancel"},
            frozenset({"delivery_id", "user_message_event"}),
        )
    )

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(asyncio.shield(request), 1)
    assert control.start_calls + control.steer_calls == 1
    assert not control.echo_started.is_set()
