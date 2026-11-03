"""Race and wire regressions for serve-owned approval reservation."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

import zeta.core.session as session_module
from tests.support.fake_backend import FakeBackend
from tests.test_server import (
    TIMEOUT,
    _close,
    _connect,
    _read_raw,
    _request,
    _socket_path,
)
from zeta.core.approval import ApprovalPolicy, ApprovalRequest
from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    ToolCall,
    ToolUseContent,
)
from zeta.server import ZetaServer
from zeta.server.protocol import ProtocolError
from zeta.server.slash_commands import ServerSlashSession


class _StoreGate:
    """Bounded worker gate: a wrong loop call times out instead of hanging."""

    def __init__(
        self, store: ConversationStore, method: str = "approval_states"
    ) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.thread_ids: list[int] = []
        self.method = method
        self.original = getattr(store, method)

    def call(self, *args: object, **kwargs: object) -> object:
        self.thread_ids.append(threading.get_ident())
        self.entered.set()
        assert self.release.wait(timeout=1), (
            f"timed out waiting to release {self.method}"
        )
        return self.original(*args, **kwargs)


async def _wait_thread_gate(gate: _StoreGate) -> None:
    assert await asyncio.to_thread(gate.entered.wait, TIMEOUT)


async def _approval_server(
    tmp_path: Path,
    approval_count: int = 1,
) -> tuple[ZetaServer, asyncio.StreamReader, asyncio.StreamWriter, ToolCall]:
    server = ZetaServer(
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
        provider="codex",
        cwd="/tmp",
        backend_factory=lambda provider, model, home: (
            FakeBackend([]),
            model or "offline",
        ),
    )
    reader, writer = await _connect(server)
    await _request(reader, writer, 1, "hello", {"protocol_version": "1.0"})
    await _request(reader, writer, 2, "new_session", {"provider": "codex"})
    calls = [
        ToolCall(f"call-{index}", "read", {"path": f"/tmp/input-{index}"})
        for index in range(1, approval_count + 1)
    ]
    server.runtime.opened.store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call) for call in calls]),
        [(call.id, call) for call in calls],
    )
    return server, reader, writer, calls[0]


def _start_event(call: ToolCall) -> StreamEvent:
    return StreamEvent(StreamEventType.TOOL_APPROVAL_START, tool_call=call)


def _end_event(call: ToolCall, decision: str = "abort") -> StreamEvent:
    return StreamEvent(
        StreamEventType.TOOL_APPROVAL_END,
        tool_call=call,
        data={"decision": decision},
    )


async def _gated_start(
    server: ZetaServer, call: ToolCall, monkeypatch: pytest.MonkeyPatch
) -> tuple[asyncio.Task[None], _StoreGate]:
    store = server.runtime.opened.store
    gate = _StoreGate(store)
    monkeypatch.setattr(store, gate.method, gate.call)
    task = asyncio.create_task(server.control._event(_start_event(call)))
    await _wait_thread_gate(gate)
    return task, gate


@pytest.mark.asyncio
async def test_approval_start_disconnect_while_lookup_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        start, gate = await _gated_start(server, call, monkeypatch)
        released = asyncio.create_task(server.control.controller_released())
        await asyncio.sleep(0)
        assert not released.done()
        gate.release.set()
        await asyncio.wait_for(asyncio.gather(start, released), TIMEOUT)
        state = server.runtime.opened.store.approval_states()[call.id]
        assert state[1] == "abort"
        assert gate.thread_ids != [threading.get_ident()]
        assert server.control.pending_approvals_now() == []
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_start_terminal_event_while_lookup_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    notifications: list[str] = []
    original = server.control._notify

    async def record(event: str, *args: object, **kwargs: object) -> bool:
        notifications.append(event)
        return await original(event, *args, **kwargs)

    monkeypatch.setattr(server.control, "_notify", record)
    try:
        start, gate = await _gated_start(server, call, monkeypatch)
        terminal = asyncio.create_task(server.control._event(_end_event(call)))
        await asyncio.sleep(0)
        gate.release.set()
        await asyncio.wait_for(asyncio.gather(start, terminal), TIMEOUT)
        assert "approval_request" not in notifications
        assert notifications.count("approval_end") <= 1
        assert server.control.pending_approvals_now() == []
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_start_expiry_while_lookup_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        start, gate = await _gated_start(server, call, monkeypatch)
        expiry = asyncio.create_task(
            server.control._terminate_pending_approvals(reason="grace_expired")
        )
        await asyncio.sleep(0)
        gate.release.set()
        await asyncio.wait_for(asyncio.gather(start, expiry), TIMEOUT)
        resolution = server.runtime.opened.store.approval_resolution_data(call.id)
        assert resolution == {
            "request_id": call.id,
            "decision": "abort",
            "reason": "grace_expired",
        }
        assert server.control.pending_approvals_now() == []
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_start_shutdown_while_lookup_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    start, gate = await _gated_start(server, call, monkeypatch)
    closing = asyncio.create_task(server.control.close())
    await asyncio.sleep(0)
    assert not closing.done()
    gate.release.set()
    await asyncio.wait_for(asyncio.gather(start, closing), TIMEOUT)
    assert server.control.pending_approvals_now() == []
    await _close(server, writer)


@pytest.mark.asyncio
async def test_decision_rejected_while_approval_start_lookup_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        start, gate = await _gated_start(server, call, monkeypatch)
        for method in ("approve", "deny"):
            with pytest.raises(
                ProtocolError, match="not found or already resolved"
            ) as error:
                await server.control.resolve_approval(method, call.id)
            assert error.value.code == -32006
        gate.release.set()
        await asyncio.wait_for(start, TIMEOUT)
        assert [
            item["request_id"] for item in server.control.pending_approvals_now()
        ] == [call.id]
    finally:
        gate.release.set()
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("terminal_first", [False, True])
async def test_approval_start_disconnect_then_terminal_while_lookup_gated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, terminal_first: bool
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        start, gate = await _gated_start(server, call, monkeypatch)
        terminal = asyncio.create_task(server.control._event(_end_event(call)))
        disconnect = asyncio.create_task(server.control.controller_released())
        ordered = (terminal, disconnect) if terminal_first else (disconnect, terminal)
        await asyncio.sleep(0)
        gate.release.set()
        await asyncio.wait_for(asyncio.gather(start, *ordered), TIMEOUT)
        rows = [
            entry
            for entry in server.runtime.opened.store.entries
            if entry.type == "approval_resolution"
        ]
        assert len(rows) <= 1
        assert server.control.pending_approvals_now() == []
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_delegated_cleanup_overlaps_worker_resolve_exact_target(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "child")
    call = ToolCall("child-call", "read", {"path": "input"})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
    )
    server = ZetaServer(home=tmp_path / "serve", port=0, provider="codex")
    await server.start()
    policy = ApprovalPolicy(store=ConversationStore(tmp_path / "parent"))
    server.runtime._state = SimpleNamespace(policy=policy, loop=None, opened=None)
    request = ApprovalRequest(call.id, call, child_instance_id="child")
    server.control.bind_policy(policy)
    policy.register_delegated(request, store, child_instance_id="child")
    assert policy._delegated == {}
    cleanup = policy.cleanup_delegated("child")
    if cleanup is not None:
        await cleanup
    await server.control.wait_for_approval_settlement()
    assert store.approval_resolution_data(call.id)["decision"] == "abort"
    assert server.control.pending_approvals_now() == []
    store.close()
    server.runtime._state = None
    await server.close()


@pytest.mark.asyncio
async def test_same_tick_approval_reservation_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        await server.control._event(_start_event(call))
        store = server.runtime.opened.store
        gate = _StoreGate(store, "resolve_approval")
        monkeypatch.setattr(store, gate.method, gate.call)
        approve = asyncio.create_task(
            server.control.resolve_approval("approve", call.id)
        )
        await _wait_thread_gate(gate)
        abort = asyncio.create_task(server.control._terminate_pending_approvals())
        await asyncio.sleep(0)
        gate.release.set()
        results = await asyncio.gather(approve, abort, return_exceptions=True)
        assert gate.thread_ids != [threading.get_ident()]
        assert sum(not isinstance(item, BaseException) for item in results) >= 1
        rows = [
            entry
            for entry in server.runtime.opened.store.entries
            if entry.type == "approval_resolution"
        ]
        assert len(rows) == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_durable_failure_is_fail_closed_and_documented(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        await server.control._event(_start_event(call))
        events: list[tuple[str, str | None]] = []
        aborted: list[bool] = []
        original_notify = server.control._notify

        async def record_notify(event: str, *args: object, **kwargs: object) -> bool:
            error = kwargs.get("error")
            code = error.get("code") if isinstance(error, dict) else None
            events.append((event, code))
            return await original_notify(event, *args, **kwargs)

        monkeypatch.setattr(server.control, "_notify", record_notify)
        assert server.runtime.loop is not None
        monkeypatch.setattr(
            server.runtime.loop,
            "abort",
            lambda **kwargs: aborted.append(True),
        )
        store = server.runtime.opened.store
        monkeypatch.setattr(
            store,
            "resolve_approval",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failed")),
        )
        with pytest.raises(ProtocolError) as error:
            await server.control.resolve_approval("approve", call.id)
        assert error.value.code == -32000
        assert events == [("error", "server_error")]
        assert aborted == [True]
        with pytest.raises(ProtocolError) as later:
            await server.control.resolve_approval("approve", call.id)
        assert later.value.code == -32006
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_legacy_status_bytes_match_main_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ids = iter(range(1000, 1100))
    monkeypatch.setattr(uuid, "uuid4", lambda: uuid.UUID(int=next(ids)))
    monkeypatch.setattr(session_module, "_now", lambda: "2026-01-01T00:00:00+00:00")
    server, reader, writer, call = await _approval_server(tmp_path, approval_count=65)
    try:
        await server.control._event(_start_event(call))
        await _read_raw(reader)
        assert server._client is not None
        monkeypatch.setattr(
            server._client, "_session_snapshot", lambda: {"session_id": "fixed"}
        )
        store = server.runtime.opened.store
        gate = _StoreGate(store)
        monkeypatch.setattr(store, gate.method, gate.call)
        expected = (
            Path(__file__).parent
            / "fixtures"
            / "serve_approval_status_origin_main.jsonl"
        ).read_bytes()
        writer.write(b'{"jsonrpc":"2.0","id":9,"method":"status","params":{}}\n')
        await writer.drain()
        await _wait_thread_gate(gate)
        gate.release.set()
        actual = await _read_raw(reader)
        assert gate.thread_ids != [threading.get_ident()]
        assert actual == expected
        assert b"resolution_pending" not in actual
        assert b"pending_approvals_truncated" not in actual
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_legacy_disconnect_abort_is_off_loop_and_byte_identical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = (
        Path(__file__).parent
        / "fixtures"
        / "serve_approval_disconnect_origin_main.jsonl"
    ).read_bytes()
    assert expected.endswith(b"\n")
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        await server.control._event(_start_event(call))
        store = server.runtime.opened.store
        gate = _StoreGate(store, "resolve_approval")
        monkeypatch.setattr(store, gate.method, gate.call)
        released = asyncio.create_task(server.control.controller_released())
        await _wait_thread_gate(gate)
        gate.release.set()
        await released
        assert gate.thread_ids != [threading.get_ident()]
        resolution = server.runtime.opened.store.approval_resolution_data(call.id)
        actual = (json.dumps(resolution, separators=(",", ":")) + "\n").encode()
        assert actual == expected
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_stale_start_worker_cannot_cross_session_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        start, gate = await _gated_start(server, call, monkeypatch)
        server.control.session_replaced()
        gate.release.set()
        await asyncio.wait_for(start, TIMEOUT)
        assert server.control.pending_approvals_now() == []
    finally:
        await _close(server, writer)


def test_serve_bound_delegated_never_touches_policy_registry(tmp_path: Path) -> None:
    policy = ApprovalPolicy(store=ConversationStore(tmp_path / "parent"))
    child = ConversationStore(tmp_path / "child")
    call = ToolCall("child-call", "read", {"path": "input"})
    request = ApprovalRequest(call.id, call, child_instance_id="child")
    control = SimpleNamespace(
        register_delegated=lambda *args: None, cleanup_delegated=lambda *args: None
    )
    policy.bind_serve_control(control)
    policy.register_delegated(request, child, child_instance_id="child")
    assert policy._delegated == {}
    policy.cleanup_delegated("child")
    assert policy._delegated == {}
    child.close()
    policy._store.close()


@pytest.mark.asyncio
async def test_slash_and_ergonomics_read_serve_control_not_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        await server.control._event(_start_event(call))
        policy = server.runtime.policy
        assert policy is not None and server._client is not None
        monkeypatch.setattr(
            policy,
            "pending_requests",
            lambda: (_ for _ in ()).throw(AssertionError("store read")),
        )
        session = ServerSlashSession(server.runtime)
        session._approval_control = server.control
        status = session.slash_status()
        assert status.pending_approvals == ("call-1 (read)",)
        with pytest.raises(ProtocolError) as error:
            await server._client._ergonomics(
                "set_settings",
                {
                    "session_id": server.runtime.session_id,
                    "model": server.runtime.metadata.model,
                    "approval_mode": "ask",
                },
            )
        assert error.value.code == -32004
    finally:
        await _close(server, writer)
