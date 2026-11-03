"""Race and wire regressions for serve-owned approval reservation."""

from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path

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
from zeta.core.approval import ApprovalRequest
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


def _reservation_gate(
    server: ZetaServer, monkeypatch: pytest.MonkeyPatch
) -> asyncio.Event:
    entered = asyncio.Event()
    reserve = getattr(server.control, "_reserve", None)
    if reserve is not None:

        def gated_reserve(*args: object, **kwargs: object) -> bool:
            result = reserve(*args, **kwargs)
            entered.set()
            return result

        monkeypatch.setattr(server.control, "_reserve", gated_reserve)
    else:
        terminate = server.control._terminate_pending_approvals

        async def gated_terminate(*args: object, **kwargs: object) -> None:
            entered.set()
            await terminate(*args, **kwargs)

        monkeypatch.setattr(
            server.control, "_terminate_pending_approvals", gated_terminate
        )
    return entered


def _terminal_gate(
    server: ZetaServer, monkeypatch: pytest.MonkeyPatch
) -> asyncio.Event:
    entered = asyncio.Event()
    original = server.control._end_approval

    async def finish(*args: object, **kwargs: object) -> None:
        await original(*args, **kwargs)
        entered.set()

    monkeypatch.setattr(server.control, "_end_approval", finish)
    return entered


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
        reserved = _reservation_gate(server, monkeypatch)
        released = asyncio.create_task(server.control.controller_released())
        await asyncio.wait_for(reserved.wait(), TIMEOUT)
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
        terminal_entered = _terminal_gate(server, monkeypatch)
        terminal = asyncio.create_task(server.control._event(_end_event(call)))
        await asyncio.wait_for(terminal_entered.wait(), TIMEOUT)
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
        reserved = _reservation_gate(server, monkeypatch)
        expiry = asyncio.create_task(
            server.control._terminate_pending_approvals(reason="grace_expired")
        )
        await asyncio.wait_for(reserved.wait(), TIMEOUT)
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
    reserved = _reservation_gate(server, monkeypatch)
    closing = asyncio.create_task(server.control.close())
    await asyncio.wait_for(reserved.wait(), TIMEOUT)
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
        reserved = _reservation_gate(server, monkeypatch)
        terminal_entered = _terminal_gate(server, monkeypatch)
        terminal = asyncio.create_task(server.control._event(_end_event(call)))
        disconnect = asyncio.create_task(server.control.controller_released())
        ordered = (terminal, disconnect) if terminal_first else (disconnect, terminal)
        await asyncio.wait_for(
            asyncio.gather(reserved.wait(), terminal_entered.wait()), TIMEOUT
        )
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
@pytest.mark.parametrize("cleanup_first", [False, True])
async def test_delegated_cleanup_overlaps_worker_resolve_exact_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cleanup_first: bool
) -> None:
    server, _reader, writer, _foreground = await _approval_server(tmp_path)
    store = ConversationStore(tmp_path / "child")
    call = ToolCall("child-call", "read", {"path": "input"})
    store.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
    )
    policy = server.runtime.policy
    assert policy is not None
    request = ApprovalRequest(call.id, call, child_instance_id="child")
    policy.register_delegated(request, store, child_instance_id="child")
    await server.control._event(
        StreamEvent(
            StreamEventType.TOOL_APPROVAL_START,
            tool_call=call,
            data={"agent_instance_id": "child"},
        ),
        session_id="parent",
        background=True,
    )
    wire_id = server.control._approvals.wire_id(request.key)
    gate = _StoreGate(store, "resolve_approval")
    monkeypatch.setattr(store, gate.method, gate.call)
    try:
        if cleanup_first:
            cleanup = policy.cleanup_delegated("child")
            assert cleanup is not None
            await _wait_thread_gate(gate)
            with pytest.raises(ProtocolError) as loser:
                await server.control.resolve_approval("approve", wire_id)
            assert loser.value.code == -32006
            gate.release.set()
            await cleanup
            expected = "abort"
        else:
            approve = asyncio.create_task(
                server.control.resolve_approval("approve", wire_id)
            )
            await _wait_thread_gate(gate)
            assert policy.cleanup_delegated("child") is None
            gate.release.set()
            result = await approve
            assert result["decision"] == "approve"
            expected = "allow"
        await server.control.wait_for_approval_settlement()
        assert store.approval_resolution_data(call.id)["decision"] == expected
        rows = [entry for entry in store.entries if entry.type == "approval_resolution"]
        assert len(rows) == 1
        pending = await server.control.pending_approvals()
        assert all(item["request_id"] != wire_id for item in pending)
        assert policy._delegated == {}
    finally:
        gate.release.set()
        await _close(server, writer)
        store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("abort_first", [False, True])
async def test_same_tick_approval_reservation_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, abort_first: bool
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    events: list[str] = []
    original_notify = server.control._notify

    async def record(event: str, *args: object, **kwargs: object) -> bool:
        events.append(event)
        return await original_notify(event, *args, **kwargs)

    monkeypatch.setattr(server.control, "_notify", record)
    try:
        await server.control._event(_start_event(call))
        store = server.runtime.opened.store
        gate = _StoreGate(store, "resolve_approval")
        monkeypatch.setattr(store, gate.method, gate.call)
        if abort_first:
            abort = asyncio.create_task(server.control._terminate_pending_approvals())
            await _wait_thread_gate(gate)
            with pytest.raises(ProtocolError) as loser:
                await server.control.resolve_approval("approve", call.id)
            assert loser.value.code == -32006
            gate.release.set()
            await abort
        else:
            approve = asyncio.create_task(
                server.control.resolve_approval("approve", call.id)
            )
            await _wait_thread_gate(gate)
            abort = asyncio.create_task(server.control._terminate_pending_approvals())
            assert not abort.done()
            gate.release.set()
            result = await approve
            await abort
            assert result["decision"] == "approve"
        assert gate.thread_ids != [threading.get_ident()]
        rows = [
            entry
            for entry in server.runtime.opened.store.entries
            if entry.type == "approval_resolution"
        ]
        assert len(rows) == 1
        assert events.count("approval_end") == 1
    finally:
        gate.release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_always_tool_grant_waits_for_durable_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, call = await _approval_server(tmp_path)
    try:
        await server.control._event(_start_event(call))
        store = server.runtime.opened.store
        gate = _StoreGate(store, "resolve_approval")

        def fail_after_gate(*args: object, **kwargs: object) -> object:
            del args, kwargs
            gate.thread_ids.append(threading.get_ident())
            gate.entered.set()
            assert gate.release.wait(timeout=1)
            raise OSError("disk failed")

        monkeypatch.setattr(store, gate.method, fail_after_gate)
        approval = asyncio.create_task(
            _request(
                reader,
                writer,
                3,
                "approve",
                {"request_id": call.id, "scope": "always_tool"},
            )
        )
        await _wait_thread_gate(gate)
        assert server.runtime.policy is not None
        assert server.runtime.policy.session_allow == frozenset()
        gate.release.set()
        frames = await approval
        assert frames[-1]["error"]["code"] == -32000
        assert server.runtime.policy.session_allow == frozenset()
    finally:
        gate.release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_delegated_cleanup_during_request_history_write_publishes_no_ghost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, _foreground = await _approval_server(tmp_path)
    child = ConversationStore(tmp_path / "child-history-race")
    call = ToolCall("child-history-call", "read", {"path": "input"})
    child.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
    )
    policy = server.runtime.policy
    assert policy is not None
    request = ApprovalRequest(call.id, call, child_instance_id="child")
    events: list[str] = []
    original_notify = server.control._notify

    async def record_notify(event: str, *args: object, **kwargs: object) -> bool:
        events.append(event)
        return await original_notify(event, *args, **kwargs)

    monkeypatch.setattr(server.control, "_notify", record_notify)
    parent = server.runtime.opened.store
    gate = _StoreGate(parent, "append_approval_history")
    monkeypatch.setattr(parent, gate.method, gate.call)
    try:
        policy.register_delegated(request, child, child_instance_id="child")
        start = asyncio.create_task(
            server.control._event(
                StreamEvent(
                    StreamEventType.TOOL_APPROVAL_START,
                    tool_call=call,
                    data={"agent_instance_id": "child"},
                ),
                session_id="parent",
                background=True,
            )
        )
        await _wait_thread_gate(gate)
        cleanup = policy.cleanup_delegated("child")
        assert cleanup is not None
        await cleanup
        gate.release.set()
        await start
        await server.control.wait_for_approval_settlement()

        history = [
            entry.data
            for entry in parent.entries
            if entry.type == "approval_history"
        ]
        assert [row["event"] for row in history] == [
            "approval_request",
            "approval_end",
        ]
        assert history[-1]["data"] == {
            "decision": "abort",
            "reason": "child_cleanup",
        }
        assert events == []
        assert server.control.pending_approvals_now() == []
    finally:
        gate.release.set()
        await _close(server, writer)
        child.close()


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
        durable_resolve = store.resolve_approval
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

        call2 = ToolCall("call-committed", "read", {"path": "/tmp/committed"})
        store.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(call2)]),
            [(call2.id, call2)],
        )
        await server.control._event(_start_event(call2))
        events.clear()
        aborted.clear()

        def commit_then_fail(*args: object, **kwargs: object) -> object:
            durable_resolve(*args, **kwargs)
            raise OSError("acknowledgement lost")

        monkeypatch.setattr(store, "resolve_approval", commit_then_fail)
        committed = await server.control.resolve_approval("approve", call2.id)
        assert committed["decision"] == "approve"
        assert events == [("approval_end", None)]
        assert aborted == []
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
        responsive = asyncio.Event()
        asyncio.get_running_loop().call_soon(responsive.set)
        await asyncio.wait_for(responsive.wait(), TIMEOUT)
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
        responsive = asyncio.Event()
        asyncio.get_running_loop().call_soon(responsive.set)
        await asyncio.wait_for(responsive.wait(), TIMEOUT)
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
    server, reader, writer, call = await _approval_server(tmp_path)
    try:
        start, gate = await _gated_start(server, call, monkeypatch)
        reserved = _reservation_gate(server, monkeypatch)
        replacement = asyncio.create_task(
            _request(reader, writer, "replace", "new_session", {"provider": "codex"})
        )
        await asyncio.wait_for(reserved.wait(), TIMEOUT)
        gate.release.set()
        await asyncio.wait_for(asyncio.gather(start, replacement), TIMEOUT)
        assert await server.control.pending_approvals() == []
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_bound_delegated_never_touches_policy_registry(
    tmp_path: Path,
) -> None:
    server, _reader, writer, _call = await _approval_server(tmp_path)
    child = ConversationStore(tmp_path / "child")
    call = ToolCall("child-call", "read", {"path": "input"})
    child.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
    )
    request = ApprovalRequest(call.id, call, child_instance_id="child")
    policy = server.runtime.policy
    assert policy is not None
    closed = False
    late: ConversationStore | None = None
    try:
        policy.register_delegated(request, child, child_instance_id="child")
        assert policy._delegated == {}
        cleanup = policy.cleanup_delegated("child")
        assert policy._delegated == {}
        if cleanup is not None:
            await cleanup

        await _close(server, writer)
        closed = True
        late = ConversationStore(tmp_path / "late-child")
        late_call = ToolCall("late-call", "read", {"path": "late"})
        late.append_message_with_approval_requests(
            Message(MessageRole.ASSISTANT, [ToolUseContent(late_call)]),
            [(late_call.id, late_call)],
        )
        late_request = ApprovalRequest(
            late_call.id, late_call, child_instance_id="late-child"
        )
        policy.register_delegated(
            late_request, late, child_instance_id="late-child"
        )
        await server.control.wait_for_approval_settlement()
        assert late.approval_resolution_data(late_call.id)["decision"] == "abort"
        policy.cleanup_delegated("late-child")
        assert policy._delegated == {}
    finally:
        if not closed:
            await _close(server, writer)
        if late is not None:
            late.close()
        child.close()


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
