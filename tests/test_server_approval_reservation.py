"""Race and wire regressions for serve-owned approval reservation."""

from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import threading
import uuid
from functools import partial
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
from zeta.server.server import LOADED_SESSION_MUTATIONS
from zeta.server.slash_commands import ServerSlashSession
from zeta.tools.agent.approval import ChildApprovalPolicy


def test_approval_lifecycle_state_has_one_writer_in_all_production_modules() -> None:
    root = Path(__file__).parents[1] / "src"
    lifecycle_path = root / "zeta" / "server" / "approval_lifecycle.py"
    record_fields = {
        "target",
        "wire_id",
        "generation",
        "token",
        "phase",
        "state",
        "disposition",
        "request",
        "history_published",
        "published",
        "settlement",
    }
    violations: list[str] = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        parents = {
            child: parent
            for parent in ast.walk(tree)
            for child in ast.iter_child_nodes(parent)
        }
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                targets = (
                    node.targets if isinstance(node, ast.Assign) else [node.target]
                )
            for target in targets:
                if not isinstance(target, ast.Attribute):
                    continue
                function = parents.get(node)
                while function is not None and not isinstance(
                    function, (ast.FunctionDef, ast.AsyncFunctionDef)
                ):
                    function = parents.get(function)
                function_name = function.name if function is not None else "<module>"
                writes_lifecycle_state = target.attr == "_state" and (
                    path == lifecycle_path
                    or "approval" in ast.unparse(target.value).lower()
                )
                if writes_lifecycle_state and not (
                    path == lifecycle_path and function_name in {"__init__", "_swap"}
                ):
                    violations.append(f"{path}:{target.lineno}: writes _state")
                owner = target.value.id if isinstance(target.value, ast.Name) else ""
                if target.attr in record_fields and "record" in owner.lower():
                    violations.append(
                        f"{path}:{target.lineno}: writes approval record {target.attr}"
                    )
        for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
            constructor = call.func.id if isinstance(call.func, ast.Name) else ""
            if constructor not in {"ApprovalMachineState", "ApprovalRecord"}:
                continue
            function = parents.get(call)
            while function is not None and not isinstance(
                function, (ast.FunctionDef, ast.AsyncFunctionDef)
            ):
                function = parents.get(function)
            function_name = function.name if function is not None else "<module>"
            allowed_functions = {
                "ApprovalMachineState": {"__init__", "_swap"},
                "ApprovalRecord": {"_new_record"},
            }
            if (
                path != lifecycle_path
                or function_name not in allowed_functions[constructor]
            ):
                violations.append(f"{path}:{call.lineno}: constructs {constructor}")
    assert not violations, (
        "approval state writes outside lifecycle swap seam: " + ", ".join(violations)
    )


@pytest.mark.asyncio
async def test_approval_lifecycle_transition_rejects_unknown_event(
    tmp_path: Path,
) -> None:
    server, _reader, writer, _call = await _approval_server(tmp_path)
    try:
        with pytest.raises(AssertionError, match="unhandled approval lifecycle event"):
            await server.control._approvals.transition(object())  # type: ignore[arg-type]
    finally:
        await _close(server, writer)


class _AbortEvent:
    def __init__(self) -> None:
        self._event = asyncio.Event()

    def is_set(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def set(self, **kwargs: object) -> None:
        del kwargs
        self._event.set()


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
    *,
    protocol_version: str = "1.0",
    features: list[str] | None = None,
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
    hello: dict[str, object] = {"protocol_version": protocol_version}
    if features is not None:
        hello["features"] = features
    await _request(reader, writer, 1, "hello", hello)
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
        # the async durable-writer seam does not expose the old registry-thread gate;
        # cleanup may overlap the off-loop history append without publishing a ghost.
        while ("child", call.id) not in server.control._approvals.state.records:
            await asyncio.sleep(0)
        cleanup = policy.cleanup_delegated("child")
        cleanup_task = asyncio.ensure_future(cleanup) if cleanup is not None else None
        await asyncio.gather(start, *(task for task in [cleanup_task] if task))
        await server.control.wait_for_approval_settlement()

        history = [
            entry.data for entry in parent.entries if entry.type == "approval_history"
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
        await _close(server, writer)
        child.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["foreground", "delegated"])
async def test_serve_bound_approval_writes_only_through_settlement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    server, _reader, writer, foreground = await _approval_server(tmp_path)
    child: ConversationStore | None = None
    try:
        policy = server.runtime.policy
        assert policy is not None
        if path == "foreground":
            call = foreground
            store = server.runtime.opened.store
            await server.control._event(_start_event(call))
            abort = partial(
                policy.abort_or_winner,
                call.id,
                capability=object(),  # type: ignore[arg-type]
            )
        else:
            child = ConversationStore(tmp_path / "guard-child")
            call = ToolCall("guard-child-call", "read", {"path": "input"})
            child.append_message_with_approval_requests(
                Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
                [(call.id, call)],
            )
            child_policy = ChildApprovalPolicy(policy, child, "child", "child")
            request = ApprovalRequest(call.id, call, child_instance_id="child")
            policy.register_delegated(request, child, child_instance_id="child")
            await server.control._event(
                StreamEvent(
                    StreamEventType.TOOL_APPROVAL_START,
                    tool_call=call,
                    data={"agent_instance_id": "child"},
                ),
                session_id="parent",
                background=True,
            )
            store = child
            abort = partial(
                child_policy.abort_or_winner,
                call.id,
                capability=object(),  # type: ignore[arg-type]
            )

        writers: list[tuple[str, ...]] = []
        original_resolve = store.resolve_approval

        def record_writer(*args: object, **kwargs: object) -> bool:
            writers.append(tuple(frame.function for frame in inspect.stack()))
            return original_resolve(*args, **kwargs)

        monkeypatch.setattr(store, "resolve_approval", record_writer)
        assert abort() is None
        await asyncio.sleep(0)
        await server.control.wait_for_approval_settlement()

        assert writers
        assert all("persist" in caller for caller in writers)
    finally:
        await _close(server, writer)
        if child is not None:
            child.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["foreground", "delegated"])
async def test_failed_settlement_cancellation_recovery_balances_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str
) -> None:
    server, _reader, writer, foreground = await _approval_server(tmp_path)
    child: ConversationStore | None = None
    notifications: list[str] = []
    original_notify = server.control._notify

    async def record_notify(event: str, *args: object, **kwargs: object) -> bool:
        notifications.append(event)
        return await original_notify(event, *args, **kwargs)

    monkeypatch.setattr(server.control, "_notify", record_notify)
    try:
        policy = server.runtime.policy
        assert policy is not None and server.runtime.loop is not None
        if path == "foreground":
            call = foreground
            store = server.runtime.opened.store
            authorizer = policy
            key: str | tuple[str, str] = call.id
        else:
            child = ConversationStore(tmp_path / "recover-child")
            call = ToolCall("recover-child-call", "bash", {"command": "true"})
            authorizer = ChildApprovalPolicy(policy, child, "child", "child")
            store = child
            key = ("child", call.id)

        capability = server.runtime.loop.tool_registry.resolve_tool_call_capability(
            call
        )
        if child is not None:
            request = authorizer.prepare(call, capability=capability)
            assert request is not None
            child.append_message_with_approval_requests(
                Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
                [(call.id, call, request.audit_facts(), request.audit_display())],
            )
        signal = _AbortEvent()
        authorization = asyncio.create_task(
            authorizer.authorize(call, signal, capability=capability)
        )
        if path == "delegated":
            while key not in server.control._approvals.state.records:
                await asyncio.sleep(0)
        await server.control._event(
            StreamEvent(
                StreamEventType.TOOL_APPROVAL_START,
                tool_call=call,
                data={"agent_instance_id": "child"} if path == "delegated" else {},
            ),
            session_id="parent" if path == "delegated" else None,
            background=path == "delegated",
        )

        original_resolve = store.resolve_approval
        writes = 0

        def fail_twice_then_recover(*args: object, **kwargs: object) -> bool:
            nonlocal writes
            writes += 1
            if writes <= 2:
                raise OSError("injected settlement failure")
            return original_resolve(*args, **kwargs)

        monkeypatch.setattr(store, "resolve_approval", fail_twice_then_recover)
        monkeypatch.setattr(server.runtime.loop, "abort", signal.set)
        request_id = server.control._approvals.wire_id(key)
        with pytest.raises(ProtocolError) as error:
            await server.control.resolve_approval("approve", request_id)
        assert error.value.code == -32000
        await server.control.request_approval_abort(
            call.id,
            child_instance_id="child" if path == "delegated" else None,
            reason="child_cleanup" if path == "delegated" else "turn_canceled",
        )
        assert await asyncio.wait_for(authorization, TIMEOUT) is None
        await server.control.wait_for_approval_settlement()

        assert writes == 3
        assert store.approval_resolution_data(call.id) == {
            "request_id": call.id,
            "decision": "abort",
            "reason": "child_cleanup" if path == "delegated" else "turn_canceled",
        }
        assert notifications.count("approval_request") == 1
        assert notifications.count("approval_end") == 1
        if path == "delegated":
            history = [
                entry.data
                for entry in server.runtime.opened.store.entries
                if entry.type == "approval_history"
            ]
            assert [row["event"] for row in history] == [
                "approval_request",
                "approval_end",
            ]
    finally:
        await _close(server, writer)
        if child is not None:
            child.close()


async def _make_unhealthy(
    server: ZetaServer, call: ToolCall, monkeypatch: pytest.MonkeyPatch
) -> None:
    await server.control._event(_start_event(call))
    store = server.runtime.opened.store
    monkeypatch.setattr(
        store,
        "resolve_approval",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failed")),
    )
    with pytest.raises(ProtocolError) as error:
        await server.control.resolve_approval("approve", call.id)
    assert error.value.code == -32000
    assert server.control.unhealthy()


@pytest.mark.asyncio
async def test_protocol_1_0_persistent_approval_failure_keeps_legacy_status_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, call = await _approval_server(tmp_path)
    try:
        await _make_unhealthy(server, call, monkeypatch)
        await _read_raw(reader)  # approval_request
        event = json.loads((await _read_raw(reader)).decode())
        assert event["method"] == "event"
        assert event["params"]["event"] == "error"
        assert event["params"]["error"]["code"] == "server_error"
        response = (await _request(reader, writer, 9, "status"))[-1]
        assert response == {
            "jsonrpc": "2.0",
            "id": 9,
            "result": {
                "session": response["result"]["session"],
                "state": "idle",
                "explicit_finish": False,
                "pending_approvals": [],
                "usage": {},
                "compaction_markers": 0,
            },
        }
        assert "health" not in response["result"]
        with pytest.raises(ProtocolError) as error:
            await server.control.resolve_approval("approve", call.id)
        assert error.value.code == -32000
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_observer_persistent_approval_failure_keeps_legacy_status_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _controller_reader, controller_writer, call = await _approval_server(
        tmp_path
    )
    observer_reader, observer_writer = await asyncio.open_unix_connection(
        str(server.socket_path)
    )
    await _request(
        observer_reader,
        observer_writer,
        10,
        "hello",
        {
            "protocol_version": "1.2",
            "role": "observer",
            "features": ["serve_observers"],
        },
    )
    try:
        await _make_unhealthy(server, call, monkeypatch)
        response = (await _request(observer_reader, observer_writer, 11, "status"))[-1]
        result = response["result"]
        assert set(result) == {
            "session",
            "state",
            "explicit_finish",
            "pending_approvals",
            "usage",
            "compaction_markers",
        }
        assert result["state"] == "idle"
        mutation = (
            await _request(
                observer_reader, observer_writer, 12, "send", {"text": "nope"}
            )
        )[-1]
        assert mutation["error"]["data"]["code"] == "observer_read_only"
    finally:
        await _close(server, controller_writer)
        observer_writer.close()
        await observer_writer.wait_closed()


@pytest.mark.asyncio
async def test_unhealthy_session_preserves_delivery_id_idempotency_before_mutation_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, call = await _approval_server(tmp_path)
    try:
        store = server.runtime.opened.store
        session_id = server.runtime.session_id
        store.append_many(
            [
                (
                    "client_delivery",
                    {
                        "delivery_id": "evicted-id",
                        "method": "send",
                        "status": "delivered",
                        "outcome": {"accepted": True, "session_id": session_id},
                    },
                ),
                *[
                    (
                        "client_delivery",
                        {
                            "delivery_id": f"delivery-filler-{index}",
                            "method": "steer",
                            "status": "queued",
                            "outcome": {"accepted": True},
                        },
                    )
                    for index in range(1_000)
                ],
                (
                    "client_delivery",
                    {
                        "delivery_id": "retained-send",
                        "method": "send",
                        "status": "delivered",
                        "outcome": {"accepted": True, "session_id": session_id},
                    },
                ),
                (
                    "client_delivery",
                    {
                        "delivery_id": "retained-steer",
                        "method": "steer",
                        "status": "queued",
                        "outcome": {"accepted": True},
                    },
                ),
            ]
        )
        await _make_unhealthy(server, call, monkeypatch)
        features = frozenset({"delivery_id"})
        deliveries = server.control.deliveries
        effects: list[str] = []

        async def record_message(*args: object, **kwargs: object) -> None:
            del args, kwargs
            effects.append("message")

        monkeypatch.setattr(server.control, "user_message", record_message)
        monkeypatch.setattr(
            server.control, "start_turn", lambda *args, **kwargs: effects.append("turn")
        )
        assert server.runtime.loop is not None
        monkeypatch.setattr(
            server.runtime.loop,
            "steer",
            lambda *args, **kwargs: effects.append("steer"),
        )

        assert await deliveries.send(
            {"text": "retry", "delivery_id": "retained-send"}, features
        ) == {"accepted": True, "session_id": session_id, "duplicate": True}
        assert await deliveries.steer(
            {"text": "retry", "delivery_id": "retained-steer"}, features
        ) == {"accepted": True, "duplicate": True}
        for method, params in (
            ("send", {"text": "retry", "delivery_id": "evicted-id"}),
            ("steer", {"text": "retry", "delivery_id": "evicted-id"}),
        ):
            with pytest.raises(ProtocolError) as error:
                await getattr(deliveries, method)(params, features)
            assert error.value.code == -32010
        assert (
            deliveries.status({"delivery_id": "retained-send"}, features)["status"]
            == "delivered"
        )
        assert (
            deliveries.status({"delivery_id": "evicted-id"}, features)["status"]
            == "evicted"
        )
        assert (
            deliveries.status({"delivery_id": "unknown"}, features)["status"]
            == "unknown"
        )
        for params in (
            {"text": "new", "delivery_id": "new-id"},
            {"text": "unkeyed"},
        ):
            with pytest.raises(ProtocolError) as error:
                await deliveries.send(params, features)
            assert error.value.code == -32000
        assert effects == []

        response = (
            await _request(reader, writer, 90, "resume", {"session_id": session_id})
        )[-1]
        assert "result" in response
        assert not server.control.unhealthy()
        result = await deliveries.send(
            {"text": "recovered", "delivery_id": "recovered-id"}, features
        )
        assert result == {"accepted": True, "session_id": session_id}
        assert effects == ["message", "turn"]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_unhealthy_session_rejects_every_loaded_session_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    features = ["abort_scope", "delivery_id", "turn_context"]
    server, reader, writer, call = await _approval_server(
        tmp_path, protocol_version="1.2", features=features
    )
    try:
        await _make_unhealthy(server, call, monkeypatch)
        session_id = server.runtime.session_id
        params_by_method: dict[str, dict[str, object]] = {
            "abort": {},
            "approve": {"request_id": call.id},
            "clear_steering": {},
            "deny": {"request_id": call.id},
            "fork_message": {"session_id": session_id, "message_id": "missing"},
            "rename_session": {"session_id": session_id, "name": "changed"},
            "send": {"text": "blocked"},
            "send_images": {"session_id": session_id, "text": "blocked", "images": []},
            "set_compaction": {"session_id": session_id, "mode": "eviction"},
            "set_settings": {
                "session_id": session_id,
                "model": server.runtime.model,
                "approval_mode": "ask",
            },
            "set_turn_context": {"text": "blocked"},
            "slash_run": {"session_id": session_id, "text": "/status"},
            "steer": {"text": "blocked"},
            "switch_branch": {"session_id": session_id, "head_id": "missing"},
        }
        assert params_by_method.keys() == LOADED_SESSION_MUTATIONS
        old_state = server.runtime.state
        old_metadata = server.runtime.metadata.to_dict()
        old_entries = tuple(server.runtime.opened.store.entries)
        for request_id, method in enumerate(LOADED_SESSION_MUTATIONS, 20):
            response = (
                await _request(reader, writer, request_id, method, params_by_method[method])
            )[-1]
            assert "error" in response, (method, response)
            assert response["error"]["code"] == -32000, method
            assert server.runtime.state is old_state
            assert server.runtime.metadata.to_dict() == old_metadata
            assert tuple(server.runtime.opened.store.entries) == old_entries

        read_requests = {
            "delivery_status": {"delivery_id": "unknown"},
            "list_sessions": {},
            "model_catalog": {"session_id": session_id},
            "session_history": {"session_id": session_id},
            "session_settings": {"session_id": session_id},
            "session_tree": {"session_id": session_id},
            "slash_list": {"session_id": session_id},
            "status": {},
        }
        for request_id, (method, params) in enumerate(read_requests.items(), 50):
            response = (await _request(reader, writer, request_id, method, params))[-1]
            assert "result" in response, method

        recovered = (
            await _request(reader, writer, 70, "new_session", {"provider": "codex"})
        )[-1]
        assert "result" in recovered
        assert not server.control.unhealthy()
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("recovery", ["new_session", "missing_resume", "failed_resume"])
async def test_failed_recovery_keeps_session_unhealthy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, recovery: str
) -> None:
    server, reader, writer, call = await _approval_server(tmp_path)
    try:
        await _make_unhealthy(server, call, monkeypatch)
        old_state = server.runtime.state
        method = "new_session" if recovery == "new_session" else "resume"
        params: dict[str, str] = {"provider": "codex"}
        if recovery == "missing_resume":
            params = {"session_id": "missing-session"}
        elif recovery == "failed_resume":
            opened = await asyncio.to_thread(
                server.runtime.manager.create,
                provider="codex",
                model="offline",
                cwd=tmp_path,
            )
            params = {"session_id": opened.metadata.session_id}
            await asyncio.to_thread(opened.store.close)
        if recovery != "missing_resume":

            async def fail_setup(*args: object, **kwargs: object) -> None:
                del args, kwargs
                raise OSError("setup failed")

            monkeypatch.setattr(server.runtime, "_run_setup", fail_setup)

        response = (await _request(reader, writer, 9, method, params))[-1]
        assert "error" in response
        assert server.runtime.state is old_state
        assert server.control.unhealthy()
        mutation = (await _request(reader, writer, 10, "send", {"text": "blocked"}))[-1]
        assert mutation["error"]["code"] == -32000
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_unhealthy_same_session_resume_reopens_and_clears_after_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, call = await _approval_server(tmp_path)
    try:
        await _make_unhealthy(server, call, monkeypatch)
        session_id = server.runtime.session_id
        old_store = server.runtime.opened.store
        response = (
            await _request(reader, writer, 9, "resume", {"session_id": session_id})
        )[-1]
        assert "result" in response
        assert server.runtime.session_id == session_id
        assert server.runtime.opened.store is not old_store
        assert not server.control.unhealthy()
        assert (await _request(reader, writer, 10, "send", {"text": "recovered"}))[-1][
            "result"
        ]["accepted"] is True
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_unhealthy_same_session_resume_failure_does_not_clear_health(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, reader, writer, call = await _approval_server(tmp_path)
    try:
        await _make_unhealthy(server, call, monkeypatch)
        session_id = server.runtime.session_id
        monkeypatch.setattr(
            server.runtime,
            "_run_setup",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("replay failed")),
        )
        response = (
            await _request(reader, writer, 9, "resume", {"session_id": session_id})
        )[-1]
        assert response["error"]["code"] == -32000
        assert server.control.unhealthy()
        assert server.runtime.opened is None
        mutation = (await _request(reader, writer, 10, "send", {"text": "blocked"}))[-1]
        assert mutation["error"]["code"] == -32003
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_forced_resume_cancel_during_close_finishes_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    old_state = server.runtime.state
    assert old_state is not None
    old_store = old_state.opened.store
    session_dir = old_store.session_dir
    entered = asyncio.Event()
    release = asyncio.Event()
    completed = asyncio.Event()
    original_close = old_state.loop.close

    async def gated_close(*args: object, **kwargs: object) -> tuple[str, ...]:
        entered.set()
        await release.wait()
        result = await original_close(*args, **kwargs)
        completed.set()
        return result

    monkeypatch.setattr(old_state.loop, "close", gated_close)
    try:
        await _make_unhealthy(server, call, monkeypatch)
        resume = asyncio.create_task(
            server.runtime.resume_session(old_state.session_id, force=True)
        )
        await asyncio.wait_for(entered.wait(), TIMEOUT)
        resume.cancel()
        await asyncio.sleep(0)
        assert server.runtime.state is old_state
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await resume
        assert completed.is_set()
        assert server.runtime.state is None
        assert old_store._closed
        assert not old_state.loop._background_child_watchers
        assert not [task for task in old_state.loop._tracked_tasks if not task.done()]
        assert old_state.loop.step_delivery is None
        mount_task = old_state.loop._mcp_mount_task
        assert mount_task is None or mount_task.done()
        reopened = server.runtime.manager.open(old_state.session_id)
        try:
            assert reopened.metadata.session_id == old_state.session_id
        finally:
            reopened.store.close()
        fd_directory = "/proc/self/fd" if os.path.isdir("/proc/self/fd") else "/dev/fd"
        leaked_targets = []
        for name in os.listdir(fd_directory):
            try:
                target = os.path.realpath(f"{fd_directory}/{name}")
            except OSError:
                continue
            if target == str(session_dir) or target.startswith(f"{session_dir}{os.sep}"):
                leaked_targets.append(target)
        assert leaked_targets == []
    finally:
        release.set()
        if not old_store._closed:
            await server.runtime._close_state(old_state)
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["close", "setup", "install"])
async def test_forced_resume_cancellation_at_each_boundary_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        await _make_unhealthy(server, call, monkeypatch)
        runtime = server.runtime
        session_id = runtime.session_id
        original_close = runtime._close_state
        original_setup = runtime._run_setup
        cancelled = asyncio.CancelledError()

        async def cancel_close(state):
            await original_close(state)
            if boundary == "close":
                raise cancelled

        async def cancel_setup(*args, **kwargs):
            if boundary == "setup":
                raise cancelled
            return await original_setup(*args, **kwargs)

        def cancel_install(state):
            if boundary == "install":
                raise cancelled

        monkeypatch.setattr(runtime, "_close_state", cancel_close)
        monkeypatch.setattr(runtime, "_run_setup", cancel_setup)
        monkeypatch.setattr(runtime, "_bind_background_event_sink", cancel_install)
        with pytest.raises(asyncio.CancelledError):
            await runtime.resume_session(session_id, force=True)
        assert runtime.opened is None
        assert runtime.state is None
        assert server.control.unhealthy()
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
        assert aborted == [True]

        await server.control.request_approval_abort(
            call.id,
            reason="turn_canceled",
        )
        await server.control.wait_for_approval_settlement()

        assert events == [
            ("error", "server_error"),
            ("approval_end", None),
        ]
        assert server.control.pending_approvals_now() == []
        assert not server.control.has_pending_approvals()
        assert server.control._approvals.state.records == {}
        with pytest.raises(ProtocolError) as later:
            await server.control.resolve_approval("approve", call.id)
        assert later.value.code == -32000

    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_persistent_failure_removes_many_live_child_approvals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, _call = await _approval_server(tmp_path)
    stores: list[ConversationStore] = []
    events: list[str] = []
    original_notify = server.control._notify

    async def record_notify(event: str, *args: object, **kwargs: object) -> bool:
        events.append(event)
        return await original_notify(event, *args, **kwargs)

    monkeypatch.setattr(server.control, "_notify", record_notify)
    assert server.runtime.loop is not None
    monkeypatch.setattr(server.runtime.loop, "abort", lambda **kwargs: None)
    policy = server.runtime.policy
    assert policy is not None
    try:
        for index in range(4):
            child_id = f"child-{index}"
            call = ToolCall(f"child-call-{index}", "read", {"path": "input"})
            store = ConversationStore(tmp_path / child_id)
            stores.append(store)
            store.append_message_with_approval_requests(
                Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
                [(call.id, call)],
            )
            request = ApprovalRequest(call.id, call, child_instance_id=child_id)
            policy.register_delegated(request, store, child_instance_id=child_id)
            await server.control._event(
                StreamEvent(
                    StreamEventType.TOOL_APPROVAL_START,
                    tool_call=call,
                    data={"agent_instance_id": child_id},
                ),
                session_id="parent",
                background=True,
            )
            monkeypatch.setattr(
                store,
                "resolve_approval",
                lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failed")),
            )

        first_key = ("child-0", "child-call-0")
        with pytest.raises(ProtocolError) as error:
            await server.control.resolve_approval(
                "approve", server.control._approvals.wire_id(first_key)
            )
        assert error.value.code == -32000

        for index in range(4):
            await server.control.request_approval_abort(
                f"child-call-{index}",
                child_instance_id=f"child-{index}",
                reason="child_cleanup",
            )
        await server.control.wait_for_approval_settlement()

        assert events.count("error") == 1
        assert events.count("approval_end") == 4
        assert not server.control.has_pending_approvals()
        assert server.control._approvals.state.records == {}
    finally:
        await _close(server, writer)
        for store in stores:
            store.close()


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
        policy.register_delegated(late_request, late, child_instance_id="late-child")
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


_APPROVAL_SETTLEMENT_CASES = [
    *(
        pytest.param(
            requested,
            "none",
            write,
            "foreground",
            id=f"foreground-{requested}-none-{write}",
        )
        for requested in ("allow", "always_tool", "deny")
        for write in ("ok", "raise")
    ),
    *(
        pytest.param(
            requested,
            "cancel",
            write,
            "foreground",
            id=f"foreground-{requested}-cancel-{write}",
        )
        for requested in ("allow", "always_tool", "deny")
        for write in ("ok", "raise")
    ),
    pytest.param(
        "allow", "disconnect", "ok", "foreground", id="foreground-allow-disconnect-ok"
    ),
    pytest.param("deny", "expiry", "ok", "foreground", id="foreground-deny-expiry-ok"),
    pytest.param(
        "always_tool",
        "shutdown",
        "ok",
        "foreground",
        id="foreground-always_tool-shutdown-ok",
    ),
    *(
        pytest.param(
            requested, event, "ok", "delegated", id=f"delegated-{requested}-{event}-ok"
        )
        for requested in ("allow", "deny")
        for event in ("none", "child_cleanup")
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "requested, competing_event, write_result, path", _APPROVAL_SETTLEMENT_CASES
)
async def test_confirmed_durable_outcome_controls_all_approval_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    requested: str,
    competing_event: str,
    write_result: str,
    path: str,
) -> None:
    """Cover feasible ownership races; always_tool is foreground-only, and child cleanup is delegated-only."""
    server, reader, writer, foreground_call = await _approval_server(tmp_path)
    child: ConversationStore | None = None
    release = threading.Event()
    try:
        policy = server.runtime.policy
        assert policy is not None
        if path == "foreground":
            call = foreground_call
            store = server.runtime.opened.store
            await server.control._event(_start_event(call))
            request_id = call.id
        else:
            child = ConversationStore(tmp_path / "matrix-child")
            call = ToolCall("matrix-child-call", "read", {"path": "input"})
            child.append_message_with_approval_requests(
                Message(MessageRole.ASSISTANT, [ToolUseContent(call)]),
                [(call.id, call)],
            )
            request = ApprovalRequest(call.id, call, child_instance_id="child")
            policy.register_delegated(request, child, child_instance_id="child")
            await server.control._event(
                StreamEvent(
                    StreamEventType.TOOL_APPROVAL_START,
                    tool_call=call,
                    data={"agent_instance_id": "child"},
                ),
                session_id="parent",
                background=True,
            )
            store = child
            request_id = server.control._approvals.wire_id(request.key)

        original_resolve = store.resolve_approval
        entered = threading.Event()
        calls = 0
        writer_paths: list[tuple[str, ...]] = []

        def gated_resolve(*args: object, **kwargs: object) -> bool:
            nonlocal calls
            calls += 1
            writer_paths.append(tuple(frame.function for frame in inspect.stack()))
            entered.set()
            assert release.wait(timeout=1)
            if write_result == "raise" and calls == 1:
                raise OSError("matrix write failed")
            return original_resolve(*args, **kwargs)

        monkeypatch.setattr(store, "resolve_approval", gated_resolve)
        method = "deny" if requested == "deny" else "approve"
        params = {"request_id": request_id}
        if requested == "always_tool":
            params["scope"] = "always_tool"
        rpc = asyncio.create_task(_request(reader, writer, 40, method, params))
        assert await asyncio.to_thread(entered.wait, TIMEOUT)

        reasons = {
            "cancel": "turn_canceled",
            "disconnect": "controller_disconnected",
            "child_cleanup": "child_cleanup",
            "expiry": "grace_expired",
            "shutdown": "controller_disconnected",
        }
        if competing_event != "none":
            original_resolve(call.id, "abort", reason=reasons[competing_event])
        release.set()
        frames = await rpc
        await server.control.wait_for_approval_settlement()

        resolution = store.approval_resolution_data(call.id)
        assert resolution is not None
        response = frames[-1]
        assert "error" not in response
        expected_rpc = (
            "approve" if resolution["decision"] == "allow" else resolution["decision"]
        )
        assert response["result"]["decision"] == expected_rpc
        granted = policy.session_allow != frozenset()
        assert granted is (
            requested == "always_tool" and resolution["decision"] == "allow"
        )
        key = call.id if path == "foreground" else ("child", call.id)
        assert key not in server.control._approvals.state.records
        assert writer_paths
        assert all("persist" in caller for caller in writer_paths)
        if path == "delegated":
            history = [
                entry.data
                for entry in server.runtime.opened.store.entries
                if entry.type == "approval_history"
            ]
            assert [row["event"] for row in history] == [
                "approval_request",
                "approval_end",
            ]
    finally:
        release.set()
        await _close(server, writer)
        if child is not None:
            child.close()


@pytest.mark.asyncio
async def test_persistent_approval_failure_aborts_session_owned_children(
    tmp_path: Path,
) -> None:
    server, _reader, writer, _foreground = await _approval_server(tmp_path)
    child = ConversationStore(tmp_path / "session-child")
    call = ToolCall("session-child-call", "read", {"path": "input"})
    child.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
    )
    policy = server.runtime.policy
    assert policy is not None
    request = ApprovalRequest(call.id, call, child_instance_id="child")
    policy.register_delegated(request, child, child_instance_id="child")
    try:
        await server.control._event(
            StreamEvent(
                StreamEventType.TOOL_APPROVAL_START,
                tool_call=call,
                data={"agent_instance_id": "child"},
            ),
            session_id="parent",
            background=True,
        )
        await server.control.controller_released()
        resolution = child.approval_resolution_data(call.id)
        assert resolution is not None
        assert resolution["decision"] == "abort"
        assert resolution["reason"] in {
            "child_cleanup",
            "controller_disconnected",
        }
        assert not server.control._approvals.state.records
        assert not server.control.has_pending_approvals()
    finally:
        await _close(server, writer)
        child.close()


@pytest.mark.asyncio
async def test_concurrent_delegated_history_uses_store_durable_writer(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path / "concurrent-history")
    tool_call = ToolCall("concurrent-call", "read", {"path": "input"})

    async def write_pair(request_id: str) -> None:
        request = {
            "event": "approval_request",
            "request_id": request_id,
            "tool_call": tool_call.to_dict(),
            "delegated": True,
            "agent_instance_id": request_id,
        }
        end = {
            "event": "approval_end",
            "request_id": request_id,
            "tool_call": tool_call.to_dict(),
            "data": {"decision": "abort", "reason": "child_cleanup"},
            "reason": "child_cleanup",
        }
        await store.append_approval_history_async(request)
        await store.append_approval_history_async(end)

    try:
        await asyncio.gather(write_pair("wire-a"), write_pair("wire-b"))
        history = [
            entry.data for entry in store.entries if entry.type == "approval_history"
        ]
        assert len(history) == 4
        for request_id in ("wire-a", "wire-b"):
            rows = [row for row in history if row["request_id"] == request_id]
            assert [row["event"] for row in rows] == [
                "approval_request",
                "approval_end",
            ]
    finally:
        store.close()


@pytest.mark.asyncio
async def test_approval_lifecycle_has_no_records_after_release(
    tmp_path: Path,
) -> None:
    server, _reader, writer, call = await _approval_server(tmp_path)
    try:
        await server.control._event(_start_event(call))
        await server.control.controller_released()
        await server.control.wait_for_approval_settlement()
        assert server.control._approvals.state.records == {}
        assert server.control._approvals.active_keys() == ()
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_persistence_failure_requests_turn_cancel_without_awaiting_reentrant_transition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, _call = await _approval_server(tmp_path)
    try:
        called: list[bool] = []
        monkeypatch.setattr(
            server.control,
            "request_turn_failure",
            lambda: called.append(True),
        )
        server.control.request_turn_failure()
        await asyncio.sleep(0)
        assert called == [True]
    finally:
        await _close(server, writer)


async def test_failed_delegated_request_write_settles_cleanup_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, _foreground = await _approval_server(tmp_path)
    child = ConversationStore(tmp_path / "failed-start-child")
    call = ToolCall("failed-start-call", "read", {"path": "input"})
    child.append_message_with_approval_requests(
        Message(MessageRole.ASSISTANT, [ToolUseContent(call)]), [(call.id, call)]
    )
    policy = server.runtime.policy
    assert policy is not None
    request = ApprovalRequest(call.id, call, child_instance_id="child")
    parent = server.runtime.opened.store

    async def fail_history(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("request history failed")

    monkeypatch.setattr(parent, "append_approval_history_async", fail_history)
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
        while ("child", call.id) not in server.control._approvals.state.records:
            await asyncio.sleep(0)
        cleanup = policy.cleanup_delegated("child")
        cleanup_task = asyncio.ensure_future(cleanup) if cleanup is not None else None
        await asyncio.gather(start, *(task for task in [cleanup_task] if task))

        assert child.approval_resolution_data(call.id) == {
            "request_id": call.id,
            "decision": "abort",
            "reason": "child_cleanup",
        }
        assert ("child", call.id) not in server.control._approvals.state.records
        history = [
            entry.data for entry in parent.entries if entry.type == "approval_history"
        ]
        assert history == []
    finally:
        await _close(server, writer)
        child.close()

async def _wait_controller_release(server: ZetaServer) -> None:
    for _ in range(100):
        if server.connections is not None and server.connections.current_controller() is None:
            return
        await asyncio.sleep(0)
    raise AssertionError("controller was not released")


@pytest.mark.asyncio
async def test_approval_reconnect_drop_replay_and_approve_same_id(
    tmp_path: Path,
) -> None:
    server, reader, writer, call = await _approval_server(
        tmp_path, protocol_version="1.2", features=["approval_reconnect"]
    )
    replay_reader: asyncio.StreamReader | None = None
    replay_writer: asyncio.StreamWriter | None = None
    try:
        await server.control._event(_start_event(call))
        request = json.loads(await _read_raw(reader))["params"]
        assert request["request_id"] == call.id
        writer.close()
        await writer.wait_closed()
        await _wait_controller_release(server)

        replay_reader, replay_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        hello = await _request(
            replay_reader,
            replay_writer,
            10,
            "hello",
            {"protocol_version": "1.2", "features": ["approval_reconnect"]},
        )
        assert hello[-1]["result"]["capabilities"]["features"] == [
            "approval_reconnect"
        ]
        replay = json.loads(await _read_raw(replay_reader))["params"]
        assert replay["event"] == "approval_request"
        assert replay["request_id"] == call.id
        status = (await _request(replay_reader, replay_writer, 11, "status"))[-1]
        assert status["result"]["pending_approvals"][0]["request_id"] == call.id

        approved = (
            await _request(
                replay_reader,
                replay_writer,
                12,
                "approve",
                {"request_id": call.id},
            )
        )[-1]
        assert approved["result"]["request_id"] == call.id
        assert server.runtime.opened.store.approval_resolution_data(call.id)[
            "decision"
        ] == "allow"
    finally:
        if replay_writer is not None:
            await _close(server, replay_writer)
        else:
            await server.close()


@pytest.mark.asyncio
async def test_approval_reconnect_starting_claim_before_lookup_completion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(
        tmp_path, protocol_version="1.2", features=["approval_reconnect"]
    )
    events: list[tuple[str, str | None]] = []
    original = server.control._notify

    async def record(
        event: str, session_id: str | None = None, **fields: object
    ) -> bool:
        request_id = fields.get("request_id")
        events.append((event, request_id if isinstance(request_id, str) else None))
        return await original(event, session_id, **fields)

    monkeypatch.setattr(server.control, "_notify", record)
    try:
        start, gate = await _gated_start(server, call, monkeypatch)
        await server.control.controller_released(reconnect=True)
        await server.control.controller_claimed(reconnect=True)
        gate.release.set()
        await start
        assert ("approval_request", call.id) not in events
        await server.control.controller_ready(reconnect=True)
        assert events.count(("approval_request", call.id)) == 1
        assert server.control.pending_approvals_now()[0]["request_id"] == call.id
    finally:
        gate.release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_reconnect_expiry_aborts_once(tmp_path: Path) -> None:
    server, _reader, writer, call = await _approval_server(
        tmp_path,
        protocol_version="1.2",
        features=["approval_reconnect"],
        approval_reconnect_grace=0,
    )
    try:
        await server.control._event(_start_event(call))
        await server.control.controller_released(reconnect=True)
        await server.control.wait_for_approval_settlement()
        rows = [
            row
            for row in server.runtime.opened.store.entries
            if row.type == "approval_resolution" and row.data["request_id"] == call.id
        ]
        assert len(rows) == 1
        assert rows[0].data["reason"] == "grace_expired"
        with pytest.raises(ProtocolError) as late:
            await server.control.resolve_approval("approve", call.id)
        assert late.value.code == -32006
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_reconnect_status_during_reserved_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _reader, writer, call = await _approval_server(
        tmp_path, protocol_version="1.2", features=["approval_reconnect"]
    )
    try:
        await server.control._event(_start_event(call))
        store = server.runtime.opened.store
        gate = _StoreGate(store, "resolve_approval")
        monkeypatch.setattr(store, gate.method, gate.call)
        decision = asyncio.create_task(server.control.resolve_approval("approve", call.id))
        await _wait_thread_gate(gate)
        status = server.control.pending_approvals_now(reconnect=True)
        assert status == [
            {
                **server.control._approvals.observe(
                    server.control._approvals.state.records[call.id].request
                ),
                "resolution_pending": "approve",
            }
        ]
        with pytest.raises(ProtocolError) as competing:
            await server.control.resolve_approval("deny", call.id)
        assert competing.value.code == -32006
        gate.release.set()
        await decision
    finally:
        gate.release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_approval_reconnect_status_bound_is_negotiated(tmp_path: Path) -> None:
    server, _reader, writer, first = await _approval_server(
        tmp_path,
        approval_count=65,
        protocol_version="1.2",
        features=["approval_reconnect"],
    )
    try:
        states = server.runtime.opened.store.approval_states()
        for call, _decision in states.values():
            await server.control._event(_start_event(call))
        negotiated, truncated = server.control.pending_approvals_now(
            reconnect=True, include_truncated=True
        )
        assert len(negotiated) == 64
        assert truncated is True
        legacy = await server.control.pending_approvals()
        assert len(legacy) == 65
        assert all("resolution_pending" not in item for item in legacy)
        assert first.id == negotiated[0]["request_id"]
    finally:
        await _close(server, writer)
