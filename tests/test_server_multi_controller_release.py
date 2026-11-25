from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_in_flight_replay import GatedBackend
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent
from zeta.server import ZetaServer
from zeta.server.control import ServeControl
from zeta.server.handlers import lifecycle
from zeta.server.runtime import ServerRuntime
from zeta.server.server import _Client
from zeta.server.session_controllers import SessionController

_FEATURES = ("multi_controller", "session_park", "resume_attach", "approval_reconnect")


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", (
    "reclaim", "legacy-refusal", "expiry-restart", "duplicate", "explicit-close", "peer-loss",
    "cancel-validate", "cancel-claim", "cancel-bind", "cancel-response", "cancel-ready",
    "fail-activate", "fail-commit-hello", "fail-commit-session", "fail-response", "fail-ready",
    "shutdown-parked", "shutdown-setup",
))
async def test_multi_release_table(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    tool = ToolUseContent(ToolCall("write-1", "write", {"path": str(tmp_path / "out"), "content": "x"}))
    done = TextContent("done")
    backend = GatedBackend([([tool], [tool]), ([done], [done])])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    clients: list[ReplayClient] = []
    work_due, idle_due, hold_close = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def connect() -> ReplayClient:
        client = await await_or_fail(connect_replay(server, transport), what="open socket")
        clients.append(client)
        return client

    async def hello(client: ReplayClient, sid: str) -> dict[str, Any]:
        return await client.request(1, "hello", {
            "protocol_version": "1.1", "features": list(_FEATURES), "session_id": sid,
        })

    async def create(client: ReplayClient, request_id: int) -> str:
        reply = await client.request(request_id, "new_session", {"approval_mode": "ask", "model": "gpt-5.6-sol"})
        assert "result" in reply, reply
        return reply["result"]["session"]["session_id"]

    def watch_close(record: SessionController, due: asyncio.Event) -> tuple[asyncio.Event, asyncio.Event]:
        closing, closed = asyncio.Event(), asyncio.Event()
        close = record.runtime.close

        async def deadline() -> None:
            await await_or_fail(due.wait(), what="advance this record's deadline", timeout=20)

        async def close_runtime() -> None:
            closing.set()
            if row == "expiry-restart" and due is work_due:
                await await_or_fail(hold_close.wait(), what="settle parked close")
            await await_or_fail(close(), what="close runtime")
            closed.set()

        # test-contract: seam replace the deadline and observe the normal retirement finalizer.
        monkeypatch.setattr(record.control, "_grace_deadline", deadline)
        monkeypatch.setattr(record.runtime, "close", close_runtime)
        return closing, closed

    await await_or_fail(server.start(), what="start serve")
    try:
        owner = await connect()
        assert "result" in await negotiate(owner, replay=False, extra=(*_FEATURES, "new_session_settings"))
        parked_id = await create(owner, 2)
        parked = server.controllers.resident(parked_id)
        assert parked is not None
        work_closing, work_closed = watch_close(parked, work_due)
        assert "result" in await owner.request(3, "send", {"text": "write"})
        approval = await owner.event("approval_request")
        approval_id = approval["request_id"]
        active_reply = await owner.request(4, "new_session", {})
        active_id = active_reply["result"]["session"]["session_id"]
        status = (await owner.request(5, "status", {}))["result"]
        assert [(p["session_id"], p["pending_approvals"]) for p in status["parked_sessions"]] == [(parked_id, 1)]
        active = server.controllers.resident(active_id)
        assert active is not None
        _, idle_closed = watch_close(active, idle_due)
        if row.startswith("shutdown-"):
            await _shutdown_release(server, owner, parked, row, monkeypatch)
            await await_or_fail(work_closed.wait(), what="parked runtime closed despite release failure")
            await await_or_fail(idle_closed.wait(), what="attached runtime closed despite release failure")
            return
        peer, = server.connections.clients()
        disconnected = asyncio.Event()
        unregister = server.connections.unregister

        def observe_unregister(connection: _Client) -> None:
            unregister(connection)
            if connection is peer:
                disconnected.set()

        # test-contract: seam observe when the owner has lost both physical and semantic admission.
        monkeypatch.setattr(server.connections, "unregister", observe_unregister)
        if row == "explicit-close":
            await await_or_fail(peer.close(), what="explicit server-side client close")
        elif row == "peer-loss":
            await _lose_during_request(server, owner, peer, monkeypatch)
        else:
            await await_or_fail(owner.close(), what="owner socket EOF")
        await await_or_fail(disconnected.wait(), what="owner EOF settled")

        if row in {"legacy-refusal", "expiry-restart"}:
            idle_due.set()
            await await_or_fail(idle_closed.wait(), what="retire only the idle record")
            legacy = await connect()
            refused = await negotiate(legacy, replay=False, extra=("session_park", "resume_attach", "approval_reconnect"))
            leaked = (await legacy.request(2, "status", {}))["result"]["parked_sessions"] if "result" in refused else []
            assert refused.get("error", {}).get("code") == -32001, (
                f"legacy hello crossed a parked approval's live grace: {leaked}"
            )
            if row == "legacy-refusal":
                return
            work_due.set()
            await await_or_fail(work_closing.wait(), what="parked record expiry started normal close")
            legacy = await connect()
            refused = await negotiate(legacy, replay=False, extra=("session_park",))
            assert refused.get("error", {}).get("code") == -32001, "legacy hello crossed unsettled parked close"
            hold_close.set()
            await await_or_fail(work_closed.wait(), what="parked record close settled")
            legacy = await connect()
            assert "result" in await negotiate(legacy, replay=False, extra=("session_park",))
            assert (await legacy.request(2, "status", {}))["result"].get("parked_sessions", []) == []
            await await_or_fail(server.close(), what="close serve before restart")
            server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
            await await_or_fail(server.start(), what="restart serve")
            restarted = await connect()
            assert "result" in await negotiate(restarted, replay=False, extra=_FEATURES)
            resumed = await restarted.request(2, "resume", {"session_id": parked_id})
            assert resumed["result"]["session"]["session_id"] == parked_id
            status = (await restarted.request(3, "status", {}))["result"]
            assert status["pending_approvals"] == []
            return

        if row.startswith(("cancel-", "fail-")):
            await _interrupt_hello(server, await connect(), parked, parked_id, row, monkeypatch)
        reconnect = await connect()
        if row == "duplicate":
            reclaimed = await _race_hello(reconnect, await connect(), parked, hello, parked_id, monkeypatch)
        else:
            reclaimed = await hello(reconnect, parked_id)
        assert "result" in reclaimed, f"parked approval cannot be reclaimed by its exact id: {reclaimed}"
        if row == "duplicate":
            await await_or_fail(peer.close(), what="repeat old release after exact reclaim")
        status = (await reconnect.request(2, "status", {}))["result"]
        assert status["session"]["session_id"] == parked_id
        assert [a["request_id"] for a in status["pending_approvals"]] == [approval_id]
        assert status["parked_sessions"] == []
        assert "result" in await reconnect.request(3, "approve", {"request_id": approval_id})
    finally:
        backend.open_all()
        work_due.set()
        idle_due.set()
        hold_close.set()
        for client in clients:
            await await_or_fail(client.close(), what="close socket")
        await await_or_fail(server.close(), what="finish serve")


async def _interrupt_hello(
    server: ZetaServer, client: ReplayClient, record: SessionController, sid: str,
    row: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    action, step = row.split("-", 1)
    control = record.control
    entered, resume, disconnected = asyncio.Event(), asyncio.Event(), asyncio.Event()
    params: dict[str, object] = {"protocol_version": "1.1", "features": list(_FEATURES), "session_id": sid}
    if step in {"validate", "bind"}:
        params.update(features=[*_FEATURES, "session_mcp_bindings"], mcp_bindings=[])
    registry = server.connections
    assert registry is not None
    unregister = registry.unregister
    failed_peer: _Client | None = None

    def observe_unregister(connection: _Client) -> None:
        unregister(connection)
        if connection is failed_peer:
            disconnected.set()

    async def route(peer: _Client, params: dict[str, Any]) -> tuple[bool, object]:
        nonlocal failed_peer
        failed_peer = peer
        return await original_route(peer, params)

    original_route = _Client._route_hello
    seam, method = {
        "validate": (lifecycle, "reconnect_mcp_bindings"),
        "claim": (control, "controller_claimed"),
        "bind": (control, "reattach_mcp_bindings"),
        "activate": (registry, "activate_controller"),
        "commit-hello": (server.controllers, "commit_hello"),
        "commit-session": (server.controllers, "commit_session"),
        "response": (_Client, "_write"),
        "ready": (control, "controller_ready"),
    }[step]
    original = getattr(seam, method)

    def fail(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError(f"injected {step} publication failure")

    async def interrupt(*args: Any, **kwargs: Any) -> Any:
        if action == "fail":
            fail()
        entered.set()
        await await_or_fail(resume.wait(), what=f"resume the {step} await")
        return await await_or_fail(original(*args, **kwargs), what=f"complete {step}")

    with monkeypatch.context() as patch:
        # test-contract: seam observe the actual server-side handler and block or fail one await/publication.
        patch.setattr(_Client, "_route_hello", route)
        # test-contract: seam wait for rollback and physical close before retrying the exact id.
        patch.setattr(registry, "unregister", observe_unregister)
        if step in {"validate", "bind"}:
            # The empty set makes the binding validation path runnable without a live MCP server.
            # test-contract: seam configure the binding prerequisite without connecting to MCP.
            patch.setattr(control, "_mcp_binding_required", True)
        # test-contract: seam inject only at the selected await or publication.
        patch.setattr(seam, method, fail if step in {"activate", "commit-hello", "commit-session"} else interrupt)
        try:
            await client.send(1, "hello", params)
            if action == "cancel":
                await await_or_fail(entered.wait(), what=f"hello entered {step}")
                assert failed_peer is not None and failed_peer._run_task is not None
                failed_peer._run_task.cancel()
                resume.set()
            await await_or_fail(disconnected.wait(), what=f"failed {step} hello disconnected")
            if step in {"activate", "commit-hello", "commit-session"}:
                assert (await client.response(1))["error"]["code"] == -32000
            elif step == "ready":
                assert "result" in await client.response(1)
            else:
                # test-contract: transport distinguish closed-before-publication from a hello response.
                assert await await_or_fail(client._read_or_eof(), what="failed hello EOF") is None
        finally:
            resume.set()


async def _race_hello(
    first: ReplayClient, second: ReplayClient, record: SessionController,
    hello: Callable[[ReplayClient, str], Awaitable[dict[str, Any]]], sid: str, monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    entered, release = asyncio.Event(), asyncio.Event()
    claim = record.control.controller_claimed

    async def hold_claim(*args: Any, **kwargs: Any) -> bool:
        entered.set()
        await await_or_fail(release.wait(), what="let winning reconnect claim")
        return await await_or_fail(claim(*args, **kwargs), what="claim after race")

    with monkeypatch.context() as patch:
        # test-contract: seam delay the first hello after reserving the exact session, before claiming it.
        patch.setattr(record.control, "controller_claimed", hold_claim)
        winning = asyncio.create_task(hello(first, sid))
        try:
            await await_or_fail(entered.wait(), what="winning hello reserved session")
            denied = await hello(second, sid)
            assert denied.get("error", {}).get("code") == -32014, denied
        finally:
            release.set()
            reclaimed = await await_or_fail(winning, what="winning hello settled")
    return reclaimed


async def _lose_during_request(
    server: ZetaServer, client: ReplayClient, peer: _Client, monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered, lost, release = asyncio.Event(), asyncio.Event(), asyncio.Event()
    registry = server.connections
    assert registry is not None
    mark_lost = registry.mark_lost
    dispatch = peer._dispatch

    async def hold_request(*args: Any, **kwargs: Any) -> Any:
        entered.set()
        await await_or_fail(release.wait(), what="finish request after peer loss")
        return await await_or_fail(dispatch(*args, **kwargs), what="settle in-flight status")

    def observe_loss(connection: _Client) -> asyncio.Event | None:
        result = mark_lost(connection)
        if connection is peer:
            lost.set()
        return result

    with monkeypatch.context() as patch:
        # test-contract: seam hold a real request so the run loop takes the mid-request EOF path.
        patch.setattr(peer, "_dispatch", hold_request)
        # test-contract: seam observe peer EOF before unblocking the request.
        patch.setattr(registry, "mark_lost", observe_loss)
        try:
            await client.send(6, "status", {})
            await await_or_fail(entered.wait(), what="in-flight status before EOF")
            await await_or_fail(client.close(), what="peer loss during a request")
            await await_or_fail(lost.wait(), what="server saw mid-request EOF")
        finally:
            release.set()


async def _shutdown_release(
    server: ZetaServer, client: ReplayClient, parked: SessionController, row: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    setup_started, unblock = asyncio.Event(), asyncio.Event()
    target = parked.runtime
    create = ServerRuntime.create_session
    release = ServeControl.begin_controller_release
    message = f"{row} semantic release failed"

    async def hold_create(runtime: ServerRuntime, *args: Any, **kwargs: Any) -> None:
        nonlocal target
        target = runtime
        setup_started.set()
        await await_or_fail(unblock.wait(), what="hold create before it opens the runtime")
        await await_or_fail(create(runtime, *args, **kwargs), what="finish create")

    def fail_release(control: ServeControl, **kwargs: Any) -> asyncio.Task[None]:
        if control.runtime is target:
            raise RuntimeError(message)
        return release(control, **kwargs)

    try:
        if row == "shutdown-setup":
            # test-contract: seam reserve a new pair before the first awaited runtime operation.
            monkeypatch.setattr(ServerRuntime, "create_session", hold_create)
            await client.send(6, "new_session", {})
            await await_or_fail(setup_started.wait(), what="provisional setup reached its first await")
        # test-contract: seam a release failure from this record must reach the public shutdown waiter.
        monkeypatch.setattr(ServeControl, "begin_controller_release", fail_release)
        with pytest.raises(RuntimeError, match=message):
            await await_or_fail(server.close(), what="shutdown settles after semantic release error")
    finally:
        unblock.set()
