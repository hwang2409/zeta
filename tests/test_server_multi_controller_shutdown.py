from __future__ import annotations

import asyncio
import contextlib
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from tests.test_server_multi_controller_admission import _close_server
from tests.test_server_replay_table import _backend_factory


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize(
    "boundary",
    ("semantic", "flush", "workers", "cancel", "client", "controllers", "two_failures"),
)
@pytest.mark.asyncio
async def test_public_multi_controller_shutdown_state_table(
    tmp_path: Path, transport: Transport, boundary: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.server.runtime import ServerRuntime

    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    clients = [await connect_replay(server, transport) for _ in range(4)]
    provisional_request = None
    release = asyncio.Event()
    shutdown_gate = asyncio.Event()
    try:
        features = ("multi_controller", "session_park", "resume_attach")
        for client in clients:
            assert "result" in await negotiate(client, replay=False, extra=features)
        for request_id, client in enumerate(clients[:-1], 2):
            assert "result" in await client.request(request_id, "new_session", {})

        entered = asyncio.Event()
        original_create = ServerRuntime.create_session

        async def gated_create(runtime: ServerRuntime, *args: Any, **kwargs: Any):
            entered.set()
            await await_or_fail(release.wait(), what="release provisional create")
            return await original_create(runtime, *args, **kwargs)

        monkeypatch.setattr(ServerRuntime, "create_session", gated_create)
        provisional_request = asyncio.create_task(clients[-1].request(9, "new_session", {}))
        await await_or_fail(entered.wait(), what="provisional create entered")
        registry = server.connections
        physical_clients = registry.clients()
        records = server.controllers.records()
        first_client, second_client = physical_clients[:2]
        first_error = RuntimeError("first shutdown failure")
        second_error = RuntimeError("second shutdown failure")
        failures = []
        release_failure = {"semantic", "two_failures"}
        if boundary in release_failure:
            def fail_release():
                failures.append(first_error)
                raise first_error
            # test-contract: seam patches the exact client's semantic release.
            monkeypatch.setattr(first_client, "_begin_controller_release", fail_release)
        if boundary in {"flush", "two_failures"}:
            original_flush = server.connections.flush
            async def fail_flush(connection, *args, **kwargs):
                if connection is (second_client if boundary == "two_failures" else first_client):
                    error = second_error if boundary == "two_failures" else first_error
                    failures.append(error)
                    raise error
                return await original_flush(connection, *args, **kwargs)
            monkeypatch.setattr(server.connections, "flush", fail_flush)
        shutdown_entered = asyncio.Event()
        if boundary == "workers":
            original_workers = server.connections.stop_workers
            async def fail_workers(*args):
                await original_workers(*args)
                raise first_error
            monkeypatch.setattr(server.connections, "stop_workers", fail_workers)
        if boundary == "cancel":
            original_workers = server.connections.stop_workers
            async def gated_workers(*args):
                shutdown_entered.set()
                await await_or_fail(shutdown_gate.wait(), what="shutdown worker gate")
                return await original_workers(*args)
            monkeypatch.setattr(server.connections, "stop_workers", gated_workers)
        if boundary == "client":
            original_writer_close = first_client.writer.close
            def fail_writer_close():
                original_writer_close()
                raise first_error
            monkeypatch.setattr(first_client.writer, "close", fail_writer_close)
        if boundary == "controllers":
            original_controllers_close = server.controllers.close
            async def fail_controllers_close(*args, **kwargs):
                await original_controllers_close(*args, **kwargs)
                raise first_error
            monkeypatch.setattr(server.controllers, "close", fail_controllers_close)
        closes: dict[int, int] = {id(record.runtime): 0 for record in records}
        for record in records:
            original_close = record.runtime.close

            async def counted_close(
                *, _runtime=record.runtime, _close=original_close
            ) -> None:
                closes[id(_runtime)] += 1
                await _close()

            monkeypatch.setattr(record.runtime, "close", counted_close)

        if boundary == "cancel":
            close_task = asyncio.create_task(server.close())
            await await_or_fail(shutdown_entered.wait(), what="shutdown remains gated")
            close_task.cancel()
            delivered = asyncio.Event()
            # The cancel wake is queued before this callback. Keep the drain held
            # until the caller has had a chance to return out of order.
            asyncio.get_running_loop().call_soon(delivered.set)
            await await_or_fail(delivered.wait(), what="deliver caller cancellation")
            assert not close_task.done(), "caller cancellation escaped before shutdown drained"
            shutdown_gate.set()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(close_task, what="cancelled shutdown")
            await await_or_fail(server.close(), what="deferred shutdown")
        else:
            with pytest.raises(RuntimeError) as caught:
                await await_or_fail(server.close(), what="four-controller server shutdown")
            assert caught.value is first_error
            if boundary == "two_failures":
                assert failures == [first_error, second_error]
        assert provisional_request.done()
        assert all(client._run_task is None or client._run_task.done() for client in physical_clients)
        assert registry.clients() == ()
        assert registry._observers == []
        assert server.connections is None
        assert server._server is None
        assert server.controllers.records() == ()
        assert set(closes.values()) == {1}
        await server.close()
    finally:
        shutdown_gate.set()
        release.set()
        if provisional_request is not None and not provisional_request.done():
            provisional_request.cancel()
        if provisional_request is not None:
            with contextlib.suppress(BaseException):
                await await_or_fail(provisional_request, what="provisional request settles")
        for client in clients:
            await client.close()
        if server.connections is not None:
            await server.close()


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("grace_seconds", (0, 90))
@pytest.mark.asyncio
async def test_idle_multi_grace_expires(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch, grace_seconds: int
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    server.approval_reconnect_grace = grace_seconds
    await await_or_fail(server.start(), what="start idle grace server")
    clients = []
    deadline, expire = asyncio.Event(), asyncio.Event()
    try:
        owner = await connect_replay(server, transport)
        clients.append(owner)
        assert "result" in await negotiate(owner, replay=False, extra=("multi_controller",))
        session_id = (await owner.request(2, "new_session", {}))["result"]["session"]["session_id"]
        record = server.controllers.resident(session_id)
        assert record is not None

        async def clock():
            deadline.set()
            await await_or_fail(expire.wait(), what="idle grace clock expires")

        # test-contract: seam advance the existing deadline explicitly, not by elapsed time.
        monkeypatch.setattr(record.control, "_grace_deadline", clock)
        registry = server.connections
        retiring = asyncio.Event()
        close_runtime = record.runtime.close

        async def runtime_close():
            retiring.set()
            await close_runtime()

        # test-contract: seam wait for expiry to enter the real close path.
        monkeypatch.setattr(record.runtime, "close", runtime_close)
        peer, = registry.clients()
        unregistered = asyncio.Event()
        unregister = registry.unregister

        def observe_unregister(client: Any) -> None:
            unregister(client)
            if client is peer:
                unregistered.set()

        # test-contract: observe the detached owner's release, not transport timing.
        monkeypatch.setattr(registry, "unregister", observe_unregister)
        await await_or_fail(owner.close(), what="disconnect idle owner")
        await await_or_fail(unregistered.wait(), what="idle owner release")
        if grace_seconds:
            await await_or_fail(deadline.wait(), what="idle deadline is armed")
            assert record.runtime.opened is not None
            assert server.controllers.controller_mode == "multi"
            expire.set()
            await await_or_fail(retiring.wait(), what="idle expiry starts retirement")
        await await_or_fail(server.controllers.wait_closed(session_id), what="idle record close")
        fresh = await connect_replay(server, transport)
        clients.append(fresh)
        assert "result" in await negotiate(fresh, replay=False), "idle grace kept multi admission after release"
    finally:
        expire.set()
        await await_or_fail(_close_server(server, clients), what="close idle grace server")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_shutdown_drains_setup_generation_even_if_setup_finishes_on_cancel(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    from zeta.server.control import ServeControl

    server = replay_server(tmp_path, transport, _backend_factory)
    await await_or_fail(server.start(), what="start setup shutdown server")
    client = await connect_replay(server, transport)
    entered = asyncio.Event()
    cancelled = asyncio.Event()
    finish = asyncio.Event()
    incoming = closing = None
    try:
        assert "result" in await negotiate(client, replay=False, extra=("multi_controller",))
        original = ServeControl.controller_ready

        async def finish_setup(control: ServeControl, **kwargs: Any) -> None:
            entered.set()
            try:
                await await_or_fail(finish.wait(), what="finish setup work")
            except asyncio.CancelledError:
                cancelled.set()
                # The setup worker returns a completed result on hand-back.
                await await_or_fail(finish.wait(), what="hand back completed setup")
            await original(control, **kwargs)

        # test-contract: gate setup after its session ID has been reserved.
        monkeypatch.setattr(ServeControl, "controller_ready", finish_setup)
        await client.send(2, "new_session", {})
        await await_or_fail(entered.wait(), what="setup reservation")
        _, incoming = server.controllers.records()
        closing = asyncio.create_task(server.close())
        await await_or_fail(cancelled.wait(), what="shutdown cancels the setup caller")
        assert incoming.runtime.opened is not None
        finish.set()
        await await_or_fail(closing, what="shutdown after setup hand-back")
        assert incoming.runtime.opened is None, "shutdown omitted the setup-held runtime generation"
        assert server.controllers.records() == ()
    finally:
        finish.set()
        await await_or_fail(client.close(), what="close setup client")
        if closing is not None:
            await await_or_fail(closing, what="setup shutdown settles")
        await await_or_fail(server.close(), what="repeat setup shutdown")
        if incoming is not None and incoming in server.controllers.records():
            # Clean up the leaked generation when this test is run on the base.
            await await_or_fail(server.controllers.retire(incoming), what="dispose leaked setup")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_frozen_shutdown_drains_all_owners(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.test_server_in_flight_replay import HOLD, GatedBackend
    from zeta.protocol.types import TextContent
    from zeta.server import login
    from zeta.server.control import ServeControl
    from zeta.server.protocol import EventDescriptor, ProtocolError
    from zeta.server.session_replay import ReplayScopeId

    text = TextContent("held turn")
    backend = GatedBackend([([text, HOLD], [text])])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    clients = []
    features = ("multi_controller", "session_park", "resume_attach", "turn_reconnect", "approval_reconnect")

    async def connect(*, observer=False, reconnect=None):
        client = await connect_replay(server, transport)
        clients.append(client)
        if reconnect is None:
            hello = await negotiate(client, observer=observer, extra=() if observer else features, protocol_version="1.2")
        else:
            hello = await client.request(1, "hello", {
                "protocol_version": "1.2", "features": list(features), "session_id": reconnect,
            })
        assert "result" in hello, hello
        return client, registry.clients()[-1]

    async def create(client, request_id=2):
        result = await client.request(request_id, "new_session", {})
        return result["result"]["session"]["session_id"]

    async def detach(client, peer):
        released = asyncio.Event()
        original = registry.unregister

        def unregister(connection):
            original(connection)
            if connection is peer:
                released.set()

        with monkeypatch.context() as patch:
            # test-contract: seam signal when the physical row has been removed.
            patch.setattr(registry, "unregister", unregister)
            await await_or_fail(client.close(), what="detach grace peer")
            await await_or_fail(released.wait(), what="grace detached")

    close_gate, close_entered = asyncio.Event(), asyncio.Event()
    setup_gate, setup_entered, setup_cancelled = (asyncio.Event() for _ in range(3))
    login_gate, publishing, login_cancelled, published = (asyncio.Event() for _ in range(4))
    use_gate, replay_gate, replay_entered = (asyncio.Event() for _ in range(3))
    idle_clock = asyncio.Event()
    approval_gate, approval_entered = asyncio.Event(), asyncio.Event()
    record_error = RuntimeError("record close failed")
    writer_error = RuntimeError("writer close failed")
    failures = []
    closing = waiter = None
    await await_or_fail(server.start(), what="start composed shutdown")
    registry = server.connections
    try:
        _unattached, unattached_peer = await connect()
        # Verify both exact reconnects before placing both generations back in grace.
        for _ in range(2):
            detached, peer = await connect()
            sid = await create(detached)
            record = server.controllers.resident(sid)
            assert record is not None

            async def held_idle_deadline():
                await await_or_fail(idle_clock.wait(), what="idle grace clock")

            # test-contract: seam replace the grace clock, not the owner or expiry path.
            monkeypatch.setattr(record.control, "_grace_deadline", held_idle_deadline)
            await detach(detached, peer)
            detached, peer = await connect(reconnect=sid)
            assert server.controllers.controller_for(peer) is record
            await detach(detached, peer)

        a, a_peer = await connect()
        parked_sid = await create(a)
        assert "result" in await a.request(3, "send", {"text": "hold"})
        await await_or_fail(backend.held[0].wait(), what="turn held for parking")
        sid_a = await create(a, 4)
        parked = server.controllers.parked_record(parked_sid)
        assert parked is not None and parked.parked_by is a_peer
        b, b_peer = await connect()
        old_b = server.controllers.resident(await create(b))
        assert old_b is not None
        original_close = old_b.control.close

        async def failing_close():
            close_entered.set()
            await await_or_fail(close_gate.wait(), what="finish previously started record close")
            await original_close()
            failures.append(record_error)
            raise record_error

        # test-contract: hold and fail one real retire without stopping later records.
        monkeypatch.setattr(old_b.control, "close", failing_close)
        sid_b = await create(b, 3)
        await await_or_fail(close_entered.wait(), what="old B already closing")
        observer, observer_peer = await connect(observer=True)
        joined = await observer.request(2, "session_history_and_subscribe", {"session_id": sid_a})
        assert "result" in joined, joined
        assert registry.replay.scope_for(observer_peer) == ReplayScopeId.top(sid_a)

        active_a = server.controllers.controller_for(a_peer)
        assert active_a is not None
        original_settle = active_a.control.wait_for_approval_settlement

        async def settle_approval(*args):
            approval_entered.set()
            await await_or_fail(approval_gate.wait(), what="finish approval settlement")
            await original_settle(*args)

        # test-contract: hold the approval drain at the control/runtime seam.
        monkeypatch.setattr(active_a.control, "wait_for_approval_settlement", settle_approval)

        original_drain = registry.drain_record_events

        async def gated_replay(record):
            if registry.replay.controller_for(observer_peer) is record:
                replay_entered.set()
                await await_or_fail(replay_gate.wait(), what="finish replay finalizer")
            await original_drain(record)

        # test-contract: hold the exact observer generation's finalizer.
        monkeypatch.setattr(registry, "drain_record_events", gated_replay)
        use = server.controllers.owned(b_peer)
        completion = asyncio.create_task(await_or_fail(use_gate.wait(), what="finish transferred completion"))
        use.transfer_to(completion)
        assert use.runtime.session_id == sid_b

        async def provider_login(provider, _pkce, *, on_authorization_url, on_publication_commit, publication_owner, **kwargs):
            on_authorization_url("https://authorize.invalid/")
            if provider == "codex":
                try:
                    await await_or_fail(login_gate.wait(), what="pending provider callback")
                except asyncio.CancelledError:
                    login_cancelled.set()
                    raise
            else:
                async def publish():
                    assert on_publication_commit()
                    publishing.set()
                    await await_or_fail(login_gate.wait(), what="committed provider publication")
                    published.set()

                await publication_owner(publish)

        # test-contract: seam synthetic provider, real Logins and AccountOperations owners.
        monkeypatch.setattr(login.Logins, "_prepare", lambda self, provider, account: provider)
        monkeypatch.setattr(login, "run_login", provider_login)
        assert (await a.request(5, "login_start", {"provider": "codex"}))["result"]["state"] == "pending"
        assert (await b.request(5, "login_start", {"provider": "claude"}))["result"]["state"] == "pending"
        await await_or_fail(publishing.wait(), what="account publication has committed")

        setup, _ = await connect()
        original_ready = ServeControl.controller_ready

        async def setup_ready(control, **kwargs):
            setup_entered.set()
            try:
                await await_or_fail(setup_gate.wait(), what="settle setup")
            except asyncio.CancelledError:
                setup_cancelled.set()
                await await_or_fail(setup_gate.wait(), what="settle setup hand-back")
            await original_ready(control, **kwargs)

        # test-contract: hold setup after its session reservation but before commit.
        monkeypatch.setattr(ServeControl, "controller_ready", setup_ready)
        await setup.send(2, "new_session", {})
        await await_or_fail(setup_entered.wait(), what="setup is reserved")
        records = server.controllers.records()
        assert len(records) >= 7
        peers = registry.clients()
        closes = []
        for record in records:
            close_runtime = record.runtime.close

            async def observed_close(_record=record, _close=close_runtime):
                assert published.is_set(), "runtime closed before committed publication"
                assert server._server is not None, "listener closed before runtime"
                closes.append(_record)
                await _close()

            # test-contract: count every physical runtime close and its phase.
            monkeypatch.setattr(record.runtime, "close", observed_close)

        original_writer_close = unattached_peer.writer.close

        def failed_writer():
            original_writer_close()
            failures.append(writer_error)
            raise writer_error

        # test-contract: fail the unattached peer's physical writer close.
        monkeypatch.setattr(unattached_peer.writer, "close", failed_writer)
        closing = asyncio.create_task(server.close())
        await await_or_fail(login_cancelled.wait(), what="pre-commit login cancels")
        assert closes == []
        assert not closing.done()
        refused = await connect_replay(server, transport)
        clients.append(refused)
        # The listener still answers physical attempts, but never admits a row.
        assert await refused._read_or_eof(what="post-freeze connect is refused") is None
        assert registry.clients() == peers
        with pytest.raises(ProtocolError):
            server.controllers.owned(a_peer)
        with pytest.raises(ProtocolError):
            server.controllers.selected(a_peer, sid_a)
        assert registry.publish_session(a_peer, EventDescriptor.TURN_START, sid_a).kind == "not_writable"

        login_gate.set()
        await await_or_fail(setup_cancelled.wait(), what="cancel setup after the login owners settle")
        assert records[-1] not in closes
        setup_gate.set()
        await await_or_fail(approval_entered.wait(), what="approval drain is held")
        assert active_a not in closes, "runtime closed before approval settlement"
        approval_gate.set()
        await await_or_fail(replay_entered.wait(), what="replay finalizer is held")
        assert registry.replay.scope_for(observer_peer) == ReplayScopeId.top(sid_a)
        assert active_a not in closes, "runtime closed before replay drained"
        assert use.record not in closes, "runtime closed before transferred use settled"
        assert old_b not in closes, "an already retiring runtime was closed twice"
        assert not closing.done()
        waiter = asyncio.create_task(server.close())
        waiter_entered = asyncio.Event()
        asyncio.get_running_loop().call_soon(waiter_entered.set)
        await await_or_fail(waiter_entered.wait(), what="second close waiter started")
        waiter.cancel("second caller cancelled")
        setup_gate.set()
        replay_gate.set()
        close_gate.set()
        use_gate.set()
        backend.open_all()
        with pytest.raises(RuntimeError) as first:
            await await_or_fail(closing, what="drain all frozen owners")
        assert first.value is record_error
        with pytest.raises(asyncio.CancelledError, match="second caller cancelled"):
            await await_or_fail(waiter, what="return caller cancellation after drain")
        assert failures == [record_error, writer_error]
        assert len(closes) == len(records) and set(closes) == set(records)
        assert registry.clients() == () and registry._observers == []
        assert registry.replay.scopes() == ()
        assert all(peer._closed for peer in peers)
        assert server.controllers.records() == () and server.controllers.controller_mode == "empty"
        assert server._server is None
        await await_or_fail(server.close(), what="repeat close is a no-op")
        assert failures == [record_error, writer_error] and len(closes) == len(records)
    finally:
        for gate in (close_gate, setup_gate, login_gate, use_gate, replay_gate, idle_clock, approval_gate):
            gate.set()
        backend.open_all()
        if closing is not None:
            with contextlib.suppress(BaseException):
                await await_or_fail(closing, what="composed shutdown settles")
        if waiter is not None:
            with contextlib.suppress(asyncio.CancelledError):
                await await_or_fail(waiter, what="cancelled close waiter settles")
        with contextlib.suppress(RuntimeError):
            await await_or_fail(_close_server(server, clients), what="dispose composed shutdown peers")


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("exit_path", ("pre_hello", "refused_hello"))
@pytest.mark.asyncio
async def test_closed_unclaimed_connection_settles_its_output_writer(
    tmp_path: Path, transport: Transport, exit_path: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    registry = server.connections
    holder = None
    probe = None
    writers: list[asyncio.Task[None]] = []  # strong refs: GC must not hide a pending writer
    registered = asyncio.Event()
    gone = asyncio.Event()
    original_register, original_unregister = registry.register, registry.unregister

    def register(connection: Any) -> None:
        original_register(connection)
        writers.append(registry.output_state(connection).writer_task)
        registered.set()

    def unregister(connection: Any) -> None:
        original_unregister(connection)
        gone.set()

    try:
        if exit_path == "refused_hello":
            holder = await connect_replay(server, transport)
            assert "result" in await negotiate(holder, replay=False)
        # test-contract: seam observes the physical row's register and unregister, not timing.
        monkeypatch.setattr(registry, "register", register)
        monkeypatch.setattr(registry, "unregister", unregister)
        probe = await connect_replay(server, transport)
        await await_or_fail(registered.wait(), what="probe registered")
        (writer,) = writers
        if exit_path == "refused_hello":
            refused = await negotiate(probe, replay=False)
            assert refused["error"]["code"] == -32001, refused
        await probe.close()
        await await_or_fail(gone.wait(), what="probe unregistered")
        assert writer.done() or writer.cancelling(), "closed connection left its writer pending"
        clients = [client for client in (holder,) if client is not None]
        await await_or_fail(_close_server(server, clients), what="server close")
        holder = None
        assert writer.done(), "server close left the closed connection's writer pending"
        assert not [
            task for task in asyncio.all_tasks()
            if getattr(task.get_coro(), "__qualname__", "") == "ConnectionRegistry._output_writer"
        ], "a serve output writer outlived server close"
    finally:
        with contextlib.suppress(RuntimeError):
            await await_or_fail(
                _close_server(server, [c for c in (holder, probe) if c is not None]),
                what="dispose writer probe",
            )
