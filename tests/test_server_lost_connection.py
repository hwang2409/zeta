"""Serve releases the controller slot when a connection is lost mid-request.

Each test drives a real serve server (unix and websocket) over real sockets. A
request handler is gated by an ``asyncio.Event`` so the test controls when the
handler is running, suspended, cancelled, or finished, with no sleeps and no
wall-clock thresholds. The server's loss-handling deadlines are instance
attributes the tests override.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from collections.abc import Callable
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.replay_transport import (
    ReplayClient,
    Transport,
    connect_replay,
    negotiate,
    replay_server,
)
from zeta.server import ZetaServer
from zeta.server import control as control_module
from zeta.server.handlers import registry
from zeta.server.protocol import EventDescriptor, MethodDescriptor

pytest_plugins = ("tests.support.socket_fixtures",)


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


async def _spin(cond: Callable[[], bool], *, what: str) -> None:
    """Yield to the loop until ``cond`` holds; deterministic, no wall clock."""
    try:
        async with asyncio.timeout(10):
            while not cond():
                await asyncio.sleep(0)
    except TimeoutError:
        raise AssertionError(f"{what} did not happen within 10 s") from None


async def _settle_until(cond: Callable[[], bool], *, turns: int = 2000) -> None:
    """Yield up to ``turns`` event-loop turns for ``cond``, then return anyway.

    Used for the "the server observed the peer EOF" barrier so that a run
    against a base that has no loss handling still proceeds to the behavior
    assertion instead of hanging; on the branch ``cond`` holds within a few
    turns. Bounded by loop turns, not by wall clock.
    """
    for _ in range(turns):
        if cond():
            return
        await asyncio.sleep(0)


def _gate_handler(
    descriptor: MethodDescriptor,
    monkeypatch: pytest.MonkeyPatch,
    *,
    started: asyncio.Event,
    gate: asyncio.Event,
    cancelled: asyncio.Event | None = None,
    finished: asyncio.Event | None = None,
    hold_through_cancel: asyncio.Event | None = None,
    result: object | None = None,
) -> None:
    """Replace one handler with a gated test handler.

    ``started`` fires when the handler begins. The handler then waits on
    ``gate`` and fires ``finished`` when it completes normally. If the request
    task is cancelled it sets ``cancelled``; when ``hold_through_cancel`` is
    given it swallows the cancel and keeps running (holding the slot) until that
    event is set, modelling a cancel-resistant handler.
    """

    async def handler(client: object, request_id: object, params: object, controller: object) -> object:
        started.set()
        try:
            await asyncio.wait_for(gate.wait(), 10)
        except asyncio.CancelledError:
            if cancelled is not None:
                cancelled.set()
            if hold_through_cancel is not None:
                await asyncio.wait_for(hold_through_cancel.wait(), 10)
                if finished is not None:
                    finished.set()
                return result if result is not None else {"ok": True}
            raise
        if finished is not None:
            finished.set()
        return result if result is not None else {"ok": True}

    monkeypatch.setitem(registry.HANDLERS, descriptor, handler)


def _spy_handler(
    descriptor: MethodDescriptor,
    monkeypatch: pytest.MonkeyPatch,
    *,
    ran: asyncio.Event,
) -> None:
    async def handler(client: object, request_id: object, params: object, controller: object) -> object:
        ran.set()
        return {"ok": True}

    monkeypatch.setitem(registry.HANDLERS, descriptor, handler)


def _lose(client: ReplayClient) -> None:
    """Half-close the client write side (TCP FIN); the server reads EOF."""
    client.writer.write_eof()


async def _make_server(tmp_path: Path, transport: Transport) -> ZetaServer:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    return server


def _is_lost(server: ZetaServer) -> bool:
    controller = server.connections.legacy_controller_holder
    return controller is not None and getattr(controller, "lost", False)


def _released(server: ZetaServer) -> bool:
    return server.connections.legacy_controller_holder is None


def _contended(server: ZetaServer) -> bool:
    controller = server.connections.legacy_controller_holder
    event = getattr(controller, "contended", None) if controller is not None else None
    return event is not None and event.is_set()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_contended_cancel_releases_slot(tmp_path: Path, transport: Transport) -> None:
    """Test 1: with a zero contended drain, a contending controller hello
    cancels the lost holder's handler and claims the slot."""
    server = await _make_server(tmp_path, transport)
    server.lost_request_contended_drain = 0.0
    started = asyncio.Event()
    gate = asyncio.Event()
    cancelled = asyncio.Event()
    monkeypatch = pytest.MonkeyPatch()
    _gate_handler(
        MethodDescriptor.STATUS, monkeypatch,
        started=started, gate=gate, cancelled=cancelled,
    )
    a = await connect_replay(server, transport)
    b = await connect_replay(server, transport)
    try:
        await negotiate(a)
        await a.send(2, "status", {})
        await await_or_fail(started.wait(), what="A status handler started")
        _lose(a)
        await _settle_until(lambda: _is_lost(server))

        hello = await b.request(2, "hello", {"protocol_version": "1.1", "features": []})
        assert "result" in hello, hello
        assert cancelled.is_set(), "the lost holder's handler was not cancelled"
    finally:
        monkeypatch.undo()
        gate.set()
        await a.close()
        await b.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_live_holder_refused_immediately(tmp_path: Path, transport: Transport) -> None:
    """Test 3 (passes on base): a live controller still refuses a second
    controller hello at once with -32001, with no wait."""
    server = await _make_server(tmp_path, transport)
    a = await connect_replay(server, transport)
    b = await connect_replay(server, transport)
    try:
        await negotiate(a)
        hello = await b.request(2, "hello", {"protocol_version": "1.1", "features": []})
        assert hello["error"]["code"] == -32001, hello
    finally:
        await a.close()
        await b.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_contended_drain_keeps_side_effect(tmp_path: Path, transport: Transport) -> None:
    """Test 2 (fail-first): with a large contended drain, a handler that
    finishes while a controller hello waits keeps its side effect and is not
    cancelled; the hello then claims the slot."""
    server = await _make_server(tmp_path, transport)
    server.lost_request_contended_drain = 30.0
    started = asyncio.Event()
    gate = asyncio.Event()
    cancelled = asyncio.Event()
    finished = asyncio.Event()
    monkeypatch = pytest.MonkeyPatch()
    _gate_handler(
        MethodDescriptor.STATUS, monkeypatch,
        started=started, gate=gate, cancelled=cancelled, finished=finished,
    )
    a = await connect_replay(server, transport)
    b = await connect_replay(server, transport)
    try:
        await negotiate(a)
        await a.send(2, "status", {})
        await await_or_fail(started.wait(), what="A status handler started")
        _lose(a)
        await _settle_until(lambda: _is_lost(server))
        await b.send(2, "hello", {"protocol_version": "1.1", "features": []})
        await _settle_until(lambda: _contended(server))
        # the handler finishes during the drain, before the drain elapses
        gate.set()
        hello = await b.response(2)
        assert "result" in hello, hello
        assert finished.is_set(), "the draining handler did not keep its side effect"
        assert not cancelled.is_set(), "the handler was cancelled during the drain"
    finally:
        monkeypatch.undo()
        gate.set()
        await a.close()
        await b.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_no_new_request_after_loss_reply_still_written(
    tmp_path: Path, transport: Transport
) -> None:
    """Test 4 (fail-first): a request pipelined behind the running one never
    runs after EOF; on unix/tcp the reply to the running request is still
    written because the write side is not closed (F4)."""
    server = await _make_server(tmp_path, transport)
    started = asyncio.Event()
    gate = asyncio.Event()
    ran = asyncio.Event()
    monkeypatch = pytest.MonkeyPatch()
    _gate_handler(MethodDescriptor.STATUS, monkeypatch, started=started, gate=gate)
    _spy_handler(MethodDescriptor.LIST_SESSIONS, monkeypatch, ran=ran)
    a = await connect_replay(server, transport)
    try:
        await negotiate(a)
        await a.send(2, "status", {})
        await a.send(3, "list_sessions", {})
        await await_or_fail(started.wait(), what="A status handler started")
        _lose(a)
        await _settle_until(lambda: _is_lost(server))
        gate.set()
        await _settle_until(ran.is_set)
        assert not ran.is_set(), "a pipelined request ran after the connection was lost"
        if transport == "unix":
            reply = await a.response(2)
            assert "result" in reply, reply
    finally:
        monkeypatch.undo()
        gate.set()
        await a.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_cancel_resistant_handler_holds_slot(
    tmp_path: Path, transport: Transport, caplog: pytest.LogCaptureFixture
) -> None:
    """Test 5 (behavior; new-only path, proven by mutation): a handler that
    swallows the cancel keeps the slot held; the contending hello is refused at
    the handoff deadline, the stuck-holder dump repeats, and once the handler
    ends a later hello succeeds."""
    server = await _make_server(tmp_path, transport)
    server.lost_request_contended_drain = 0.0
    server.lost_request_cancel_bound = 0.01
    started = asyncio.Event()
    gate = asyncio.Event()
    cancelled = asyncio.Event()
    release_hold = asyncio.Event()
    monkeypatch = pytest.MonkeyPatch()
    _gate_handler(
        MethodDescriptor.STATUS, monkeypatch,
        started=started, gate=gate, cancelled=cancelled,
        hold_through_cancel=release_hold,
    )
    a = await connect_replay(server, transport)
    b = await connect_replay(server, transport)
    c = await connect_replay(server, transport)

    def _dumps() -> int:
        return sum(
            1 for record in caplog.records
            if record.name == "zeta.server.diagnostics"
        )

    try:
        with caplog.at_level(logging.WARNING, logger="zeta.server.diagnostics"):
            await negotiate(a)
            await a.send(2, "status", {})
            await await_or_fail(started.wait(), what="A status handler started")
            _lose(a)
            await _settle_until(lambda: _is_lost(server))
            # the hello contends; the handler swallows the cancel and holds the slot.
            monkeypatch.setattr(control_module, "CONTROLLER_HANDOFF_DEADLINE_SECONDS", 0.5)
            hello = await b.request(2, "hello", {"protocol_version": "1.1", "features": []})
            assert hello["error"]["code"] == -32001, hello
            await await_or_fail(cancelled.wait(), what="holder saw the cancel")
            assert not _released(server), "the slot was released while the handler held it"
            await _spin(lambda: _dumps() >= 2, what="stuck-holder dump repeats")
            # let the handler end; the slot is then released and a later hello wins.
            release_hold.set()
            await _spin(lambda: _released(server), what="slot released after handler ends")
            hello2 = await c.request(3, "hello", {"protocol_version": "1.1", "features": []})
            assert "result" in hello2, hello2
    finally:
        monkeypatch.undo()
        gate.set()
        release_hold.set()
        await a.close()
        await b.close()
        await c.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_one_shared_handoff_deadline(tmp_path: Path, transport: Transport) -> None:
    """Test 6 (behavior; new-only path, proven by mutation): the waiting
    controller hello passes the time left on one handoff deadline to
    controller_claimed, not a fresh full deadline."""
    server = await _make_server(tmp_path, transport)
    server.lost_request_contended_drain = 0.0
    deadline = control_module.CONTROLLER_HANDOFF_DEADLINE_SECONDS
    captured: list[float] = []
    original = server.controllers.attached().control.controller_claimed

    async def spy(*args: object, **kwargs: object) -> bool:
        captured.append(float(kwargs["deadline"]))
        return await original(*args, **kwargs)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(server.controllers.attached().control, "controller_claimed", spy)
    started = asyncio.Event()
    gate = asyncio.Event()
    _gate_handler(MethodDescriptor.STATUS, monkeypatch, started=started, gate=gate)
    a = await connect_replay(server, transport)
    b = await connect_replay(server, transport)
    try:
        await negotiate(a)
        await a.send(2, "status", {})
        await await_or_fail(started.wait(), what="A status handler started")
        _lose(a)
        await _settle_until(lambda: _is_lost(server))
        await b.send(2, "hello", {"protocol_version": "1.1", "features": []})
        await _settle_until(lambda: _contended(server))
        gate.set()
        hello = await b.response(2)
        assert "result" in hello, hello
        assert captured, "controller_claimed was not called"
        # the remaining time is strictly less than a fresh full deadline, so the
        # lost-holder wait and the claim share one budget.
        assert captured[-1] < deadline, (captured, deadline)
    finally:
        monkeypatch.undo()
        gate.set()
        await a.close()
        await b.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_no_contender_no_cancel(tmp_path: Path, transport: Transport) -> None:
    """Test 7 (fail-first via the pipelined spy; regression guard for cancel):
    with no contending controller, a lost holder's handler is not cancelled and
    runs to completion, but no new request from the lost connection runs."""
    server = await _make_server(tmp_path, transport)
    started = asyncio.Event()
    gate = asyncio.Event()
    cancelled = asyncio.Event()
    finished = asyncio.Event()
    ran = asyncio.Event()
    monkeypatch = pytest.MonkeyPatch()
    _gate_handler(
        MethodDescriptor.STATUS, monkeypatch,
        started=started, gate=gate, cancelled=cancelled, finished=finished,
    )
    _spy_handler(MethodDescriptor.LIST_SESSIONS, monkeypatch, ran=ran)
    a = await connect_replay(server, transport)
    try:
        await negotiate(a)
        await a.send(2, "status", {})
        await a.send(3, "list_sessions", {})
        await await_or_fail(started.wait(), what="A status handler started")
        _lose(a)
        await _settle_until(lambda: _is_lost(server))
        # no contender: the handler keeps running, not cancelled.
        await _settle_until(cancelled.is_set)
        assert not cancelled.is_set(), "handler was cancelled with no contender"
        gate.set()
        await _settle_until(ran.is_set)
        assert not ran.is_set(), "a pipelined request ran after the connection was lost"
        await _spin(finished.is_set, what="handler finished its side effect")
        await _spin(lambda: _released(server), what="connection released via loss path")
        assert not cancelled.is_set()
    finally:
        monkeypatch.undo()
        gate.set()
        await a.close()
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_turn_reconnect_through_loss_path(tmp_path: Path, transport: Transport) -> None:
    """Test 8: a turn outlives a controller lost while one of its requests runs;
    a reconnecting controller with turn_reconnect reclaims the running turn."""
    from tests.test_server_turn_reconnect import RECONNECT, HeldBackend
    from zeta.protocol.types import TextContent

    backend = HeldBackend([([TextContent("the reply")], True)])
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await server.start()
    started = asyncio.Event()
    gate = asyncio.Event()
    monkeypatch = pytest.MonkeyPatch()
    _gate_handler(MethodDescriptor.LIST_SESSIONS, monkeypatch, started=started, gate=gate)
    a = await connect_replay(server, transport)
    b = await connect_replay(server, transport)
    try:
        await negotiate(a, extra=tuple(RECONNECT), protocol_version="1.2")
        await a.request(2, "new_session", {"provider": "codex"})
        await a.send(3, "send", {"text": "hi"})
        await await_or_fail(backend.streaming[0].wait(), what="turn mid-stream")
        # lose A while a request handler runs; the turn keeps running.
        await a.send(4, "list_sessions", {})
        await await_or_fail(started.wait(), what="A request handler started")
        _lose(a)
        await _settle_until(lambda: _is_lost(server))
        gate.set()
        await _spin(lambda: _released(server), what="A released via loss path")
        assert server.controllers.attached().runtime.state.status == "running", "the turn was lost on release"

        await negotiate(b, extra=tuple(RECONNECT), protocol_version="1.2")
        status = (await b.request(5, "status", {}))["result"]
        assert status["state"] == "running", status
        backend.release[0].set()
        await b.event("agent_end")
    finally:
        monkeypatch.undo()
        for release in backend.release:
            release.set()
        gate.set()
        await a.close()
        await b.close()
        await server.close()


@pytest.mark.asyncio
async def test_parked_output_drain_on_lost_holder_releases_slot(tmp_path: Path) -> None:
    """Test 9 (fail-first, unix only): a live-event publish parked in the output
    writer's ``drain()`` (the holder stopped reading and the kernel send buffer
    is full) keeps ``callers_settled`` clear, so a request behind it is stuck at
    ``wait_until_settled`` and never dispatches.

    On the branch the run loop reads the socket concurrently, so a lost holder's
    EOF is still seen: the slot is released, the stuck handler is cancelled, the
    connection closes, and the parked publish ends; a contending controller
    hello wins within one handoff deadline. On base origin/main the run loop is
    blocked at ``wait_until_settled`` before it can loop back to read, so the EOF
    is never read, the slot is held, and the hello is refused with -32001.

    Unix only: on TCP a client half-close or RST errors the parked ``drain()``,
    which resolves the caller and unblocks the base loop, so the bug does not
    reproduce there. This mirrors the achilles RCA (a unix socket whose peer
    stopped reading after a resume).
    """
    server = await _make_server(tmp_path, "unix")
    server.lost_request_contended_drain = 0.0
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(control_module, "CONTROLLER_HANDOFF_DEADLINE_SECONDS", 0.5)
    a = await connect_replay(server, "unix")
    b = await connect_replay(server, "unix")
    publishes: list[asyncio.Task[bool]] = []
    try:
        await negotiate(a)
        # shrink the client receive buffer so the kernel buffers fill fast.
        a.writer.get_extra_info("socket").setsockopt(
            socket.SOL_SOCKET, socket.SO_RCVBUF, 8192
        )
        state = server.connections.legacy_controller_holder
        conn, output = state.connection, state.output
        # publish big live events the holder never reads; the writer parks in
        # drain() with its caller still counted, so callers_settled stays clear.
        publishes = [
            asyncio.create_task(
                server.connections.send(
                    conn,
                    conn.codec.notification(
                        EventDescriptor.CONFIG_CHANGED, None, data={"blob": "x" * 900_000}
                    ),
                )
            )
            for _ in range(4)
        ]
        await _settle_until(lambda: output.pending_callers > 0)
        assert output.pending_callers > 0 and not all(p.done() for p in publishes), (
            "the publish did not park in the output writer's drain()"
        )
        # a request queued behind the parked publish; its handler cannot pass
        # wait_until_settled. on base the run loop blocks here before reading on.
        await a.send(2, "session_settings", {})
        # the holder is gone (FIN); its write side is half-closed.
        _lose(a)
        await _settle_until(lambda: _is_lost(server))
        hello = await b.request(2, "hello", {"protocol_version": "1.1", "features": []})
        assert "result" in hello, hello
        await _spin(lambda: all(p.done() for p in publishes), what="parked publish ends")
    finally:
        monkeypatch.undo()
        for publish in publishes:
            publish.cancel()
        await a.close()
        await b.close()
        await server.close()
