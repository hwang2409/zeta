"""Close-table test: one row per way a session record could close (or be
claimed again) before the work that uses it settles: success, cancel at each
await of a delivery, approval, setup, or worker read, failure at a publication
or setup step, restart with a committed but unpublished delivery, and duplicate
input. Each row drives a real serve server over a real unix or websocket
socket; gates are events the test opens, never sleeps or wall-clock bounds.

Every row ends with the same public checks:
- no two non-closing records hold one session;
- every control or runtime close ran only after the row's gate opened (its
  uses and finalizers had settled), once per record;
- no record is left closing, and each publication the row counts ran once;
- a fresh controller connection can claim the server after settlement (rows
  that shut the server down check a restarted server instead).
"""

from __future__ import annotations

import asyncio
import itertools
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
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
from tests.test_server_in_flight_replay import HOLD, GatedBackend
from tests.test_server_session_park import FEATURES
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent
from zeta.server import ZetaServer
from zeta.server.handlers import history as history_module

pytest_plugins = ("tests.support.socket_fixtures",)


async def _until(cond: Callable[[], bool], *, what: str) -> None:
    try:
        async with asyncio.timeout(10):
            while not cond():
                await asyncio.sleep(0)
    except TimeoutError:
        raise AssertionError(f"{what} did not happen within 10 s") from None


async def _turns(cond: Callable[[], bool], turns: int = 2000) -> None:
    """Yield up to ``turns`` loop turns for ``cond``; bounded by turns, not time."""
    for _ in range(turns):
        if cond():
            return
        await asyncio.sleep(0)


class Hold:
    """A gate no cancel opens: models a commit or publication that is already
    under way when its caller is cancelled. A cancel that arrived while held is
    raised once the gate opens."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.opened = asyncio.Event()
        self.count = 0

    async def __call__(self) -> None:
        self.count += 1
        self.entered.set()
        cancelled = False
        while not self.opened.is_set():
            try:
                await asyncio.wait_for(asyncio.shield(self.opened.wait()), 10)
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def wait_entered(self) -> None:
        await await_or_fail(self.entered.wait(), what="the gated step starts")


@dataclass
class Run:
    tmp_path: Path
    transport: Transport
    monkeypatch: pytest.MonkeyPatch
    server: ZetaServer | None = None
    backend: GatedBackend | None = None
    clients: list[ReplayClient] = field(default_factory=list)
    hold: Hold = field(default_factory=Hold)
    # (record label, hook, whether the row's gate had opened by then)
    closes: list[tuple[str, str, bool]] = field(default_factory=list)
    shut_down: bool = False
    _ids: itertools.count[int] = field(default_factory=lambda: itertools.count(10))

    async def serve(self, replies: list[tuple[list[object], list[Any]]] | None = None) -> ZetaServer:
        self.backend = GatedBackend(replies or [([TextContent("ok")], [TextContent("ok")])] * 4)
        backend = self.backend
        self.server = replay_server(self.tmp_path, self.transport, lambda *_: (backend, "offline"))
        await self.server.start()
        return self.server

    async def open(self) -> ReplayClient:
        assert self.server is not None
        client = await connect_replay(self.server, self.transport)
        self.clients.append(client)
        response = await negotiate(client, extra=tuple(FEATURES), protocol_version="1.2")
        assert "result" in response, response
        return client

    async def call(self, client: ReplayClient, method: str, params: dict[str, object] | None = None) -> dict[str, Any]:
        return await client.request(next(self._ids), method, params or {})

    async def send(self, client: ReplayClient, method: str, params: dict[str, object]) -> int:
        request_id = next(self._ids)
        await client.send(request_id, method, params)
        return request_id

    async def new(self, client: ReplayClient, mode: str = "allow") -> str:
        response = await self.call(
            client,
            "new_session",
            {"provider": "codex", "model": "gpt-5.6-sol", "approval_mode": mode},
        )
        return response["result"]["session"]["session_id"]

    def watch(self, label: str, record: Any) -> None:
        """Record each control/runtime close of ``record`` and whether the row's
        gate had opened by then."""
        for hook, owner in (("control", record.control), ("runtime", record.runtime)):
            original = owner.close

            async def close(*args: Any, _hook: str = hook, _original: Any = original, **kwargs: Any) -> Any:
                self.closes.append((label, _hook, self.hold.opened.is_set()))
                return await _original(*args, **kwargs)

            self.monkeypatch.setattr(owner, "close", close)

    def count(self, owner: Any, name: str) -> list[int]:
        """Count calls of ``owner.name`` (sync or async) without changing it."""
        calls = [0]
        original = getattr(owner, name)

        def counted(*args: Any, **kwargs: Any) -> Any:
            calls[0] += 1
            return original(*args, **kwargs)

        self.monkeypatch.setattr(owner, name, counted)
        return calls

    async def shutdown_while_held(self) -> asyncio.Task[None]:
        """Start server close while the row's gate holds; the close must not
        reach any control or runtime close until the gate opens."""
        assert self.server is not None
        closing = asyncio.create_task(self.server.close())
        await _turns(lambda: bool(self.closes) or closing.done())
        assert self.closes == [], "a session closed under work that still used it"
        assert not closing.done()
        self.shut_down = True
        return closing

    async def finish(self, closing: asyncio.Task[None] | None = None) -> None:
        self.hold.opened.set()
        if closing is not None:
            await await_or_fail(closing, what="server close")
        assert self.server is not None
        await _until(lambda: not any(getattr(r, "closing", False) for r in self.server.controllers.records()), what="closes settle")
        # each record closes control then runtime once, both after the gate opened
        labels = {label for label, _hook, _open in self.closes}
        for label in labels:
            assert [(hook, opened) for name, hook, opened in self.closes if name == label] == [
                ("control", True),
                ("runtime", True),
            ], self.closes
        live = [r.runtime.session_id for r in self.server.controllers.records() if not getattr(r, "closing", False) and r.runtime.opened is not None]
        assert len(live) == len(set(live)), live
        if not self.shut_down:
            for client in self.clients:
                await client.close()
            self.clients.clear()
            fresh = await self.open()
            assert "result" in await self.call(fresh, "status")


# --- delivery ---------------------------------------------------------------


async def _delivery_held_at_commit(run: Run, *, before_commit: bool) -> tuple[Any, list[int], list[int], str]:
    server = await run.serve()
    client = await run.open()
    await run.new(client)
    record = server.controllers.attached()
    run.watch("a", record)
    store = record.runtime.opened.store
    original = store.append_client_delivery_async

    async def append(*args: Any, **kwargs: Any) -> Any:
        if before_commit:
            await run.hold()
            return await original(*args, **kwargs)
        result = await original(*args, **kwargs)
        await run.hold()
        return result

    run.monkeypatch.setattr(store, "append_client_delivery_async", append)
    turns = run.count(record.control, "start_turn")
    echoes = run.count(record.control, "user_message")
    await run.send(client, "send", {"text": "hi", "delivery_id": "d-1"})
    await run.hold.wait_entered()
    return store, turns, echoes, "d-1"


async def delivery_after_commit(run: Run) -> None:
    """Fail-first: the commit landed, then server close starts (and on unix
    cancels the send). The session closes only after the delivery's finalizer
    ran: one durable row, one turn start, at most one echo."""
    store, turns, echoes, delivery_id = await _delivery_held_at_commit(run, before_commit=False)
    closing = await run.shutdown_while_held()
    await run.finish(closing)
    assert store.client_delivery(delivery_id) is not None
    assert turns[0] == 1 and echoes[0] <= 1, (turns, echoes)


async def delivery_cancel_before_commit(run: Run) -> None:
    """Cancel at the commit await (unix: server close cancels the send before its
    row is written): no early close, and a turn starts exactly when the row
    exists."""
    store, turns, echoes, delivery_id = await _delivery_held_at_commit(run, before_commit=True)
    closing = await run.shutdown_while_held()
    await run.finish(closing)
    committed = store.client_delivery(delivery_id) is not None
    assert turns[0] == int(committed) and echoes[0] <= turns[0], (committed, turns, echoes)
    if run.transport == "unix":
        assert not committed


async def delivery_duplicate(run: Run) -> None:
    """Duplicate input: the second send with one delivery id is a duplicate."""
    server = await run.serve()
    client = await run.open()
    await run.new(client)
    control = server.controllers.attached().control
    turns = run.count(control, "start_turn")
    first = await run.call(client, "send", {"text": "hi", "delivery_id": "d-1"})
    second = await run.call(client, "send", {"text": "hi", "delivery_id": "d-1"})
    assert first["result"]["accepted"] and second["result"]["duplicate"], (first, second)
    assert turns[0] == 1
    assert server.controllers.attached().uses == set()
    run.hold.opened.set()
    await run.finish()


async def delivery_restart(run: Run) -> None:
    """Restart with a committed delivery whose echo never published: the restarted
    server answers the same delivery id as a duplicate and starts no turn."""
    store, turns, _echoes, delivery_id = await _delivery_held_at_commit(run, before_commit=False)
    session_id = store.session_id
    closing = await run.shutdown_while_held()
    await run.finish(closing)
    assert turns[0] == 1
    for client in run.clients:
        await client.close()
    run.clients.clear()
    run.closes.clear()
    run.shut_down = False
    server = await run.serve()
    client = await run.open()
    await run.call(client, "resume", {"session_id": session_id})
    control = server.controllers.attached().control
    restarted_turns = run.count(control, "start_turn")
    again = await run.call(client, "send", {"text": "hi", "delivery_id": delivery_id})
    assert again["result"]["duplicate"], again
    assert restarted_turns[0] == 0
    await run.finish()


# --- approval ---------------------------------------------------------------


async def _approval(run: Run) -> tuple[Any, ReplayClient, str]:
    target = run.tmp_path / "out.txt"
    write = ToolUseContent(ToolCall("w-1", "write", {"path": str(target), "content": "x"}))
    server = await run.serve([
        ([TextContent("writing")], [TextContent("writing"), write]),
        *[([TextContent("done")], [TextContent("done")])] * 3,
    ])
    client = await run.open()
    await run.new(client, mode="ask")
    record = server.controllers.attached()
    run.watch("a", record)
    await run.call(client, "send", {"text": "write"})
    await _until(record.control.has_pending_approvals, what="the approval request")
    status = await run.call(client, "status")
    return record, client, status["result"]["pending_approvals"][0]["request_id"]


async def approval_after_decision(run: Run) -> None:
    """Fail-first: the decision is durable, its terminal publication is held, and
    server close cancels the approve request. The session closes only after the
    settlement publishes, once."""
    record, client, request_id = await _approval(run)
    approvals = record.control._approvals
    original = approvals._publish_end

    async def publish_end(*args: Any, **kwargs: Any) -> Any:
        await run.hold()
        return await original(*args, **kwargs)

    # test-contract: seam gates the one approval publication step
    run.monkeypatch.setattr(approvals, "_publish_end", publish_end)
    store = record.runtime.opened.store
    await run.send(client, "approve", {"request_id": request_id})
    await run.hold.wait_entered()
    assert store.approval_resolution_now(request_id) is not None
    closing = await run.shutdown_while_held()
    await run.finish(closing)
    assert run.hold.count == 1
    assert store.approval_resolution_now(request_id)["decision"] == "allow"


async def approval_publication_fails(run: Run) -> None:
    """Failure at the terminal publication step: the approve request reports it,
    the durable decision stays, and nothing closes early."""
    record, client, request_id = await _approval(run)
    approvals = record.control._approvals

    async def publish_end(*args: Any, **kwargs: Any) -> Any:
        await run.hold()
        raise OSError("publication failed")

    # test-contract: seam gates the one approval publication step
    run.monkeypatch.setattr(approvals, "_publish_end", publish_end)
    store = record.runtime.opened.store
    sent = await run.send(client, "approve", {"request_id": request_id})
    await run.hold.wait_entered()
    run.hold.opened.set()
    response = await client.response(sent)
    assert "error" in response, response
    assert store.approval_resolution_now(request_id)["decision"] == "allow"
    assert run.hold.count == 1
    run.backend.open_all()
    await run.finish()


async def approval_duplicate(run: Run) -> None:
    """Duplicate input: a second approve of one request is refused and adds no
    decision or publication."""
    record, client, request_id = await _approval(run)
    publications = run.count(record.control._approvals, "_publish_end")
    first = await run.call(client, "approve", {"request_id": request_id})
    second = await run.call(client, "approve", {"request_id": request_id})
    assert first["result"]["accepted"] and "error" in second, (first, second)
    assert publications[0] == 1
    run.hold.opened.set()
    await _until(lambda: not record.uses, what="the settlement ends")
    await run.finish()


# --- setup ------------------------------------------------------------------


HELD = ([TextContent("held"), HOLD], [TextContent("held")])
OK = ([TextContent("ok")], [TextContent("ok")])


async def _busy_a(run: Run, replies: list[Any] | None = None) -> tuple[ZetaServer, ReplayClient, str]:
    server = await run.serve(replies or [HELD, OK, OK, OK])
    client = await run.open()
    a = await run.new(client)
    run.watch("a", server.controllers.attached())
    await run.call(client, "send", {"text": "hold"})
    await await_or_fail(run.backend.held[0].wait(), what="A's turn is busy")
    return server, client, a


def _gate_new_pair(run: Run, server: ZetaServer, method: str, fail: bool) -> list[Any]:
    """Make the next new pair's ``method`` wait at the row's gate, then fail."""
    incoming: list[Any] = []
    new_pair = server.new_pair

    def gated_pair() -> Any:
        reservation = new_pair()
        record = getattr(reservation, "record", reservation)
        incoming.append(record)
        run.watch("incoming", record)

        async def setup(*args: Any, **kwargs: Any) -> None:
            await run.hold()
            if fail:
                raise RuntimeError("setup failed")

        run.monkeypatch.setattr(record.runtime, method, setup)
        return reservation

    run.monkeypatch.setattr(server, "new_pair", gated_pair)
    return incoming


async def _old_session_serves(run: Run, client: ReplayClient, a: str) -> None:
    assert run.server.controllers.attached().runtime.session_id == a
    run.backend.open_all()
    await _until(lambda: not run.server.controllers.attached().control.state_change_busy(), what="A's turn ends")
    sent = await run.call(client, "send", {"text": "still here", "delivery_id": "after-failure"})
    assert sent["result"]["accepted"], sent


async def setup_failure_stays_reserved(run: Run) -> None:
    """Fail-first: a parked new_session fails its setup. The new generation stays
    resident (closing) until its cleanup ends, the error answers only after
    that, and the old session still serves."""
    server, client, a = await _busy_a(run)
    incoming = _gate_new_pair(run, server, "create_session", fail=True)
    sent = await run.send(client, "new_session", {"provider": "codex"})
    await run.hold.wait_entered()
    run.hold.opened.set()
    await _until(lambda: bool(incoming) and getattr(incoming[0], "closing", True), what="setup fails")
    assert incoming[0] in server.controllers.records(), "a failed setup released its generation before cleanup"
    response = await client.response(sent)
    assert "error" in response, response
    assert incoming[0] not in server.controllers.records()
    await _old_session_serves(run, client, a)
    await run.finish()


async def resume_failure_keeps_old(run: Run) -> None:
    """Failed acquire: a parked resume fails its setup; only the target
    generation closes and the old attachment keeps serving."""
    # A held, B's first turn, B held, then plain replies
    server, client, a = await _busy_a(run, [HELD, OK, HELD, OK, OK])
    b = await run.new(client)  # parks A
    await run.call(client, "send", {"text": "b"})
    await _until(lambda: not server.controllers.attached().control.state_change_busy(), what="B's turn ends")
    run.watch("b", server.controllers.attached())
    run.backend.gates[0].set()
    await _until(lambda: server.parking.get(a) is None, what="A's park ends")
    await await_or_fail(server.controllers.wait_closed(a), what="A closes")
    run.closes.clear()
    # A is on disk now; hold B busy so the resume takes a new pair
    await run.call(client, "send", {"text": "hold b"})
    await await_or_fail(run.backend.held[1].wait(), what="B's turn is busy")
    incoming = _gate_new_pair(run, server, "resume_session", fail=True)
    sent = await run.send(client, "resume", {"session_id": a})
    await run.hold.wait_entered()
    run.hold.opened.set()
    response = await client.response(sent)
    assert "error" in response, response
    assert incoming and incoming[0] not in server.controllers.records()
    await _old_session_serves(run, client, b)
    await run.finish()


async def setup_cancelled_by_shutdown(run: Run) -> None:
    """Cancel at the setup's open await: server close cancels new_session while
    the new pair opens. The new generation stays resident until its cleanup;
    nothing closes before the setup ends."""
    server, client, _a = await _busy_a(run)
    incoming = _gate_new_pair(run, server, "create_session", fail=False)
    await run.send(client, "new_session", {"provider": "codex"})
    await run.hold.wait_entered()
    run.backend.open_all()
    closing = await run.shutdown_while_held()
    assert incoming[0] in server.controllers.records()
    await run.finish(closing)
    assert incoming[0] not in server.controllers.records()


async def resume_waits_for_closing(run: Run) -> None:
    """A closing generation is not claimable: resuming its session waits until
    its close ends, then reopens it."""
    server, client, a = await _busy_a(run)
    await run.new(client)  # parks A
    parked = server.parking.get(a)
    original = parked.control.close
    started = asyncio.Event()

    async def held_close() -> None:
        started.set()
        await run.hold()
        await original()

    run.monkeypatch.setattr(parked.control, "close", held_close)
    run.backend.open_all()
    await await_or_fail(started.wait(), what="A's idle close starts")
    sent = await run.send(client, "resume", {"session_id": a})
    await _turns(lambda: bool(client.buffered))
    assert parked in server.controllers.records() and parked.closing
    assert all(frame.get("id") != sent for frame in client.buffered), "resumed a closing session"
    run.hold.opened.set()
    response = await client.response(sent)
    assert response["result"]["session"]["session_id"] == a, response
    assert server.controllers.attached() is not parked
    await run.finish()


# --- worker read ------------------------------------------------------------


async def worker_read_cancelled(run: Run) -> None:
    """Cancel at the worker-read await: server close cancels a history read whose
    worker still runs; the session closes after the worker returns."""
    server = await run.serve()
    client = await run.open()
    session_id = await run.new(client)
    run.watch("a", server.controllers.attached())
    reading = threading.Event()
    opened = threading.Event()
    original = history_module.history

    def gated(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        reading.set()
        assert opened.wait(10), "history worker was not released"
        return result

    run.monkeypatch.setattr(history_module, "history", gated)
    await run.send(client, "session_history", {"session_id": session_id})
    assert await await_or_fail(asyncio.to_thread(reading.wait, 10), what="worker reads")
    try:
        closing = await run.shutdown_while_held()
    finally:
        opened.set()
    await run.finish(closing)


async def success(run: Run) -> None:
    """Success: a send, its turn, and an idle switch settle with no hold and no
    close; the final server close then closes the pair once."""
    server = await run.serve()
    client = await run.open()
    await run.new(client)
    run.watch("a", server.controllers.attached())
    run.hold.opened.set()
    sent = await run.call(client, "send", {"text": "hi", "delivery_id": "ok-1"})
    assert sent["result"]["accepted"], sent
    await _until(lambda: not server.controllers.attached().control.state_change_busy(), what="the turn ends")
    record = server.controllers.attached()
    await run.new(client)  # A is idle: the same pair opens the new session
    assert server.controllers.attached() is record and record.uses == set()
    # the reused pair closed only its old session's runtime state, never the control
    assert run.closes == [("a", "runtime", True)], run.closes
    run.closes.clear()
    await run.finish()


ROWS: dict[str, Callable[[Run], Awaitable[None]]] = {
    "success": success,
    "delivery_after_commit": delivery_after_commit,
    "delivery_cancel_before_commit": delivery_cancel_before_commit,
    "delivery_duplicate": delivery_duplicate,
    "delivery_restart": delivery_restart,
    "approval_after_decision": approval_after_decision,
    "approval_publication_fails": approval_publication_fails,
    "approval_duplicate": approval_duplicate,
    "setup_failure_stays_reserved": setup_failure_stays_reserved,
    "setup_cancelled_by_shutdown": setup_cancelled_by_shutdown,
    "resume_failure_keeps_old": resume_failure_keeps_old,
    "resume_waits_for_closing": resume_waits_for_closing,
    "worker_read_cancelled": worker_read_cancelled,
}


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("row", list(ROWS))
async def test_session_close_table(
    tmp_path: Path, transport: Transport, row: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = Run(tmp_path, transport, monkeypatch)
    try:
        await ROWS[row](run)
    finally:
        run.hold.opened.set()
        if run.backend is not None:
            run.backend.open_all()
        for client in run.clients:
            await client.close()
        if run.server is not None:
            await await_or_fail(run.server.close(), what="final server close")
