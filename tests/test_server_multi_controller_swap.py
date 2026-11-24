"""Multi-controller swap table through the real serve socket: each row sets up
the connection's outgoing session, swaps (``new_session`` or ``resume``), and
checks the one outgoing decision (park, close, refuse, or nothing), the parked
count, one record per session id, and that the next turn runs.

Restart is not a row: the swap writes no stored state of its own; the stores it
opens or creates are covered by the resume and park tables."""
from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from dataclasses import dataclass
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
from tests.test_server_multi_controller_resume import _legacy_resident, _records
from tests.test_server_session_park import _until
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent
from zeta.server import ZetaServer
from zeta.server.control import ServeControl
from zeta.server.handlers import lifecycle
from zeta.server.runtime import ServerRuntime

MULTI = ("multi_controller", "new_session_settings")
PARK = (*MULTI, "session_park", "resume_attach")
LEGACY = ("new_session_settings",)
SETTINGS = {"provider": "codex", "model": "gpt-5.6-sol"}
HELD = ([TextContent("held"), HOLD], [TextContent("held")])
NEXT = ([TextContent("next")], [TextContent("next")])


@dataclass(frozen=True)
class Row:
    features: tuple[str, ...]
    setup: str  # see _setup
    route: str  # new_session | resume_disk | resume_parked | resume_own
    error: str | None = None  # parked_limit | busy
    parked: int = 0  # parked rows after the swap
    outgoing_kept: bool = False  # the outgoing session still has its record
    inject: tuple[str, str] | None = None  # (cancel | fail, seam)
    duplicate: bool = False


ROWS = {
    # parked cap: only a swap that parks counts, and the target never does
    "cap_idle_new": Row(PARK, "cap_idle", "new_session", parked=3),
    "cap_idle_resume_disk": Row(PARK, "cap_idle", "resume_disk", parked=3),
    "cap_idle_resume_parked": Row(PARK, "cap_idle", "resume_parked", parked=2),
    "cap_busy_resume_parked": Row(PARK, "cap_busy", "resume_parked", parked=3, outgoing_kept=True),
    "cap_busy_new": Row(PARK, "cap_busy", "new_session", error="parked_limit", parked=3, outgoing_kept=True),
    "cap_busy_duplicate_resume_parked": Row(
        PARK, "cap_busy", "resume_parked", parked=3, outgoing_kept=True, duplicate=True
    ),
    # park / close / refuse
    "busy_park_new": Row(PARK, "busy", "new_session", parked=1, outgoing_kept=True),
    "busy_new": Row(MULTI, "busy", "new_session", error="busy", outgoing_kept=True),
    "idle_new": Row(MULTI, "idle", "new_session"),
    "idle_park_resume_disk": Row(PARK, "idle", "resume_disk"),
    # minor 4: without session_park, background work closes as in legacy
    "background_new": Row(MULTI, "background", "new_session"),
    "background_new_legacy": Row(LEGACY, "background", "new_session"),
    "background_park_new": Row(PARK, "background", "new_session", parked=1, outgoing_kept=True),
    # minor 2: an unhealthy outgoing closes, never parks beside the new record
    "unhealthy_work_park_resume_own": Row(PARK, "unhealthy_work", "resume_own"),
    "unhealthy_busy_park_new": Row(PARK, "unhealthy_busy", "new_session"),
    # cancel at each await, failure at each publish step
    **{
        f"{action}_park_new_{seam}": Row(
            PARK, "busy", "new_session", parked=1, outgoing_kept=True, inject=(action, seam)
        )
        for action in ("cancel", "fail") for seam in ("create", "ready", "notice", "grant")
    },
    **{
        f"{action}_park_resume_parked_{seam}": Row(
            PARK, "parked_busy", "resume_parked", parked=1, outgoing_kept=True, inject=(action, seam)
        )
        for action in ("cancel", "fail") for seam in ("claim", "grant")
    },
    **{
        f"{action}_close_new_{seam}": Row(MULTI, "idle", "new_session", inject=(action, seam))
        for action in ("cancel", "fail") for seam in ("create", "ready", "notice", "grant")
    },
    "cancel_close_new_release": Row(MULTI, "idle", "new_session", inject=("cancel", "release")),
    **{
        f"{action}_close_resume_disk_reopen": Row(MULTI, "idle", "resume_disk", inject=(action, "reopen"))
        for action in ("cancel", "fail")
    },
}


def _seam(name: str) -> tuple[object, str]:
    return {
        "create": (ServerRuntime, "create_session"),
        "reopen": (ServerRuntime, "resume_session"),
        "ready": (ServeControl, "controller_ready"),
        "notice": (lifecycle, "_publish_project_registry_notice"),
        "grant": (ServeControl, "publish_session_grant_reset"),
        "claim": (ServeControl, "controller_claimed"),
        "release": (ServeControl, "_disconnect_turn"),
    }[name]


async def _settled_turn(client: ReplayClient, session_id: str) -> None:
    while (end := await await_or_fail(client.event(match={"session_id": session_id}), what="turn end"))[
        "event"
    ] not in ("turn_end", "error"):
        pass
    assert end["event"] == "turn_end", f"the turn failed: {end}"


async def _turn(client: ReplayClient, request_id: int, session_id: str) -> None:
    sent = await client.request(request_id, "send", {"text": f"turn {request_id}"})
    assert sent.get("result", {}).get("accepted") is True, sent
    await _settled_turn(client, session_id)


async def _parked(client: ReplayClient, request_id: int) -> int:
    return len((await client.request(request_id, "status", {}))["result"].get("parked_sessions", []))


class Setup:
    """Drive one row's outgoing state; request ids grow from 10."""

    def __init__(self, server: ZetaServer, backend: GatedBackend, client: ReplayClient, go: Path) -> None:
        self.server, self.backend, self.client, self.go = server, backend, client, go
        self.next_id = 10
        self.gates = 0
        self.parked: list[str] = []

    def rid(self) -> int:
        self.next_id += 1
        return self.next_id

    async def new(self, mode: str = "allow") -> str:
        reply = await self.client.request(self.rid(), "new_session", {**SETTINGS, "approval_mode": mode})
        assert "result" in reply, reply
        return reply["result"]["session"]["session_id"]

    async def hold(self) -> str:
        session_id = await self.new()
        assert "result" in await self.client.request(self.rid(), "send", {"text": "hold"})
        await await_or_fail(self.backend.held[self.gates].wait(), what="the held turn")
        self.gates += 1
        return session_id

    async def background(self) -> str:
        session_id = await self.new()
        await _turn(self.client, self.rid(), session_id)
        (record,) = _records(self.server, session_id)
        await _until(lambda: not record.control.state_change_busy(), "the turn's end step")
        assert record.control.holds_work(), "the background task must hold work"
        return session_id


def _replies(setup: str, go: Path) -> list[tuple[list[object], list[Any]]]:
    task = ToolUseContent(ToolCall("bg-1", "run_background", {
        "command": f"while [ ! -e {go} ]; do sleep 0.05; done",
    }))
    held = {"cap_idle": 3, "cap_busy": 4, "busy": 1, "unhealthy_busy": 1, "parked_busy": 2}.get(setup, 0)
    background = [([TextContent("starting")], [TextContent("starting"), task])] if setup in (
        "background", "unhealthy_work"
    ) else []
    return [*background, *[HELD] * held, *[NEXT] * 8]


async def _setup(row: Row, run: Setup, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str | None]:
    """Return (outgoing session id, disk session id or None)."""
    disk = None
    if row.route == "resume_disk":
        disk = await run.new()
    if row.setup.startswith("cap_"):
        run.parked = [await run.hold() for _ in range(3)]
        return (await run.hold() if row.setup == "cap_busy" else await run.new()), disk
    if row.setup == "parked_busy":
        run.parked = [await run.hold()]
        return await run.hold(), disk
    if row.setup == "idle":
        return await run.new(), disk
    if row.setup == "busy":
        return await run.hold(), disk
    if row.setup == "background":
        return await run.background(), disk
    outgoing = await (run.background() if row.setup == "unhealthy_work" else run.hold())
    (record,) = _records(run.server, outgoing)
    # stands in for failed approval storage on the outgoing session
    monkeypatch.setattr(record.control, "unhealthy", lambda: True)
    return outgoing, disk


def _no_leaked_reservation(server: ZetaServer) -> None:
    for record in server.controllers.records():
        assert not record.control._session_swap_active, "the swap left the outgoing reserved"
        if record.parked_by is not None:
            assert record.closing or server.parking.get(record.runtime.session_id) is record, (
                "the swap left a provisional parked charge"
            )


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("name", tuple(ROWS))
@pytest.mark.asyncio
async def test_multi_swap_outgoing_table(
    tmp_path: Path, transport: Transport, name: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    row = ROWS[name]
    go = tmp_path / "go"
    backend = GatedBackend(_replies(row.setup, go))
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start swap server")
    clients: list[ReplayClient] = []
    try:
        client = await connect_replay(server, transport)
        clients.append(client)
        assert "result" in await negotiate(client, replay=False, extra=row.features)
        run = Setup(server, backend, client, go)
        outgoing, disk = await _setup(row, run, monkeypatch)
        target = {
            "resume_disk": disk, "resume_parked": run.parked[0] if run.parked else None, "resume_own": outgoing,
        }.get(row.route)
        request: tuple[str, dict[str, Any]] = (
            ("new_session", SETTINGS) if target is None else ("resume", {"session_id": target})
        )
        if row.inject is not None:
            await _injected(server, run, row, request, outgoing, transport, clients, monkeypatch)
            return

        (old,) = _records(server, outgoing)
        if row.duplicate:
            await client.send(90, *request)
            await client.send(91, *request)
            replies = [await client.response(90), await client.response(91)]
            assert all("result" in reply for reply in replies), replies
            reply = replies[0]
        else:
            reply = await client.request(90, *request)

        if row.error is not None:
            error = reply.get("error", {})
            assert error.get("code") == -32004, reply
            if row.error == "parked_limit":
                assert error.get("data", {}).get("reason") == "parked_limit", reply
            assert await _parked(client, 91) == row.parked
            status = (await client.request(92, "status", {}))["result"]
            assert status["session"]["session_id"] == outgoing, "a refused swap must keep the session"
            backend.open_all()
            await _settled_turn(client, outgoing)
            await _turn(client, 93, outgoing)
            return
        assert "result" in reply, f"the swap was refused: {reply.get('error')}"
        current = reply["result"]["session"]["session_id"]
        if target is not None:
            assert current == target
        assert await _parked(client, 92) == row.parked, "wrong parked count after the swap"
        assert len(_records(server, current)) == 1, "one resident record per session id"
        if current != outgoing:
            kept = len(_records(server, outgoing))
            assert kept == int(row.outgoing_kept), f"the outgoing session has {kept} records"
        if old.close_task is not None:
            await await_or_fail(asyncio.shield(old.close_task), what="outgoing closes")
        (record,) = _records(server, current)
        # a closed background task's exit may wake the session once
        await _until(lambda: not record.control.holds_work() or row.route == "resume_parked" or record is old, "idle")
        backend.open_all()
        if row.route == "resume_parked":
            await _settled_turn(client, current)  # its held turn ends now
        await _turn(client, 93, current)
    finally:
        go.touch()
        backend.open_all()
        for client in clients:
            await client.close()
        await await_or_fail(server.close(), what="close swap server")


async def _injected(
    server: ZetaServer, run: Setup, row: Row, request: tuple[str, dict[str, Any]], outgoing: str,
    transport: Transport, clients: list[ReplayClient], monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cancel the swap's request at one await, or fail one publish step; the
    swap must release every reservation it took."""
    assert row.inject is not None
    action, step = row.inject
    owner, method = _seam(step)
    original = getattr(owner, method)
    entered, release, settled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    calls: list[None] = []

    async def interrupt(*args: Any, **kwargs: Any) -> Any:
        calls.append(None)
        if len(calls) > 1:
            return await original(*args, **kwargs)
        if action == "fail":
            entered.set()
            raise RuntimeError(f"injected {step} failure")
        task = asyncio.current_task()
        assert task is not None
        task.add_done_callback(lambda _: settled.set())
        entered.set()
        await await_or_fail(release.wait(), what=f"release {step}")
        return await original(*args, **kwargs)

    client = run.client
    peer = server.connections.clients()[0]
    fresh = None
    if action == "cancel":
        # a second controller keeps multi mode while the cancelled one's sessions are in grace
        fresh = await connect_replay(server, transport)
        clients.append(fresh)
        assert "result" in await negotiate(fresh, replay=False, extra=(*MULTI, "resume_attach"))
    with monkeypatch.context() as patch:
        # test-contract: seam hold or fail the swap at one await or publish step.
        patch.setattr(owner, method, interrupt)
        try:
            await client.send(90, *request)
            await await_or_fail(entered.wait(), what=f"the swap reaches {step}")
            if action == "cancel":
                assert peer._run_task is not None
                peer._run_task.cancel()
                if step == "release":
                    release.set()  # the release is shielded: the cancel waits for it, then raises
                await await_or_fail(settled.wait(), what="the cancelled swap settles")
                with contextlib.suppress(asyncio.CancelledError):
                    await await_or_fail(peer._run_task, what="the cancelled handler ends")
            else:
                reply = await client.response(90)
                assert "error" in reply, f"the injected {step} failure succeeded: {reply}"
        finally:
            release.set()
    _no_leaked_reservation(server)
    run.backend.open_all()
    if action == "fail":
        # the connection keeps its session and its parked count, and can swap again
        status = (await client.request(91, "status", {}))["result"]
        assert status["session"]["session_id"] == outgoing, "a failed swap must keep the session"
        # a failed claim releases the parked target (its turn is cut), as before this owner
        kept = 0 if row.route == "resume_parked" else len(run.parked)
        assert len(status.get("parked_sessions", [])) == kept
        if row.setup != "idle":
            await _settled_turn(client, outgoing)
        await _turn(client, 92, outgoing)
        retried = await client.request(93, *request)
        assert "result" in retried, f"a retry after the failed swap was refused: {retried}"
        return
    # a fresh controller resumes the outgoing session and runs its next turn
    assert fresh is not None
    await _until(lambda: peer not in server.connections.clients(), "the cancelled connection leaves")
    await await_or_fail(server.controllers.wait_closed(outgoing), what="outgoing settles")
    resumed = await fresh.request(2, "resume", {"session_id": outgoing})
    assert "result" in resumed, resumed
    assert len(_records(server, outgoing)) == 1, "one resident record per session id"
    await _turn(fresh, 3, outgoing)


@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.asyncio
async def test_resume_resident_does_not_reserve_shared_sessionless_record(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Connections without a session share one unopened record: one connection's
    resume of a resident session does not make another's first open refuse."""
    backend = GatedBackend([NEXT] * 4)
    server = replay_server(tmp_path, transport, lambda *_: (backend, "offline"))
    await await_or_fail(server.start(), what="start resident server")
    clients: list[ReplayClient] = []
    entered, release = asyncio.Event(), asyncio.Event()
    original: Callable[..., Any] = ServeControl.controller_claimed

    async def held_claim(self: ServeControl, *args: Any, **kwargs: Any) -> Any:
        entered.set()
        await await_or_fail(release.wait(), what="release the claim")
        return await original(self, *args, **kwargs)

    try:
        resident = await _legacy_resident(server, transport, monkeypatch=monkeypatch)
        first = await connect_replay(server, transport)
        second = await connect_replay(server, transport)
        clients += [first, second]
        assert "result" in await negotiate(first, replay=False, extra=MULTI)
        assert "result" in await negotiate(second, replay=False, extra=MULTI)
        # test-contract: seam the first connection's resident claim is held open by the test
        monkeypatch.setattr(ServeControl, "controller_claimed", held_claim)
        await first.send(2, "resume", {"session_id": resident})
        await await_or_fail(entered.wait(), what="resume in its claim")

        opened = await second.request(2, "new_session", SETTINGS)

        assert "result" in opened, f"the shared sessionless record was reserved: {opened.get('error')}"
        release.set()
        resumed = await first.response(2)
        assert resumed.get("result", {}).get("session", {}).get("session_id") == resident, resumed
        await _turn(first, 3, resident)
        await _turn(second, 3, opened["result"]["session"]["session_id"])
    finally:
        release.set()
        backend.open_all()
        for client in clients:
            await client.close()
        await await_or_fail(server.close(), what="close resident server")
