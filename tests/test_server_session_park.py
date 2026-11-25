"""real serve-socket coverage for `session_park`: `new_session` and `resume` park a
busy session instead of refusing, and the parked session closes once idle."""
from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from tests.support.replay_transport import replay_server
from tests.support.server_client import event, frames_until_event, read, send
from tests.test_server import _connect
from tests.test_server_in_flight_replay import HOLD, GatedBackend, _server
from tests.test_server_turn_reconnect import RECONNECT, _close, _drop
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent
from zeta.server import ZetaServer

BUSY = -32004
FEATURES = [
    *RECONNECT,
    "new_session_settings",
    "resume_attach",
    "session_replay_seq",
    "in_flight_replay",
    "session_park",
]
NO_PARK = [item for item in FEATURES if item != "session_park"]


class Client:
    """One controller connection; every request returns all frames up to its response."""

    def __init__(self, reader, writer) -> None:
        self.reader = reader
        self.writer = writer
        self.next_id = 100

    async def frames(self, method: str, params: dict[str, Any] | None = None):
        self.next_id += 1
        await send(self.reader, self.writer, self.next_id, method, params or {})
        frames = [await read(self.reader)]
        while frames[-1].get("id") != self.next_id:
            frames.append(await read(self.reader))
        return frames

    async def call(self, method: str, params: dict[str, Any] | None = None):
        frames = await self.frames(method, params)
        assert "error" not in frames[-1], frames[-1]
        return frames[-1]["result"]

    async def error(self, method: str, params: dict[str, Any] | None = None):
        frames = await self.frames(method, params)
        assert "error" in frames[-1], frames[-1]
        return frames[-1]["error"]

    async def new(self, mode: str = "allow") -> str:
        result = await self.call("new_session", {
            "provider": "codex", "model": "gpt-5.6-sol", "approval_mode": mode,
        })
        return result["session"]["session_id"]


async def _open(server: ZetaServer, features: list[str]) -> Client:
    if server.connections is None:
        reader, writer = await _connect(server)
    else:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", server.port), 10
        )
    client = Client(reader, writer)
    await client.call("hello", {"protocol_version": "1.2", "features": features})
    return client


async def _until(predicate: Callable[[], bool], what: str) -> None:
    try:
        async with asyncio.timeout(10):
            while not predicate():
                await asyncio.sleep(0)
    except TimeoutError:
        raise AssertionError(f"timed out waiting for {what}") from None


async def _closed(server: ZetaServer, session_id: str) -> None:
    await _until(lambda: server.parking.get(session_id) is None, f"{session_id} closing")
    await await_or_fail(server.controllers.wait_closed(session_id), what=f"{session_id} closed")


def _stream_text(frames: list[dict[str, Any]], stream_id: str) -> str:
    return "".join(
        frame["params"]["delta"]
        for frame in frames
        if frame.get("method") == "event"
        and frame["params"].get("stream_id") == stream_id
        and frame["params"]["event"] == "assistant_delta"
    )


async def _join_text(client: Client, session_id: str) -> str:
    """The reply text a join replays from the session's in-flight buffer."""
    joined = await client.call("session_history_and_subscribe", {"session_id": session_id})
    # the held frames follow the join response; a status round trip reads them all
    after = await client.frames("status")
    return _stream_text(after, joined["replay"]["stream_id"])


def _event_names(frames: list[dict[str, Any]], session_id: str) -> list[str]:
    return [
        frame["params"]["event"]
        for frame in frames
        if frame.get("method") == "event"
        and frame["params"].get("session_id") == session_id
        and frame["params"]["event"] != "parked_session_changed"
    ]


@pytest.mark.asyncio
async def test_new_session_parks_busy_turn_and_each_session_keeps_its_reply(
    tmp_path: Path,
) -> None:
    backend = GatedBackend([
        ([TextContent("alpha-1"), HOLD, TextContent("alpha-2"), HOLD],
         [TextContent("alpha-1alpha-2")]),
        ([TextContent("beta-1"), HOLD], [TextContent("beta-1")]),
    ])
    server = _server(tmp_path, backend)
    client = None
    try:
        client = await _open(server, FEATURES)
        await client.new()
        idle_swap_status = await client.call("status")
        manager = server.controllers.attached().runtime.manager
        a = await client.new()
        await client.call("send", {"text": "go"})
        await await_or_fail(backend.held[0].wait(), what="alpha-1 streamed")

        b = await client.new()  # without session_park: -32004
        assert b != a
        assert server.controllers.attached().runtime.manager is manager
        b_status = await client.call("status")
        assert b_status["project_context_health"] == idle_swap_status["project_context_health"]
        assert [row["session_id"] for row in b_status["parked_sessions"]] == [a]

        await client.call("send", {"text": "go"})
        await await_or_fail(backend.held[2].wait(), what="beta-1 streamed")
        assert await _join_text(client, b) == "beta-1"

        backend.gates[0].set()
        await await_or_fail(backend.held[1].wait(), what="alpha-2 streamed while parked")
        backend.gates[2].set()
        frames = await frames_until_event(client.reader, "agent_end")
        assert frames[-1]["params"]["session_id"] == b
        assert _event_names(frames, a) == []
        status = await client.call("status")
        assert status["parked_sessions"] == [
            {"session_id": a, "state": "running", "pending_approvals": 0}
        ]

        resumed = await client.call("resume", {"session_id": a})
        assert resumed["attached"] == "live_turn"
        assert await _join_text(client, a) == "alpha-1alpha-2"
        backend.gates[1].set()
        frames = await frames_until_event(client.reader, "agent_end")
        turn_end = next(f for f in frames if f["params"].get("event") == "turn_end")
        assert turn_end["params"]["session_id"] == a
        assert (await client.call("status"))["last_turn_outcome"]["reason"] == "completed"
    finally:
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_parked_approval_is_read_from_join_after_resume(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    call = ToolUseContent(ToolCall("w-1", "write", {"path": str(target), "content": "x"}))
    backend = GatedBackend([
        ([TextContent("writing"), HOLD], [TextContent("writing"), call]),
        ([TextContent("beta")], [TextContent("beta")]),
        ([TextContent("written")], [TextContent("written")]),
    ])
    server = replay_server(tmp_path, "unix", lambda *_: (backend, "offline"))
    client = None
    try:
        client = await _open(server, [*FEATURES, "state_read_v1"])
        a = await client.new(mode="ask")
        await client.call("send", {"text": "write"})
        await await_or_fail(backend.held[0].wait(), what="A mid-reply")
        b = await client.new()
        parked = server.parking.get(a).control

        backend.gates[0].set()
        await _until(parked.has_pending_approvals, "parked approval")
        # B's turn start closes stale approvals of B's own store only (#1357)
        seen = await client.frames("send", {"text": "go"})
        seen += await frames_until_event(client.reader, "agent_end")
        assert seen[-1]["params"]["session_id"] == b
        seen += await client.frames("status")
        status = seen[-1]["result"]
        assert [(row["session_id"], row["pending_approvals"])
                for row in status["parked_sessions"]] == [(a, 1)]
        assert {"session_id": a, "state": "running", "pending_approvals": 1} in [
            {key: frame["params"]["data"][key] for key in ("session_id", "state", "pending_approvals")}
            for frame in seen
            if frame.get("params", {}).get("event") == "parked_session_changed"
        ]

        rows = (await client.call("state_sessions"))["items"]
        assert {row["session_id"]: row["status"] for row in rows} == {a: "approval", b: "idle"}

        resume = await client.frames("resume", {"session_id": a})
        seen += resume
        assert resume[-1]["result"]["attached"] == "live_turn"
        assert "approval_request" not in [
            frame.get("params", {}).get("event") for frame in seen
        ]
        joined = await client.call("session_history_and_subscribe", {"session_id": a})
        durable = await server.controllers.attached().control.pending_approvals()
        request_id = joined["status"]["pending_approvals"][0]["request_id"]
        assert request_id == durable[0]["request_id"]

        await client.call("approve", {"request_id": request_id})
        frames = await frames_until_event(client.reader, "agent_end")
        turn_end = next(f for f in frames if f["params"].get("event") == "turn_end")
        assert turn_end["params"]["session_id"] == a
        assert (await client.call("status"))["last_turn_outcome"]["reason"] == "completed"
        assert target.read_text() == "x"
    finally:
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_without_session_park_new_session_during_turn_is_still_busy(
    tmp_path: Path,
) -> None:
    backend = GatedBackend([([TextContent("held"), HOLD], [TextContent("held")])])
    server = _server(tmp_path, backend)
    client = None
    try:
        reader, writer = await _connect(server)
        client = Client(reader, writer)
        hello = await client.call("hello", {
            "protocol_version": "1.2", "features": ["session_park", "new_session_settings", *RECONNECT],
        })
        assert "session_park" not in hello["capabilities"]["features"]
        await client.new()
        await client.call("send", {"text": "go"})
        await await_or_fail(backend.held[0].wait(), what="mid-reply")

        error = await client.error("new_session", {"provider": "codex"})
        assert error["code"] == BUSY
        assert "data" not in error
        assert "parked_sessions" not in await client.call("status")
    finally:
        backend.open_all()
        await _close(server, client and client.writer)


async def _parked_turn(server: ZetaServer, backend: GatedBackend, features: list[str]):
    client = await _open(server, features)
    a = await client.new()
    await client.call("send", {"text": "go"})
    await await_or_fail(backend.held[0].wait(), what="A mid-reply")
    await client.new()
    return client, a, server.parking.get(a).control


async def _parked_changes(client: Client, until: str) -> list[dict[str, Any]]:
    """Read frames until a ``parked_session_changed`` with state ``until``."""
    changes: list[dict[str, Any]] = []
    while not changes or changes[-1]["data"]["state"] != until:
        changes.append(await event(client.reader, "parked_session_changed"))
    return changes


async def _outcome(server: ZetaServer, client: Client, session_id: str) -> str:
    """Wait for the session's pair to close, then read its last turn from disk."""
    await _closed(server, session_id)
    await client.call("resume", {"session_id": session_id})
    return (await client.call("status"))["last_turn_outcome"]["reason"]


@pytest.mark.asyncio
async def test_disconnect_without_turn_reconnect_cancels_parked_turn(
    tmp_path: Path,
) -> None:
    backend = GatedBackend([([TextContent("held"), HOLD], [TextContent("held")])])
    server = _server(tmp_path, backend)
    features = [item for item in FEATURES if item != "turn_reconnect"]
    client = client2 = None
    try:
        client, a, _ = await _parked_turn(server, backend, features)
        await _drop(client.writer, server)
        client2 = await _open(server, features)
        assert await _outcome(server, client2, a) == "disconnect"
    finally:
        backend.open_all()
        await _close(server, client2 and client2.writer, client and client.writer)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("features", "reason"),
    [(FEATURES, "completed"), (NO_PARK, "disconnect")],
    ids=["reclaim_with_session_park", "reclaim_without_session_park"],
)
async def test_reclaim_inside_grace_keeps_parked_turn_only_with_session_park(
    tmp_path: Path, features: list[str], reason: str
) -> None:
    backend = GatedBackend([([TextContent("held"), HOLD], [TextContent("held")])])
    server = _server(tmp_path, backend)
    deadline = asyncio.Event()
    client = client2 = None
    try:
        client, a, parked = await _parked_turn(server, backend, FEATURES)

        async def injected_deadline() -> None:
            await await_or_fail(deadline.wait(), what="injected grace deadline")

        parked._grace_deadline = injected_deadline
        await _drop(client.writer, server)
        assert parked._grace_task is not None

        client2 = await _open(server, features)
        status = await client2.call("status")
        if features is NO_PARK:
            assert "parked_sessions" not in status
        # the old deadline ends now; a pair that was not reclaimed is cancelled by it
        deadline.set()
        grace = parked._grace_task
        if grace is not None:
            await await_or_fail(asyncio.shield(grace), what="grace expiry")
        backend.open_all()
        assert await _outcome(server, client2, a) == reason
    finally:
        deadline.set()
        backend.open_all()
        await _close(server, client2 and client2.writer, client and client.writer)


@pytest.mark.asyncio
async def test_park_limit_refuses_fourth_busy_swap_but_resume_swaps_one_for_one(
    tmp_path: Path,
) -> None:
    backend = GatedBackend([
        ([TextContent(f"turn-{index}"), HOLD], [TextContent(f"turn-{index}")])
        for index in range(4)
    ])
    server = _server(tmp_path, backend)
    client = None
    try:
        client = await _open(server, FEATURES)
        sessions = []
        for index in range(4):
            sessions.append(await client.new())
            await client.call("send", {"text": "go"})
            await await_or_fail(backend.held[index].wait(), what=f"turn {index}")

        error = await client.error("new_session", {"provider": "codex"})
        assert error["code"] == BUSY
        assert error["data"] == {"reason": "parked_limit", "parked": sessions[:3]}

        resumed = await client.call("resume", {"session_id": sessions[0]})
        assert resumed["attached"] == "live_turn"
        status = await client.call("status")
        assert [row["session_id"] for row in status["parked_sessions"]] == [
            *sessions[1:]
        ]
    finally:
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_resume_during_idle_close_waits_for_it_and_new_session_does_not(
    tmp_path: Path,
) -> None:
    backend = GatedBackend([([TextContent("held"), HOLD], [TextContent("held")])])
    server = _server(tmp_path, backend)
    close_started = asyncio.Event()
    close_gate = asyncio.Event()
    client = None
    try:
        client, a, parked = await _parked_turn(server, backend, FEATURES)
        original_close = parked.runtime.close

        async def held_close() -> None:
            close_started.set()
            await await_or_fail(close_gate.wait(), what="held close")
            await original_close()

        parked.runtime.close = held_close
        backend.gates[0].set()
        await await_or_fail(close_started.wait(), what="idle close started")
        assert a in server.controllers.closing_session_ids()

        await await_or_fail(client.new(), what="new_session during the close")

        waiting = asyncio.Event()
        original_wait = server.controllers.wait_closed

        async def wait_closed(session_id: str) -> None:
            if session_id in server.controllers.closing_session_ids():
                waiting.set()
            await original_wait(session_id)

        server.controllers.wait_closed = wait_closed  # type: ignore[method-assign]
        resume = asyncio.create_task(client.call("resume", {"session_id": a}))
        try:
            await await_or_fail(waiting.wait(), what="resume waits for the close")
            assert not resume.done() and a in server.controllers.closing_session_ids()
            close_gate.set()
            result = await await_or_fail(resume, what="resume")
        finally:
            close_gate.set()
            if not resume.done():
                resume.cancel()
        assert result["session"]["session_id"] == a
        assert "attached" not in result
        assert server.controllers.attached().control is not parked
        assert server.controllers.attached().runtime.opened.store.last_turn_end().data["reason"] == "completed"
    finally:
        close_gate.set()
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_turn_ending_during_incoming_setup_closes_instead_of_parking(
    tmp_path: Path,
) -> None:
    """Forced order: the outgoing turn ends while the new session is set up, so
    the swap commit finds it idle and closes it instead of parking it."""
    backend = GatedBackend([([TextContent("held"), HOLD], [TextContent("held")])])
    server = _server(tmp_path, backend)
    setup_started = asyncio.Event()
    setup_gate = asyncio.Event()
    client = None
    try:
        client = await _open(server, FEATURES)
        a = await client.new()
        await client.call("send", {"text": "go"})
        await await_or_fail(backend.held[0].wait(), what="A mid-reply")
        outgoing = server.controllers.attached().control
        original_new_pair = server.new_pair

        def new_pair():
            pair = original_new_pair()
            create = pair.runtime.create_session

            async def held_create(**kwargs):
                setup_started.set()
                await await_or_fail(setup_gate.wait(), what="held setup")
                return await create(**kwargs)

            pair.runtime.create_session = held_create
            return pair

        server.new_pair = new_pair
        created = asyncio.create_task(client.new())
        try:
            await await_or_fail(setup_started.wait(), what="incoming setup")
            backend.gates[0].set()
            await _until(lambda: not outgoing.holds_work(), "A's turn ends during setup")
            setup_gate.set()
            b = await await_or_fail(created, what="new_session")
        finally:
            setup_gate.set()
            if not created.done():
                created.cancel()
        assert b != a
        await _closed(server, a)
        assert (await client.call("status"))["parked_sessions"] == []
    finally:
        setup_gate.set()
        backend.open_all()
        await _close(server, client and client.writer)
