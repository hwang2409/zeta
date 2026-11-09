"""child_run_events under pressure: rejoin convergence, byte and record caps, bounds.

Every test crosses the real `zeta serve` unix socket and authenticated websocket.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

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
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent
from zeta.server import ZetaServer
from zeta.server.protocol import EventDescriptor
from zeta.server.server import _Client
from zeta.server.session_replay import (
    REPLAY_SCOPE_MAX_BYTES,
    REPLAY_SCOPE_MAX_RECORDS,
    ReplayScopeId,
)

pytest_plugins = ("tests.support.socket_fixtures",)

TRANSPORTS: tuple[Transport, ...] = ("unix", "websocket")
# feature 1 charges frames only (no scope or pending-join record): 64 fit, 65 is stale
SCOPE_RECORDS = 64
BUSY_CHILDREN = 63


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


@asynccontextmanager
async def _serve(
    tmp_path: Path, transport: Transport
) -> AsyncIterator[tuple[ZetaServer, ReplayClient, str, list[ConversationStore]]]:
    """A real server and client with child_run_events; closes every child store it got."""
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    stores: list[ConversationStore] = []
    try:
        hello = await negotiate(
            client, extra=("child_run_history", "child_run_events"), protocol_version="1.2"
        )
        assert "child_run_events" in hello["result"]["capabilities"]["features"]
        created = await client.request(2, "new_session", {"provider": "codex"})
        yield server, client, created["result"]["session"]["session_id"], stores
    finally:
        for store in stores:
            store.close()
        await client.close()
        await server.close()


def _start_child(
    server: ZetaServer,
    stores: list[ConversationStore],
    run_id: str,
    parent: ConversationStore | None = None,
) -> ConversationStore:
    """Install the durable start row in the immediate parent, then start the lifecycle."""
    root = server.runtime.opened.store
    parent = parent or root
    parent_run_id = run_id.rpartition(":")[0]
    parent.append_child_run_started(
        run_id,
        parent_run_id if parent is not root else None,
        "worker",
        f"task {run_id}",
        "2026-01-01T00:00:00+00:00",
    )
    child = ConversationStore(
        parent.session_dir / "agents",
        session_id=run_id.rpartition(":")[2],
        _storage_lease=root._storage_lease,
    )
    stores.append(child)
    child.start_agent_lifecycle(
        handle=run_id,
        started_at="2026-01-01T00:00:00+00:00",
        depth=run_id.count(":"),
        agent_type="worker",
        description=f"task {run_id}",
    )
    return child


async def _lifecycle_applied(server: ZetaServer) -> None:
    """Wait until every lifecycle callback posted so far ran on the serve loop.

    Sinks post with call_soon_threadsafe in order, so a later loop callback runs after them.
    """
    loop = asyncio.get_running_loop()
    done = loop.create_future()
    loop.call_soon_threadsafe(done.set_result, None)
    await await_or_fail(done, what="lifecycle callbacks applied")


async def _publish_turns(server: ZetaServer, session_id: str, count: int) -> None:
    for index in range(count):
        await await_or_fail(
            server.control._notify(
                EventDescriptor.TURN_START, session_id, data={"turn_id": f"turn-{index}"}
            ),
            what="publish replay event",
        )


def _reduce(rows: dict[str, tuple[int, dict[str, Any]]], event: dict[str, Any]) -> None:
    """The client reducer for one stream: keep the freshest row of each run."""
    current = rows.get(event["run_id"])
    if current is not None and event["revision"] <= current[0]:
        return
    rows[event["run_id"]] = (event["revision"], event["child"])


def _gate_history_read(server: ZetaServer, monkeypatch: pytest.MonkeyPatch) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold the next join's off-loop newest-page read until released."""
    assert server.connections is not None
    entered, release = asyncio.Event(), asyncio.Event()
    original = server.connections.run_read

    async def gated(connection: Any, read: Any, *, session_id: str | None = None) -> Any:
        if not entered.is_set():
            entered.set()
            await await_or_fail(release.wait(), what="release join history read")
        return await original(connection, read, session_id=session_id)

    # test-contract: seam hold the real fixed-worker history read of one join
    monkeypatch.setattr(server.connections, "run_read", gated)
    return entered, release


def _capture_baseline(server: ZetaServer, monkeypatch: pytest.MonkeyPatch) -> list[dict[str, tuple[dict[str, object], int]]]:
    """Record the lifecycle projection each admission's baseline builder read."""
    assert server.connections is not None
    projection = server.connections.child_runs
    original = projection.with_baseline
    captured: list[dict[str, tuple[dict[str, object], int]]] = []

    def capture(result: dict[str, Any]) -> dict[str, Any]:
        captured.append(
            {run_id: state for run_id in list(projection._rows) if (state := projection.current(run_id))}
        )
        return original(result)

    # test-contract: seam observe the real baseline builder at response admission
    monkeypatch.setattr(projection, "with_baseline", capture)
    return captured


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_rejoin_after_stale_converges_while_63_children_stay_busy(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve(tmp_path, transport) as (server, client, session_id, stores):
        assert server.connections is not None
        replay = server.connections.replay
        children = {
            f"{session_id}:{n}": _start_child(server, stores, f"{session_id}:{n}")
            for n in range(1, BUSY_CHILDREN + 1)
        }
        scope = ReplayScopeId.top(session_id)

        # 1. a real stale result: 65 small non-progress events overflow a pending join
        entered, release = _gate_history_read(server, monkeypatch)
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(entered.wait(), what="first join read held")
        await _publish_turns(server, session_id, SCOPE_RECORDS + 1)
        release.set()
        stale = await client.response(3)
        assert stale.get("error") == {
            "code": -32009,
            "message": "replay continuity was invalidated",
            "data": {"code": "stale_event_sequence"},
        }, stale

        # 2-3. a fresh join; every child keeps changing while its read is held
        entered, release = _gate_history_read(server, monkeypatch)
        captured = _capture_baseline(server, monkeypatch)
        await client.send(4, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(entered.wait(), what="rejoin read held")
        for step in ("held-a", "held-b"):
            for child in children.values():
                child.update_agent_lifecycle(current_step=step)
        await _lifecycle_applied(server)
        held = (
            replay.is_pending(server._client),
            len(replay.pending_tail(server._client)),
            scope in replay._scopes,
            replay._next_sequences.get(server._client),
        )
        release.set()
        # producers keep changing after the read returns, before admission
        for child in children.values():
            child.update_agent_lifecycle(current_step="released")
        joined = await client.response(4)

        # 5. the response carries the latest revision of each active run in its page
        assert joined.get("error") is None, "the rejoin did not converge"
        # 4. 126 progress callbacks made no pending record, charge, or seq
        assert held == (True, 0, False, 1)
        stream_id = joined["result"]["replay"]["stream_id"]
        messages = joined["result"]["messages"]
        starts = [row["run_id"] for row in messages if row.get("event") == "child_started"]
        snapshot = [row for row in messages if row.get("event") == "child_progress"]
        assert len(captured) == 1
        admitted = captured[0]
        assert len(snapshot) <= len(starts) <= len(messages)
        assert sorted(row["run_id"] for row in snapshot) == sorted(
            run_id for run_id in starts if run_id in admitted
        )
        rows: dict[str, tuple[int, dict[str, Any]]] = {}
        for row in snapshot:
            assert (row["child"], row["revision"]) == admitted[row["run_id"]]
            assert row["child"]["current_step"] == "released"
            _reduce(rows, row)
        assert len(snapshot) == BUSY_CHILDREN

        # 6. children stay busy: later revisions are live from seq 1, consecutive, fresher
        for child in children.values():
            child.update_agent_lifecycle(current_step="live")
        events = [
            await await_or_fail(client.event("child_progress"), what="live progress")
            for _ in range(BUSY_CHILDREN)
        ]
        assert [event["stream_id"] for event in events] == [stream_id] * BUSY_CHILDREN
        assert [event["seq"] for event in events] == list(range(1, BUSY_CHILDREN + 1))
        for event in events:
            assert event["revision"] > admitted[event["run_id"]][1]
            _reduce(rows, event)
        assert {run_id: row["current_step"] for run_id, (_, row) in rows.items()} == dict.fromkeys(
            children, "live"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_progress_between_snapshot_and_response_write_is_next_live_seq(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve(tmp_path, transport) as (server, client, session_id, stores):
        run_id = f"{session_id}:1"
        child = _start_child(server, stores, run_id)
        captured = _capture_baseline(server, monkeypatch)
        write_started, release_write = asyncio.Event(), asyncio.Event()
        original_write = _Client._write_physical

        async def gated(owner: _Client, payload: bytes) -> bool:
            if payload.startswith(b'{"jsonrpc":"2.0","id":3,') and not write_started.is_set():
                write_started.set()
                await await_or_fail(release_write.wait(), what="release response write")
            return await original_write(owner, payload)

        # test-contract: seam hold the real physical write of the join response
        monkeypatch.setattr(_Client, "_write_physical", gated)
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(write_started.wait(), what="response write started")
        assert len(captured) == 1, "snapshot was not captured before the response write"
        revision = captured[0][run_id][1]
        child.update_agent_lifecycle(current_step="r+1")
        await _lifecycle_applied(server)
        release_write.set()
        response = await client.read()
        assert response.get("id") == 3, f"an event preceded the join response: {response}"
        # the connection serves its next request only after the join call returned
        assert "error" not in await client.request(4, "status", {})
        child.update_agent_lifecycle(current_step="r+2")
        await _lifecycle_applied(server)

        stream_id = response["result"]["replay"]["stream_id"]
        baseline = [row for row in response["result"]["messages"] if row.get("event") == "child_progress"]
        assert [(row["run_id"], row["revision"]) for row in baseline] == [(run_id, revision)]
        rows: dict[str, tuple[int, dict[str, Any]]] = {}
        _reduce(rows, baseline[0])
        first = (await client.read())["params"]
        assert (first["event"], first["stream_id"], first["seq"], first["revision"]) == (
            "child_progress", stream_id, 1, revision + 1,
        ), "progress after the snapshot was not the next live seq"
        second = (await client.read())["params"]
        assert (second["event"], second["stream_id"], second["seq"], second["revision"]) == (
            "child_progress", stream_id, 2, revision + 2,
        )
        _reduce(rows, first)
        _reduce(rows, second)
        assert rows[run_id][1]["current_step"] == "r+2"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_64_small_frames_join_and_65th_is_stale_while_child_scope_stays_gap_free(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert REPLAY_SCOPE_MAX_RECORDS == SCOPE_RECORDS
    async with _serve(tmp_path, transport) as (server, client, session_id, stores):
        parent = _start_child(server, stores, f"{session_id}:1")
        nested = _start_child(server, stores, f"{session_id}:1:1", parent)
        observer = await connect_replay(server, transport)
        try:
            await negotiate(
                observer,
                observer=True,
                extra=("child_run_history", "child_run_events"),
                protocol_version="1.2",
            )
            child_join = await observer.request(
                3, "child_run_events", {"session_id": session_id, "run_id": f"{session_id}:1"}
            )
            child_stream = child_join["result"]["replay"]["stream_id"]
            nested.update_agent_lifecycle(current_step="before")
            before = await await_or_fail(observer.event("child_progress"), what="child stream seq 1")

            # 64 small ordinary frames fit one pending join: response, then seq 1..64
            entered, release = _gate_history_read(server, monkeypatch)
            await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
            await await_or_fail(entered.wait(), what="join read held")
            await _publish_turns(server, session_id, SCOPE_RECORDS)
            release.set()
            first = await client.read()
            assert first.get("id") == 3 and "result" in first, first
            stream_id = first["result"]["replay"]["stream_id"]
            tail = [(await client.read())["params"] for _ in range(SCOPE_RECORDS)]
            assert [(event["event"], event["stream_id"]) for event in tail] == [
                ("turn_start", stream_id)
            ] * SCOPE_RECORDS
            assert [event["seq"] for event in tail] == list(range(1, SCOPE_RECORDS + 1))

            # the 65th invalidates only the growing scope
            entered, release = _gate_history_read(server, monkeypatch)
            await client.send(4, "session_history_and_subscribe", {"session_id": session_id})
            await await_or_fail(entered.wait(), what="rejoin read held")
            await _publish_turns(server, session_id, SCOPE_RECORDS + 1)
            release.set()
            stale = await client.response(4)
            assert stale.get("error", {}).get("data") == {"code": "stale_event_sequence"}, (
                "frame 65 did not invalidate the pending join"
            )
            nested.update_agent_lifecycle(current_step="after")
            after = await await_or_fail(observer.event("child_progress"), what="child stream seq 2")
            assert [(event["stream_id"], event["seq"]) for event in (before, after)] == [
                (child_stream, 1), (child_stream, 2)
            ], "the unrelated child stream lost continuity"
            assert after["child"]["current_step"] == "after"
            assert all(
                frame.get("params", {}).get("event") != "replay_stream_ended"
                for frame in observer.buffered
            )
        finally:
            await observer.close()
