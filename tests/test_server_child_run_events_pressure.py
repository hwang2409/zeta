"""child_run_events under pressure: rejoin convergence, byte and record caps, bounds.

Every test crosses the real `zeta serve` unix socket and authenticated websocket.
"""

from __future__ import annotations

import asyncio
import json
import os
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
from zeta.core import agent_progress
from zeta.core.agent_progress import remove_live_agent_lifecycle
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent, ToolResult
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
            client, extra=("child_run_history", "child_run_events", "child_runs"), protocol_version="1.2"
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
async def test_pending_progress_from_63_busy_children_converges_to_one_baseline(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve(tmp_path, transport) as (server, client, session_id, stores):
        children = [
            _start_child(server, stores, f"{session_id}:{index}")
            for index in range(1, BUSY_CHILDREN + 1)
        ]
        await _lifecycle_applied(server)
        entered, release = _gate_history_read(server, monkeypatch)

        await client.send(
            3, "session_history_and_subscribe", {"session_id": session_id}
        )
        await await_or_fail(entered.wait(), what="join history read held")
        for index, child in enumerate(children, 1):
            child.update_agent_lifecycle(current_step=f"pending-{index}")
        await _lifecycle_applied(server)
        release.set()

        joined = await client.response(3)
        assert "error" not in joined, joined
        baseline = {
            row["run_id"]: row
            for row in joined["result"]["messages"]
            if row.get("event") == "child_progress"
        }
        assert len(baseline) == BUSY_CHILDREN
        selected = children[-1]
        selected_run = selected.agent_handle()
        selected.update_agent_lifecycle(current_step="after-join")
        await _lifecycle_applied(server)

        event = await client.event("child_progress")
        assert (
            event["run_id"],
            event["seq"],
            event["revision"],
            event["child"]["current_step"],
        ) == (
            selected_run,
            1,
            baseline[selected_run]["revision"] + 1,
            "after-join",
        ), "pending progress escaped the admission baseline"


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
                extra=("child_run_history", "child_run_events", "child_runs"),
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


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_byte_pressure_invalidates_before_the_record_cap_at_exact_codec_bytes(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve(tmp_path, transport) as (server, client, session_id, _stores):
        assert server.connections is not None and server._client is not None
        replay = server.connections.replay
        scope = ReplayScopeId.top(session_id)
        codec = server._client.codec
        encoded: list[int] = []
        original = codec.notification

        def record(event: EventDescriptor, *args: Any, **fields: Any) -> Any:
            frame = original(event, *args, **fields)
            if event is EventDescriptor.TURN_START and "stream_id" in fields:
                encoded.append(len(frame))
            return frame

        # test-contract: seam observe the complete frames the real codec returned
        monkeypatch.setattr(codec, "notification", record)
        entered, release = _gate_history_read(server, monkeypatch)
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(entered.wait(), what="join read held")
        large = "x" * 200_000
        charged: list[int] = []
        while scope in replay._scopes or not charged:
            await await_or_fail(
                server.control._notify(
                    EventDescriptor.TURN_START, session_id, data={"turn_id": large}
                ),
                what="publish large frame",
            )
            if scope in replay._scopes:
                charged.append(replay._scopes[scope].bytes)
        admitted = len(charged)
        release.set()
        stale = await client.response(3)
        assert stale.get("error", {}).get("data") == {"code": "stale_event_sequence"}, (
            "byte pressure did not invalidate the pending join"
        )
        assert [frame for frame in client.buffered if "seq" in frame.get("params", {})] == []
        # every admitted frame was charged its complete codec bytes, below the cap
        assert charged == [sum(encoded[: n + 1]) for n in range(admitted)]
        assert charged[-1] <= REPLAY_SCOPE_MAX_BYTES < charged[-1] + encoded[admitted]
        # bytes bound first: far below the record cap, and nothing kept a charge or seq
        assert len(encoded) == admitted + 1 < SCOPE_RECORDS
        assert scope not in replay._scopes
        assert not replay.is_pending(server._client)
        assert replay.stream_id(server._client) is None


MAX_LIFECYCLE_FRAME_BYTES = 16_000


def _maximal_lifecycle(run_id: str) -> dict[str, object]:
    """A terminal lifecycle whose every field reaches the row's exported bound."""
    return {
        "handle": run_id,
        "description": "d" * 9_000,
        "agent_type": "a" * 900,
        "state": "s" * 900,
        "current_step": "c" * 9_000,
        "started_at": "t" * 900,
        "finished_at": "f" * 900,
        "turns_used": 2**63 - 1,
        "tool_calls": 2**63 - 1,
        "elapsed": 999_999_999_999.999,
    }


def _wire_size(frame: dict[str, Any]) -> int:
    """The exact codec bytes of a frame the client read: compact UTF-8 JSON plus newline."""
    return len((json.dumps(frame, separators=(",", ":"), ensure_ascii=False) + "\n").encode())


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_64_maximal_lifecycle_frames_fit_the_scope_byte_cap(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve(tmp_path, transport) as (server, client, session_id, stores):
        assert server.connections is not None
        replay = server.connections.replay
        _start_child(server, stores, f"{session_id}:1")
        scope = ReplayScopeId.child(f"{session_id}:1")
        joined = await client.request(
            3, "child_run_events", {"session_id": session_id, "run_id": f"{session_id}:1"}
        )
        stream_id = joined["result"]["replay"]["stream_id"]
        control = replay._scopes[scope].bytes if scope in replay._scopes else 0
        held, release = asyncio.Event(), asyncio.Event()
        original_write = _Client._write_physical

        async def gated(owner: _Client, payload: bytes) -> bool:
            if b'"seq":1,' in payload and not held.is_set():
                held.set()
                await await_or_fail(release.wait(), what="release first lifecycle frame")
            return await original_write(owner, payload)

        # test-contract: seam keep every admitted frame charged until all are admitted
        monkeypatch.setattr(_Client, "_write_physical", gated)
        prefix = f"{session_id}:1:"
        for index in range(SCOPE_RECORDS):
            run_id = prefix + "9" * (256 - len(prefix) - 2) + f"{index + 10}"
            assert len(run_id.encode()) == 256
            agent_progress.publish_child_ended(
                run_id, 2**62 + index, _maximal_lifecycle(run_id)
            )
        await _lifecycle_applied(server)
        await await_or_fail(held.wait(), what="first lifecycle frame write")
        charged = replay._scopes[scope]
        totals = (charged.records, charged.bytes)
        release.set()
        frames = [await client.read() for _ in range(SCOPE_RECORDS)]
        events = [frame["params"] for frame in frames]
        assert [(event["event"], event["stream_id"], event["seq"]) for event in events] == [
            ("child_ended", stream_id, seq) for seq in range(1, SCOPE_RECORDS + 1)
        ]
        sizes = [_wire_size(frame) for frame in frames]
        assert totals == (SCOPE_RECORDS, sum(sizes)), "charges are not the complete frame bytes"
        assert max(sizes) <= MAX_LIFECYCLE_FRAME_BYTES
        assert control + sum(sizes) <= REPLAY_SCOPE_MAX_BYTES
        # the fixture reaches the row bound: every bounded text is at its limit
        row = events[0]["child"]
        assert len(json.dumps(row["description"], ensure_ascii=False).encode()) == 4_000
        assert len(json.dumps(row["current_step"], ensure_ascii=False).encode()) == 4_000


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
@pytest.mark.parametrize(
    ("charges", "victim"),
    (
        pytest.param(((500, 500), (500, 500)), 0, id="equal-earliest-created"),
        pytest.param(((1000,), (500, 500)), 1, id="equal-bytes-more-records"),
        pytest.param(((500, 500), (1001,)), 1, id="more-bytes-fewer-records"),
    ),
)
async def test_global_pressure_evicts_by_bytes_then_records_then_creation_order(
    tmp_path: Path,
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    charges: tuple[tuple[int, ...], tuple[int, ...]],
    victim: int,
) -> None:
    async with _serve(tmp_path, transport) as (server, client, session_id, stores):
        assert server.connections is not None
        replay = server.connections.replay
        runs = [f"{session_id}:1", f"{session_id}:2"]
        for run_id in runs:
            _start_child(server, stores, run_id)
        await _lifecycle_applied(server)
        joined = await client.request(3, "session_history_and_subscribe", {"session_id": session_id})
        stream_id = joined["result"]["replay"]["stream_id"]
        arrivals = [asyncio.Event() for _ in runs]
        release = asyncio.Event()
        original = server.connections.run_read

        async def gated(connection: Any, read: Any, *, session_id: str | None = None) -> Any:
            next(event for event in arrivals if not event.is_set()).set()
            await await_or_fail(release.wait(), what="release child join reads")
            return await original(connection, read, session_id=session_id)

        # test-contract: seam hold both real child join reads while pressure is applied
        monkeypatch.setattr(server.connections, "run_read", gated)
        observers = [await connect_replay(server, transport) for _ in runs]
        try:
            for observer, run_id in zip(observers, runs, strict=True):
                await negotiate(
                    observer,
                    observer=True,
                    extra=("child_run_history", "child_run_events", "child_runs"),
                    protocol_version="1.2",
                )
                await observer.send(3, "child_run_events", {"session_id": session_id, "run_id": run_id})
                await await_or_fail(
                    arrivals[runs.index(run_id)].wait(), what="child join read held"
                )
            # creation order is first charge order: run 1's scope, then run 2's
            for run_id, sizes in zip(runs, charges, strict=True):
                for size in sizes:
                    assert replay.reserve(ReplayScopeId.child(run_id), size) is None
            monkeypatch.setattr(
                "zeta.server.session_replay.REPLAY_MAX_RECORDS", sum(map(len, charges))
            )
            await _publish_turns(server, session_id, 1)
            release.set()
            replies = [await observer.response(3) for observer in observers]
            codes = [reply.get("error", {}).get("data", {}).get("code") for reply in replies]
            expected = [None, None]
            expected[victim] = "stale_event_sequence"
            assert codes == expected, f"wrong global victim: {codes}"
            event = await await_or_fail(client.event("turn_start"), what="candidate frame")
            assert (event["stream_id"], event["seq"]) == (stream_id, 1)
        finally:
            release.set()
            for observer in observers:
                await observer.close()



def _push_older(store: ConversationStore) -> None:
    """Append enough history after earlier rows to move them to an older page.

    History bounds each assistant row near 8 KB, so 140 rows exceed one 1 MiB page.
    """
    for _ in range(140):
        store.append_message(Message(MessageRole.ASSISTANT, [TextContent("x" * 10_000)]))


def _rows(messages: list[dict[str, Any]], event: str) -> list[str]:
    return [row["run_id"] for row in messages if row.get("event") == event]


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_snapshot_rows_come_only_from_newest_page_starts_without_scans(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> None:
    async with _serve(tmp_path, transport) as (server, client, session_id, stores):
        assert server.connections is not None
        root = server.runtime.opened.store
        older = [f"{session_id}:{n}" for n in (1, 2, 3)]
        newer = [f"{session_id}:{n}" for n in (4, 5, 6)]
        for run_id in older:
            _start_child(server, stores, run_id)
        _push_older(root)
        children = {run_id: _start_child(server, stores, run_id) for run_id in newer}
        parent = children[newer[0]]
        nested_old = _start_child(server, stores, f"{newer[0]}:1", parent)
        _push_older(parent)
        nested_new = _start_child(server, stores, f"{newer[0]}:2", parent)
        nested_old.update_agent_lifecycle(current_step="older start, still running")
        await _lifecycle_applied(server)

        # count the projection reads and directory walks of each admission
        projection = server.connections.child_runs
        counts = {"current": 0, "walks": 0}
        admitting = False
        original_current = projection.current
        original_baseline = projection.with_baseline

        def current(run_id: str) -> Any:
            counts["current"] += admitting
            return original_current(run_id)

        def with_baseline(result: dict[str, Any]) -> dict[str, Any]:
            nonlocal admitting
            admitting = True
            try:
                return original_baseline(result)
            finally:
                admitting = False

        def counted(original: Any) -> Any:
            def walk(*args: Any, **kwargs: Any) -> Any:
                counts["walks"] += admitting
                return original(*args, **kwargs)

            return walk

        # test-contract: seam count the baseline builder's lifecycle reads and directory walks
        monkeypatch.setattr(projection, "current", current)
        monkeypatch.setattr(projection, "with_baseline", with_baseline)
        monkeypatch.setattr(Path, "iterdir", counted(Path.iterdir))
        monkeypatch.setattr(Path, "glob", counted(Path.glob))
        monkeypatch.setattr(os, "scandir", counted(os.scandir))
        monkeypatch.setattr(os, "listdir", counted(os.listdir))

        # bullet 1 and 4: the top join snapshots only the newest page's starts
        joined = await client.request(3, "session_history_and_subscribe", {"session_id": session_id})
        messages = joined["result"]["messages"]
        assert joined["result"]["next_before"] is not None, "older starts were not paged out"
        starts = _rows(messages, "child_started")
        assert starts == newer
        assert _rows(messages, "child_progress") == newer
        assert counts == {"current": len(starts), "walks": 0}

        # bullet 2: an older start is found by paging, its current row by a one-run join
        counts.update(current=0, walks=0)
        page = await client.request(
            4, "child_run_events", {"session_id": session_id, "run_id": newer[0]}
        )
        assert _rows(page["result"]["messages"], "child_started") == [nested_new.agent_handle()]
        assert _rows(page["result"]["messages"], "child_progress") == [nested_new.agent_handle()]
        assert counts["walks"] == 0 and counts["current"] == 1
        history = await client.request(
            5,
            "child_run_history",
            {
                "session_id": session_id,
                "run_id": newer[0],
                "before": page["result"]["next_before"],
                "history_validator": page["result"]["history_validator"],
            },
        )
        older_page = history["result"]["messages"]
        assert _rows(older_page, "child_started") == [nested_old.agent_handle()]
        assert _rows(older_page, "child_progress") == []
        found = await client.request(
            6, "child_run_events", {"session_id": session_id, "run_id": nested_old.agent_handle()}
        )
        assert "error" not in found, found
        row, revision = original_current(nested_old.agent_handle())
        assert (found["result"]["child"], found["result"]["revision"]) == (row, revision)
        assert found["result"]["child"]["current_step"] == "older start, still running"


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", TRANSPORTS)
async def test_restart_registers_the_projection_before_children_and_reads_durable_lifecycle_rows(
    tmp_path: Path, transport: Transport
) -> None:
    # before the restart: run 1 ends with a durable receipt, run 2 is running
    async with _serve(tmp_path, transport) as (first, _client, session_id, stores):
        assert first.connections is not None
        first_sink = first.connections.child_runs.sink
        root = first.runtime.opened.store
        ended = _start_child(first, stores, f"{session_id}:1")
        running = _start_child(first, stores, f"{session_id}:2")
        running.update_agent_lifecycle(current_step="before restart")
        ended.finish_agent_lifecycle("completed", final_result="done")
        root.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [TextContent("done")],
                tool_result=ToolResult(
                    "call-1",
                    "done",
                    structured_content={"child_instance_id": f"{session_id}:1", "status": "completed"},
                ),
            )
        )
        await _lifecycle_applied(first)
    # the process that ran run 2 is gone: its live lifecycle goes with it
    remove_live_agent_lifecycle(running.agent_lifecycle_path)
    assert first_sink not in agent_progress._sinks

    second = replay_server(tmp_path, transport, _backend_factory)
    await second.start()
    restarted: list[ConversationStore] = []
    try:
        assert second.connections is not None
        at_start = (
            second.runtime.opened is None,
            second.connections.child_runs.sink in agent_progress._sinks,
        )
        client = await connect_replay(second, transport)
        try:
            await negotiate(
                client, extra=("child_run_history", "child_run_events", "child_runs"), protocol_version="1.2"
            )
            resumed = await client.request(2, "resume", {"session_id": session_id})
            assert "error" not in resumed, resumed
            fresh = _start_child(second, restarted, f"{session_id}:3")
            fresh.update_agent_lifecycle(current_step="after restart")
            await _lifecycle_applied(second)
            joined = await client.request(3, "session_history_and_subscribe", {"session_id": session_id})
            messages = joined["result"]["messages"]
            # start and end come from durable rows
            assert _rows(messages, "child_started") == [f"{session_id}:{n}" for n in (1, 2, 3)]
            ends = [row for row in messages if row.get("event") == "child_ended"]
            assert [(row["run_id"], row["child"]["status"]) for row in ends] == [
                (f"{session_id}:1", "completed")
            ]
            assert "finished_at" in ends[0]["child"]
            # current progress comes only from this process's projection
            progress = [row for row in messages if row.get("event") == "child_progress"]
            assert [item["run_id"] for item in progress] == [f"{session_id}:3"], (
                "the restarted server's projection missed a child started after restart"
            )
            row, revision = second.connections.child_runs.current(f"{session_id}:3")
            assert [(item["run_id"], item["revision"], item["child"]) for item in progress] == [
                (f"{session_id}:3", revision, row)
            ]
            assert row["current_step"] == "after restart"
            # the sink was registered at start, before a session existed, so before any child
            assert at_start == (True, True)
        finally:
            await client.close()
    finally:
        for store in restarted:
            store.close()
        await second.close()
