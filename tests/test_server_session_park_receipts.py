"""`session_park`: a task-exit receipt keeps a parked pair resident only while the
exit is still being delivered."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.test_server_in_flight_replay import GatedBackend, _server
from tests.test_server_session_park import FEATURES, _closed, _open, _until
from tests.test_server_turn_reconnect import _close
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent


def _task_backend(go: Path, *, wake: bool) -> GatedBackend:
    command = f"while [ ! -e {go} ]; do sleep 0.05; done; echo done"
    start = ToolUseContent(ToolCall("bg-1", "run_background", {"command": command}))
    replies = [
        ([TextContent("starting")], [TextContent("starting"), start]),
        ([TextContent("started")], [TextContent("started")]),
    ]
    if wake:
        replies.append(([TextContent("woke")], [TextContent("woke")]))
    return GatedBackend(replies)


@pytest.mark.asyncio
async def test_resumed_session_with_a_stale_exit_receipt_closes_when_left(
    tmp_path: Path,
) -> None:
    go = tmp_path / "go"
    backend = _task_backend(go, wake=True)
    server = _server(tmp_path, backend)
    client = None
    try:
        client = await _open(server, FEATURES)
        a = await client.new()
        session_dir = server.runtime.opened.store.session_dir
        await client.call("send", {"text": "start a task"})
        await _until(
            lambda: backend.calls == 2 and not server.control.state_change_busy(),
            "A's turn ends while its task runs",
        )
        go.touch()
        await _until(
            lambda: backend.calls == 3 and not server.control.holds_work(),
            "the task exit is delivered and its wake turn ends",
        )
        await client.new()
        await _closed(server, a)

        # a crash between the durable exit notice and the receipt commit leaves this row
        path = session_dir / "background_tasks.json"
        rows = json.loads(path.read_text(encoding="utf-8"))
        assert [row["receipt_pending"] for row in rows] == [False]
        rows[0]["receipt_pending"] = True
        path.write_text(json.dumps(rows), encoding="utf-8")

        await client.call("resume", {"session_id": a})
        await client.new()
        status = await client.call("status")
        assert status["parked_sessions"] == [], "a delivered exit must not park A"
        await _closed(server, a)
    finally:
        go.touch()
        backend.open_all()
        await _close(server, client and client.writer)


@pytest.mark.asyncio
async def test_parked_session_closes_when_its_exit_receipt_clears_without_a_wake(
    tmp_path: Path,
) -> None:
    go = tmp_path / "go"
    backend = _task_backend(go, wake=False)
    server = _server(tmp_path, backend)
    client = None
    try:
        client = await _open(server, FEATURES)
        a = await client.new()
        await client.call("send", {"text": "start a task"})
        await _until(
            lambda: backend.calls == 2 and not server.control.state_change_busy(),
            "A's turn ends while its task runs",
        )
        await client.new()
        parked = server.parking._parked[a]
        # m-b: the client requires host commands but none are bound, so no wake can start
        parked._host_binding_required = True
        parked._active_host_commands = None

        go.touch()
        await _closed(server, a)
        assert backend.calls == 2  # no wake turn ran; the receipt clear closed A
    finally:
        go.touch()
        backend.open_all()
        await _close(server, client and client.writer)
