"""`session_park` table test: one row per way a parked pair could miss its idle
signal or stay pinned forever (success, cancel at each await, failure at each
publish step, restart, duplicate input), each through the real serve socket.

Every row ends with the same check: no parked pair that holds no work survives."""
from __future__ import annotations

import asyncio
import itertools
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.support.branch_reads import resident_entries
from tests.support.owned import await_or_fail
from tests.test_server_in_flight_replay import HOLD, GatedBackend, _server
from tests.test_server_session_park import (
    FEATURES,
    Client,
    _closed,
    _open,
    _parked_changes,
    _parked_turn,
    _until,
)
from tests.test_server_turn_reconnect import _close
from zeta.core.store import ConversationStore
from zeta.core.store._validation import TASK_EXITED_NOTIFICATION_KIND
from zeta.protocol.types import TextContent, ToolCall, ToolUseContent
from zeta.server import ZetaServer
from zeta.server.control import ServeControl

GRANTS = [*FEATURES, "session_grants"]


@dataclass
class Run:
    """One row's server, backend, and clients; the test closes them."""

    tmp_path: Path
    monkeypatch: pytest.MonkeyPatch
    caplog: pytest.LogCaptureFixture
    server: ZetaServer | None = None
    backend: GatedBackend | None = None
    clients: list[Client] = field(default_factory=list)

    @property
    def go(self) -> Path:
        return self.tmp_path / "go"

    def serve(self, replies: list[tuple[list[object], list[Any]]]) -> ZetaServer:
        self.backend = GatedBackend(replies)
        self.server = _server(self.tmp_path, self.backend)
        return self.server

    async def open(self, features: list[str] = FEATURES) -> Client:
        assert self.server is not None
        client = await _open(self.server, features)
        self.clients.append(client)
        return client

    async def parked_turn(self, features: list[str] = FEATURES):
        """A parked mid-reply: its turn waits at gate 0."""
        assert self.server is not None and self.backend is not None
        client, a, parked = await _parked_turn(self.server, self.backend, features)
        self.clients.append(client)
        return client, a, parked

    async def parked_task(self) -> tuple[Client, str, ServeControl]:
        """A parked after its turn ended with a background task still running. The
        client requires host commands but binds none, so no wake can start (m-b):
        only the exit delivery holds the pair."""
        server = self.serve(_task_replies(self.go))
        client = await self.open()
        a = await client.new()
        await client.call("send", {"text": "start a task"})
        await _until(
            lambda: self.backend.calls == 2 and not server.controllers.attached().control.state_change_busy(),
            "A's turn ends while its task runs",
        )
        await client.new()
        parked = _parked(server, a)
        parked._host_binding_required = True
        parked._active_host_commands = None
        return client, a, parked


def _task_replies(go: Path) -> list[tuple[list[object], list[Any]]]:
    command = f"while [ ! -e {go} ]; do sleep 0.05; done; echo done"
    start = ToolUseContent(ToolCall("bg-1", "run_background", {"command": command}))
    return [
        ([TextContent("starting")], [TextContent("starting"), start]),
        ([TextContent("started")], [TextContent("started")]),
        ([TextContent("woke")], [TextContent("woke")]),
    ]


def _held_reply() -> list[tuple[list[object], list[Any]]]:
    return [([TextContent("held"), HOLD], [TextContent("held")])]


def _task_exits(server: ZetaServer) -> int:
    assert server.controllers.attached().runtime.opened is not None
    return sum(
        entry.type == "notification"
        and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND
        for entry in resident_entries(server.controllers.attached().runtime.opened.store)
    )


def _parked(server: ZetaServer, session_id: str) -> ServeControl:
    """A's parked pair, read through ``controls()`` (also on the pre-4b base)."""
    (pair,) = [
        control
        for control in server.controls()
        if control is not server.controllers.attached().control and control.runtime.session_id == session_id
    ]
    return pair


def _registry(control: ServeControl):
    assert control.runtime.loop is not None
    return control.runtime.loop.tool_registry.background_tasks


def _connection_task() -> asyncio.Task[Any]:
    """The serve task of the only connected client; its requests run inside it."""
    tasks = [
        task
        for task in asyncio.all_tasks()
        if getattr(task.get_coro(), "__qualname__", "") == "ZetaServer._accept"
    ]
    assert len(tasks) == 1, tasks
    return tasks[0]


async def _cancel(task: asyncio.Task[Any]) -> None:
    task.cancel()
    await await_or_fail(asyncio.wait([task]), what="cancelled request")


def _grant_resets(frames: list[dict[str, Any]], session_id: str) -> int:
    return sum(
        frame.get("params", {}).get("event") == "session_grants_changed"
        and frame["params"].get("session_id") == session_id
        and frame["params"]["data"]["change"] == "reset"
        for frame in frames
    )


# --- success ---------------------------------------------------------------


async def s1_turn_ends(run: Run) -> None:
    run.serve(_held_reply())
    client, a, _ = await run.parked_turn()
    run.backend.gates[0].set()
    changes = await await_or_fail(_parked_changes(client, "closed"), what="A closed")
    assert {change["session_id"] for change in changes} == {a}
    *_, idle, closed = [change["data"] for change in changes]
    assert idle["session_id"] == a
    assert idle["state"] == "idle"
    assert idle["pending_approvals"] == 0
    assert idle["last_turn_outcome"]["reason"] == "completed"
    assert closed == {**idle, "state": "closed"}


async def s2_task_and_wake(run: Run) -> None:
    server = run.serve(_task_replies(run.go))
    client = await run.open()
    a = await client.new()
    await client.call("send", {"text": "start a task"})
    await _until(
        lambda: run.backend.calls == 2 and not server.controllers.attached().control.state_change_busy(),
        "A's turn ends while its task runs",
    )
    await client.new()
    status = await client.call("status")
    assert [row["session_id"] for row in status["parked_sessions"]] == [a]

    run.go.touch()
    await _closed(server, a)
    assert run.backend.calls == 3  # the wake turn ran before the close

    await client.call("resume", {"session_id": a})
    joined = await client.call("session_history_and_subscribe", {"session_id": a})
    replies = [
        block["text"]
        for row in joined["messages"]
        if row.get("role") == "assistant"
        for block in row["content"]
        if block.get("type") == "text"
    ]
    assert replies[-1] == "woke"


async def s3_exit_without_wake(run: Run) -> None:
    _client, a, _ = await run.parked_task()
    run.go.touch()
    await _closed(run.server, a)
    assert run.backend.calls == 2  # no wake turn ran; the settled delivery closed A


# --- failure at each publish step --------------------------------------------


async def _failed_delivery_closes_and_reopen_recovers(
    run: Run, fail: Callable[[ServeControl, asyncio.Event], None]
) -> None:
    client, a, parked = await run.parked_task()
    hit = asyncio.Event()
    fail(parked, hit)
    run.go.touch()
    await await_or_fail(hit.wait(), what="the injected delivery failure")
    # the receipt stays pending on disk, so closing is the way to the retry
    await _closed(run.server, a)
    await client.call("resume", {"session_id": a})
    await _until(lambda: _task_exits(run.server) == 1, "the exit is in A's history")
    assert _task_exits(run.server) == 1


async def x1_exit_sink_fails(run: Run) -> None:
    def fail(parked: ServeControl, hit: asyncio.Event) -> None:
        async def sink(_item: object) -> None:
            hit.set()
            raise OSError("session store write failed")

        # test-contract: seam the exit sink is the delivery step a store write failure breaks
        run.monkeypatch.setattr(_registry(parked), "_notification_sink", sink)

    await _failed_delivery_closes_and_reopen_recovers(run, fail)


async def x2_receipt_commit_fails(run: Run) -> None:
    def fail(parked: ServeControl, hit: asyncio.Event) -> None:
        registry = _registry(parked)
        original = registry._commit_notice

        async def commit(notice: object) -> None:
            if notice is None and not hit.is_set():
                hit.set()
                raise OSError("background task rows write failed")
            await original(notice)

        # test-contract: seam the receipt commit is the publish step a disk failure breaks
        run.monkeypatch.setattr(registry, "_commit_notice", commit)

    # the exit was appended before the failed commit: the reopen must not add one
    await _failed_delivery_closes_and_reopen_recovers(run, fail)


async def x3_close_fails(run: Run) -> None:
    server = run.serve(_held_reply())
    client, a, parked = await run.parked_turn()
    original = parked.close

    async def close() -> None:
        await original()
        raise RuntimeError("injected close failure")

    parked.close = close
    run.caplog.set_level(logging.ERROR, logger="zeta.server.session_controllers")
    run.backend.gates[0].set()
    await _until(lambda: server.parking.get(a) is None, "A leaves the parked set")

    def logged() -> list[tuple[str, type[BaseException]]]:
        return [
            (record.getMessage(), type(record.exc_info[1]))
            for record in run.caplog.records
            if record.name == "zeta.server.session_controllers"
        ]

    await _until(logged, "the failed close is logged")
    assert logged() == [(f"closing session {a} failed", RuntimeError)]
    assert server.controllers.closing_session_ids() == []
    result = await client.call("resume", {"session_id": a})
    assert result["session"]["session_id"] == a
    assert "attached" not in result


async def x4_setup_fails_before_commit(run: Run) -> None:
    server = run.serve(_held_reply())
    client = await run.open()
    a = await client.new()
    await client.call("send", {"text": "go"})
    await await_or_fail(run.backend.held[0].wait(), what="A mid-reply")
    pairs: list[ServeControl] = []
    original_new_pair = server.new_pair

    def new_pair() -> ServeControl:
        pair = original_new_pair()

        async def create_session(**_kwargs: object) -> None:
            raise OSError("injected setup failure")

        pair.runtime.create_session = create_session
        pairs.append(pair)
        return pair

    run.monkeypatch.setattr(server, "new_pair", new_pair)
    await client.error("new_session", {"provider": "codex"})
    assert server.controllers.attached().control.runtime.session_id == a
    assert [pair.runtime.opened for pair in pairs] == [None]
    assert (await client.call("status"))["parked_sessions"] == []
    run.backend.gates[0].set()


# --- cancel at each await ------------------------------------------------------


async def c1_cancel_incoming_setup(run: Run) -> None:
    server = run.serve(_held_reply())
    client = await run.open()
    await client.new()
    await client.call("send", {"text": "go"})
    await await_or_fail(run.backend.held[0].wait(), what="A mid-reply")
    outgoing = server.controllers.attached().control
    started = asyncio.Event()
    gate = asyncio.Event()
    pairs: list[ServeControl] = []
    original_new_pair = server.new_pair

    def new_pair() -> ServeControl:
        pair = original_new_pair()
        create = pair.runtime.create_session

        async def held_create(**kwargs: object) -> None:
            started.set()
            await await_or_fail(gate.wait(), what="held setup")
            await create(**kwargs)

        pair.runtime.create_session = held_create
        pairs.append(pair)
        return pair

    run.monkeypatch.setattr(server, "new_pair", new_pair)
    created = asyncio.create_task(client.new())
    try:
        await await_or_fail(started.wait(), what="incoming setup")
        await _cancel(_connection_task())
    finally:
        gate.set()
        created.cancel()
        await asyncio.wait([created], timeout=10)
    assert [pair.runtime.opened for pair in pairs] == [None]
    assert list(server.controls()) == [outgoing]
    assert outgoing.state_change_busy()
    run.backend.gates[0].set()


async def c2_cancel_resume_waiting_for_close(run: Run) -> None:
    server = run.serve(_held_reply())
    client, a, parked = await run.parked_turn()
    close_started = asyncio.Event()
    close_gate = asyncio.Event()
    original_close = parked.runtime.close

    async def held_close() -> None:
        close_started.set()
        await await_or_fail(close_gate.wait(), what="held close")
        await original_close()

    parked.runtime.close = held_close
    waiting = asyncio.Event()
    original_wait = server.controllers.wait_closed

    async def wait_closed(session_id: str) -> None:
        if session_id in server.controllers.closing_session_ids():
            waiting.set()
        await original_wait(session_id)

    run.monkeypatch.setattr(server.controllers, "wait_closed", wait_closed)
    run.backend.gates[0].set()
    await await_or_fail(close_started.wait(), what="idle close started")
    (closing,) = [record.close_task for record in server.controllers.records() if record.closing]
    resume = asyncio.create_task(client.call("resume", {"session_id": a}))
    try:
        await await_or_fail(waiting.wait(), what="resume waits for the close")
        await _cancel(_connection_task())
    finally:
        close_gate.set()
        resume.cancel()
        await asyncio.wait([resume], timeout=10)
    # the cancelled request did not cancel the close
    await await_or_fail(asyncio.wait([closing]), what="A's close")
    assert not closing.cancelled() and closing.exception() is None
    client2 = await run.open()
    result = await client2.call("resume", {"session_id": a})
    assert result["session"]["session_id"] == a
    assert "attached" not in result


async def c3_cancel_exit_delivery(run: Run) -> None:
    def fail(parked: ServeControl, hit: asyncio.Event) -> None:
        async def sink(_item: object) -> None:
            hit.set()
            raise asyncio.CancelledError

        # test-contract: seam the exit sink is the delivery step a cancel interrupts
        run.monkeypatch.setattr(_registry(parked), "_notification_sink", sink)

    await _failed_delivery_closes_and_reopen_recovers(run, fail)


# --- restart -------------------------------------------------------------------


async def r1_stale_receipt_at_load(run: Run) -> None:
    server = run.serve(_task_replies(run.go))
    client = await run.open()
    a = await client.new()
    session_dir = server.controllers.attached().runtime.opened.store.session_dir
    await client.call("send", {"text": "start a task"})
    await _until(
        lambda: run.backend.calls == 2 and not server.controllers.attached().control.state_change_busy(),
        "A's turn ends while its task runs",
    )
    run.go.touch()
    await _until(
        lambda: run.backend.calls == 3 and not server.controllers.attached().control.holds_work(),
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


async def _unhealthy(run: Run, client: Client) -> None:
    """Fail the pending approval's durable write through the wire."""
    control = run.server.controllers.attached().control
    await _until(control.has_pending_approvals, "A's approval")
    request_id = (await client.call("status"))["pending_approvals"][0]["request_id"]
    with run.monkeypatch.context() as patch:
        patch.setattr(
            ConversationStore,
            "resolve_approval",
            lambda *args, **kwargs: (_ for _ in ()).throw(OSError("disk failed")),
        )
        error = await client.error("approve", {"request_id": request_id})
    assert error["code"] == -32000
    assert control.unhealthy()
    await _until(lambda: not control.state_change_busy(), "A's failed turn ends")


async def u1_unhealthy_alone_closes(run: Run) -> None:
    """An unhealthy pair whose failed turn ended holds no work: the failed
    approval leaves no record, so leaving it closes it instead of parking it."""
    target = run.tmp_path / "out.txt"
    write = ToolUseContent(ToolCall("w-1", "write", {"path": str(target), "content": "x"}))
    server = run.serve([([TextContent("writing")], [TextContent("writing"), write])])
    client = await run.open()
    a = await client.new(mode="ask")
    await client.call("send", {"text": "write"})
    await _unhealthy(run, client)
    assert not server.controllers.attached().control.holds_work()
    await client.new()
    assert (await client.call("status"))["parked_sessions"] == []
    await _closed(server, a)


async def r3_unhealthy_parked_resume(run: Run) -> None:
    command = f"while [ ! -e {run.go} ]; do sleep 0.05; done"
    start = ToolUseContent(ToolCall("bg-1", "run_background", {"command": command}))
    target = run.tmp_path / "out.txt"
    write = ToolUseContent(ToolCall("w-1", "write", {"path": str(target), "content": "x"}))
    server = run.serve([
        ([TextContent("starting")], [TextContent("starting"), start]),
        ([TextContent("writing")], [TextContent("writing"), write]),
        *[([TextContent("after")], [TextContent("after")])] * 4,
    ])
    client = await run.open()
    a = await client.new(mode="ask")
    await client.call("send", {"text": "start a task, then write"})
    await _until(server.controllers.attached().control.has_pending_approvals, "the task approval")
    first = (await client.call("status"))["pending_approvals"][0]["request_id"]
    await client.call("approve", {"request_id": first})
    await _unhealthy(run, client)
    parked = server.controllers.attached().control
    assert parked.holds_work()  # its task still runs

    await client.new()
    assert _parked(server, a) is parked
    result = await client.call("resume", {"session_id": a})
    assert "attached" not in result
    assert server.controllers.attached().control is not parked, "resume must reopen an unhealthy parked session"
    assert not server.controllers.attached().control.unhealthy()
    assert parked.runtime.opened is None


# --- duplicate input -------------------------------------------------------------


async def g1_parked_resume_resets_grants(run: Run) -> None:
    run.serve(_held_reply())
    client, a, _ = await run.parked_turn(GRANTS)
    frames = await client.frames("resume", {"session_id": a})
    assert frames[-1]["result"]["attached"] == "live_turn"
    assert _grant_resets(frames, a) == 1, "resume of a parked session resets grants"
    run.backend.gates[0].set()


async def d1_dropped_frames_same_row(run: Run) -> None:
    run.serve([(
        [TextContent("a"), HOLD, TextContent("b"), TextContent("c")],
        [TextContent("abc")],
    )])
    client, _a, _ = await run.parked_turn()
    run.backend.gates[0].set()
    changes = await await_or_fail(_parked_changes(client, "closed"), what="A closed")
    datas = [change["data"] for change in changes]
    # the dropped b and c deltas re-check an unchanged row: one event per real change
    assert all(prev != nxt for prev, nxt in itertools.pairwise(datas)), datas


async def d2_stale_check_after_resume(run: Run) -> None:
    """Forced order: a parked pair's idle check is scheduled, then `resume`
    attaches the pair before the check runs; the late check must not close it."""
    server = run.serve(_held_reply())
    client, a, parked = await run.parked_turn()
    late_checks: list[ServeControl] = []
    original_check = server.parking._check
    # test-contract: seam holds the scheduled idle check to force the resume-first order
    run.monkeypatch.setattr(server.parking, "_check", late_checks.append)
    run.backend.gates[0].set()
    # dropped frames of the parked turn schedule checks too; wait for an idle one
    await _until(
        lambda: parked in late_checks and not parked.holds_work(),
        "A's idle check scheduled",
    )
    resumed = await client.call("resume", {"session_id": a})
    assert "attached" not in resumed
    for control in late_checks:
        original_check(control)
    assert server.controllers.attached().control is parked
    assert a not in server.controllers.closing_session_ids()
    assert (await client.call("status"))["session"]["session_id"] == a


async def d3_resume_parked_twice(run: Run) -> None:
    server = run.serve(_held_reply())
    client, a, parked = await run.parked_turn(GRANTS)
    first = await client.frames("resume", {"session_id": a})
    second = await client.frames("resume", {"session_id": a})
    assert [frames[-1]["result"]["attached"] for frames in (first, second)] == [
        "live_turn",
        "live_turn",
    ]
    # the second resume finds A attached: no new pair, no parked_limit, no reset
    assert [_grant_resets(frames, a) for frames in (first, second)] == [1, 0]
    assert server.controllers.attached().control is parked
    assert (await client.call("status"))["parked_sessions"] == []
    run.backend.gates[0].set()


ROWS: list[Callable[[Run], Awaitable[None]]] = [
    s1_turn_ends,
    s2_task_and_wake,
    s3_exit_without_wake,
    x1_exit_sink_fails,
    x2_receipt_commit_fails,
    x3_close_fails,
    x4_setup_fails_before_commit,
    c1_cancel_incoming_setup,
    c2_cancel_resume_waiting_for_close,
    c3_cancel_exit_delivery,
    r1_stale_receipt_at_load,
    u1_unhealthy_alone_closes,
    r3_unhealthy_parked_resume,
    g1_parked_resume_resets_grants,
    d1_dropped_frames_same_row,
    d2_stale_check_after_resume,
    d3_resume_parked_twice,
]


@pytest.mark.asyncio
@pytest.mark.parametrize("row", ROWS, ids=[row.__name__ for row in ROWS])
async def test_parked_pair(
    row: Callable[[Run], Awaitable[None]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    run = Run(tmp_path, monkeypatch, caplog)
    try:
        await row(run)
        server = run.server
        assert server is not None
        await _until(
            lambda: all(
                pair.holds_work()
                for pair in server.controls()
                if pair is not server.controllers.attached().control
            ),
            "every parked pair that holds no work closes",
        )
    finally:
        run.go.touch()
        if run.backend is not None:
            run.backend.open_all()
        if run.server is not None:
            await _close(run.server, *(client.writer for client in run.clients))
