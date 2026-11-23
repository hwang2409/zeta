"""session_history_read: a bounded, observer-safe page of another same-project session."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import shutil
import threading
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.serve_contract import ServeClient
from tests.support.sockets import short_socket_path
from zeta.core.session_execution import execution_owner_state
from zeta.core.session_files import child_directory, session_root
from zeta.core.store import ConversationStore
from zeta.project_registry import ProjectRegistry
from zeta.protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    with_message_origin,
)
from zeta.server import ZetaServer
from zeta.server.handlers.resources import session_read
from zeta.storage_format import WRITER_EPOCH

METHOD = "session_history_read"
FEATURES = ["serve_observers", "session_history_read_v1", "session_history_backward", "session_history_fidelity"]


class Harness:
    def __init__(self, tmp_path: Path, *, launch_project: bool) -> None:
        cwd = tmp_path / "launch"
        cwd.mkdir()
        self.server = ZetaServer(
            home=tmp_path / "home", cwd=cwd, provider="codex", socket_path=short_socket_path(tmp_path),
            backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
        )
        # keep the background retention writer idle
        (self.server.home / "settings.toml").write_text("[retention]\ndays = 0\n")
        self.writers: list[asyncio.StreamWriter] = []
        registry = ProjectRegistry(self.server.home / "projects")
        self.project = registry.create_project("alpha", "repo", str(cwd.resolve()) if launch_project else None).project_id
        self.other_project = registry.create_project("beta", "repo").project_id

    def session(self, project_id: str | None, turns: int = 0, *, keep_open: bool = False):
        opened = self.server.runtime.manager.create(
            provider="codex", model="offline", cwd=self.server.runtime.cwd, project_id=project_id, auto_project=False,
        )
        for turn in range(turns):
            append_turn(opened.store, f"turn {turn}")
        if keep_open:
            return opened
        opened.store.close()
        return opened.metadata.session_id

    def transcript(self, session_id: str) -> Path:
        return self.server.home / "sessions" / session_id / "conversation.jsonl"

    async def client(self, *, role: str = "observer") -> ServeClient:
        reader, writer = await asyncio.open_unix_connection(str(self.server.socket_path), limit=2 * 1024 * 1024)
        self.writers.append(writer)
        client = ServeClient(reader, writer)
        hello = {"protocol_version": "1.2", "features": FEATURES}
        if role == "observer":
            hello["role"] = "observer"
        reply = await client.request("hello", "hello", hello)
        assert "session_history_read_v1" in reply["result"]["capabilities"]["features"], reply
        return client


def append_turn(store, text: str) -> None:
    store.append_message(with_message_origin(Message(MessageRole.USER, [TextContent(f"{text} question")]), MessageOrigin.USER))
    store.append_message(Message(MessageRole.ASSISTANT, [TextContent(f"{text} answer")]))


async def start(tmp_path: Path, attach: bool = True, *, projectless: bool = False) -> tuple[Harness, str | None]:
    harness = await asyncio.to_thread(Harness, tmp_path, launch_project=not projectless)
    attached = None
    if attach:
        attached = await asyncio.to_thread(harness.session, None if projectless else harness.project)
        await harness.server.runtime.resume_session(attached)
    await harness.server.start()
    return harness, attached


async def read(client: ServeClient, request_id: str, **params) -> dict:
    return await client.request(request_id, METHOD, params)


def texts(page: dict) -> list[str]:
    return [part["text"] for row in page["messages"] for part in row.get("content", ()) if part.get("type") == "text"]


class GatedRead:
    """Pause every session read inside the worker thread until released."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0
        self.fail = fail
        real = session_read._read

        def gated(*args):
            self.calls += 1
            self.entered.set()
            assert self.release.wait(10), "test never released the gated read"
            if self.fail:
                raise RuntimeError("injected read failure")
            return real(*args)

        # test-contract: seam hold the read inside the worker thread for a forced ordering
        monkeypatch.setattr(session_read, "_read", gated)

    async def wait_entered(self) -> None:
        await await_or_fail(asyncio.to_thread(self.entered.wait, 10), what="gated read entry")


async def slot_free() -> None:
    pending = session_read._in_flight.get(asyncio.get_running_loop())
    if pending is not None:
        await await_or_fail(asyncio.wait([pending]), what="session read slot release")
    assert asyncio.get_running_loop() not in session_read._in_flight


@pytest.mark.asyncio
async def test_observer_reads_same_project_session_as_its_session_history(tmp_path: Path) -> None:
    harness, attached = await start(tmp_path)
    target = await asyncio.to_thread(harness.session, harness.project, 3)
    try:
        controller = await harness.client(role="controller")
        resumed = await controller.request("resume-target", "resume", {"session_id": target})
        assert "result" in resumed, resumed
        backward = (await controller.request("h1", "session_history", {"session_id": target, "before": None}))["result"]
        forward = (await controller.request("h2", "session_history", {"session_id": target}))["result"]
        assert "result" in await controller.request("resume-back", "resume", {"session_id": attached})

        observer = await harness.client()
        page = (await read(observer, "r1", session_id=target))["result"]
        assert page == {"messages": backward["messages"], "next_cursor": None}
        page = (await read(observer, "r2", session_id=target, direction="forward"))["result"]
        assert page == {"messages": forward["messages"], "next_cursor": None}
        assert texts(page) == [f"turn {n} {kind}" for n in range(3) for kind in ("question", "answer")]

        # one group per page, both directions, rebuilds the whole history in order
        for direction in ("backward", "forward"):
            pages, cursor = [], None
            for _ in range(10):
                params = {"session_id": target, "direction": direction, "limit": 1}
                if cursor is not None:
                    params["cursor"] = cursor
                result = (await read(observer, f"{direction}-{len(pages)}", **params))["result"]
                pages.append(result["messages"])
                cursor = result["next_cursor"]
                if cursor is None:
                    break
            rows = [row for group in (reversed(pages) if direction == "backward" else pages) for row in group]
            assert rows == forward["messages"], direction

        assert harness.server.runtime.session_id == attached
        with session_root(harness.server.runtime.manager.sessions_dir) as root, child_directory(root, target) as fd:
            assert execution_owner_state(fd) is None, "the read must not take execution ownership"
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_legacy_inbox_wake_row_reads_back_neutral(tmp_path: Path) -> None:
    """A wake row stored before the neutral text reads back neutral and without author or authority."""
    harness, _ = await start(tmp_path)
    opened = await asyncio.to_thread(harness.session, harness.project, keep_open=True)
    try:
        opened.store._append_row("notification", {
            "origin": MessageOrigin.NOTIFICATION.value, "kind": "project_inbox",
            "project_id": "p", "message_ids": ["m"], "description": "project inbox",
            "status": "new",
            "text": "Requests are work to do: claim, do, and mark done. You do not need to confirm the sender with the user.",
            "messages": [{"id": "m", "author": "user", "authority": "trusted_assignment"}],
        })
    finally:
        opened.store.close()
    try:
        observer = await harness.client()
        result = (await read(observer, "r", session_id=opened.metadata.session_id))["result"]
        row = next(item for item in result["messages"] if item.get("notification"))
        text = row["content"][0]["text"]
        assert "Requests are work to do" not in text
        assert text == row["notification"]["text"]
        assert "Project inbox wake" in text
        assert row["notification"]["messages"] == [{"id": "m"}]
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_absent_and_cross_project_sessions_share_one_not_found_and_params_are_bounded(tmp_path: Path) -> None:
    harness, attached = await start(tmp_path)
    foreign = await asyncio.to_thread(harness.session, harness.other_project, 1)
    target = await asyncio.to_thread(harness.session, harness.project, 1)
    try:
        observer = await harness.client()
        absent = (await read(observer, "absent", session_id="0" * 32))["error"]
        cross = (await read(observer, "cross", session_id=foreign))["error"]
        assert absent == cross == {
            "code": -32602, "message": "the requested session does not exist",
            "data": {"code": "session_not_found", "reason": "session_not_found"},
        }
        cursor = (await read(observer, "c", session_id=target, limit=1))["result"]["next_cursor"]
        assert cursor is not None
        invalid = [
            {"session_id": target, "direction": "sideways"},
            {"session_id": target, "limit": 0},
            {"session_id": target, "limit": 129},
            {"session_id": target, "max_bytes": 1024 * 1024 + 1},
            {"session_id": target, "cursor": "not-a-cursor"},
            {"session_id": "../" + target},
            {"session_id": target, "before": None},
            {"session_id": attached},
            {"session_id": target, "direction": "forward", "cursor": cursor},  # bound to backward
            {"session_id": foreign, "cursor": cursor},  # bound to another session
        ]
        for index, params in enumerate(invalid):
            error = (await observer.request(f"invalid-{index}", METHOD, params))["error"]
            assert (error["code"], error["data"]) == (-32602, {"code": "invalid_params"}), params
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
@pytest.mark.parametrize("attach", [True, False], ids=["attached", "idle"])
async def test_projectless_sessions_read_each_other_and_never_cross_the_project_boundary(
    tmp_path: Path, attach: bool
) -> None:
    harness, _ = await start(tmp_path, attach=attach, projectless=True)
    target = await asyncio.to_thread(harness.session, None, 2)
    in_project = await asyncio.to_thread(harness.session, harness.project, 1)
    # a child-agent store under the target, at sessions/<target>/agents/1
    child = ConversationStore(harness.server.home / "sessions" / target / "agents", session_id="1", cwd=str(tmp_path))
    try:
        append_turn(child, "child")
    finally:
        child.close()
    not_found = {
        "code": -32602, "message": "the requested session does not exist",
        "data": {"code": "session_not_found", "reason": "session_not_found"},
    }
    try:
        observer = await harness.client()
        reply = await read(observer, "same", session_id=target, direction="forward")
        assert "result" in reply, reply
        assert texts(reply["result"]) == [f"turn {n} {kind}" for n in range(2) for kind in ("question", "answer")]
        assert (await read(observer, "cross", session_id=in_project))["error"] == not_found
        # child sessions are not reachable by id: by its number, its instance id, or its path
        assert (await read(observer, "child-number", session_id="1"))["error"] == not_found
        assert (await read(observer, "child-instance", session_id=f"{target}:1"))["error"] == not_found
        error = (await read(observer, "child-path", session_id=f"{target}/agents/1"))["error"]
        assert (error["code"], error["data"]) == (-32602, {"code": "invalid_params"})

        # a project caller does not see project-less sessions either
        assert "result" in await (await harness.client(role="controller")).request(
            "attach-project", "resume", {"session_id": in_project},
        )
        assert (await read(observer, "reverse", session_id=target))["error"] == not_found
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_idle_history_reads_only_the_launch_project(tmp_path: Path) -> None:
    harness, _ = await start(tmp_path, attach=False)
    target = await asyncio.to_thread(harness.session, harness.project, 1)
    foreign = await asyncio.to_thread(harness.session, harness.other_project, 1)
    projectless = await asyncio.to_thread(harness.session, None, 1)
    try:
        observer = await harness.client()
        reply = await read(observer, "same", session_id=target)
        assert "result" in reply, reply
        assert texts(reply["result"]) == ["turn 0 question", "turn 0 answer"]
        assert harness.server.runtime.opened is None
        absent = (await read(observer, "absent", session_id="0" * 32))["error"]
        assert (absent["code"], absent["data"]["code"]) == (-32602, "session_not_found")
        assert (await read(observer, "foreign", session_id=foreign))["error"] == absent
        assert (await read(observer, "projectless", session_id=projectless))["error"] == absent

        controller = await harness.client(role="controller")
        created = await controller.request("new", "new_session", {})
        assert "result" in created, created
        assert harness.server.runtime.metadata.project_id == harness.project
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_idle_history_read_rejects_malformed_launch_project(tmp_path: Path) -> None:
    harness, _ = await start(tmp_path, attach=False)
    try:
        target = await asyncio.to_thread(harness.session, None, 1)
        observer = await harness.client()
        record = harness.server.home / "projects" / harness.project / "project.json"
        await asyncio.to_thread(record.write_text, "{")

        reply = await read(observer, "unresolved", session_id=target)
        assert "error" in reply, reply
        assert (reply["error"]["code"], reply["error"]["data"]["code"]) == (-32005, "state_unavailable")
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_idle_history_read_discards_rows_after_target_reassociation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    harness, _ = await start(tmp_path, attach=False, projectless=True)
    owner = await asyncio.to_thread(harness.session, None, 1, keep_open=True)
    manager = harness.server.runtime.manager
    real_page = session_read._page

    def reassociate_before_page_returns(*args):
        page = real_page(*args)
        assert texts(page) == ["turn 0 question", "turn 0 answer"]
        manager.associate_project(owner.metadata, harness.project)
        return page

    # test-contract: seam force a durable reassociation after projection, before _page returns
    monkeypatch.setattr(session_read, "_page", reassociate_before_page_returns)
    try:
        observer = await harness.client()
        absent = await read(observer, "absent", session_id="0" * 32)
        reply = await read(observer, "reassociated", session_id=owner.metadata.session_id)
        assert "error" in reply, reply
        assert reply["error"] == absent["error"]
    finally:
        await asyncio.to_thread(owner.store.close)
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
@pytest.mark.parametrize("transition", ["attach", "attach-detach", "detach"])
async def test_history_read_discards_rows_when_the_attachment_changes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, transition: str
) -> None:
    harness, _ = await start(tmp_path, attach=transition == "detach")
    target = await asyncio.to_thread(harness.session, harness.project, 1)
    foreign = await asyncio.to_thread(harness.session, harness.other_project)
    gate = GatedRead(monkeypatch)
    pending = None
    try:
        observer = await harness.client()
        controller = await harness.client(role="controller")
        pending = asyncio.create_task(read(observer, "held", session_id=target))
        await gate.wait_entered()
        if transition != "detach":
            attached = await controller.request("attach", "resume", {"session_id": foreign})
            assert "result" in attached, attached
        if transition != "attach":
            # Serve has no detach RPC; exercise the real runtime teardown without closing the socket.
            await harness.server.runtime.close()
            assert harness.server.runtime.opened is None
        gate.release.set()
        reply = await await_or_fail(pending, what="history reply after attachment change")
        assert "error" in reply, reply
        assert (reply["error"]["code"], reply["error"]["data"]["code"]) == (-32005, "state_unavailable")
    finally:
        gate.release.set()
        try:
            if pending is not None and not pending.done():
                await await_or_fail(pending, what="finish gated history read")
        finally:
            await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_live_writer_torn_tail_is_ignored_and_owner_is_never_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, _ = await start(tmp_path)
    owner = await asyncio.to_thread(harness.session, harness.project, 2, keep_open=True)
    target = owner.metadata.session_id
    try:
        observer = await harness.client()
        first = (await read(observer, "first", session_id=target, direction="forward", limit=1))["result"]
        assert first["next_cursor"] is not None

        # the owner is mid-append: it holds the append lock and the last line is torn
        lock_fd = os.open(harness.transcript(target).with_name(".lock"), os.O_RDWR)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX)
            size = harness.transcript(target).stat().st_size
            with harness.transcript(target).open("ab") as log:
                log.write(b'{"type": "message", "data": {"torn')
            page = (await read(observer, "torn", session_id=target))["result"]
            assert texts(page) == [f"turn {n} {kind}" for n in range(2) for kind in ("question", "answer")]
            os.truncate(harness.transcript(target), size)
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

        # the reader holds its snapshot inside the worker thread while the owner appends
        gate = GatedRead(monkeypatch)
        pending = asyncio.ensure_future(read(observer, "paused", session_id=target))
        await gate.wait_entered()
        await await_or_fail(asyncio.to_thread(append_turn, owner.store, "turn 2"), what="owner append during a read")
        gate.release.set()
        await await_or_fail(pending, what="paused read")
        page = (await read(observer, "newest", session_id=target))["result"]
        assert texts(page)[-2:] == ["turn 2 question", "turn 2 answer"]
        # an append keeps the groups below the first page's end: the continuation still works
        rest = (await read(observer, "rest", session_id=target, direction="forward", cursor=first["next_cursor"]))
        assert texts(rest["result"])[-2:] == ["turn 2 question", "turn 2 answer"], rest
    finally:
        await asyncio.to_thread(owner.store.close)
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_replaced_source_makes_a_continuation_stale(tmp_path: Path) -> None:
    harness, _ = await start(tmp_path)
    target = await asyncio.to_thread(harness.session, harness.project, 3)
    try:
        observer = await harness.client()
        first = (await read(observer, "first", session_id=target, limit=1))["result"]
        path = harness.transcript(target)
        shutil.copyfile(path, path.with_name("replacement"))
        os.replace(path.with_name("replacement"), path)
        stale = (await read(observer, "stale", session_id=target, cursor=first["next_cursor"]))["error"]
        assert (stale["code"], stale["data"]["code"]) == (-32009, "stale_history_snapshot")
        assert "result" in await read(observer, "newest", session_id=target)
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_corrupt_transcript_and_newer_format_are_state_unavailable_without_detail(tmp_path: Path) -> None:
    harness, _ = await start(tmp_path)
    target = await asyncio.to_thread(harness.session, harness.project, 1)
    try:
        observer = await harness.client()
        with harness.transcript(target).open("ab") as log:
            log.write(b"not json at all\n")
        raw = await observer.request_raw("corrupt", METHOD, {"session_id": target})
        error = json.loads(raw)["error"]
        assert (error["code"], error["data"]["code"]) == (-32005, "state_unavailable")
        assert str(tmp_path) not in raw and "conversation" not in raw

        healthy = await asyncio.to_thread(harness.session, harness.project, 1)
        manifest = harness.server.home / "storage-format.json"
        original = manifest.read_bytes()
        manifest.write_text(json.dumps({"writer_epoch": WRITER_EPOCH + 1}))
        try:
            raw = await observer.request_raw("newer", METHOD, {"session_id": healthy})
        finally:
            manifest.write_bytes(original)
        error = json.loads(raw)["error"]
        assert (error["code"], error["data"]["code"]) == (-32005, "state_unavailable")
        assert str(tmp_path) not in raw and "format" not in raw
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_limit_max_bytes_and_result_too_large(tmp_path: Path) -> None:
    harness, _ = await start(tmp_path)
    target = await asyncio.to_thread(harness.session, harness.project, 65)
    try:
        observer = await harness.client()
        full = (await read(observer, "full", session_id=target, limit=128))["result"]
        assert len(full["messages"]) == 128 and full["next_cursor"] is not None
        rest = (await read(observer, "rest", session_id=target, limit=128, cursor=full["next_cursor"]))["result"]
        assert texts(rest) == ["turn 0 question", "turn 0 answer"]
        assert rest["next_cursor"] is None

        cap = len(json.dumps({"jsonrpc": "2.0", "id": "small", "result": full}, separators=(",", ":"))) // 3
        small_raw = await observer.request_raw("small", METHOD, {"session_id": target, "limit": 128, "max_bytes": cap})
        small = json.loads(small_raw)["result"]
        assert len(small_raw.encode()) <= cap + 1  # +1 for the frame newline
        assert 0 < len(small["messages"]) < 128
        assert small["messages"] == full["messages"][-len(small["messages"]):]
        assert small["next_cursor"] is not None

        error = (await read(observer, "tiny", session_id=target, max_bytes=100))["error"]
        assert (error["code"], error["data"]["code"]) == (-32007, "result_too_large")
    finally:
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
async def test_slow_read_holds_the_only_slot_without_blocking_other_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, attached = await start(tmp_path)
    target = await asyncio.to_thread(harness.session, harness.project, 1)
    try:
        first, second, third = [await harness.client() for _ in range(3)]
        gate = GatedRead(monkeypatch)
        pending = asyncio.ensure_future(read(first, "slow", session_id=target))
        await gate.wait_entered()

        # another client's history read uses the shared read workers while the slow read waits
        other = await await_or_fail(
            second.request("other", "session_history", {"session_id": attached, "before": None}),
            what="another client's request during a slow session read",
        )
        assert "result" in other, other
        busy = (await read(second, "busy", session_id=target))["error"]
        assert busy == {
            "code": -32004, "message": "operation is blocked by active turn, tool, or background-agent work",
            "data": {"code": "operation_busy", "reason": "operation_busy", "retry_after_ms": 2000},
        }
        assert gate.calls == 1, "a busy call must not start a second read"

        gate.release.set()
        assert "result" in await await_or_fail(pending, what="slow read")
        await slot_free()
        assert "result" in await read(third, "after", session_id=target)
        assert gate.calls == 2
    finally:
        gate.release.set()
        await close_serve(harness.server, *harness.writers)


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["disconnect", "deadline", "failure"])
async def test_slot_frees_only_after_the_worker_read_ends(
    ending: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    harness, _ = await start(tmp_path)
    target = await asyncio.to_thread(harness.session, harness.project, 1)
    try:
        first, second = [await harness.client() for _ in range(2)]
        if ending == "deadline":
            # test-contract: complexity shrink the named deadline; the gate, not the clock, orders the steps
            monkeypatch.setattr(session_read, "READ_DEADLINE_SECONDS", 0)
        gate = GatedRead(monkeypatch, fail=ending == "failure")
        pending = asyncio.ensure_future(read(first, "held", session_id=target))
        await gate.wait_entered()
        if ending == "disconnect":
            harness.writers.remove(first_writer := first._writer)
            first_writer.close()
            pending.cancel()
        if ending == "deadline":
            error = (await await_or_fail(pending, what="deadline reply"))["error"]
            assert (error["code"], error["data"]["code"]) == (-32005, "state_unavailable")
        # the worker thread still runs: the slot stays taken
        busy = (await read(second, "busy", session_id=target))["error"]
        assert busy["code"] == -32004
        gate.release.set()
        if ending == "failure":
            error = (await await_or_fail(pending, what="failed read"))["error"]
            assert error == {"code": -32000, "message": "unexpected server or storage failure"}
        await slot_free()
        gate.fail = False
        monkeypatch.setattr(session_read, "READ_DEADLINE_SECONDS", 10)
        assert "result" in await read(second, "after", session_id=target)
        assert gate.calls == 2
    finally:
        await close_serve(harness.server, *harness.writers)
