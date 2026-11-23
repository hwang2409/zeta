"""A page that closes its socket during a join reply must free the controller slot."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator
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
from zeta.server import ZetaServer
from zeta.server.server import _Client

pytest_plugins = ("tests.support.socket_fixtures",)

JOIN_REPLY = b'"id":3'


def _backend_factory(provider: str, model: str | None, home: Path) -> tuple[FakeBackend, str]:
    del provider, home
    return FakeBackend([]), model or "offline"


async def _started(tmp_path: Path, transport: Transport) -> tuple[ZetaServer, ReplayClient, str]:
    server = replay_server(tmp_path, transport, _backend_factory)
    await server.start()
    client = await connect_replay(server, transport)
    await negotiate(client)
    created = await client.request(2, "new_session", {"provider": "codex"})
    return server, client, created["result"]["session"]["session_id"]


@contextlib.asynccontextmanager
async def _closing(
    server: ZetaServer, client: ReplayClient, resume: asyncio.Event | None = None
) -> AsyncIterator[None]:
    try:
        yield
    except BaseException:
        if resume is not None:
            resume.set()
        await client.close()
        # the bug also wedges server close: keep the first failure as the report
        with contextlib.suppress(Exception, pytest.fail.Exception):
            await await_or_fail(server.close(), what="server close after a failure")
        raise
    await client.close()
    await await_or_fail(server.close(), what="server close")


def _watch_release(server: ZetaServer, monkeypatch: pytest.MonkeyPatch) -> asyncio.Event:
    assert server.connections is not None
    connection, released = next(iter(server.connections.clients()), None), asyncio.Event()
    original = server.connections.release

    def release(target: Any) -> None:
        original(target)
        if target is connection:
            released.set()

    # test-contract: seam observe the real connection release after its transport closes
    monkeypatch.setattr(server.connections, "release", release)
    return released


async def _assert_new_controller(server: ZetaServer, transport: Transport) -> None:
    replacement = await connect_replay(server, transport)
    try:
        hello = await negotiate(replacement)
        assert "error" not in hello, f"the old connection kept the controller slot: {hello}"
    finally:
        await replacement.close()


def _hold_write(
    monkeypatch: pytest.MonkeyPatch, marker: bytes = JOIN_REPLY
) -> tuple[asyncio.Event, asyncio.Event]:
    """Hold one frame's physical write until the test resumes it."""
    entered, resume = asyncio.Event(), asyncio.Event()
    original = _Client._write_physical

    async def gated(target: _Client, payload: bytes) -> bool:
        if marker in payload and not entered.is_set():
            entered.set()
            await await_or_fail(resume.wait(), what="resume the held write")
        return await original(target, payload)

    # test-contract: seam hold one frame at its physical write
    monkeypatch.setattr(_Client, "_write_physical", gated)
    return entered, resume


WRITER_FAILED = "serve output writer failed"


def _writer_failures(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.getMessage().startswith(WRITER_FAILED)]


async def _assert_released_and_reusable(
    server: ZetaServer, transport: Transport, output: Any, released: asyncio.Event
) -> None:
    await await_or_fail(released.wait(), what="old connection released its controller slot")
    assert (output.retained_bytes, output.retained_frames) == (0, 0)
    assert server.connections is not None
    assert (server.connections.replay._bytes, server.connections.replay._records) == (0, 0)
    await _assert_new_controller(server, transport)


async def _started_owner(
    tmp_path: Path, transport: Transport, monkeypatch: pytest.MonkeyPatch
) -> tuple[ZetaServer, ReplayClient, str, Any, Any, asyncio.Event]:
    server, client, session_id = await _started(tmp_path, transport)
    assert server.connections is not None
    owner = next(iter(server.connections.clients()), None)
    output = server.connections._connections[owner].output
    return server, client, session_id, owner, output, _watch_release(server, monkeypatch)


@pytest.mark.asyncio
async def test_page_close_during_join_reply_releases_controller_slot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """A browser reload: the socket closes while the join reply is written, so the
    write raises (websockets' ConnectionClosed), the writer's exception branch."""
    caplog.set_level(logging.ERROR, logger="zeta.server.connections")
    server, client, session_id, owner, output, released = await _started_owner(
        tmp_path, "websocket", monkeypatch
    )
    entered, resume = _hold_write(monkeypatch)
    async with _closing(server, client, resume):
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(entered.wait(), what="join reply write started")
        await client.close()
        # test-contract: seam wait until serve's websocket saw the close, so the held write fails
        await await_or_fail(owner.writer._connection.wait_closed(), what="serve saw the close")
        resume.set()
        await _assert_released_and_reusable(server, "websocket", output, released)
        # the write failure path itself must hold, not the dead-writer guard
        assert _writer_failures(caplog) == []


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_cancelled_join_reply_write_releases_controller_slot(
    tmp_path: Path,
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The join reply's write task is cancelled (the writer's cancel branch)."""
    caplog.set_level(logging.ERROR, logger="zeta.server.connections")
    server, client, session_id, _, output, released = await _started_owner(
        tmp_path, transport, monkeypatch
    )
    entered, resume = _hold_write(monkeypatch)
    async with _closing(server, client, resume):
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(entered.wait(), what="join reply write started")
        assert output.current_write is not None
        # test-contract: seam cancel the in-flight frame write, as a write owner close does
        output.current_write.cancel()
        await _assert_released_and_reusable(server, transport, output, released)
        # the cancel path itself must hold, not the dead-writer guard
        assert _writer_failures(caplog) == []


HELD_EVENT = b'"id":99'


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
@pytest.mark.parametrize("charged", (False, True), ids=("after_uncharge", "before_uncharge"))
async def test_output_writer_failure_closes_connection_and_releases_slot(
    tmp_path: Path,
    transport: Transport,
    charged: bool,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """An unexpected exception in the output writer closes the connection: the frame
    in flight is uncharged, and the request loop's reply queued behind it fails, so
    the loop reads EOF and releases the slot."""
    caplog.set_level(logging.ERROR, logger="zeta.server.connections")
    server, client, _, owner, output, released = await _started_owner(
        tmp_path, transport, monkeypatch
    )
    connections = server.connections
    assert connections is not None
    entered, resume = _hold_write(monkeypatch, HELD_EVENT)
    original = connections._end_frame_write

    failed = False

    def broken(target: Any, item: Any) -> None:
        nonlocal failed
        held = not failed and HELD_EVENT in item.payload
        failed = failed or held
        if held and charged:
            raise RuntimeError("injected output writer failure")
        original(target, item)
        if held:
            raise RuntimeError("injected output writer failure")

    queued = asyncio.Event()
    enqueue = connections._enqueue

    def watch_enqueue(target: Any, frame: Any, **kwargs: Any) -> None:
        enqueue(target, frame, **kwargs)
        if b'"id":4' in frame.payload:
            queued.set()

    # test-contract: seam an unexpected bug inside the output writer task
    monkeypatch.setattr(connections, "_end_frame_write", broken)
    # test-contract: seam observe the status reply entering the output queue
    monkeypatch.setattr(connections, "_enqueue", watch_enqueue)
    async with _closing(server, client, resume):
        # a live event: the writer sends it with no caller waiting on it
        connections._publish_observer(
            connections._connections[owner], owner.codec.response(99, {})
        )
        await await_or_fail(entered.wait(), what="held frame write started")
        # the request loop's reply waits in the queue behind the held frame
        await client.send(4, "status", {})
        await await_or_fail(queued.wait(), what="status reply queued")
        assert output.current_frame is not None and HELD_EVENT in output.current_frame.payload
        resume.set()
        await _assert_released_and_reusable(server, transport, output, released)
        assert len(_writer_failures(caplog)) == 1



@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ("unix", "websocket"))
async def test_reply_after_writer_failure_fails_at_once_and_releases_slot(
    tmp_path: Path,
    transport: Transport,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A handler still running when the writer dies must not queue its reply into
    the dead queue: the reply fails at once, so the loop reads EOF and releases."""
    caplog.set_level(logging.ERROR, logger="zeta.server.connections")
    server, client, session_id, owner, output, released = await _started_owner(
        tmp_path, transport, monkeypatch
    )
    connections = server.connections
    assert connections is not None
    reading, finish_read = asyncio.Event(), asyncio.Event()
    read_history = connections.read_replay_history

    async def held_read(*args: Any, **kwargs: Any) -> Any:
        reading.set()
        await await_or_fail(finish_read.wait(), what="finish the held history read")
        return await read_history(*args, **kwargs)

    original = connections._end_frame_write

    def broken(target: Any, item: Any) -> None:
        original(target, item)
        if HELD_EVENT in item.payload:
            raise RuntimeError("injected output writer failure")

    # test-contract: seam hold the join handler at its off-loop history read
    monkeypatch.setattr(connections, "read_replay_history", held_read)
    # test-contract: seam an unexpected bug inside the output writer task
    monkeypatch.setattr(connections, "_end_frame_write", broken)
    async with _closing(server, client, finish_read):
        await client.send(3, "session_history_and_subscribe", {"session_id": session_id})
        await await_or_fail(reading.wait(), what="join handler is reading history")
        # a live event trips the guard while the join handler is mid-way
        connections._publish_observer(
            connections._connections[owner], owner.codec.response(99, {})
        )
        writer = output.writer_task
        assert writer is not None
        await await_or_fail(writer, what="output writer ended")
        assert len(_writer_failures(caplog)) == 1
        # a late send fails at once instead of waiting on a queue nobody drains
        late = owner.codec.response(98, {})
        assert await await_or_fail(connections.send(owner, late), what="late send") is False
        finish_read.set()
        await _assert_released_and_reusable(server, transport, output, released)
