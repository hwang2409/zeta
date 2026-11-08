"""Serve close while a resume's wake turn is still streaming (bench serve probe order)."""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.support.sockets import short_socket_path
from zeta.core.session import SessionManager
from zeta.core.session_execution import RUNTIME_LEASE_NAME
from zeta.protocol.types import (
    Message,
    StreamEvent,
    StreamEventType,
    ToolSchema,
)
from zeta.providers.completion import owned_completion
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)


class BlockedStream:
    """Starts a stream, then waits for a release that never comes."""

    def __init__(self) -> None:
        self.streaming = asyncio.Event()
        self.release = asyncio.Event()
        self.ended: str | None = None

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        try:
            yield StreamEvent(StreamEventType.MESSAGE_START)
            self.streaming.set()
            await self.release.wait()
            self.ended = "released"
        except asyncio.CancelledError:
            self.ended = "cancelled"
            raise
        except GeneratorExit:
            self.ended = "closed"
            raise


def _session_with_undelivered_notification(home: Path) -> str:
    opened = SessionManager(home).create(
        provider="codex", model="offline", cwd=home, auto_project=False
    )
    try:
        opened.store.append_agent_notification(
            "child:1",
            child_session_path=str(home / "child"),
            description="child",
            status="completed",
            text="done",
        )
        return opened.metadata.session_id
    finally:
        opened.store.close()


def _reopen(home: Path, sid: str) -> None:
    SessionManager(home).open(sid).store.close()


async def _request(reader, writer, ident: int, method: str, params: dict) -> dict:
    writer.write((json.dumps({"jsonrpc": "2.0", "id": ident, "method": method, "params": params}) + "\n").encode())
    await writer.drain()
    while True:
        raw = await await_or_fail(reader.readline(), what=f"{method} reply")
        assert raw, f"{method}: serve closed the connection"
        frame = json.loads(raw)
        if frame.get("id") == ident:
            assert "result" in frame, frame
            return frame["result"]


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["connected", "disconnected_first"])
async def test_close_during_resume_wake_turn_cancels_turn_and_releases_session(
    tmp_path: Path, client: str
) -> None:
    sid = await await_or_fail(
        asyncio.to_thread(_session_with_undelivered_notification, tmp_path),
        what="session setup",
    )
    backend = BlockedStream()
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=short_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "offline"),
        auto_memory=False,
        approval_reconnect_grace=0,
    )
    writer = None
    closed = False
    try:
        await await_or_fail(server.start(), what="serve start")
        reader, writer = await await_or_fail(
            asyncio.open_unix_connection(str(server.socket_path), limit=1 << 26),
            what="serve connect",
        )
        # The same hello and resume the bench serve probe sends.
        await _request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["session_history_fidelity", "session_host_commands"],
            },
        )
        await _request(reader, writer, 2, "resume", {"session_id": sid})
        await await_or_fail(backend.streaming.wait(), what="wake turn stream start")
        if client == "disconnected_first":
            writer.close()
            await await_or_fail(writer.wait_closed(), what="client close")

        await await_or_fail(server.close(), timeout=30, what="serve close")
        closed = True

        assert backend.ended in {"cancelled", "closed"}
        lease = os.open(tmp_path / "sessions" / sid / RUNTIME_LEASE_NAME, os.O_RDWR)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if still held
        finally:
            os.close(lease)
        await await_or_fail(asyncio.to_thread(_reopen, tmp_path, sid), what="reopen")
    finally:
        backend.release.set()
        if writer is not None:
            writer.close()
        if not closed:
            await await_or_fail(server.close(), timeout=30, what="serve cleanup close")
