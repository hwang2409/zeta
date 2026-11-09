"""Serve close while a resume's wake turn is still streaming (bench serve probe order).

scripts/bench/bench.py's serve probe resumes a session with an undelivered child
notification, so resume starts a wake turn, and then closes the server. These
tests hold that wake turn mid-stream and prove close still returns, ends the
turn's stream, and frees the session for the next owner. The order that does
hang is pinned in tests/test_completion_owner.py (strict xfail).
"""

from __future__ import annotations

import asyncio
import fcntl
import json
import os
import threading
from collections.abc import AsyncIterator, Sequence
from pathlib import Path

import httpx
import pytest

from tests.support.owned import await_or_fail
from tests.support.server_client import request
from tests.support.sockets import short_socket_path
from zeta.core.session import SessionManager
from zeta.core.session_execution import RUNTIME_LEASE_NAME
from zeta.protocol.types import (
    CompletionBackend,
    Message,
    StreamEvent,
    StreamEventType,
    ToolSchema,
)
from zeta.providers.completion import owned_completion
from zeta.providers.ollama import OllamaBackend
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)


class BlockedStream:
    """A scripted backend whose stream starts and then waits for a release."""

    def __init__(self) -> None:
        self.streaming = threading.Event()
        self.ended = threading.Event()  # the stream ended without a release
        self.release = asyncio.Event()

    def backend(self) -> CompletionBackend:
        return self  # type: ignore[return-value]

    @owned_completion
    async def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        try:
            yield StreamEvent(StreamEventType.MESSAGE_START)
            self.streaming.set()
            await asyncio.wait_for(self.release.wait(), 60)  # hang guard only
        finally:
            if not self.release.is_set():
                self.ended.set()


class BlockedOllama(httpx.AsyncByteStream):
    """The production Ollama client on a transport whose chat stream sends one
    line and then waits for a release."""

    def __init__(self) -> None:
        self.streaming = threading.Event()
        self.ended = threading.Event()  # the client closed the stream unreleased
        self.release = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        line = {"message": {"role": "assistant", "content": "."}, "done": False}
        yield json.dumps(line).encode() + b"\n"
        self.streaming.set()
        await asyncio.wait_for(self.release.wait(), 60)  # hang guard only

    async def aclose(self) -> None:
        if not self.release.is_set():
            self.ended.set()

    def backend(self) -> CompletionBackend:
        def reply(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/chat", request.url
            return httpx.Response(
                200, headers={"Content-Type": "application/x-ndjson"}, stream=self
            )

        client = httpx.AsyncClient(transport=httpx.MockTransport(reply))
        return OllamaBackend(model="stub", base_url="http://ollama.test", client=client)


def _session_with_undelivered_notification(home: Path) -> str:
    opened = SessionManager(home).create(
        provider="codex", model="stub", cwd=home, auto_project=False
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


@pytest.mark.asyncio
@pytest.mark.parametrize("client", ["connected", "disconnected_first"])
@pytest.mark.parametrize("stream", [BlockedStream, BlockedOllama])
async def test_close_during_resume_wake_turn_ends_turn_and_releases_session(
    tmp_path: Path, client: str, stream: type[BlockedStream | BlockedOllama]
) -> None:
    blocked = stream()
    backend = blocked.backend()
    sid = await await_or_fail(
        asyncio.to_thread(_session_with_undelivered_notification, tmp_path),
        what="session setup",
    )
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=short_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (backend, model or "stub"),
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
        [hello] = await request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["session_history_fidelity", "session_host_commands"],
            },
        )
        assert "result" in hello, hello
        [resume] = await request(reader, writer, 2, "resume", {"session_id": sid})
        assert "result" in resume, resume
        assert await asyncio.to_thread(blocked.streaming.wait, 30), "no wake turn"
        if client == "disconnected_first":  # the bench order
            writer.close()
            await await_or_fail(writer.wait_closed(), what="client close")

        await await_or_fail(server.close(), timeout=30, what="serve close")
        closed = True

        # Close ended the wake turn's stream instead of leaving it running.
        assert await asyncio.to_thread(blocked.ended.wait, 30), "stream still open"
        lease = os.open(tmp_path / "sessions" / sid / RUNTIME_LEASE_NAME, os.O_RDWR)
        try:
            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if still held
        finally:
            os.close(lease)
        await await_or_fail(asyncio.to_thread(_reopen, tmp_path, sid), what="reopen")
    finally:
        blocked.release.set()
        if writer is not None:
            writer.close()
        if not closed:
            await await_or_fail(server.close(), timeout=30, what="serve cleanup close")

