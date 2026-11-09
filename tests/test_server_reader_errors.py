"""A peer that stops reading must end its serve connection through the EOF path."""

from __future__ import annotations

import asyncio
import json
import socket
from pathlib import Path
from typing import Any

import pytest

import zeta.server.server as server_module
from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.server_client import request
from tests.support.sockets import short_socket_path
from zeta.server import ZetaServer

_HELLO = (
    json.dumps(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "hello",
            "params": {"protocol_version": "1.0"},
        }
    )
    + "\n"
).encode()


@pytest.mark.asyncio
@pytest.mark.parametrize("read_site", ["frame", "oversized_discard"])
async def test_failed_write_to_closed_reader_ends_connection_without_unhandled_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, read_site: str
) -> None:
    payload = _HELLO
    if read_site == "oversized_discard":
        # The hello reply fails while an oversized partial frame waits in the
        # buffer, so the next read happens in the discard loop.
        monkeypatch.setattr(server_module, "MAX_FRAME_BYTES", 1024)
        payload += b"x" * 2048
    loop = asyncio.get_running_loop()
    unhandled: list[dict[str, Any]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: unhandled.append(context))
    server = await asyncio.to_thread(
        ZetaServer,
        home=tmp_path,
        socket_path=short_socket_path(tmp_path),
        provider="codex",
    )
    accepted: asyncio.Future[asyncio.Task[Any]] = loop.create_future()
    accept = server._accept

    async def record_accept(reader: Any, writer: Any) -> None:
        if not accepted.done():
            accepted.set_result(asyncio.current_task())
        await accept(reader, writer)

    server._accept = record_accept
    peer = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    writer = None
    try:
        await await_or_fail(server.start(), what="server start")
        peer.connect(str(server.socket_path))
        # The peer is gone before the server reads, so the hello reply fails
        # with EPIPE and asyncio stores that error in the stream reader.
        peer.sendall(payload)
        peer.close()
        accept_task = await await_or_fail(accepted, what="server accept")
        # The listener's done callback runs before asyncio.wait's callback.
        await await_or_fail(
            asyncio.wait({accept_task}), what="closed-reader connection end"
        )

        assert [
            (context.get("message"), repr(context.get("exception")))
            for context in unhandled
        ] == []

        reader, writer = await await_or_fail(
            asyncio.open_unix_connection(str(server.socket_path)),
            what="second client connect",
        )
        hello = await request(
            reader, writer, 1, "hello", {"protocol_version": "1.0"}
        )
        assert "result" in hello[-1], hello
    finally:
        peer.close()
        try:
            await close_serve(server, *([writer] if writer is not None else []))
        finally:
            loop.set_exception_handler(previous_handler)
