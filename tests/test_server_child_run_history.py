"""real transport checks for child run history negotiation and validation."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.serve_close import close_serve
from tests.support.server_client import request
from tests.support.sockets import short_socket_path
from zeta.server import ZetaServer


async def _open(tmp_path: Path, features: list[str]):
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=short_socket_path(tmp_path),
        approval_reconnect_grace=0,
    )
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    await request(
        reader,
        writer,
        1,
        "hello",
        {"protocol_version": "1.2", "features": features},
    )
    session_id = (
        await request(reader, writer, 2, "new_session", {"provider": "codex"})
    )[-1]["result"]["session"]["session_id"]
    return server, reader, writer, session_id


@pytest.mark.asyncio
async def test_child_history_requires_negotiated_feature_and_valid_run_id(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(tmp_path, [])
    try:
        reply = (
            await request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": "not-a-run"},
            )
        )[-1]
        assert reply["error"]["code"] == -32601
    finally:
        await close_serve(server, writer)

    server, reader, writer, session_id = await _open(tmp_path / "second", ["child_run_history"])
    try:
        reply = (
            await request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": "../../etc"},
            )
        )[-1]
        assert reply["error"]["code"] == -32602
    finally:
        await close_serve(server, writer)
