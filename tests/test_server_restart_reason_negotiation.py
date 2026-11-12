"""Serve negotiation of the approval_restart_reason wire feature."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.serve_close import close_serve
from tests.support.server_client import request
from tests.support.sockets import short_socket_path as _socket_path
from zeta.protocol.types import ToolCall
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)


async def _open(tmp_path: Path, features: list[str]):
    server = ZetaServer(
        provider="codex",
        home=tmp_path,
        socket_path=_socket_path(tmp_path),
    )
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(server.socket_path))
    hello = (
        await request(
            reader, writer, 1, "hello", {"protocol_version": "1.2", "features": features}
        )
    )[-1]["result"]
    session_id = (
        await request(reader, writer, 2, "new_session", {"provider": "codex"})
    )[-1]["result"]["session"]["session_id"]
    return server, reader, writer, session_id, hello


def _seed_restart_end(server: ZetaServer) -> None:
    store = server.runtime.opened.store
    store.append_approval_request("call-r", ToolCall("call-r", "bash", {"command": "true"}))
    store.resolve_approval("call-r", "abort", reason="server_restarted")


async def _approval_end_row(reader, writer, session_id: str) -> dict:
    messages = (
        await request(reader, writer, 90, "session_history", {"session_id": session_id})
    )[-1]["result"]["messages"]
    ends = [row for row in messages if row.get("event") == "approval_end"]
    assert len(ends) == 1, f"expected one approval_end row, got {ends!r}"
    return ends[0]


@pytest.mark.asyncio
async def test_negotiating_client_sees_server_restarted_reason(tmp_path: Path) -> None:
    server, reader, writer, session_id, hello = await _open(
        tmp_path, ["approval_restart_reason"]
    )
    try:
        assert "approval_restart_reason" in hello["capabilities"]["features"]
        _seed_restart_end(server)
        row = await _approval_end_row(reader, writer, session_id)
        assert row["data"]["decision"] == "abort"
        assert row["data"]["reason"] == "server_restarted"
        assert row["reason"] == "server_restarted"
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_legacy_client_gets_terminal_end_without_restart_reason(
    tmp_path: Path,
) -> None:
    # A client that did not negotiate the feature (fulcrum is live) still learns
    # the request is terminal, but never sees the unknown reason.
    server, reader, writer, session_id, _hello = await _open(tmp_path, [])
    try:
        _seed_restart_end(server)
        row = await _approval_end_row(reader, writer, session_id)
        assert row["data"]["decision"] == "abort"
        assert "reason" not in row["data"]
        assert "reason" not in row
    finally:
        await close_serve(server, writer)
