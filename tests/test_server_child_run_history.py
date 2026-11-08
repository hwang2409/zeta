"""real transport checks for child run history negotiation and validation."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

from tests.support.serve_close import close_serve
from tests.support.server_client import request
from tests.support.sockets import short_socket_path
from zeta.core.store import ConversationStore
from zeta.protocol.types import Message, MessageRole, TextContent
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


@pytest.mark.asyncio
async def test_child_history_reads_newest_and_continuation_over_unix_socket(
    tmp_path: Path,
) -> None:
    server, reader, writer, session_id = await _open(
        tmp_path, ["child_run_history", "session_history_fidelity"]
    )
    child = ConversationStore(
        server.runtime.opened.store.session_dir / "agents",
        session_id="1",
        _storage_lease=server.runtime.opened.store._storage_lease,
    )
    child.start_agent_lifecycle(
        handle=f"{session_id}:1",
        started_at="2026-01-01T00:00:00+00:00",
        depth=1,
        agent_type="worker",
        description="child",
    )
    for index in range(130):
        child.append_message(
            Message(MessageRole.ASSISTANT, [TextContent(f"message-{index}")])
        )
    child.close()
    try:
        newest_reply = (
            await request(
                reader,
                writer,
                3,
                "child_run_history",
                {"session_id": session_id, "run_id": f"{session_id}:1"},
            )
        )[-1]
        assert "result" in newest_reply, newest_reply
        newest = newest_reply["result"]
        assert len(newest["messages"]) == 130
        assert newest["messages"][0]["content"][0]["text"] == "message-0"
        assert newest["messages"][-1]["content"][0]["text"] == "message-129"
        assert newest.get("next_before") is None
        continuation = (
            await request(
                reader,
                writer,
                4,
                "child_run_history",
                {
                    "session_id": session_id,
                    "run_id": f"{session_id}:1",
                    "before": 65,
                    "history_validator": newest["history_validator"],
                },
            )
        )[-1]["result"]
        assert continuation["messages"][0]["content"][0]["text"] == "message-0"
        assert continuation["messages"][-1]["content"][0]["text"] == "message-64"
        child_path = server.runtime.opened.store.session_dir / "agents" / "1"
        rows = (child_path / "conversation.jsonl").read_text().splitlines(True)
        replacement = child_path / "conversation.replacement"
        replacement.write_text("".join(rows[:-1]))
        os.replace(replacement, child_path / "conversation.jsonl")
        stale = (
            await request(
                reader,
                writer,
                5,
                "child_run_history",
                {
                    "session_id": session_id,
                    "run_id": f"{session_id}:1",
                    "before": 65,
                    "history_validator": newest["history_validator"],
                },
            )
        )[-1]
        assert stale["error"]["code"] == -32009
    finally:
        await close_serve(server, writer)
