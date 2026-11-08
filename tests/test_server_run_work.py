from __future__ import annotations

import asyncio
import shlex
import sys
from pathlib import Path

import pytest

from tests.support.serve_close import close_serve
from tests.support.server_client import event, request
from zeta.server import ZetaServer

TIMEOUT = 30


async def _connect(server: ZetaServer) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    await asyncio.wait_for(server.start(), TIMEOUT)
    assert server.port is not None
    return await asyncio.wait_for(
        asyncio.open_connection("127.0.0.1", server.port), TIMEOUT
    )


@pytest.mark.asyncio
async def test_run_work_lists_and_cancels_real_task_and_publishes_todo(
    tmp_path: Path,
) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        hello = (
            await request(
                reader,
                writer,
                "hello",
                "hello",
                {"protocol_version": "1.2", "features": ["run_work"]},
            )
        )[-1]
        assert "run_work" in hello["result"]["capabilities"]["features"]
        session = (
            await request(
                reader, writer, "new", "new_session", {"provider": "codex"}
            )
        )[-1]["result"]["session"]
        session_id = session["session_id"]
        registry = server.runtime.loop.tool_registry.background_tasks
        command = (
            f"{shlex.quote(sys.executable)} -c "
            + shlex.quote("import sys; sys.stdin.buffer.read()")
        )
        task_id, _pid = await registry.start(command, tmp_path)
        started = await event(reader, "background_task_updated")
        assert started["task"] == {
            "id": task_id,
            "status": "running",
            "command": command,
            "elapsed_seconds": started["task"]["elapsed_seconds"],
            "cancelable": True,
        }
        assert started["task"]["elapsed_seconds"] >= 0

        todo = [{"content": "finish transport proof", "status": "in_progress"}]
        server.runtime.opened.store.set_todo_items(todo)
        assert (await event(reader, "todo_updated"))["todo"] == todo

        listed = (
            await request(
                reader,
                writer,
                "list",
                "run_work_list",
                {"session_id": session_id},
            )
        )[-1]["result"]
        assert listed["todo"] == todo
        assert listed["background_tasks"][0]["id"] == task_id
        assert listed["background_tasks"][0]["status"] == "running"
        assert listed["background_tasks"][0]["cancelable"] is True
        assert listed["background_tasks"][0]["elapsed_seconds"] >= 0

        canceled = (
            await request(
                reader,
                writer,
                "cancel",
                "background_task_cancel",
                {"session_id": session_id, "task_id": task_id},
            )
        )[-1]["result"]["task"]
        assert canceled["id"] == task_id
        assert canceled["status"] == "killed"
        assert canceled["cancelable"] is False
        terminal = await event(reader, "background_task_updated")
        assert terminal["task"] == canceled
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_run_work_requires_negotiated_feature(tmp_path: Path) -> None:
    server = ZetaServer(home=tmp_path, port=0, provider="codex")
    reader, writer = await _connect(server)
    try:
        await request(
            reader,
            writer,
            "hello",
            "hello",
            {"protocol_version": "1.2", "features": []},
        )
        response = (
            await request(
                reader,
                writer,
                "list",
                "run_work_list",
                {"session_id": "missing"},
            )
        )[-1]
        assert response["error"]["code"] == -32601
    finally:
        await close_serve(server, writer)
