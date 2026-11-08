from __future__ import annotations

import asyncio
import importlib
import json
import os
import threading
from contextlib import suppress
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.sockets import short_socket_path
from tests.test_server import (
    _connect,
    _frames_until_eof,
    _request,
)
from zeta.server import ZetaServer

pytest_plugins = ("tests.support.socket_fixtures",)


def _server(tmp_path: Path, cwd: Path) -> ZetaServer:
    return ZetaServer(
        home=tmp_path / "home",
        cwd=cwd,
        provider="codex",
        socket_path=short_socket_path(tmp_path),
        backend_factory=lambda provider, model, home: (
            FakeBackend([]),
            model or "offline",
        ),
    )


async def _hello(reader, writer, *, role: str = "controller", features=()) -> dict:
    return (
        await _request(
            reader,
            writer,
            f"{role}-hello",
            "hello",
            {
                "protocol_version": "1.2",
                "role": role,
                "features": list(features),
            },
        )
    )[-1]


@pytest.mark.asyncio
async def test_path_complete_uses_tui_projection_over_real_unix_transport(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "😀folder").mkdir()
    (workspace / "😀file name.txt").touch()
    (workspace / ".hidden").touch()
    for index in range(51):
        (workspace / f"many-{index:02}.txt").touch()

    server = _server(tmp_path, workspace)
    reader, writer = await _connect(server)
    try:
        hello = await _hello(reader, writer, features=["path_completion"])
        assert "path_completion" in hello["result"]["capabilities"]["features"]
        created = (await _request(reader, writer, "new", "new_session"))[-1]
        session_id = created["result"]["session"]["session_id"]
        before_cursor = "😀 before @😀f"
        response = (
            await _request(
                reader,
                writer,
                "complete",
                "path_complete",
                {
                    "session_id": session_id,
                    "text_before_cursor": before_cursor,
                },
            )
        )[-1]
        entries = response["result"]["entries"]

        assert response["result"]["truncated"] is False
        assert entries == [
            {
                "text": "😀folder/",
                "start_position": -2,
                "display": "@😀folder/",
                "kind": "directory",
            },
            {
                "text": '"😀file name.txt"',
                "start_position": -2,
                "display": '@"😀file name.txt"',
                "kind": "file",
            },
        ]
        first = entries[0]
        replaced = before_cursor[: len(before_cursor) + first["start_position"]]
        replaced += first["text"]
        assert replaced == "😀 before @😀folder/"

        quoted = (
            await _request(
                reader,
                writer,
                "quoted",
                "path_complete",
                {
                    "session_id": session_id,
                    "text_before_cursor": '😀 before @"😀f',
                },
            )
        )[-1]["result"]
        assert [entry["text"] for entry in quoted["entries"]] == [
            "😀folder/",
            '😀file name.txt"',
        ]
        assert {entry["start_position"] for entry in quoted["entries"]} == {-2}

        hidden = (
            await _request(
                reader,
                writer,
                "hidden",
                "path_complete",
                {"session_id": session_id, "text_before_cursor": "@."},
            )
        )[-1]["result"]
        assert [entry["text"] for entry in hidden["entries"]] == [
            ".zeta/",
            ".hidden",
        ]

        capped = (
            await _request(
                reader,
                writer,
                "capped",
                "path_complete",
                {"session_id": session_id, "text_before_cursor": "@many-"},
            )
        )[-1]["result"]
        assert capped["truncated"] is True
        assert len(capped["entries"]) == 50

        oversized = (
            await _request(
                reader,
                writer,
                "oversized",
                "path_complete",
                {"session_id": session_id, "text_before_cursor": "😀" * 1025},
            )
        )[-1]
        assert oversized["error"]["code"] == -32602
        inactive = (
            await _request(
                reader,
                writer,
                "inactive",
                "path_complete",
                {"session_id": "missing", "text_before_cursor": "@"},
            )
        )[-1]
        assert inactive["error"]["code"] == -32003
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_path_complete_negotiation_validation_and_observer_access(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    server = _server(tmp_path, workspace)
    reader, writer = await _connect(server)
    observer_reader = observer_writer = None
    try:
        await _hello(reader, writer, features=["serve_observers"])
        created = (await _request(reader, writer, "new", "new_session"))[-1]
        session_id = created["result"]["session"]["session_id"]
        unnegotiated = (
            await _request(
                reader,
                writer,
                "missing-feature",
                "path_complete",
                {"session_id": session_id, "text_before_cursor": "@"},
            )
        )[-1]
        assert unnegotiated["error"]["code"] == -32601

        observer_reader, observer_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        observer_hello = await _hello(
            observer_reader,
            observer_writer,
            role="observer",
            features=["serve_observers", "path_completion"],
        )
        assert "path_completion" in observer_hello["result"]["capabilities"]["features"]
        rejected = (
            await _request(
                observer_reader,
                observer_writer,
                "observer-path",
                "path_complete",
                {"session_id": session_id, "text_before_cursor": "@"},
            )
        )[-1]
        assert rejected["error"]["code"] == -32008
    finally:
        if observer_writer is not None:
            observer_writer.close()
            await await_or_fail(
                observer_writer.wait_closed(), what="observer writer close"
            )
        await close_serve(server, writer)


@pytest.mark.asyncio
async def test_cancelled_path_complete_publishes_no_response_or_event(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "file.txt").touch()
    server = _server(tmp_path, workspace)
    reader, writer = await _connect(server)
    observer_reader = observer_writer = None
    request_task: asyncio.Task[None] | None = None
    entered = threading.Event()
    release = threading.Event()
    worker_finished = threading.Event()
    path_module = None
    original_scandir = None

    def gated_scandir(path: int | str | bytes | os.PathLike[str] | os.PathLike[bytes]):
        if not isinstance(path, int) and Path(path) == workspace:
            entered.set()
            assert release.wait(10), "path scan gate was not released"
            worker_finished.set()
        assert original_scandir is not None
        return original_scandir(path)

    try:
        hello = await _hello(
            reader,
            writer,
            features=["path_completion", "serve_observers"],
        )
        assert "path_completion" in hello["result"]["capabilities"]["features"]
        path_module = importlib.import_module("zeta.core.path_completion")
        original_scandir = path_module.os.scandir
        # test-contract: seam blocks the actual synchronous path scanner thread.
        monkeypatch.setattr(path_module.os, "scandir", gated_scandir)
        created = (await _request(reader, writer, "new", "new_session"))[-1]
        session_id = created["result"]["session"]["session_id"]
        observer_reader, observer_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _hello(
            observer_reader,
            observer_writer,
            role="observer",
            features=["serve_observers"],
        )

        writer.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": "cancelled-path",
                    "method": "path_complete",
                    "params": {
                        "session_id": session_id,
                        "text_before_cursor": "@f",
                    },
                }
            ).encode()
            + b"\n"
        )
        await await_or_fail(writer.drain(), what="path request write")
        assert await await_or_fail(
            asyncio.to_thread(entered.wait, 10), what="path scan gate"
        )

        status = await await_or_fail(
            _request(observer_reader, observer_writer, "live", "status"),
            what="second connection status while path scan is blocked",
        )
        assert status[-1]["id"] == "live"
        assert not release.is_set()

        client = server._client
        assert client is not None
        request_task = next(
            task
            for task in asyncio.all_tasks()
            if task is not asyncio.current_task()
            and any(frame.f_locals.get("client") is client for frame in task.get_stack())
        )
        request_task.cancel()
        with suppress(asyncio.CancelledError):
            await await_or_fail(request_task, what="cancelled path request")
        release.set()
        assert await await_or_fail(
            asyncio.to_thread(worker_finished.wait, 10), what="path scanner drain"
        )

        frames = await _frames_until_eof(reader)
        assert not any(
            frame.get("id") == "cancelled-path"
            or frame.get("params", {}).get("event") is not None
            for frame in frames
        )
    finally:
        release.set()
        if request_task is not None and not request_task.done():
            request_task.cancel()
        await close_serve(server, observer_writer or writer)
