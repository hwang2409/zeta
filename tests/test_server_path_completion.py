from __future__ import annotations

import asyncio
import importlib
import json
import os
import threading
from contextlib import suppress
from pathlib import Path

import pytest
from prompt_toolkit.document import Document

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.serve_contract import ServeContract
from tests.support.sockets import short_socket_path
from tests.test_server import (
    _connect,
    _frames_until_eof,
    _request,
)
from zeta.server import ZetaServer
from zeta.tui.completion import PathCompleter

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
        assert response.get("result") is not None, response
        entries = response["result"]["entries"]

        assert "path_completion" in hello["result"]["capabilities"]["features"]
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
        invalid_unicode = (
            await _request(
                reader,
                writer,
                "invalid-unicode",
                "path_complete",
                {"session_id": session_id, "text_before_cursor": "\ud800"},
            )
        )[-1]
        assert invalid_unicode["error"]["code"] == -32602
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
    negotiated_root = tmp_path / "negotiated"
    workspace = negotiated_root / "workspace"
    workspace.mkdir(parents=True)
    server = _server(negotiated_root, workspace)
    reader, writer = await _connect(server)
    observer_reader = observer_writer = None
    try:
        hello = await _hello(
            reader,
            writer,
            features=["serve_observers", "path_completion"],
        )
        created = (await _request(reader, writer, "new", "new_session"))[-1]
        session_id = created["result"]["session"]["session_id"]
        completed = (
            await _request(
                reader,
                writer,
                "complete",
                "path_complete",
                {"session_id": session_id, "text_before_cursor": "plain text"},
            )
        )[-1]
        assert completed.get("result") == {"entries": [], "truncated": False}, completed
        assert "path_completion" in hello["result"]["capabilities"]["features"]

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

    unnegotiated_root = tmp_path / "unnegotiated"
    unnegotiated_workspace = unnegotiated_root / "workspace"
    unnegotiated_workspace.mkdir(parents=True)
    server = _server(unnegotiated_root, unnegotiated_workspace)
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer)
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
    finally:
        await close_serve(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("transport", ["unix", "websocket"])
async def test_path_complete_access_and_negotiation_over_each_transport(
    tmp_path: Path, transport: str
) -> None:
    root = tmp_path / transport
    workspace = root / "workspace"
    workspace.mkdir(parents=True)
    contract = await ServeContract.start(root, transport)  # type: ignore[arg-type]
    client = await contract.connect()
    observer = None
    try:
        hello = await client.request(
            "hello", "hello", {"protocol_version": "1.2", "features": ["path_completion", "serve_observers"]}
        )
        created = await client.request("new", "new_session")
        session_id = created["result"]["session"]["session_id"]
        completed = await client.request(
            "complete", "path_complete", {"session_id": session_id, "text_before_cursor": "plain text"}
        )
        assert completed["result"] == {"entries": [], "truncated": False}
        assert "path_completion" in hello["result"]["capabilities"]["features"]

        observer = await contract.connect("observer")
        await observer.request(
            "observer-hello", "hello", {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers", "path_completion"]}
        )
        rejected = await observer.request(
            "observer-path", "path_complete", {"session_id": session_id, "text_before_cursor": "@"}
        )
        assert rejected["error"]["code"] == -32008
    finally:
        await contract.close()

    contract = await ServeContract.start(root / "unnegotiated", transport)  # type: ignore[arg-type]
    client = await contract.connect()
    try:
        await client.request("hello", "hello", {"protocol_version": "1.2", "features": []})
        created = await client.request("new", "new_session")
        session_id = created["result"]["session"]["session_id"]
        rejected = await client.request(
            "missing-feature", "path_complete", {"session_id": session_id, "text_before_cursor": "@"}
        )
        assert rejected["error"]["code"] == -32601
    finally:
        await contract.close()


@pytest.mark.asyncio
async def test_path_complete_symlink_matches_tui_without_outside_traversal(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (outside / "secret.txt").touch()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    server = _server(tmp_path, workspace)
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, features=["path_completion"])
        created = (await _request(reader, writer, "new", "new_session"))[-1]
        session_id = created["result"]["session"]["session_id"]
        result = (
            await _request(
                reader, writer, "link", "path_complete", {"session_id": session_id, "text_before_cursor": "@"}
            )
        )[-1]["result"]
        assert [entry["text"] for entry in result["entries"]] == ["link/"]
        assert list(PathCompleter(workspace).get_completions(Document("@link/"), None)) == []
        served = (
            await _request(
                reader, writer, "outside", "path_complete", {"session_id": session_id, "text_before_cursor": "@link/"}
            )
        )[-1]["result"]
        assert served == {"entries": [], "truncated": False}
    finally:
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
    scanner_observed_cancel = threading.Event()
    path_module = None
    original_scandir = None
    original_complete_paths = None

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
        created = (await _request(reader, writer, "new", "new_session"))[-1]
        session_id = created["result"]["session"]["session_id"]
        probe = (
            await _request(
                reader,
                writer,
                "probe-path",
                "path_complete",
                {"session_id": session_id, "text_before_cursor": "plain text"},
            )
        )[-1]
        assert probe.get("result") == {"entries": [], "truncated": False}, probe
        assert "path_completion" in hello["result"]["capabilities"]["features"]
        path_module = importlib.import_module("zeta.core.path_completion")
        handler_module = importlib.import_module("zeta.server.handlers.path_completion")
        original_scandir = path_module.os.scandir
        original_complete_paths = handler_module.complete_paths

        def tracked_complete_paths(*args, **kwargs):
            result = original_complete_paths(*args, **kwargs)
            cancel_event = kwargs.get("cancel_event")
            if cancel_event is not None and cancel_event.is_set():
                scanner_observed_cancel.set()
            worker_finished.set()
            return result

        # test-contract: seam blocks the actual synchronous path scanner thread.
        monkeypatch.setattr(path_module.os, "scandir", gated_scandir)
        monkeypatch.setattr(handler_module, "complete_paths", tracked_complete_paths)
        observer_reader, observer_writer = await asyncio.open_unix_connection(
            str(server.socket_path)
        )
        await _hello(
            observer_reader,
            observer_writer,
            role="observer",
            features=["serve_observers"],
        )

        # test-contract: transport cancelled before any response; a response-reading helper would hang.
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
        await asyncio.sleep(0)
        assert not request_task.done()
        release.set()
        with suppress(asyncio.CancelledError):
            await await_or_fail(request_task, what="cancelled path request")
        assert await await_or_fail(
            asyncio.to_thread(worker_finished.wait, 10), what="path scanner drain"
        )
        assert scanner_observed_cancel.is_set()

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
