from __future__ import annotations

import asyncio
import fcntl
import json
import os
import random
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.test_server import _close, _connect, _request
from zeta.project_inbox import ProjectInbox, ProjectInboxScanner
from zeta.project_registry import ProjectRegistry
from zeta.server import ZetaServer, slash_commands
from zeta.server.project_requests import ProjectRequests
from zeta.server.protocol import FrameCodec
from zeta.server.slash_commands import run_command


def _server(tmp_path: Path) -> ZetaServer:
    cwd = tmp_path / "launch"
    cwd.mkdir(parents=True)
    return ZetaServer(
        home=tmp_path / "home",
        cwd=cwd,
        provider="codex",
        port=0,
        backend_factory=lambda provider, model, home: (
            FakeBackend([]),
            model or "offline",
        ),
    )


async def _hello(reader, writer, features: list[str] | None = None) -> dict:
    params: dict[str, object] = {"protocol_version": "1.0", "client_version": "1.1"}
    if features is not None:
        params["features"] = features
    return (await _request(reader, writer, 1, "hello", params))[-1]["result"]


def _mtimes(root: Path) -> dict[str, int]:
    return {
        str(path.relative_to(root)): path.stat(follow_symlinks=False).st_mtime_ns
        for path in root.rglob("*")
    }


async def _await_setup_reapers(server) -> None:
    """Wait (bounded) for the runtime-owned cleanup of canceled session setups."""
    reapers = list(server.runtime._setup_reapers)
    assert reapers, "a canceled setup must be handed to a reaper"
    await asyncio.wait_for(asyncio.gather(*reapers), 2)


def _hold_registry_lock(
    path: Path,
) -> tuple[threading.Event, threading.Event, threading.Thread]:
    acquired = threading.Event()
    release = threading.Event()

    def hold() -> None:
        fd = os.open(path, os.O_RDWR)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            acquired.set()
            release.wait(2)
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    thread = threading.Thread(target=hold)
    thread.start()
    assert acquired.wait(2)
    return acquired, release, thread


def _wait_for_path(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        time.sleep(0.01)
    return path.exists()


def _filesystem_snapshot(root: Path) -> dict[str, tuple[int, int, int]]:
    paths = [root, *root.rglob("*")] if root.exists() else []
    return {
        "." if path == root else str(path.relative_to(root)): (
            path.stat(follow_symlinks=False).st_size,
            path.stat(follow_symlinks=False).st_mtime_ns,
            path.stat(follow_symlinks=False).st_mode,
        )
        for path in paths
    }


@pytest.mark.asyncio
async def test_new_session_promotes_healthy_cwd_project_after_malformed_probe(
    tmp_path: Path,
) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "healthy"
    unrelated = tmp_path / "unrelated"
    cwd.mkdir()
    unrelated.mkdir()
    await asyncio.to_thread(subprocess.run, ["git", "init", "-q"], cwd=cwd, check=True)
    registry = ProjectRegistry(home / "projects")
    healthy = registry.create_project("healthy", "git", cwd)
    malformed = registry.create_project("malformed", "git", unrelated)
    (registry.root / malformed.project_id / "project.json").write_text(
        "{", encoding="utf-8"
    )
    server = ZetaServer(
        home=home,
        cwd=cwd,
        provider="codex",
        port=0,
        backend_factory=lambda provider, model, home: (
            FakeBackend([]),
            model or "offline",
        ),
    )
    try:
        await server.start()
        assert server.runtime.manager.project_registry_unavailable is not None

        metadata = await server.runtime.create_session()

        assert metadata.project_id == healthy.project_id
        assert server.runtime.loop.root_project_id == healthy.project_id
    finally:
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["new_session", "resume"])
async def test_session_project_preparation_does_not_block_ping(
    tmp_path: Path, method: str
) -> None:
    server = _server(tmp_path)
    root = server.runtime.cwd
    await asyncio.to_thread(subprocess.run, ["git", "init", "-q"], cwd=root, check=True)
    registry = ProjectRegistry(server.home / "projects")
    registry.create_project("launch", "git", root)
    try:
        if method == "resume":
            first = await server.runtime.create_session()
            await server.runtime.create_session()
            request = {"session_id": first.session_id}
        else:
            request = {}
        _acquired, release, holder = _hold_registry_lock(
            server.home / "projects" / ".lock"
        )
        operation = asyncio.create_task(
            server.runtime.create_session()
            if method == "new_session"
            else server.runtime.resume_session(request["session_id"])
        )
        progress = asyncio.create_task(asyncio.sleep(0))
        await progress
        assert not operation.done()
        release.set()
        holder.join(2)
        await operation
    finally:
        if "release" in locals():
            release.set()
            holder.join(2)
        await server.close()


@pytest.mark.asyncio
async def test_serve_resume_cancel_closes_store_after_preparation_finishes(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    first = await server.runtime.create_session()
    await server.runtime.create_session()
    reader, writer = await _connect(server)
    await _hello(reader, writer)
    client = server._client
    assert client is not None
    opened: list[object] = []
    preparation_started = threading.Event()
    preparation_release = threading.Event()
    preparation_done = threading.Event()
    preparation_active: set[int] = set()
    original_open = server.runtime.manager.open
    original_prepare = server.runtime._prepare_resumed_session

    def tracked_open(session_id: str):
        result = original_open(session_id)
        opened.append(result.store)
        return result

    def tracked_prepare(opened_session, session_cwd):
        ident = threading.get_ident()
        preparation_active.add(ident)
        preparation_started.set()
        try:
            assert preparation_release.wait(2)
            return original_prepare(opened_session, session_cwd)
        finally:
            preparation_active.discard(ident)
            preparation_done.set()

    server.runtime.manager.open = tracked_open
    server.runtime._prepare_resumed_session = tracked_prepare
    operation = asyncio.create_task(
        client._dispatch(2, "resume", {"session_id": first.session_id})
    )
    try:
        assert await asyncio.to_thread(preparation_started.wait, 2)
        operation.cancel()
        preparation_release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        await _await_setup_reapers(server)
        assert await asyncio.to_thread(preparation_done.wait, 2)
        assert opened and opened[0]._closed
        assert not preparation_active
    finally:
        preparation_release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_resume_cancel_reaper_waits_for_composition_before_closing_store(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    first = await server.runtime.create_session()
    await server.runtime.create_session()
    reader, writer = await _connect(server)
    await _hello(reader, writer)
    client = server._client
    assert client is not None
    opened: list[object] = []
    compose_started = threading.Event()
    compose_done = threading.Event()
    worker_saw_closed: list[bool] = []
    original_open = server.runtime.manager.open
    original_compose = server.runtime._compose

    def tracked_open(session_id: str):
        result = original_open(session_id)
        opened.append(result.store)
        return result

    def tracked_compose(**kwargs):
        compose_started.set()
        time.sleep(0.2)
        worker_saw_closed.append(kwargs["opened"].store._closed)
        try:
            return original_compose(**kwargs)
        finally:
            compose_done.set()

    server.runtime.manager.open = tracked_open
    server.runtime._compose = tracked_compose
    close_count = 0
    original_close = None
    operation = asyncio.create_task(
        client._dispatch(2, "resume", {"session_id": first.session_id})
    )
    try:
        assert await asyncio.to_thread(compose_started.wait, 2)
        assert opened
        original_close = opened[0].close

        def tracked_close():
            nonlocal close_count
            close_count += 1
            return original_close()

        opened[0].close = tracked_close
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        await _await_setup_reapers(server)
        assert compose_done.is_set()
        assert worker_saw_closed == [False]
        assert close_count == 1
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_create_cancel_reaps_after_compose_finishes(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    reader, writer = await _connect(server)
    await _hello(reader, writer)
    client = server._client
    assert client is not None
    compose_started = threading.Event()
    compose_release = threading.Event()
    compose_done = threading.Event()
    worker_saw_closed: list[bool] = []
    compositions: list[object] = []
    close_count = 0
    original_compose = server.runtime._compose

    def tracked_compose(**kwargs):
        nonlocal close_count
        compose_started.set()
        assert compose_release.wait(2)
        composition = original_compose(**kwargs)
        original_close = composition.opened.store.close

        def tracked_close():
            nonlocal close_count
            close_count += 1
            return original_close()

        composition.opened.store.close = tracked_close
        compositions.append(composition)
        worker_saw_closed.append(composition.opened.store._closed)
        compose_done.set()
        return composition

    server.runtime._compose = tracked_compose
    operation = asyncio.create_task(client._dispatch(2, "new_session", {}))
    try:
        assert await asyncio.to_thread(compose_started.wait, 2)
        operation.cancel()
        compose_release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 2)
        await _await_setup_reapers(server)
        assert compose_done.is_set()
        assert compositions
        store = compositions[0].opened.store
        assert store._closed
        assert close_count == 1
        assert worker_saw_closed == [False]
        assert server.runtime.state is None
    finally:
        compose_release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_resume_cancel_reaper_closes_store_after_compose(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    first = await server.runtime.create_session()
    await server.runtime.create_session()
    previous = server.runtime.session_id
    reader, writer = await _connect(server)
    await _hello(reader, writer)
    client = server._client
    assert client is not None
    compose_started = threading.Event()
    compose_release = threading.Event()
    worker_saw_closed: list[bool] = []
    opened: list[object] = []
    close_count = 0
    original_open = server.runtime.manager.open
    original_compose = server.runtime._compose

    def tracked_open(session_id: str):
        result = original_open(session_id)
        opened.append(result.store)
        original_close = result.store.close

        def tracked_close():
            nonlocal close_count
            close_count += 1
            return original_close()

        result.store.close = tracked_close
        return result

    def tracked_compose(**kwargs):
        compose_started.set()
        assert compose_release.wait(2)
        composition = original_compose(**kwargs)
        worker_saw_closed.append(composition.opened.store._closed)
        return composition

    server.runtime.manager.open = tracked_open
    server.runtime._compose = tracked_compose
    operation = asyncio.create_task(
        client._dispatch(2, "resume", {"session_id": first.session_id})
    )
    try:
        assert await asyncio.to_thread(compose_started.wait, 2)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 2)
        assert not compose_release.is_set()
        assert len(server.runtime._setup_reapers) == 1
        reaper = next(iter(server.runtime._setup_reapers))
        compose_release.set()
        await asyncio.wait_for(reaper, 2)
        assert opened and opened[0]._closed
        assert close_count == 1
        assert worker_saw_closed == [False]
        assert server.runtime.session_id == previous
    finally:
        compose_release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_create_cancel_reaper_closes_store_after_compose(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    reader, writer = await _connect(server)
    await _hello(reader, writer)
    client = server._client
    assert client is not None
    compose_started = threading.Event()
    compose_release = threading.Event()
    worker_saw_closed: list[bool] = []
    compositions: list[object] = []
    close_count = 0
    original_compose = server.runtime._compose

    def tracked_compose(**kwargs):
        nonlocal close_count
        compose_started.set()
        assert compose_release.wait(2)
        composition = original_compose(**kwargs)
        original_close = composition.opened.store.close

        def tracked_close():
            nonlocal close_count
            close_count += 1
            return original_close()

        composition.opened.store.close = tracked_close
        compositions.append(composition)
        worker_saw_closed.append(composition.opened.store._closed)
        return composition

    server.runtime._compose = tracked_compose
    operation = asyncio.create_task(client._dispatch(2, "new_session", {}))
    try:
        assert await asyncio.to_thread(compose_started.wait, 2)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 2)
        assert not compose_release.is_set()
        assert len(server.runtime._setup_reapers) == 1
        reaper = next(iter(server.runtime._setup_reapers))
        compose_release.set()
        await asyncio.wait_for(reaper, 2)
        assert compositions and compositions[0].opened.store._closed
        assert close_count == 1
        assert worker_saw_closed == [False]
        assert server.runtime.state is None
    finally:
        compose_release.set()
        await _close(server, writer)


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["new_session", "resume"])
async def test_serve_repeated_cancel_still_hands_setup_to_reaper(
    tmp_path: Path, method: str
) -> None:
    server = _server(tmp_path)
    params: dict[str, object] = {}
    if method == "resume":
        created = await server.runtime.create_session()
        await server.runtime.create_session()
        params = {"session_id": created.session_id}
    reader, writer = await _connect(server)
    await _hello(reader, writer)
    client = server._client
    assert client is not None
    compose_started = threading.Event()
    compose_release = threading.Event()
    compositions: list[object] = []
    original_compose = server.runtime._compose

    def tracked_compose(**kwargs):
        compose_started.set()
        assert compose_release.wait(5)
        composition = original_compose(**kwargs)
        compositions.append(composition)
        return composition

    server.runtime._compose = tracked_compose
    operation = asyncio.create_task(client._dispatch(2, method, params))
    try:
        assert await asyncio.to_thread(compose_started.wait, 2)
        operation.cancel()
        for _ in range(5):
            await asyncio.sleep(0)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 2)
        assert not compose_release.is_set()
        assert len(server.runtime._setup_reapers) == 1
        reaper = next(iter(server.runtime._setup_reapers))
        compose_release.set()
        await asyncio.wait_for(reaper, 2)
        assert len(compositions) == 1
        assert compositions[0].opened.store._closed
    finally:
        compose_release.set()
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_shutdown_reports_blocked_setup_reaper(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    server = _server(tmp_path)
    first = await server.runtime.create_session()
    await server.runtime.create_session()
    reader, writer = await _connect(server)
    await _hello(reader, writer)
    client = server._client
    assert client is not None
    compose_started = threading.Event()
    compose_release = threading.Event()
    compositions: list[object] = []
    close_count = 0
    original_compose = server.runtime._compose

    def tracked_compose(**kwargs):
        nonlocal close_count
        compose_started.set()
        assert compose_release.wait(10)
        composition = original_compose(**kwargs)
        original_close = composition.opened.store.close

        def tracked_close():
            nonlocal close_count
            close_count += 1
            return original_close()

        composition.opened.store.close = tracked_close
        compositions.append(composition)
        return composition

    server.runtime._compose = tracked_compose
    operation = asyncio.create_task(
        client._dispatch(2, "resume", {"session_id": first.session_id})
    )
    shutdown = False
    try:
        assert await asyncio.to_thread(compose_started.wait, 2)
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(operation, 2)
        assert not compose_release.is_set()
        await asyncio.wait_for(server.close(), 2)
        shutdown = True
        assert any(
            "abandoned session setup task(s) still running at shutdown"
            in record.message
            for record in caplog.records
        )
        compose_release.set()
        reaper = next(iter(server.runtime._setup_reapers))
        await asyncio.wait_for(reaper, 2)
        assert compositions and compositions[0].opened.store._closed
        assert close_count == 1
        assert server.runtime.state is None
    finally:
        compose_release.set()
        if not shutdown:
            await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_project_instructions_does_not_block_loop_tick(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    root = server.runtime.cwd
    await asyncio.to_thread(subprocess.run, ["git", "init", "-q"], cwd=root, check=True)
    registry = ProjectRegistry(server.home / "projects")
    registry.create_project("launch", "git", root)
    await server.runtime.create_session()
    try:
        _acquired, release, holder = _hold_registry_lock(
            server.home / "projects" / ".lock"
        )
        operation = asyncio.create_task(
            run_command(server.runtime, "/project-instructions")
        )
        progress = asyncio.create_task(asyncio.sleep(0))
        await progress
        assert not operation.done()
        release.set()
        holder.join(2)
        result = await operation
        assert result["kind"] == "output"
    finally:
        if "release" in locals():
            release.set()
            holder.join(2)
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["new_session", "resume", "slash_run"])
async def test_serve_blocking_git_does_not_block_ping(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    server = _server(tmp_path)
    root = server.runtime.cwd
    await asyncio.to_thread(subprocess.run, ["git", "init", "-q"], cwd=root, check=True)
    registry = ProjectRegistry(server.home / "projects")
    registry.create_project("launch", "git", root)
    first = await server.runtime.create_session()
    if method == "resume":
        await server.runtime.create_session()
    real_git = shutil.which("git")
    assert real_git is not None
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    git_started = tmp_path / "git-started"
    git_release = tmp_path / "git-release"
    (fake_bin / "git").write_text(
        f'#!/bin/sh\ntouch {git_started}\nwhile [ ! -e {git_release} ]; do sleep 0.01; done\nexec {real_git} "$@"\n',
        encoding="utf-8",
    )
    (fake_bin / "git").chmod(0o755)
    monkeypatch.setenv("PATH", f"{fake_bin}{os.pathsep}{os.environ['PATH']}")
    reader, writer = await _connect(server)
    release = None
    holder = None
    try:
        await _hello(reader, writer, ["ping"])
        event_loop_thread = threading.get_ident()
        if method == "slash_run":
            registry_started = threading.Event()
            registry_release = threading.Event()
            registry_thread: list[int] = []
            original_build_registry = slash_commands.build_registry

            def blocking_build_registry(runtime):
                registry_thread.append(threading.get_ident())
                registry_started.set()
                registry_release.wait(1)
                return original_build_registry(runtime)

            monkeypatch.setattr(
                slash_commands, "build_registry", blocking_build_registry
            )
        client = server._client
        assert client is not None
        _acquired, release, holder = _hold_registry_lock(
            server.home / "projects" / ".lock"
        )
        params = (
            {}
            if method == "new_session"
            else {"session_id": first.session_id}
            if method == "resume"
            else {"text": "/project-instructions", "session_id": first.session_id}
        )
        request_method = {
            "new_session": "new_session",
            "resume": "resume",
            "slash_run": "slash_run",
        }[method]
        operation = asyncio.create_task(client._dispatch(2, request_method, params))
        await asyncio.to_thread(_wait_for_path, git_started, 1)
        if method == "slash_run":
            assert await asyncio.to_thread(registry_started.wait, 1)
        assert not operation.done()
        ping = asyncio.create_task(client._dispatch(3, "ping", {}))
        assert await asyncio.wait_for(ping, 0.25) == {"pong": True}
        git_release.touch()
        if method == "slash_run":
            registry_release.set()
        release.set()
        holder.join(2)
        await asyncio.wait_for(operation, 8)
        if method == "slash_run":
            assert registry_thread and registry_thread[0] != event_loop_thread
    finally:
        if release is not None:
            release.set()
        if holder is not None:
            holder.join(2)
        await _close(server, writer)


@pytest.mark.asyncio
async def test_serve_shutdown_with_registry_reader_held_exits_bounded(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    ProjectRegistry(server.home / "projects").create_project("launch", "repo")
    reader, writer = await _connect(server)
    await _hello(reader, writer, ["projects"])
    _acquired, release, holder = _hold_registry_lock(server.home / "projects" / ".lock")
    request = asyncio.create_task(_request(reader, writer, 2, "list_projects"))
    try:
        await asyncio.sleep(0.05)
        await asyncio.wait_for(server.close(), 2)
        assert holder.is_alive()
    finally:
        release.set()
        holder.join(2)
        await asyncio.gather(request, return_exceptions=True)


@pytest.mark.asyncio
async def test_project_requests_run_registry_reads_off_event_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    registry.update_memory(project.project_id, {"brief.md": "content\n"})
    calls: list[tuple[str, int]] = []
    original_list = ProjectRegistry.list_projects
    original_memory_log = ProjectRegistry.memory_log
    original_read = ProjectInbox.read

    def list_projects(self: ProjectRegistry):
        calls.append(("list", threading.get_ident()))
        return original_list(self)

    def memory_log(self: ProjectRegistry, project_id: str, **kwargs: object):
        calls.append(("memory", threading.get_ident()))
        return original_memory_log(self, project_id, **kwargs)

    def read(self: ProjectInbox, project_id: str):
        calls.append(("inbox", threading.get_ident()))
        return original_read(self, project_id)

    monkeypatch.setattr(ProjectRegistry, "list_projects", list_projects)
    monkeypatch.setattr(ProjectRegistry, "memory_log", memory_log)
    monkeypatch.setattr(ProjectInbox, "read", read)
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        event_loop_thread = threading.get_ident()
        await _request(reader, writer, 2, "list_projects")
        await _request(
            reader, writer, 3, "project_memory_log", {"project_id": project.project_id}
        )
        await _request(
            reader, writer, 4, "project_inbox", {"project_id": project.project_id}
        )
    finally:
        await _close(server, writer)
    assert {kind for kind, _thread in calls} >= {"list", "memory", "inbox"}
    assert all(thread != event_loop_thread for _kind, thread in calls)


@pytest.mark.asyncio
async def test_project_requests_do_not_modify_filesystem(tmp_path: Path) -> None:
    empty_server = _server(tmp_path / "empty")
    reader, writer = await _connect(empty_server)
    try:
        await _hello(reader, writer, ["projects"])
        before = _filesystem_snapshot(empty_server.home)
        response = await _request(reader, writer, 2, "list_projects")
        assert response[-1]["result"]["projects"] == []
        assert _filesystem_snapshot(empty_server.home) == before
    finally:
        await _close(empty_server, writer)

    server = _server(tmp_path / "populated")
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    sender = registry.create_project("sender", "repo")
    registry.update_memory(project.project_id, {"brief.md": "initial\n"})
    state = registry.compare_and_swap_memory(
        project.project_id,
        expected_digest=registry.memory_digest(project.project_id),
        updates={"brief.md": "authoritative\n"},
        provenance={"session_id": "c" * 32, "seq_start": 1, "seq_end": 2},
    )
    mirror = server.home / "projects" / project.project_id / "memory" / "brief.md"
    header = mirror.read_text().splitlines(keepends=True)[0]
    mirror_mode = mirror.stat().st_mode
    mirror.chmod(0o600)
    mirror.write_text(header + "stale mirror\n")
    mirror.chmod(mirror_mode)
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    message_id = inbox.send(
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=project.project_id,
        kind="info",
        title="stale claim",
        body="body",
    )
    assert inbox.claim(project.project_id, message_id, "b" * 32) is not None
    registry_lock = server.home / "projects" / ".lock"
    registry_lock.unlink()

    requests = [
        ("list_projects", {}),
        ("project_show", {"project_id": project.project_id}),
        ("project_memory_log", {"project_id": project.project_id}),
        (
            "project_memory_log",
            {
                "project_id": project.project_id,
                "version_id": state.version,
                "file": "brief.md",
            },
        ),
        ("list_sessions", {"project_id": project.project_id}),
        ("project_inbox", {"project_id": project.project_id, "status": "claimed"}),
    ]
    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects", "list_sessions_paging"])
        for request_id, (method, params) in enumerate(requests, 2):
            before = _filesystem_snapshot(server.home)
            response = await _request(reader, writer, request_id, method, params)
            assert "result" in response[-1]
            assert _filesystem_snapshot(server.home) == before, method
        assert mirror.read_text() == header + "stale mirror\n"
        inbox_response = response[-1]["result"]
        assert [item["id"] for item in inbox_response["messages"]] == [message_id]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_request_error_classification(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    sender = registry.create_project("sender", "repo")
    registry.update_memory(project.project_id, {"brief.md": "content\n"})
    ProjectInbox(registry, sessions_root=server.home / "sessions").send(
        from_project=sender.project_id,
        from_session="d" * 32,
        to_project=project.project_id,
        kind="info",
        title="message",
        body="body",
    )
    inbox_new = server.home / "projects" / project.project_id / "inbox" / "new"
    (inbox_new / "unexpected.txt").write_text("malformed storage\n")
    pointer = server.home / "projects" / project.project_id / "memory-current.json"
    pointer.write_text("{not-json\n")

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        malformed = await _request(
            reader, writer, 2, "project_show", {"project_id": project.project_id}
        )
        assert malformed[-1]["error"] == {
            "code": -32000,
            "message": "project storage is invalid or unavailable",
        }
        malformed_inbox = await _request(
            reader, writer, 3, "project_inbox", {"project_id": project.project_id}
        )
        inbox_result = malformed_inbox[-1]["result"]
        assert len(inbox_result["messages"]) == 1
        assert inbox_result["invalid"] == [
            {
                "filename": "unexpected.txt",
                "reason": "unexpected inbox file: unexpected.txt",
                "status": "new",
            }
        ]
        unknown_id = "p_" + "f" * 32
        unknown = await _request(
            reader, writer, 4, "project_show", {"project_id": unknown_id}
        )
        assert unknown[-1]["error"] == {
            "code": -32602,
            "message": f"project not found: {unknown_id}",
            "data": {"code": "project_not_found", "project_id": unknown_id},
        }
        invalid = await _request(reader, writer, 5, "list_projects", {"limit": 0})
        assert invalid[-1]["error"] == {
            "code": -32602,
            "message": "limit is out of range",
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_inbox_rejects_missing_required_directory(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    sender = registry.create_project("sender", "repo")
    ProjectInbox(registry, sessions_root=server.home / "sessions").send(
        from_project=sender.project_id,
        from_session="d" * 32,
        to_project=project.project_id,
        kind="info",
        title="message",
        body="body",
    )
    (registry.root / project.project_id / "inbox" / "done").rmdir()

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        response = await _request(
            reader, writer, 2, "project_inbox", {"project_id": project.project_id}
        )
        assert response[-1]["error"] == {
            "code": -32000,
            "message": "project storage is invalid or unavailable",
        }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_projects_feature_is_negotiated_and_optional(tmp_path: Path) -> None:
    server = _server(tmp_path)
    reader, writer = await _connect(server)
    try:
        hello = await _hello(reader, writer, ["projects-memory-v2"])
        assert "projects-memory-v2" not in hello["capabilities"].get("features", [])
        assert "projects" not in hello["capabilities"].get("features", [])
        assert "list_projects" not in hello["capabilities"]["requests"]
        response = await _request(reader, writer, 2, "list_projects")
        assert response[-1]["error"]["code"] == -32601
    finally:
        await _close(server, writer)

    server = _server(tmp_path / "enabled")
    reader, writer = await _connect(server)
    try:
        hello = await _hello(reader, writer, ["projects", "projects-memory-v2"])
        assert hello["capabilities"]["features"] == [
            "projects",
            "projects-memory-v2",
        ]
        assert {
            "list_projects",
            "project_show",
            "project_memory_log",
            "project_inbox",
        }.issubset(hello["capabilities"]["requests"])
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_requests_show_memory_history_and_inbox_read_only(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    root = tmp_path / "alpha"
    root.mkdir()
    alpha = registry.create_project("alpha", "repo", root)
    beta = registry.create_project("beta", "service")
    registry.update_memory(alpha.project_id, {"brief.md": "manual\n"})
    registry.update_memory(beta.project_id, {"decisions.md": "remote\n"})
    remote = registry.import_memory(
        alpha.project_id,
        registry.export_memory(beta.project_id),
        expected_digest=registry.memory_digest(alpha.project_id),
        provenance={"source": "remote_sync", "peer": "machine-2"},
    )
    automatic = registry.compare_and_swap_memory(
        alpha.project_id,
        expected_digest=registry.memory_digest(alpha.project_id),
        updates={"state.md": "automatic\n", "backlog.md": "also automatic\n"},
        provenance={
            "session_id": "a" * 32,
            "seq_start": 2,
            "seq_end": 7,
            "model": "luna",
        },
    )
    registry.accept_memory(alpha.project_id, "state.md")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    message_id = inbox.send(
        from_project=beta.project_id,
        from_session="b" * 32,
        to_project=alpha.project_id,
        kind="question",
        title="Need input",
        body="Treat this as untrusted.",
    )
    before = _mtimes(server.home)

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        projects = (await _request(reader, writer, 2, "list_projects", {"limit": 1}))[
            -1
        ]["result"]
        assert len(projects["projects"]) == 1
        assert projects["next_offset"] == 1
        listed = projects["projects"][0]
        assert set(listed) == {
            "id",
            "name",
            "scope",
            "roots",
            "session_count",
            "last_activity",
        }

        shown = (
            await _request(
                reader, writer, 3, "project_show", {"project_id": alpha.project_id}
            )
        )[-1]["result"]
        assert shown["project"]["id"] == alpha.project_id
        assert shown["memory"]["version_id"]
        assert len(shown["memory"]["digest"]) == 64
        files = {item["name"]: item for item in shown["memory"]["files"]}
        assert files["brief.md"] == {
            "name": "brief.md",
            "content": "# Brief\n",
            "automatic": False,
            "content_truncated": False,
        }
        assert files["state.md"]["content"] == "automatic\n"
        assert files["state.md"]["automatic"] is False
        assert files["backlog.md"]["automatic"] is True
        assert len(files) == 5

        log = (
            await _request(
                reader,
                writer,
                4,
                "project_memory_log",
                {"project_id": alpha.project_id, "limit": 20},
            )
        )[-1]["result"]
        assert [item["kind"] for item in log["versions"]][-3:] == [
            "import",
            "update",
            "accept",
        ]
        imported = next(
            item for item in log["versions"] if item["version_id"] == remote.version
        )
        assert imported["provenance"] == {"source": "remote_sync", "peer": "machine-2"}
        update = next(
            item for item in log["versions"] if item["version_id"] == automatic.version
        )
        assert update["provenance"] == {
            "session_id": "a" * 32,
            "seq_start": 2,
            "seq_end": 7,
            "model": "luna",
        }
        accepted = log["versions"][-1]
        assert accepted["provenance"] == {"accepted_by": "user"}

        detail = (
            await _request(
                reader,
                writer,
                5,
                "project_memory_log",
                {
                    "project_id": alpha.project_id,
                    "version_id": automatic.version,
                    "file": "state.md",
                },
            )
        )[-1]["result"]
        assert detail["version"]["content"] == "automatic\n"
        assert detail["version"]["content_truncated"] is False
        assert (
            "-" in detail["version"]["diff"]
            and "+automatic" in detail["version"]["diff"]
        )
        assert detail["version"]["diff_truncated"] is False
        missing = await _request(
            reader,
            writer,
            "missing-version",
            "project_memory_log",
            {
                "project_id": alpha.project_id,
                "version_id": "f" * 32,
                "file": "state.md",
            },
        )
        assert missing[-1]["error"]["code"] == -32602

        result = (
            await _request(
                reader,
                writer,
                6,
                "project_inbox",
                {"project_id": alpha.project_id, "status": "new"},
            )
        )[-1]["result"]
        assert result == {
            "status": "new",
            "messages": [result["messages"][0]],
            "untrusted": False,
            "invalid": [],
            "next_offset": None,
        }
        assert result["messages"][0]["id"] == message_id
        assert result["messages"][0]["origin"] == "local"
        assert _mtimes(server.home) == before

        await _request(reader, writer, 7, "new_session", {})
        attached_before = _mtimes(server.home)
        attached = (
            await _request(
                reader, writer, 8, "project_show", {"project_id": alpha.project_id}
            )
        )[-1]["result"]
        assert attached["project"]["id"] == alpha.project_id
        assert _mtimes(server.home) == attached_before
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_list_sessions_filters_project_and_keeps_lineage(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    other = registry.create_project("beta", "repo")
    parent = await asyncio.to_thread(
        lambda: server.runtime.manager.create(
            provider="codex",
            model="offline",
            cwd=server.runtime.cwd,
            project_id=project.project_id,
            project_role="orchestrator",
            name="parent",
            auto_project=False,
        )
    )
    child = await asyncio.to_thread(
        lambda: server.runtime.manager.create(
            provider="codex",
            model="offline",
            cwd=server.runtime.cwd,
            project_id=project.project_id,
            project_role="worker",
            parent_session_id=parent.metadata.session_id,
            name="child",
            auto_project=False,
        )
    )
    await asyncio.to_thread(
        lambda: server.runtime.manager.create(
            provider="codex",
            model="offline",
            cwd=server.runtime.cwd,
            project_id=other.project_id,
            project_role="session",
            auto_project=False,
        )
    )
    parent.store.close()
    child.store.close()
    before = _mtimes(server.home)

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects", "list_sessions_paging"])
        result = (
            await _request(
                reader, writer, 2, "list_sessions", {"project_id": project.project_id}
            )
        )[-1]["result"]
        assert {item["name"] for item in result["sessions"]} == {"parent", "child"}
        child_wire = next(
            item for item in result["sessions"] if item["name"] == "child"
        )
        assert child_wire["project_role"] == "worker"
        assert child_wire["parent_session_id"] == parent.metadata.session_id
        projects = (await _request(reader, writer, 3, "list_projects"))[-1]["result"]
        summary = next(
            item for item in projects["projects"] if item["id"] == project.project_id
        )
        assert summary["session_count"] == 2
        assert summary["last_activity"] >= parent.metadata.updated_at
        invalid = await _request(
            reader, writer, 4, "list_sessions", {"project_id": None}
        )
        assert invalid[-1]["error"]["code"] == -32602
        assert _mtimes(server.home) == before
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_requests_bound_large_views_and_report_unknown_project(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("large", "repo")
    sender = registry.create_project("sender", "repo")
    for index, name in enumerate(
        ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")
    ):
        content = ("\x01" if index == 0 else str(index)) * (128 * 1024)
        registry.update_memory(project.project_id, {name: content})
    ProjectInbox(registry, sessions_root=server.home / "sessions").send(
        from_project=sender.project_id,
        from_session="c" * 32,
        to_project=project.project_id,
        kind="info",
        title="large message",
        body="x" * (2 * 1024 * 1024),
    )

    reader, writer = await _connect(server)
    reader._limit = 1_048_577  # type: ignore[attr-defined]  # protocol frame limit
    try:
        await _hello(reader, writer, ["projects"])
        shown_frames = await _request(
            reader, writer, 2, "project_show", {"project_id": project.project_id}
        )
        assert "result" in shown_frames[-1]
        assert len(str(shown_frames[-1]).encode()) < 1_048_576
        assert any(
            item["content_truncated"]
            for item in shown_frames[-1]["result"]["memory"]["files"]
        )
        log_frames = await _request(
            reader,
            writer,
            3,
            "project_memory_log",
            {"project_id": project.project_id, "limit": 100},
        )
        assert "result" in log_frames[-1]
        assert len(str(log_frames[-1]).encode()) < 1_048_576
        inbox_frames = await _request(
            reader, writer, 4, "project_inbox", {"project_id": project.project_id}
        )
        inbox_result = inbox_frames[-1]["result"]
        message = inbox_result["messages"][0]
        assert message["origin"] == "local"
        assert inbox_result["untrusted"] is False
        assert "body" in message["truncated_fields"]
        assert len(str(inbox_frames[-1]).encode()) < 1_048_576

        for method in ("project_show", "project_memory_log", "project_inbox"):
            frames = await _request(
                reader, writer, method, method, {"project_id": "p_" + "f" * 32}
            )
            assert frames[-1]["error"]["code"] == -32602
            assert frames[-1]["error"]["data"] == {
                "code": "project_not_found",
                "project_id": "p_" + "f" * 32,
            }
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_inbox_isolates_and_reports_invalid_messages(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    target = registry.create_project("target", "repo")
    sender = registry.create_project("sender", "repo")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    valid_id = inbox.send(
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=target.project_id,
        kind="info",
        title="valid",
        body="body",
    )
    corrupt_name = f"{'e' * 32}.json"
    corrupt_path = (
        server.home / "projects" / target.project_id / "inbox" / "new" / corrupt_name
    )
    corrupt_path.write_text("{not json")

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        result = (
            await _request(
                reader, writer, 2, "project_inbox", {"project_id": target.project_id}
            )
        )[-1]["result"]
        assert [message["id"] for message in result["messages"]] == [valid_id]
        assert result["invalid"] == [
            {
                "filename": corrupt_name,
                "reason": f"malformed inbox message: {corrupt_name}",
                "status": "new",
            }
        ]
    finally:
        await _close(server, writer)


@pytest.mark.asyncio
async def test_project_inbox_trust_is_derived_from_returned_page(
    tmp_path: Path,
) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    target = registry.create_project("target", "repo")
    sender = registry.create_project("sender", "repo")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    local_id = inbox.send(
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=target.project_id,
        kind="info",
        title="local",
        body="local body",
        message_id="0" * 32,
    )
    external_id = inbox.send(
        from_project=sender.project_id,
        from_session="b" * 32,
        to_project=target.project_id,
        kind="info",
        title="external",
        body="external body",
        message_id="f" * 32,
    )
    external_path = (
        server.home
        / "projects"
        / target.project_id
        / "inbox"
        / "new"
        / f"{external_id}.json"
    )
    external = json.loads(external_path.read_text())
    external["origin"] = "remote"
    external_path.write_text(
        json.dumps(external, sort_keys=True, separators=(",", ":")) + "\n"
    )

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        local_page = (
            await _request(
                reader,
                writer,
                2,
                "project_inbox",
                {"project_id": target.project_id, "limit": 1},
            )
        )[-1]["result"]
        assert [message["id"] for message in local_page["messages"]] == [local_id]
        assert local_page["messages"][0]["origin"] == "local"
        assert local_page["untrusted"] is False
        assert local_page["next_offset"] == 1

        external_page = (
            await _request(
                reader,
                writer,
                3,
                "project_inbox",
                {"project_id": target.project_id, "offset": 1, "limit": 1},
            )
        )[-1]["result"]
        assert [message["id"] for message in external_page["messages"]] == [external_id]
        assert external_page["messages"][0]["origin"] == "remote"
        assert external_page["untrusted"] is True
        assert external_page["next_offset"] is None
    finally:
        await _close(server, writer)


def test_project_inbox_omits_too_deep_unknown_field(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    sender = registry.create_project("sender", "repo")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    message_id = inbox.send(
        from_project=sender.project_id,
        from_session="d" * 32,
        to_project=project.project_id,
        kind="info",
        title="deep optional metadata",
        body="body",
    )
    path = registry.root / project.project_id / "inbox" / "new" / f"{message_id}.json"
    record = json.loads(path.read_text())
    nested: object = "leaf"
    for _ in range(2_000):
        nested = [nested]
    record["future_metadata"] = nested

    requests = ProjectRequests(
        home=server.home,
        runtime=server.runtime,
        codec=FrameCodec(),
        registry=registry,
    )
    projected = requests._bounded_message("request", record)

    assert "future_metadata" not in projected
    assert projected["truncated_fields"] == ["future_metadata"]


@pytest.mark.asyncio
async def test_malformed_inbox_fuzz_corpus_is_isolated(tmp_path: Path) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    project = registry.create_project("alpha", "repo")
    sender = registry.create_project("sender", "repo")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    valid_ids = {
        inbox.send(
            from_project=sender.project_id,
            from_session="d" * 32,
            to_project=project.project_id,
            kind="info",
            title=f"valid {index}",
            body="body",
        )
        for index in range(4)
    }
    new_dir = registry.root / project.project_id / "inbox" / "new"
    template = json.loads((new_dir / f"{next(iter(valid_ids))}.json").read_text())
    rng = random.Random(392)
    required_fields = ["from", "to_project", "kind", "title", "body", "created_at"]

    for index in range(256):
        message_id = f"{index + 1024:032x}"
        path = new_dir / f"{message_id}.json"
        record = {**template, "id": message_id}
        case = index % 8
        if case == 0:
            path.write_bytes(
                b'{"schema_version":1,"id":'
                + b"9" * (4_301 + rng.randrange(700))
                + b"}"
            )
        elif case == 1:
            depth = 65 + rng.randrange(160)
            prefix = json.dumps(record, separators=(",", ":"))[:-1]
            path.write_text(
                prefix + ',"future":' + "[" * depth + "0" + "]" * depth + "}"
            )
        elif case == 2:
            path.write_bytes(b'{"schema_version":1,"title":"\xff\xfe"}')
        elif case == 3:
            path.write_bytes(b'{"schema_version":1,"title":"bad\x00value"}')
        elif case == 4:
            record[rng.choice(required_fields)] = rng.choice([False, 7, None, []])
            path.write_text(json.dumps(record))
        elif case == 5:
            path.write_text(json.dumps(record)[: rng.randrange(1, 30)])
        elif case == 6:
            path.write_text(json.dumps([record]))
        elif index == 7:
            path.write_text(json.dumps({**record, "title": "x" * (10 * 1024 * 1024)}))
        else:
            record["title"] = ["x" * (50_000 + rng.randrange(50_000))]
            path.write_text(json.dumps(record))

    listed = inbox.list(project.project_id)
    assert {item["id"] for item in listed["new"]} == valid_ids
    scanner = ProjectInboxScanner(
        registry, project.project_id, sessions_root=server.home / "sessions"
    )
    assert set(scanner.scan() or ()) == valid_ids
    for index in range(256):
        assert inbox.claim(project.project_id, f"{index + 1024:032x}", "e" * 32) is None

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        response = await _request(
            reader,
            writer,
            2,
            "project_inbox",
            {"project_id": project.project_id, "status": "new", "limit": 1_000},
        )
        assert {item["id"] for item in response[-1]["result"]["messages"]} == valid_ids
    finally:
        await _close(server, writer)

    claimed = inbox.claim(project.project_id, next(iter(valid_ids)), "e" * 32)
    assert claimed is not None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "field_value",
    [
        {"payload": "x" * (2 * 1024 * 1024)},
        ["x" * 1024 for _ in range(2048)],
    ],
    ids=["dict", "list"],
)
async def test_project_inbox_omits_large_unknown_structured_fields(
    tmp_path: Path, field_value: object
) -> None:
    server = _server(tmp_path)
    registry = ProjectRegistry(server.home / "projects")
    target = registry.create_project("target", "repo")
    sender = registry.create_project("sender", "repo")
    inbox = ProjectInbox(registry, sessions_root=server.home / "sessions")
    message_id = inbox.send(
        from_project=sender.project_id,
        from_session="a" * 32,
        to_project=target.project_id,
        kind="info",
        title="future metadata",
        body="body",
    )
    path = (
        server.home
        / "projects"
        / target.project_id
        / "inbox"
        / "new"
        / f"{message_id}.json"
    )
    stored = json.loads(path.read_text())
    stored["future_metadata"] = field_value
    stored["to_session"] = "f" * 32
    path.write_text(json.dumps(stored) + "\n")

    reader, writer = await _connect(server)
    try:
        await _hello(reader, writer, ["projects"])
        frames = await _request(
            reader,
            writer,
            2,
            "project_inbox",
            {"project_id": target.project_id},
        )
        message = frames[-1]["result"]["messages"][0]
        assert "future_metadata" not in message
        assert "future_metadata" in message["truncated_fields"]
        assert message["to_session"] == "f" * 32
        assert json.loads(path.read_text())["future_metadata"] == field_value
    finally:
        await _close(server, writer)
