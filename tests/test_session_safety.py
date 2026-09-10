"""Filesystem regressions for session rename, deletion, and store leases."""

from __future__ import annotations

import fcntl
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

from zeta.cli import main
from zeta.core import session as session_module
from zeta.core.session import SessionError, SessionInUseError, SessionManager
from zeta.core.store import ConversationStore
from zeta.types import Message, MessageRole, TextContent


def closed_session(tmp_path):
    manager = SessionManager(tmp_path / "home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    opened.store.close()
    return manager, opened.metadata.session_id


@pytest.mark.parametrize("location", ["root", "session", ".meta.lock", "meta.json"])
def test_rename_refuses_symlinks_without_external_writes(tmp_path, location):
    manager, sid = closed_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    directory = manager.sessions_dir / sid
    if location == "root":
        manager.sessions_dir.rename(outside / "sessions")
        manager.sessions_dir.symlink_to(outside / "sessions", target_is_directory=True)
    elif location == "session":
        directory.rename(outside / sid)
        directory.symlink_to(outside / sid, target_is_directory=True)
    else:
        (directory / location).unlink()
        # The review probe: opening .meta.lock must not create this target.
        (directory / location).symlink_to(outside / "must-not-create")
    before = {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()}
    with pytest.raises(SessionError):
        manager.rename(sid, "changed")
    after = {p.relative_to(outside): p.read_bytes() for p in outside.rglob("*") if p.is_file()}
    assert after == before
    assert not (outside / "must-not-create").exists()


@pytest.mark.parametrize("name", [".meta.lock", "meta.json"])
@pytest.mark.parametrize("kind", ["fifo", "directory", "hardlink"])
def test_rename_refuses_aberrant_files(tmp_path, name, kind):
    manager, sid = closed_session(tmp_path)
    path = manager.sessions_dir / sid / name
    path.unlink()
    sentinel = tmp_path / "keep"
    sentinel.write_text("keep")
    if kind == "fifo":
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        os.link(sentinel, path)
    with pytest.raises(SessionError):
        manager.rename(sid, "changed")
    assert sentinel.read_text() == "keep"


def test_delete_refuses_every_open_store_until_last_close(tmp_path, monkeypatch):
    manager, sid = closed_session(tmp_path)
    monkeypatch.setenv("ZETA_HOME", str(manager.home))
    first = manager.open(sid).store
    second = ConversationStore(manager.sessions_dir, session_id=sid)
    try:
        with pytest.raises(SessionInUseError):
            manager.delete(sid)
        assert main(["session", "delete", sid, "--force"]) == 1
        first.append_message(Message(MessageRole.USER, [TextContent("still writable")]))
        first.close()
        with pytest.raises(SessionInUseError):
            manager.delete(sid)
    finally:
        first.close()
        second.close()
    assert main(["session", "delete", sid, "--force"]) == 0
    assert not (manager.sessions_dir / sid).exists()
    with pytest.raises(ValueError, match="closed"):
        second.append_message(Message(MessageRole.USER, [TextContent("too late")]))


def test_crashed_store_releases_lease(tmp_path):
    manager, sid = closed_session(tmp_path)
    script = """
import sys
from zeta.core.session import SessionManager
store = SessionManager(sys.argv[1]).open(sys.argv[2]).store
print('ready', flush=True)
sys.stdin.read()
"""
    with subprocess.Popen(
        [sys.executable, "-c", script, str(manager.home), sid],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
    ) as process:
        try:
            assert process.stdout.readline() == "ready\n"
            with pytest.raises(SessionInUseError):
                manager.delete(sid)
            process.kill()
            process.wait(timeout=10)
            manager.delete(sid)
            assert not (manager.sessions_dir / sid).exists()
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)


def test_rename_after_delete_never_recreates_directory(tmp_path, monkeypatch):
    manager, sid = closed_session(tmp_path)
    record_name = manager.record_name

    def delete_before_metadata_lock(metadata, *, name):
        manager.delete(sid)
        record_name(metadata, name=name)

    monkeypatch.setattr(manager, "record_name", delete_before_metadata_lock)
    with pytest.raises(SessionError):
        manager.rename(sid, "too late")
    assert not (manager.sessions_dir / sid).exists()


def test_delete_refuses_metadata_mutation_until_it_finishes(tmp_path, monkeypatch):
    manager, sid = closed_session(tmp_path)
    writing = Event()
    proceed = Event()
    write = manager._write_unlocked

    def pause_write(metadata, *, directory_fd):
        writing.set()
        assert proceed.wait(10)
        write(metadata, directory_fd=directory_fd)

    monkeypatch.setattr(manager, "_write_unlocked", pause_write)
    with ThreadPoolExecutor(max_workers=1) as pool:
        rename = pool.submit(manager.rename, sid, "saved")
        try:
            assert writing.wait(10)
            with pytest.raises(SessionInUseError):
                manager.delete(sid)
        finally:
            proceed.set()
        assert rename.result(timeout=10).name == "saved"
    manager.delete(sid)
    assert not (manager.sessions_dir / sid).exists()


def test_directory_deleted_before_lease_acquisition_stays_deleted(tmp_path, monkeypatch):
    manager, sid = closed_session(tmp_path)
    flock = fcntl.flock
    deleted = False

    def delete_before_shared_lease(fd, operation):
        nonlocal deleted
        if not deleted and operation == fcntl.LOCK_SH | fcntl.LOCK_NB:
            deleted = True
            manager.delete(sid)
        return flock(fd, operation)

    monkeypatch.setattr(fcntl, "flock", delete_before_shared_lease)
    with pytest.raises(SessionError):
        manager.rename(sid, "too late")
    assert not (manager.sessions_dir / sid).exists()


def test_resume_racing_delete_never_recreates_directory(tmp_path, monkeypatch):
    manager, sid = closed_session(tmp_path)

    def delete_before_store_open(*args, **kwargs):
        manager.delete(sid)
        return ConversationStore(*args, **kwargs)

    monkeypatch.setattr(session_module, "ConversationStore", delete_before_store_open)
    with pytest.raises(SessionError):
        manager.open(sid)
    assert not (manager.sessions_dir / sid).exists()


def test_read_only_store_holds_lease_without_creating_files(tmp_path):
    manager, sid = closed_session(tmp_path)
    directory = manager.sessions_dir / sid
    before = {p.name: p.read_bytes() for p in directory.iterdir()}
    with manager.open(sid, _read_only=True).store, pytest.raises(SessionInUseError):
        manager.delete(sid)
    assert {p.name: p.read_bytes() for p in directory.iterdir()} == before
    manager.delete(sid)


def test_failed_open_releases_lease(tmp_path):
    manager, sid = closed_session(tmp_path)
    (manager.sessions_dir / sid / "conversation.jsonl").write_bytes(b"bad\n")
    with pytest.raises(SessionError):
        manager.open(sid)
    manager.delete(sid)
    assert not (manager.sessions_dir / sid).exists()


def test_discarded_store_releases_lease(tmp_path):
    import gc

    manager, sid = closed_session(tmp_path)
    opened = manager.open(sid)
    del opened
    gc.collect()
    manager.delete(sid)
    assert not (manager.sessions_dir / sid).exists()


@pytest.mark.parametrize("name", [".lock", "conversation.jsonl", "session_state.json", "agent_lifecycle.json"])
def test_store_open_refuses_symlinks_without_creating_external_targets(tmp_path, name):
    manager, sid = closed_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    path = manager.sessions_dir / sid / name
    path.unlink(missing_ok=True)
    path.symlink_to(outside / "must-not-create")
    with pytest.raises(SessionError):
        manager.open(sid)
    assert list(outside.iterdir()) == []
    # A failed open also releases its directory lease.
    path.unlink()
    manager.delete(sid)


def test_export_refuses_external_conversation_symlink(tmp_path):
    manager, sid = closed_session(tmp_path)
    sentinel = tmp_path / "external.jsonl"
    sentinel.write_bytes(b'{"secret":"outside"}\n')
    path = manager.sessions_dir / sid / "conversation.jsonl"
    path.unlink()
    path.symlink_to(sentinel)
    with pytest.raises(SessionError):
        manager.export(sid)
    assert sentinel.read_bytes() == b'{"secret":"outside"}\n'


@pytest.mark.parametrize("operation", ["append", "repair", "state", "lifecycle", "agents"])
def test_store_uses_lifetime_descriptor_after_directory_swap(tmp_path, operation):
    manager, sid = closed_session(tmp_path)
    store = manager.open(sid).store
    original = store.session_dir
    pinned = manager.sessions_dir / "pinned"
    original.rename(pinned)
    outside = tmp_path / "outside"
    outside.mkdir()
    original.symlink_to(outside, target_is_directory=True)
    try:
        if operation == "append":
            store.append_message(Message(MessageRole.USER, [TextContent("saved")]))
            assert b"saved" in (pinned / "conversation.jsonl").read_bytes()
        elif operation == "repair":
            with (pinned / "conversation.jsonl").open("ab") as handle:
                handle.write(b'{"torn')
            with pytest.warns(RuntimeWarning, match="torn"):
                store.append_message(Message(MessageRole.USER, [TextContent("saved")]))
            assert b'"torn' not in (pinned / "conversation.jsonl").read_bytes()
        elif operation == "state":
            store.set_bash_cwd("/saved")
            assert b"/saved" in (pinned / "session_state.json").read_bytes()
        elif operation == "lifecycle":
            store._agent_lifecycle = {"state": "completed"}
            store._write_agent_lifecycle()
            store._load_session_state()
            assert store._agent_lifecycle == {"state": "completed"}
            assert (pinned / "agent_lifecycle.json").is_file()
        else:
            assert store.allocate_agent_index() == 1
            assert (pinned / "agents").is_dir()
        assert list(outside.iterdir()) == []
    finally:
        store.close()


def test_child_store_rejects_symlink_in_nested_sessions_root(tmp_path):
    manager, sid = closed_session(tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    (manager.sessions_dir / sid / "agents").symlink_to(outside, target_is_directory=True)
    with pytest.raises((SessionError, OSError)):
        ConversationStore(manager.sessions_dir / sid / "agents", session_id="1")
    with manager.open(sid).store as store, pytest.raises((SessionError, OSError)):
        store.allocate_agent_index()
    assert list(outside.iterdir()) == []


def test_session_file_helper_rejects_path_components(tmp_path):
    from zeta.core.session_files import open_session_file

    manager, sid = closed_session(tmp_path)
    with manager.open(sid).store as store:
        for name in ("../escaped", str(tmp_path / "escaped"), "sub/file"):
            with pytest.raises(SessionError):
                open_session_file(store.directory_fd, name, os.O_WRONLY | os.O_CREAT)
    assert not (tmp_path / "escaped").exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("order", ["rename-first", "delete-first"])
async def test_server_rename_and_cli_delete_serialize_across_processes(tmp_path, order):
    import asyncio
    import json

    manager, sid = closed_session(tmp_path)
    script = """
import asyncio, sys
from zeta.server import ZetaServer
async def main():
    server = ZetaServer(home=sys.argv[1], port=0, provider='fake')
    manager = server.runtime.manager
    method = '_write_unlocked' if sys.argv[2] == 'rename-first' else 'record_name'
    original = getattr(manager, method)
    def paused(*args, **kwargs):
        print('barrier', flush=True)
        assert sys.stdin.readline().strip() == 'continue'
        return original(*args, **kwargs)
    setattr(manager, method, paused)
    await server.start()
    print(server.port, flush=True)
    await asyncio.Event().wait()
asyncio.run(main())
"""
    env = {**os.environ, "ZETA_HOME": str(manager.home)}
    server = await asyncio.create_subprocess_exec(
        sys.executable, "-u", "-c", script, str(manager.home), order,
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE, env=env,
    )
    writer = None
    try:
        port = int(await asyncio.wait_for(server.stdout.readline(), 10))
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        async def request(request_id, method, params):
            writer.write((json.dumps({"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}) + "\n").encode())
            await writer.drain()
        await request(1, "hello", {"protocol_version": "1.1"})
        assert "result" in json.loads(await asyncio.wait_for(reader.readline(), 10))
        await request(2, "rename_session", {"session_id": sid, "name": "saved"})
        assert await asyncio.wait_for(server.stdout.readline(), 10) == b"barrier\n"

        async def delete_cli():
            cli = await asyncio.create_subprocess_exec(
                sys.executable, "-c", "from zeta.cli import main; raise SystemExit(main())", "session", "delete", sid, "--force",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, env=env,
            )
            stdout, stderr = await asyncio.wait_for(cli.communicate(), 10)
            return cli.returncode, stdout + stderr

        code, output = await delete_cli()
        if order == "rename-first":
            assert code == 1, output
            assert b"in use" in output
        else:
            assert code == 0, output
        server.stdin.write(b"continue\n")
        await server.stdin.drain()
        response = json.loads(await asyncio.wait_for(reader.readline(), 10))
        if order == "rename-first":
            assert response["result"]["session"]["name"] == "saved"
            code, output = await delete_cli()
            assert code == 0, output
        else:
            assert "error" in response
        assert not (manager.sessions_dir / sid).exists()
    finally:
        if writer is not None:
            writer.close()
            await writer.wait_closed()
        if server.returncode is None:
            server.kill()
        await asyncio.wait_for(server.communicate(), 10)


def test_storage_operations_never_pass_full_session_paths_to_os(tmp_path, monkeypatch):
    import io
    from functools import wraps

    manager = SessionManager(tmp_path / "home")
    root = str(manager.sessions_dir)

    def checked(function):
        @wraps(function)
        def call(*args, **kwargs):
            for argument in args[:2]:
                if isinstance(argument, (str, bytes, os.PathLike)):
                    value = os.fsdecode(argument)
                    assert value != root and not value.startswith(root + "/"), (function.__name__, value)
            return function(*args, **kwargs)
        return call

    with monkeypatch.context() as patch:
        for name in ("open", "replace", "rename", "unlink", "stat", "lstat", "mkdir", "rmdir", "scandir", "listdir"):
            patch.setattr(os, name, checked(getattr(os, name)))
        patch.setattr(io, "open", checked(io.open))
        opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
        sid = opened.metadata.session_id
        store = opened.store
        store.append_message(Message(MessageRole.USER, [TextContent("saved")]))
        store.set_bash_cwd(str(tmp_path))
        store.allocate_agent_index()
        store._agent_lifecycle = {"state": "completed"}
        store._write_agent_lifecycle()
        store._load_session_state()
        assert manager.rename(sid, "renamed").name == "renamed"
        assert "saved" in manager.export(sid)
        assert [item.session_id for item in manager.list_sessions()] == [sid]
        store.close()
        manager.delete(sid)
    assert not (manager.sessions_dir / sid).exists()


def test_json_path_reader_rejects_symlink(tmp_path):
    from zeta.core.checkpoints import ConversationIntegrityError, load_session_json

    manager, sid = closed_session(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text('{"secret": "outside"}')
    path = manager.sessions_dir / sid / "agent_lifecycle.json"
    path.symlink_to(outside)
    with pytest.raises(ConversationIntegrityError):
        load_session_json(path)
    assert outside.read_text() == '{"secret": "outside"}'


def test_safe_open_rejects_hardlink_before_truncation(tmp_path):
    from zeta.core.session_files import open_session_file

    manager, sid = closed_session(tmp_path)
    outside = tmp_path / "keep"
    outside.write_text("keep")
    with manager.open(sid).store as store:
        os.link(outside, store.session_dir / "hardlink")
        with pytest.raises(SessionError):
            open_session_file(store.directory_fd, "hardlink", os.O_WRONLY | os.O_TRUNC)
    assert outside.read_text() == "keep"


@pytest.mark.parametrize("replacement", ["directory-swap", "temporary-symlink"])
async def test_background_shutdown_stays_in_pinned_directory(tmp_path, replacement):
    from zeta.server.runtime import ServerRuntime

    runtime = ServerRuntime(tmp_path / "home", cwd=tmp_path, provider="fake")
    await runtime.create_session()
    store = runtime.opened.store
    tasks = runtime.loop.tool_registry.background_tasks
    directory = store.session_dir
    outside = tmp_path / "outside"
    outside.mkdir()
    pinned = tmp_path / "pinned"
    if replacement == "directory-swap":
        directory.rename(pinned)
        directory.symlink_to(outside, target_is_directory=True)
    else:
        pinned = directory
        (directory / "background_tasks.tmp").symlink_to(outside / "must-not-create")
    try:
        task_id, _ = await tasks.start("printf contained", tmp_path, log_path=directory / "macro.log")
        await tasks.wait(task_id)
    finally:
        await runtime.close()
    assert list(outside.iterdir()) == []
    assert (pinned / "macro.log").read_text() == "contained"
    from zeta.core.checkpoints import load_session_json

    rows = load_session_json((pinned / "background_tasks.json").read_bytes())
    assert rows[0]["task_id"] == task_id
    assert rows[0]["running"] is False
    assert tasks._directory_fd is None


async def test_session_lifecycle_has_no_absolute_session_file_operations(tmp_path, monkeypatch):
    """Audit the real runtime, including tool and TUI persistence boundaries."""
    from pathlib import Path

    from zeta.persistence import DraftPersistence
    from zeta.server.runtime import ServerRuntime
    from zeta.tools.exec import run_exec_macro
    from zeta.tui.composer import build_user_message
    from zeta.types import StreamEventType, ToolCall

    home = tmp_path / "home"
    monkeypatch.setenv("ZETA_HOME", str(home))
    sessions = home / "sessions"
    events = []
    recording = True
    # session_root pins the trusted home, then opens "sessions" relative to it.
    # No absolute open of a session child belongs in this allowlist.
    root_pinning_opens = {(str(home), os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)}
    path_arguments = {"open": (0,), "os.rename": (0, 1), "os.remove": (0,), "os.mkdir": (0,)}

    def audit(event, args):
        if recording and event in path_arguments:
            events.append((event, args))

    sys.addaudithook(audit)
    runtime = None
    try:
        runtime = ServerRuntime(home, cwd=tmp_path, provider="fake")
        await runtime.create_session()
        store = runtime.opened.store
        sid = store.session_id
        turn = [event async for event in runtime.loop.run_turn("hello")]
        assert any(event.type == StreamEventType.MESSAGE_UPDATE for event in turn)
        tasks = runtime.loop.tool_registry.background_tasks
        task_id, _ = await tasks.start("printf background", tmp_path, log_path=store.session_dir / "background.log")
        await tasks.wait(task_id)
        runtime.policy.always_allow = ("exec(printf foreground)",)
        result = await run_exec_macro(
            runtime.loop.tool_registry,
            ToolCall("macro-audit", "exec", {"command": "printf foreground"}),
            store.session_dir / "foreground.log",
            stream_sink=lambda event: None,
            lifecycle_sink=lambda kind: None,
        )
        assert not result.is_error
        draft = DraftPersistence(store.session_dir / "draft", directory_fd=store.directory_fd)
        draft.schedule("draft text")
        draft.flush()
        assert draft.load() == "draft text"
        draft.clear()
        message = build_user_message("log", tmp_path, (store.session_dir / "background.log",), session_store=store)
        assert "background" in message.content[1].text
        assert runtime.manager.rename(sid, "renamed").name == "renamed"
        store.append_checkpoint("audit")
        await runtime.close()
        await runtime.resume_session(sid)
        assert runtime.loop.tool_registry.background_tasks.records[0].task_id == task_id
        await runtime.close()
        runtime.manager.delete(sid)
    finally:
        if runtime is not None:
            await runtime.close()
        recording = False
    assert set(path_arguments) <= {event for event, _ in events}
    assert any(event == "open" and (args[0], args[2]) in root_pinning_opens for event, args in events)
    violations = []
    for event, args in events:
        if event == "open" and (args[0], args[2]) in root_pinning_opens:
            continue
        for index in path_arguments[event]:
            path = args[index]
            if isinstance(path, (str, bytes)):
                path = Path(os.fsdecode(path))
                if path.is_absolute() and path.is_relative_to(sessions):
                    violations.append((event, args))
    assert not violations


async def test_background_descriptor_keeps_lease_until_registry_close(tmp_path):
    from zeta.tools._process import BackgroundTaskRegistry

    manager = SessionManager(tmp_path / "home")
    opened = manager.create(provider="fake", model="offline", cwd=tmp_path)
    store = opened.store
    tasks = BackgroundTaskRegistry(session_dir=store.session_dir, directory_fd=store.directory_fd)
    store.close()
    try:
        with pytest.raises(SessionInUseError):
            manager.delete(store.session_id)
    finally:
        await tasks.close()
    manager.delete(store.session_id)


def test_composer_persistence_survives_directory_swap(tmp_path, monkeypatch):
    from pathlib import Path
    from types import SimpleNamespace

    from zeta.persistence import DraftPersistence
    from zeta.tui import composer

    png = bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
    )
    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    outside = tmp_path / "outside"
    outside.mkdir()
    pinned = tmp_path / "pinned"
    store.session_dir.rename(pinned)
    store.session_dir.symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(composer.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(composer.shutil, "which", lambda _: "pngpaste")

    def clipboard(command, **kwargs):
        Path(command[1]).write_bytes(png)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(composer.subprocess, "run", clipboard)
    try:
        draft = DraftPersistence(store.session_dir / "draft", directory_fd=store.directory_fd)
        draft.schedule("saved")
        draft.flush()
        assert draft.load() == "saved"
        assert (pinned / "draft").is_file()
        draft.clear()
        assert not (pinned / "draft").exists()
        path = composer.paste_image(store.session_dir, directory_fd=store.directory_fd)
        assert (pinned / path.name).read_bytes() == png
        message = composer.build_user_message("image", tmp_path, (path,), session_store=store)
        assert message.content[1].mime_type == "image/png"
        app = SimpleNamespace(loop=SimpleNamespace(store=store))
        composer.ComposerAttachmentMixin._delete_staged_attachment(app, path)
        assert not (pinned / path.name).exists()
        assert list(outside.iterdir()) == []
    finally:
        store.close()


def test_child_transcript_reads_use_parent_descriptor(tmp_path):
    from zeta.tools.agent import _read_child_file
    from zeta.tui.agent_card import AgentCard

    store = ConversationStore(tmp_path / "sessions", cwd=tmp_path)
    store.allocate_agent_index()
    child = ConversationStore(store.session_dir / "agents", session_id="1", cwd=tmp_path)
    child.append_message(Message(MessageRole.ASSISTANT, [TextContent("pinned child")]))
    child.close()
    pinned = tmp_path / "pinned"
    outside = tmp_path / "outside"
    outside.mkdir()
    store.session_dir.rename(pinned)
    store.session_dir.symlink_to(outside, target_is_directory=True)
    try:
        assert b"pinned child" in _read_child_file(store, child.session_dir, "conversation.jsonl")
        assert AgentCard._tail_lines(str(child.session_dir), 5) == []
        assert list(outside.iterdir()) == []
    finally:
        store.close()
