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
