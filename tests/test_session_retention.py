from __future__ import annotations

import os
import stat

from zeta.core.session import SessionManager, SessionMetadata
from zeta.core.session_files import session_root
from zeta.session_retention import (
    RETENTION_PIN_FILENAME,
    is_retention_pinned,
    remove_retention_pin,
    write_retention_pin,
)


def _directory(path):
    path.mkdir()
    return session_root(path)


def test_pin_round_trip_and_idempotent_unpin(tmp_path):
    with _directory(tmp_path / "session") as directory_fd:
        assert not is_retention_pinned(directory_fd)
        write_retention_pin(directory_fd)
        assert is_retention_pinned(directory_fd)
        remove_retention_pin(directory_fd)
        remove_retention_pin(directory_fd)
        assert not is_retention_pinned(directory_fd)


def test_pin_write_is_atomic_and_directory_durable(tmp_path, monkeypatch):
    syncs = []
    real_fsync = os.fsync

    def record_fsync(fd):
        if stat.S_ISDIR(os.fstat(fd).st_mode):
            syncs.append(fd)
        real_fsync(fd)

    monkeypatch.setattr("zeta.core.session_files.os.fsync", record_fsync)
    with _directory(tmp_path / "session") as directory_fd:
        write_retention_pin(directory_fd)
        assert syncs
        syncs.clear()
        remove_retention_pin(directory_fd)
        assert syncs


def test_leftover_temp_file_is_pin(tmp_path):
    with _directory(tmp_path / "session") as directory_fd:
        (tmp_path / "session" / f".{RETENTION_PIN_FILENAME}.leftover.tmp").write_text(
            '{"schema_version":1,"pinned":true}'
        )
        assert is_retention_pinned(directory_fd)


def test_invalid_pin_files_fail_closed(tmp_path):
    cases = {
        "malformed": b"not json",
        "oversized": b"x" * 257,
        "wrong-version": b'{"schema_version":2,"pinned":true}',
        "false": b'{"schema_version":1,"pinned":false}',
        "empty": b"{}",
    }
    for name, content in cases.items():
        session = tmp_path / name
        with _directory(session) as directory_fd:
            pin = session / RETENTION_PIN_FILENAME
            pin.write_bytes(content)
            pin.chmod(0o600)
            assert is_retention_pinned(directory_fd)


def test_symlink_multi_link_and_nonregular_pin_files_fail_closed(tmp_path):
    session = tmp_path / "session"
    with _directory(session) as directory_fd:
        (session / "target").write_bytes(b'{"schema_version":1,"pinned":true}')
        (session / RETENTION_PIN_FILENAME).symlink_to("target")
        assert is_retention_pinned(directory_fd)
    session = tmp_path / "hardlink"
    with _directory(session) as directory_fd:
        pin = session / RETENTION_PIN_FILENAME
        pin.write_bytes(b'{"schema_version":1,"pinned":true}')
        os.link(pin, session / "other")
        assert is_retention_pinned(directory_fd)
    session = tmp_path / "directory"
    with _directory(session) as directory_fd:
        (session / RETENTION_PIN_FILENAME).mkdir()
        assert is_retention_pinned(directory_fd)
    session = tmp_path / "wrong-mode"
    with _directory(session) as directory_fd:
        pin = session / RETENTION_PIN_FILENAME
        pin.write_bytes(b'{"schema_version":1,"pinned":true}')
        pin.chmod(0o644)
        assert is_retention_pinned(directory_fd)


def test_metadata_writes_do_not_change_pin(tmp_path):
    session = tmp_path / "session"
    session.mkdir()
    manager = SessionManager(tmp_path)
    metadata = SessionMetadata.new(
        session_id="session",
        provider="codex",
        model="gpt-5.6-sol",
        cwd=str(tmp_path),
        retained_tail=10,
        compaction_budget=10,
    )
    with session_root(session) as directory_fd:
        write_retention_pin(directory_fd)
        manager._write_unlocked(metadata, directory_fd=directory_fd)
        assert is_retention_pinned(directory_fd)
        assert (session / RETENTION_PIN_FILENAME).read_bytes()
