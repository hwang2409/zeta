from __future__ import annotations

import os
import stat

from zeta.core.session_files import session_root, write_session_json
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
        remove_retention_pin(directory_fd)
    assert syncs


def test_leftover_temp_file_is_not_pin(tmp_path):
    with _directory(tmp_path / "session") as directory_fd:
        (tmp_path / "session" / f".{RETENTION_PIN_FILENAME}.leftover.tmp").write_text(
            '{"schema_version":1,"pinned":true}'
        )
        assert not is_retention_pinned(directory_fd)


def test_invalid_pin_files_fail_closed(tmp_path):
    cases = {
        "malformed": b"not json",
        "oversized": b"x" * 257,
        "wrong-version": b'{"schema_version":2,"pinned":true}',
        "false": b'{"schema_version":1,"pinned":false}',
    }
    for name, content in cases.items():
        session = tmp_path / name
        with _directory(session) as directory_fd:
            (session / RETENTION_PIN_FILENAME).write_bytes(content)
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


def test_metadata_like_writes_do_not_change_pin(tmp_path):
    with _directory(tmp_path / "session") as directory_fd:
        write_retention_pin(directory_fd)
        write_session_json(
            directory_fd, "metadata.json", {"version": 1, "updated_at": "later"}
        )
        assert is_retention_pinned(directory_fd)
        assert (tmp_path / "session" / RETENTION_PIN_FILENAME).read_bytes()
