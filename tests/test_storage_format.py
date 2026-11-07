from __future__ import annotations

import asyncio
import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

from zeta import storage_format
from zeta.core.checkpoints import ConversationIntegrityError
from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
from zeta.core.store import ConversationStore
from zeta.storage_format import (
    DeleteValue,
    ReplaceSubtree,
    SetValue,
    StorageFormatError,
    advance,
    apply_mutations,
    open_writer,
    verify,
)
from zeta.tools._shared.process import BackgroundTaskRegistry


def _try_exclusive_lock(home: str, result: multiprocessing.Queue[str]) -> None:
    fd = os.open(Path(home) / ".storage-format.lock", os.O_RDWR)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            result.put("active writers")
        else:
            result.put("exclusive lock acquired")
    finally:
        os.close(fd)


def _closed_owner_allows_exclusive_lock(home: str, owner: str, result) -> None:
    path = Path(home)
    advance(path, expected_epoch=0)
    root_lease = open_writer(path)
    if owner == "background":
        with ConversationStore(
            path / "sessions", session_id="owner", _storage_lease=root_lease
        ) as store:
            registry = BackgroundTaskRegistry(
                session_dir=store.session_dir,
                directory_fd=store.directory_fd,
                storage_lease=root_lease,
            )
            asyncio.run(registry.close())
    elif owner == "workspace":
        with ConversationStore(
            path / "sessions", session_id="owner", _storage_lease=root_lease
        ) as store:
            try:
                WorkspaceSnapshotStore(
                    store.session_dir,
                    store.session_id,
                    cap=0,
                    storage_lease=root_lease,
                )
            except ValueError:
                pass
    else:
        try:
            ConversationStore(
                path / "sessions", session_id="../unsafe", _storage_lease=root_lease
            )
        except ConversationIntegrityError:
            pass
    root_lease.close()
    _try_exclusive_lock(home, result)


def _assert_closed_owner_allows_exclusive_lock(tmp_path: Path, owner: str) -> None:
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(
        target=_closed_owner_allows_exclusive_lock,
        args=(str(tmp_path), owner, result),
    )
    process.start()
    process.join(10)
    assert process.exitcode == 0
    assert result.get(timeout=1) == "exclusive lock acquired"
    result.close()
    result.join_thread()
    process.close()


def test_writer_refuses_missing_and_too_new_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZETA_TEST_BOOTSTRAP_STORAGE_FORMAT")
    with pytest.raises(StorageFormatError, match="manifest is missing"):
        open_writer(tmp_path)
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":3}\n')
    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        open_writer(tmp_path)


def test_advance_is_exclusive_across_processes(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    lease = open_writer(tmp_path)
    context = multiprocessing.get_context("spawn")
    result = context.Queue()
    process = context.Process(target=_try_exclusive_lock, args=(str(tmp_path), result))
    process.start()
    process.join(10)
    assert process.exitcode == 0
    assert result.get(timeout=1) == "active writers"
    result.close()
    result.join_thread()
    process.close()
    lease.close()
    assert verify(tmp_path).writer_epoch == 2


def test_advanced_home_refuses_predecessor_before_transcript_parsing(
    tmp_path: Path,
) -> None:
    advance(tmp_path, expected_epoch=0)
    session = tmp_path / "sessions" / "malformed"
    session.mkdir(parents=True)
    (session / "conversation.jsonl").write_bytes(b"not-json\n")
    before = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert verify(tmp_path).writer_epoch > 1, "candidate advanced the home"
    with pytest.raises(StorageFormatError, match="newer than writer capability"):
        open_writer(tmp_path, capability=1)

    after = {
        path.relative_to(tmp_path): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert after == before


def test_writer_lock_descriptor_is_not_inherited(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    lease = open_writer(tmp_path)
    child_fds = subprocess.check_output(
        [sys.executable, "-c", "import os; print(sorted(os.listdir('/dev/fd')))"],
        text=True,
    )
    assert str(lease._entry.fd) not in child_fds
    lease.close()


def test_writer_owner_cleanup_can_reenter_registry_lock(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    lease = open_writer(tmp_path)

    storage_format._REGISTRY_LOCK.acquire()
    reentered = storage_format._REGISTRY_LOCK.acquire(blocking=False)
    try:
        if reentered:
            lease.close()
            storage_format._REGISTRY_LOCK.release()
    finally:
        storage_format._REGISTRY_LOCK.release()
    if not reentered:
        lease.close()
    assert reentered, "owner cleanup must re-enter the writer registry lock"

    result: multiprocessing.Queue[str] = multiprocessing.Queue()
    _try_exclusive_lock(str(tmp_path), result)
    assert result.get(timeout=1) == "exclusive lock acquired"
    result.close()
    result.join_thread()


def test_background_registry_releases_its_only_writer_lease(tmp_path: Path) -> None:
    _assert_closed_owner_allows_exclusive_lock(tmp_path, "background")


def test_failed_store_construction_releases_writer_lease(tmp_path: Path) -> None:
    _assert_closed_owner_allows_exclusive_lock(tmp_path, "failed-store")


def test_failed_workspace_construction_releases_writer_lease(tmp_path: Path) -> None:
    _assert_closed_owner_allows_exclusive_lock(tmp_path, "workspace")


def test_mutations_preserve_unknown_nested_keys_and_declare_deletes() -> None:
    raw = {
        "known": {"value": 1, "future": {"nested": True}},
        "records": {"a": {"known": 1, "future": "kept"}},
        "remove": True,
    }
    output = apply_mutations(
        raw,
        (
            SetValue(("known",), {"value": 2}),
            SetValue(("records", "a"), {"known": 2}),
            DeleteValue(("remove",)),
            ReplaceSubtree(("owned",), [{"replacement": True}]),
        ),
    )
    assert output == {
        "known": {"value": 2, "future": {"nested": True}},
        "records": {"a": {"known": 2, "future": "kept"}},
        "owned": [{"replacement": True}],
    }


def test_publish_reads_locked_tree_before_writing(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    lease = open_writer(tmp_path)
    directory_fd = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    (tmp_path / "state.json").write_text(
        json.dumps({"known": {"value": 1, "future": "kept"}})
    )

    def publish(fd: int, name: str, data: bytes) -> None:
        with open(
            name, "wb", opener=lambda path, flags: os.open(path, flags, dir_fd=fd)
        ) as handle:
            handle.write(data)

    try:
        lease.publish_json(
            directory_fd,
            "state.json",
            (SetValue(("known",), {"value": 2}),),
            publish,
        )
    finally:
        os.close(directory_fd)
        lease.close()
    assert json.loads((tmp_path / "state.json").read_text()) == {
        "known": {"value": 2, "future": "kept"}
    }
