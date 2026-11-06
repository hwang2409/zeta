from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
import subprocess
import sys
from pathlib import Path

import pytest

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


def test_writer_refuses_missing_and_too_new_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ZETA_TEST_BOOTSTRAP_STORAGE_FORMAT")
    with pytest.raises(StorageFormatError, match="manifest is missing"):
        open_writer(tmp_path)
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":2}\n')
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
    assert verify(tmp_path).writer_epoch == 1


def test_writer_lock_descriptor_is_not_inherited(tmp_path: Path) -> None:
    advance(tmp_path, expected_epoch=0)
    lease = open_writer(tmp_path)
    child_fds = subprocess.check_output(
        [sys.executable, "-c", "import os; print(sorted(os.listdir('/dev/fd')))"],
        text=True,
    )
    assert str(lease._entry.fd) not in child_fds
    lease.close()


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
        with open(name, "wb", opener=lambda path, flags: os.open(path, flags, dir_fd=fd)) as handle:
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
