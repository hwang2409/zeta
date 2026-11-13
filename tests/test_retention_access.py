from __future__ import annotations

import fcntl
import json
import multiprocessing
import os
from pathlib import Path

import pytest

from zeta.core.retention_access import (
    RetentionYieldError,
    acquire_after_retention_yield,
)
from zeta.locking import RETENTION_HOME_ID, domain_lock_name


def _lock_path(home: Path, domain: str) -> Path:
    return home / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, domain)


def _child_lock_holder(home_text: str, ready, command, *, crash: bool = False) -> None:
    home = Path(home_text)
    active = os.open(_lock_path(home, "planner-active"), os.O_RDWR)
    target = os.open(_lock_path(home, "target"), os.O_RDWR)
    fcntl.flock(active, fcntl.LOCK_SH)
    fcntl.flock(target, fcntl.LOCK_SH)
    ready.send(True)
    if crash:
        return
    command.recv()
    fcntl.flock(active, fcntl.LOCK_UN)
    fcntl.flock(target, fcntl.LOCK_UN)
    os.close(active)
    os.close(target)


def _child_target_holder(home_text: str, ready, command) -> None:
    target = os.open(_lock_path(Path(home_text), "target"), os.O_RDWR)
    fcntl.flock(target, fcntl.LOCK_SH)
    ready.send(True)
    command.recv()
    fcntl.flock(target, fcntl.LOCK_UN)
    os.close(target)


def _prepare(home: Path, *, epoch: int = 11) -> None:
    (home / ".domain-locks").mkdir(parents=True)
    (home / "storage-format.json").write_text(json.dumps({"writer_epoch": epoch}))
    for domain in ("planner-active", "user-open-request", "target"):
        _lock_path(home, domain).touch()


def _target(home: Path) -> int:
    return os.open(_lock_path(home, "target"), os.O_RDWR)


def _blocked(fd: int):
    def acquire() -> object:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return fd

    return acquire


def _start_target_holder(home: Path):
    context = multiprocessing.get_context("fork")
    ready_recv, ready_send = context.Pipe(False)
    command_recv, command_send = context.Pipe(False)
    process = context.Process(
        target=_child_target_holder, args=(str(home), ready_send, command_recv)
    )
    process.start()
    assert ready_recv.recv()
    return process, command_send


def _start_holder(home: Path, *, crash: bool = False):
    context = multiprocessing.get_context("fork")
    ready_recv, ready_send = context.Pipe(False)
    command_recv, command_send = context.Pipe(False)
    process = context.Process(
        target=_child_lock_holder,
        args=(str(home), ready_send, command_recv),
        kwargs={"crash": crash},
    )
    process.start()
    assert ready_recv.recv()
    return process, command_send


def test_non_retention_holder_keeps_original_typed_error(tmp_path: Path) -> None:
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":11}')
    lock_dir = tmp_path / ".domain-locks"
    lock_dir.mkdir()
    target_path = lock_dir / domain_lock_name(RETENTION_HOME_ID, "target")
    target_path.touch()
    process, command = _start_target_holder(tmp_path)
    target = _target(tmp_path)
    try:
        error = RuntimeError("original")
        with pytest.raises(RuntimeError, match="original"):
            acquire_after_retention_yield(
                tmp_path, _blocked(target), lambda _: error
            )
        assert not (lock_dir / domain_lock_name(RETENTION_HOME_ID, "planner-active")).exists()
        assert not (lock_dir / domain_lock_name(RETENTION_HOME_ID, "user-open-request")).exists()
    finally:
        os.close(target)
        command.send(True)
        process.join(10)
        assert process.exitcode == 0


def test_request_ack_and_retry_releases_the_target(tmp_path: Path) -> None:
    _prepare(tmp_path)
    process, command = _start_holder(tmp_path)
    target = _target(tmp_path)
    now = [0.0]
    waits = 0
    sent = False

    def clock() -> float:
        return now[0]

    def wait(_: float) -> None:
        nonlocal waits, sent
        waits += 1
        if not sent:
            command.send(True)
            process.join(10)
            sent = True
        now[0] += 0.1

    try:
        assert acquire_after_retention_yield(
            tmp_path, _blocked(target), lambda exc: exc,
            clock=clock, wait=wait,
        ) == target
        assert waits == 1
    finally:
        os.close(target)
        process.join(10)
        assert process.exitcode == 0


def test_timeout_then_retry_can_safely_succeed(tmp_path: Path) -> None:
    _prepare(tmp_path)
    process, command = _start_holder(tmp_path)
    target = _target(tmp_path)
    now = [0.0]

    def wait(_: float) -> None:
        command.send(True)
        process.join(10)
        now[0] = 3.0

    try:
        assert acquire_after_retention_yield(
            tmp_path, _blocked(target), lambda exc: exc,
            clock=lambda: now[0], wait=wait,
        ) == target
    finally:
        os.close(target)
        process.join(10)
        assert process.exitcode == 0


def test_timeout_with_active_planner_is_a_retention_yield_error(tmp_path: Path) -> None:
    _prepare(tmp_path)
    process, _ = _start_holder(tmp_path)
    target = _target(tmp_path)
    now = [3.0]
    try:
        with pytest.raises(RetentionYieldError, match="did not yield"):
            acquire_after_retention_yield(
                tmp_path, _blocked(target), lambda exc: exc,
                clock=lambda: now[0], wait=lambda _: now.__setitem__(0, 6.0),
            )
    finally:
        os.close(target)
        process.terminate()
        process.join(10)


def test_planner_crash_releases_the_target(tmp_path: Path) -> None:
    _prepare(tmp_path)
    process, _ = _start_holder(tmp_path, crash=True)
    target = _target(tmp_path)
    try:
        process.join(10)
        assert process.exitcode == 0
        assert acquire_after_retention_yield(
            tmp_path, _blocked(target), lambda exc: exc,
            clock=lambda: 3.0, wait=lambda _: None,
        ) == target
    finally:
        os.close(target)


@pytest.mark.asyncio
async def test_on_loop_contention_rejects_before_wait(tmp_path: Path) -> None:
    _prepare(tmp_path)
    process, _ = _start_holder(tmp_path)
    target = _target(tmp_path)
    waited = False

    def wait(_: float) -> None:
        nonlocal waited
        waited = True

    try:
        with pytest.raises(RuntimeError, match="event-loop"):
            acquire_after_retention_yield(
                tmp_path, _blocked(target), lambda exc: exc, wait=wait
            )
        assert not waited
    finally:
        os.close(target)
        process.terminate()
        process.join(10)


def test_epoch_ten_keeps_old_refusal_without_handshake_files(tmp_path: Path) -> None:
    (tmp_path / "storage-format.json").write_text('{"writer_epoch":10}')
    lock_dir = tmp_path / ".domain-locks"
    lock_dir.mkdir()
    target_path = lock_dir / domain_lock_name(RETENTION_HOME_ID, "target")
    target_path.touch()
    process, command = _start_target_holder(tmp_path)
    target = os.open(target_path, os.O_RDWR)
    try:
        with pytest.raises(RuntimeError, match="original"):
            acquire_after_retention_yield(
                tmp_path, _blocked(target), lambda _: RuntimeError("original")
            )
        assert not (lock_dir / domain_lock_name(RETENTION_HOME_ID, "planner-active")).exists()
        assert not (lock_dir / domain_lock_name(RETENTION_HOME_ID, "user-open-request")).exists()
    finally:
        os.close(target)
        command.send(True)
        process.join(10)
        assert process.exitcode == 0
