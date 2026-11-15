"""Yield retention's exclusive pass before a blocked user lock acquisition."""

from __future__ import annotations

import asyncio
import fcntl
import os
import stat
import time
from collections.abc import Callable
from pathlib import Path

from ...locking import RETENTION_HOME_ID, domain_lock_name
from ...storage_format import StorageFeature, _read_manifest, feature_enabled


class RetentionYieldError(RuntimeError):
    """Retention did not release the requested resource in time."""


_WAIT_DEADLINE = 2.0


def _existing_lock(home: Path, domain: str) -> int | None:
    path = Path(home) / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, domain)
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise OSError("retention lock must be a regular, unshared file")
    return fd


def _active(home: Path) -> int | None:
    fd = _existing_lock(home, "planner-active")
    if fd is None:
        return None
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return fd
    except BaseException:
        os.close(fd)
        raise
    os.close(fd)
    return None


def _enabled(home: Path, observed_epoch: int | None) -> bool:
    if observed_epoch is not None:
        return feature_enabled(observed_epoch, StorageFeature.SESSION_RETENTION_ACTIVATION)
    try:
        epoch = _read_manifest(Path(home), absent_epoch=0)
    except (FileNotFoundError, OSError, ValueError):
        return False
    return feature_enabled(epoch, StorageFeature.SESSION_RETENTION_ACTIVATION)


def _off_loop() -> None:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return
    from ..session_files import require_off_loop_lock

    require_off_loop_lock("retention yield")


def acquire_after_retention_yield(
    home: Path,
    acquire: Callable[[], object],
    blocked: Callable[[BaseException], BaseException],
    *,
    observed_epoch: int | None = None,
    clock: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
    timeout: float = _WAIT_DEADLINE,
) -> object:
    """Retry one refused primitive while giving an active planner one chance."""
    try:
        return acquire()
    except BlockingIOError as exc:
        first = exc
        if not _enabled(home, observed_epoch):
            raise blocked(first) from first

    active_fd = _active(home)
    if active_fd is None:
        raise blocked(first) from first
    request_fd = _existing_lock(home, "user-open-request")
    if request_fd is None:
        os.close(active_fd)
        raise blocked(first) from first
    try:
        _off_loop()
        fcntl.flock(request_fd, fcntl.LOCK_SH)
        try:
            fcntl.flock(active_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            _off_loop()
            deadline = clock() + timeout
            while True:
                try:
                    fcntl.flock(active_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    remaining = deadline - clock()
                    if remaining <= 0:
                        break
                    wait(remaining)
        try:
            return acquire()
        except BlockingIOError as second:
            try:
                fcntl.flock(active_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as still_active:
                raise RetentionYieldError(
                    "retention did not yield to user open"
                ) from still_active
            raise blocked(second) from second
    finally:
        fcntl.flock(request_fd, fcntl.LOCK_UN)
        os.close(request_fd)
        fcntl.flock(active_fd, fcntl.LOCK_UN)
        os.close(active_fd)
