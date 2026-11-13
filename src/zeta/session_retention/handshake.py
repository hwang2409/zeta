"""Cross-process coordination for retention previews and user opens."""

from __future__ import annotations

import fcntl
import os
import stat
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from ..core.session_execution import SessionInUseError
from ..locking import RETENTION_HOME_ID, domain_lock_name


class RetentionYieldError(RuntimeError):
    """Retention still owns a root after the user-open yield deadline."""


_WAIT_DEADLINE = 2.0


def _lock_file(home: Path, name: str) -> int:
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    home_fd = os.open(home, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        try:
            os.mkdir(".domain-locks", mode=0o700, dir_fd=home_fd)
        except FileExistsError:
            pass
        locks_fd = os.open(
            ".domain-locks", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=home_fd
        )
        try:
            fd = os.open(
                name,
                os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                0o600,
                dir_fd=locks_fd,
            )
        finally:
            os.close(locks_fd)
    finally:
        os.close(home_fd)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise OSError("retention lock must be a regular, unshared file")
    return fd


def _fd(home: Path, domain: str) -> int:
    return _lock_file(home, domain_lock_name(RETENTION_HOME_ID, domain))


def _existing_fd(home: Path, domain: str) -> int | None:
    path = home / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, domain)
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return None
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise OSError("retention lock must be a regular, unshared file")
    return fd


@contextmanager
def user_open_request(home: Path) -> Iterator[None]:
    fd = _fd(home, "user-open-request")
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


@contextmanager
def planner_active(home: Path) -> Iterator[None]:
    fd = _fd(home, "planner-active")
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def acquire_user_open_request(home: Path) -> int:
    fd = _fd(home, "user-open-request")
    fcntl.flock(fd, fcntl.LOCK_SH)
    return fd


def release_user_open_request(fd: int) -> None:
    fcntl.flock(fd, fcntl.LOCK_UN)
    os.close(fd)


def planner_request_pending(home: Path) -> bool:
    fd = _fd(home, "user-open-request")
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)


def _wait_exclusive(
    fd: int,
    *,
    clock: Callable[[], float],
    wait: Callable[[float], None],
    timeout: float,
) -> bool:
    deadline = clock() + timeout
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            remaining = deadline - clock()
            if remaining <= 0:
                return False
            wait(remaining)


def open_with_retention_yield(
    home: Path,
    operation: Callable[[], object],
    *,
    clock: Callable[[], float] = time.monotonic,
    wait: Callable[[float], None] = time.sleep,
    timeout: float = _WAIT_DEADLINE,
    create_locks: bool = True,
) -> object:
    """Give retention a bounded chance to leave before running one open."""
    acquire = _fd if create_locks else _existing_fd
    request_fd = acquire(home, "user-open-request")
    active_fd = acquire(home, "planner-active")
    if request_fd is None or active_fd is None:
        if request_fd is not None:
            os.close(request_fd)
        if active_fd is not None:
            os.close(active_fd)
        return operation()
    try:
        fcntl.flock(request_fd, fcntl.LOCK_SH)
        acknowledged = _wait_exclusive(
            active_fd, clock=clock, wait=wait, timeout=timeout
        )
        if acknowledged:
            try:
                return operation()
            finally:
                fcntl.flock(active_fd, fcntl.LOCK_UN)
        try:
            return operation()
        except SessionInUseError as exc:
            try:
                fcntl.flock(active_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RetentionYieldError(
                    "retention did not yield to user open"
                ) from exc
            fcntl.flock(active_fd, fcntl.LOCK_UN)
            raise
    finally:
        fcntl.flock(request_fd, fcntl.LOCK_UN)
        os.close(active_fd)
        os.close(request_fd)
