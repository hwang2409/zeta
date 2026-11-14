"""Cross-process coordination for retention previews and user opens."""

from __future__ import annotations

import fcntl
import os
import stat
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from ..locking import RETENTION_HOME_ID, domain_lock_name


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
            try:
                fd = os.open(
                    name,
                    os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK,
                    0o600,
                    dir_fd=locks_fd,
                )
            except FileNotFoundError:
                # Another owner can publish the lock directory concurrently.
                return _lock_file(home, name)
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


def ensure_handshake_files(home: Path) -> None:
    """Create the owner handshake files before retention serves requests."""
    for domain in ("planner-active", "user-open-request"):
        fd = _fd(home, domain)
        os.close(fd)


def _existing_handshake_file(home: Path, domain: str) -> int:
    path = Path(home) / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, domain)
    fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise OSError("retention lock must be a regular, unshared file")
    return fd


@contextmanager
def planner_active(home: Path) -> Iterator[None]:
    fd = _existing_handshake_file(home, "planner-active")
    try:
        fcntl.flock(fd, fcntl.LOCK_SH)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def planner_request_pending(home: Path) -> bool:
    path = Path(home) / ".domain-locks" / domain_lock_name(RETENTION_HOME_ID, "user-open-request")
    try:
        fd = os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK)
    except FileNotFoundError:
        return False
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
        fcntl.flock(fd, fcntl.LOCK_UN)
        return False
    finally:
        os.close(fd)
