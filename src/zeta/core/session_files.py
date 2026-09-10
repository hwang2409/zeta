"""Pinned session directories and cooperative lifetime leases."""

from __future__ import annotations

import fcntl
import os
import stat
from contextlib import ExitStack, contextmanager
from pathlib import Path


class SessionError(ValueError):
    """Raised when a session cannot be accessed safely."""


class SessionInUseError(SessionError):
    """Raised when an open store or metadata operation prevents deletion."""


@contextmanager
def session_directory(root: Path, session_id: str, *, exclusive: bool = False):
    """Never create directories; lease the pinned directory inode itself.

    Shared leases cover store lifetimes and metadata mutations. Deletion needs
    an exclusive lease. Closing the descriptors (including on process death)
    releases the lease without leaving a stale lock file.
    """
    try:
        with ExitStack() as cleanup:
            root_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            cleanup.callback(os.close, root_fd)
            session_fd = os.open(
                session_id, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                dir_fd=root_fd,
            )
            cleanup.callback(os.close, session_fd)
            try:
                fcntl.flock(
                    session_fd,
                    (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB,
                )
            except BlockingIOError as exc:
                raise SessionInUseError("session is currently open or in use") from exc
            # A waiter must not operate on an inode that deletion already removed.
            if os.fstat(session_fd).st_nlink == 0:
                raise SessionError(f"session {session_id} was not found")
            yield root_fd, session_fd
    except FileNotFoundError as exc:
        raise SessionError(f"session {session_id} was not found") from exc
    except OSError as exc:
        raise SessionError(f"session {session_id} could not be accessed: {exc.strerror}") from exc


def open_session_file(directory_fd: int, name: str, flags: int) -> int:
    """Open one regular, unshared file without following a link or blocking on a FIFO."""
    fd = os.open(
        name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600, dir_fd=directory_fd,
    )
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(fd)
        raise SessionError(f"session file must be a regular, unshared file: {name}")
    return fd
