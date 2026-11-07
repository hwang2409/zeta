"""Session-root identity checks and live shell cwd recovery."""

from __future__ import annotations

import errno
import os
from pathlib import Path

_DIRECTORY_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


class SessionCwdRemoved(ValueError):
    """The captured session cwd pathname no longer names a directory."""


def open_session_cwd(path: Path, identity: tuple[int, int]) -> int:
    """Open the captured session cwd, separating removal from replacement."""

    try:
        cwd_fd = os.open(path, _DIRECTORY_FLAGS)
    except OSError as exc:
        if exc.errno in {errno.ENOENT, errno.ENOTDIR}:
            raise SessionCwdRemoved(
                f"session cwd {path} was removed; use an absolute path"
            ) from exc
        raise ValueError("session cwd was replaced") from exc
    try:
        cwd_stat = os.fstat(cwd_fd)
    except OSError as exc:
        os.close(cwd_fd)
        raise ValueError("session cwd was replaced") from exc
    if (cwd_stat.st_dev, cwd_stat.st_ino) != identity:
        os.close(cwd_fd)
        raise ValueError("session cwd was replaced")
    return cwd_fd


def open_verified_directory(path: str | Path, identity: tuple[int, int]) -> int:
    """Open a separately approved directory only while its identity matches."""

    try:
        fd = os.open(path, _DIRECTORY_FLAGS)
        stat_result = os.fstat(fd)
    except OSError as exc:
        if "fd" in locals():
            os.close(fd)
        raise ValueError("approved shell cwd was replaced") from exc
    if (stat_result.st_dev, stat_result.st_ino) != identity:
        os.close(fd)
        raise ValueError("approved shell cwd was replaced")
    return fd


def nearest_existing_directory(path: str | os.PathLike[str]) -> str:
    """Return path, or its nearest ancestor that is still a directory."""

    candidate = Path(os.path.abspath(os.fspath(path)))
    while not candidate.is_dir():
        parent = candidate.parent
        if parent == candidate:
            raise ValueError(f"no existing directory contains removed shell cwd {path}")
        candidate = parent
    return str(candidate)
