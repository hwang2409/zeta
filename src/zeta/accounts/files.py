"""Small account-storage file operations."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager

from ..core.session_files import open_session_file


def read_optional_bytes(directory_fd: int, name: str) -> bytes | None:
    try:
        fd = open_session_file(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(fd, "rb") as handle:
        return handle.read()


def read_optional(directory_fd: int, name: str) -> str | None:
    try:
        fd = open_session_file(directory_fd, name, os.O_RDONLY)
    except FileNotFoundError:
        return None
    with os.fdopen(fd) as handle:
        return handle.read()


def exists(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def file_mtime_ns(directory_fd: int, name: str) -> int:
    fd = open_session_file(directory_fd, name, os.O_RDONLY)
    try:
        return os.fstat(fd).st_mtime_ns
    finally:
        os.close(fd)


@contextmanager
def owned_fd(fd: int) -> Iterator[int]:
    try:
        yield fd
    finally:
        os.close(fd)


def mkdir_child_durable(parent_fd: int, name: str) -> int:
    try:
        os.mkdir(name, mode=0o700, dir_fd=parent_fd)
    except FileExistsError:
        pass
    else:
        fsync_directory(parent_fd)
    fd = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent_fd)
    os.fchmod(fd, 0o700)
    return fd


def fsync_directory(directory_fd: int) -> None:
    os.fsync(directory_fd)
