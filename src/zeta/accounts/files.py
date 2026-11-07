"""Small account-storage file operations."""

from __future__ import annotations

import os

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
