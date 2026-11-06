"""Verified atomic replacement for user-managed files."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path


class ConcurrentFileChangeError(ValueError):
    """The opened file no longer matches the file at its path."""


def _open_current(parent_fd: int, name: str) -> int:
    try:
        return os.open(
            name,
            os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
            dir_fd=parent_fd,
        )
    except OSError as exc:
        raise ConcurrentFileChangeError(
            "target changed while operation was in progress"
        ) from exc


def _verify_current(
    file_descriptor: int,
    expected_identity: tuple[int, int],
    expected_digest: bytes | None,
) -> None:
    current_stat = os.fstat(file_descriptor)
    if not stat.S_ISREG(current_stat.st_mode):
        raise ValueError("target is no longer a regular file")
    if current_stat.st_nlink > 1:
        raise ValueError("target has multiple hard links; refusing atomic replace")
    if (current_stat.st_dev, current_stat.st_ino) != expected_identity:
        raise ConcurrentFileChangeError(
            "target changed while operation was in progress"
        )
    if expected_digest is not None:
        duplicate = os.dup(file_descriptor)
        try:
            with os.fdopen(duplicate, "rb", closefd=False) as handle:
                digest = hashlib.file_digest(handle, "sha256").digest()
        finally:
            os.close(duplicate)
        if digest != expected_digest:
            raise ConcurrentFileChangeError(
                "target changed while operation was in progress"
            )


def atomic_replace(
    file_descriptor: int,
    path: str,
    content: bytes,
    *,
    expected_digest: bytes | None = None,
) -> None:
    """Replace an opened regular file after rechecking its path identity."""

    target_stat = os.fstat(file_descriptor)
    if not stat.S_ISREG(target_stat.st_mode):
        raise ValueError(f"not a file: {path}")
    if target_stat.st_nlink > 1:
        raise ValueError("target has multiple hard links; refusing atomic replace")

    target = Path(path)
    try:
        parent_fd = os.open(
            target.parent,
            os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
        )
    except OSError as exc:
        raise ValueError(
            f"could not open parent directory: {target.parent}: {exc}"
        ) from exc

    temporary_name = f".{target.name}.{secrets.token_hex(16)}.tmp"
    temporary_fd = -1
    published = False
    try:
        temporary_fd = os.open(
            temporary_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
            stat.S_IMODE(target_stat.st_mode),
            dir_fd=parent_fd,
        )
        try:
            os.fchmod(temporary_fd, stat.S_IMODE(target_stat.st_mode))
            with os.fdopen(temporary_fd, "wb", closefd=False) as handle:
                handle.write(content)
                handle.flush()
            os.fsync(temporary_fd)
        except OSError as exc:
            raise ValueError(f"could not write file: {path}: {exc}") from exc

        current_fd = _open_current(parent_fd, target.name)
        try:
            _verify_current(
                current_fd,
                (target_stat.st_dev, target_stat.st_ino),
                expected_digest,
            )
            os.replace(
                temporary_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
            published = True
            os.fsync(parent_fd)
        finally:
            os.close(current_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if not published:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
