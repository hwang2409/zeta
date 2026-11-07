"""Verified atomic publication for user-managed files."""

from __future__ import annotations

import asyncio
import errno
import hashlib
import os
import secrets
import stat
import threading
from collections.abc import Callable
from enum import Enum, auto
from pathlib import Path
from typing import Any

from ..core.abort import AbortSignal


class ConcurrentFileChangeError(ValueError):
    """The opened file no longer matches the file at its path."""


class FilePublishedWithoutDurabilityError(RuntimeError):
    """Publication completed before cancellation, but its fsync did not."""


class FilePublicationOutcomeUnknownError(RuntimeError):
    """Cancellation overlapped the publication syscall."""


class PublicationState(Enum):
    CANCELED = auto()
    PUBLISHING = auto()
    PUBLISHED = auto()


class PublicationGuard:
    """Let cancellation and one publication syscall choose one winner."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._canceled = False
        self._published = False

    def publish(self, function: Callable[[], None]) -> None:
        with self._lock:
            if self._canceled:
                raise asyncio.CancelledError
            function()
            self._published = True

    def cancel(self) -> PublicationState:
        """Inspect or cancel publication without waiting for its syscall."""

        if not self._lock.acquire(blocking=False):
            return PublicationState.PUBLISHING
        try:
            if self._published:
                return PublicationState.PUBLISHED
            self._canceled = True
            return PublicationState.CANCELED
        finally:
            self._lock.release()


def _retrieve_worker_result(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


async def run_file_worker[Result](
    function: Callable[..., Result],
    /,
    *args: Any,
    signal: AbortSignal,
    **kwargs: Any,
) -> Result:
    """Keep ownership of file work until its disk outcome is known."""

    publication = PublicationGuard()
    task = asyncio.create_task(
        asyncio.to_thread(function, *args, publication=publication, **kwargs)
    )
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        publication_state = publication.cancel()
        if task.done():
            return task.result()
        task.add_done_callback(_retrieve_worker_result)
        if publication_state is PublicationState.PUBLISHING and signal.is_set():
            raise FilePublicationOutcomeUnknownError(
                "publication was in progress; file write outcome is unknown"
            ) from None
        if publication_state is PublicationState.PUBLISHED and signal.is_set():
            raise FilePublishedWithoutDurabilityError(
                "file was written; durability was not confirmed before cancellation"
            ) from None
        raise


def _temporary_name(name: str) -> str:
    return f".{name}.{secrets.token_hex(16)}.tmp"


def _write_temporary(
    parent_fd: int,
    temporary_name: str,
    content: bytes,
    mode: int,
) -> int:
    temporary_fd = os.open(
        temporary_name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC,
        mode,
        dir_fd=parent_fd,
    )
    try:
        with os.fdopen(temporary_fd, "wb", closefd=False) as handle:
            handle.write(content)
            handle.flush()
        os.fsync(temporary_fd)
    except BaseException:
        os.close(temporary_fd)
        raise
    return temporary_fd


def atomic_create(
    parent_fd: int,
    name: str,
    path: str,
    content: bytes,
    abort_signal: AbortSignal,
    publication: PublicationGuard,
) -> None:
    """Publish a complete new file without replacing an existing target."""

    try:
        os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    except FileNotFoundError:
        pass
    else:
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), path)

    temporary_name = _temporary_name(name)
    temporary_fd = -1
    try:
        try:
            temporary_fd = _write_temporary(
                parent_fd,
                temporary_name,
                content,
                0o666,
            )
        except OSError as exc:
            raise ValueError(f"could not write file: {path}: {exc}") from exc
        if abort_signal.is_set():
            raise asyncio.CancelledError
        publication.publish(
            lambda: os.link(
                temporary_name,
                name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        )
        os.unlink(temporary_name, dir_fd=parent_fd)
        os.fsync(parent_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        try:
            os.unlink(temporary_name, dir_fd=parent_fd)
        except FileNotFoundError:
            pass


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
    abort_signal: AbortSignal,
    publication: PublicationGuard,
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

    temporary_name = _temporary_name(target.name)
    temporary_fd = -1
    published = False
    try:
        try:
            temporary_fd = _write_temporary(
                parent_fd,
                temporary_name,
                content,
                stat.S_IMODE(target_stat.st_mode),
            )
            os.fchmod(temporary_fd, stat.S_IMODE(target_stat.st_mode))
        except OSError as exc:
            raise ValueError(f"could not write file: {path}: {exc}") from exc

        current_fd = _open_current(parent_fd, target.name)
        try:
            expected_identity = target_stat.st_dev, target_stat.st_ino
            _verify_current(current_fd, expected_identity, expected_digest)
        finally:
            os.close(current_fd)

        final_fd = _open_current(parent_fd, target.name)
        try:
            _verify_current(final_fd, expected_identity, None)
        finally:
            os.close(final_fd)
        if abort_signal.is_set():
            raise asyncio.CancelledError
        publication.publish(
            lambda: os.replace(
                temporary_name,
                target.name,
                src_dir_fd=parent_fd,
                dst_dir_fd=parent_fd,
            )
        )
        published = True
        os.fsync(parent_fd)
    finally:
        if temporary_fd >= 0:
            os.close(temporary_fd)
        if not published:
            try:
                os.unlink(temporary_name, dir_fd=parent_fd)
            except FileNotFoundError:
                pass
        os.close(parent_fd)
