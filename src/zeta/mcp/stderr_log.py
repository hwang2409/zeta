"""Private, bounded stderr logs for stdio MCP servers."""

from __future__ import annotations

import asyncio
import errno
import fcntl
import logging
import os
import stat
import time
from pathlib import Path

MCP_STDERR_LOG_FILE_BYTES = 2 * 1024 * 1024
MCP_STDERR_LOG_FILE_COUNT = 3
_READ_BYTES = 64 * 1024
_WRITE_QUEUE_BYTES = 8 * 1024 * 1024
_LOCK_TIMEOUT_SECONDS = 1.0
_LOCK_RETRY_SECONDS = 0.01

logger = logging.getLogger(__name__)


class _FileLockTimeout(TimeoutError):
    pass


class RotatingStderrLog:
    """Drain one stderr stream into securely opened, process-safe log files."""

    def __init__(self, path: Path, directory_fd: int, lock_fd: int) -> None:
        self.path = path
        self._directory_fd = directory_fd
        self._lock_fd = lock_fd

    @classmethod
    async def prepare(cls, path: Path) -> RotatingStderrLog:
        return await asyncio.to_thread(cls._prepare, path)

    @classmethod
    def _prepare(cls, path: Path) -> RotatingStderrLog:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        directory_fd = os.open(
            path.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
        )
        lock_fd: int | None = None
        try:
            os.fchmod(directory_fd, 0o700)
            lock_fd = _open_regular(
                directory_fd,
                f".{path.name}.lock",
                os.O_RDWR | os.O_CREAT,
            )
            log = cls(path, directory_fd, lock_fd)
            with log._locked():
                log._normalize_existing()
            return log
        except BaseException:
            if lock_fd is not None:
                os.close(lock_fd)
            os.close(directory_fd)
            raise

    async def drain(self, stream: asyncio.StreamReader) -> None:
        queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        queued_bytes = 0
        dropped_bytes = 0
        drop_episode = False

        def record_drop(byte_count: int) -> None:
            nonlocal dropped_bytes, drop_episode
            dropped_bytes += byte_count
            if not drop_episode:
                drop_episode = True
                logger.warning(
                    "dropping stderr bytes from MCP server log %s", self.path
                )

        async def append(chunk: bytes) -> bool:
            nonlocal dropped_bytes, drop_episode
            marker_bytes = dropped_bytes
            if not await asyncio.to_thread(self._append, chunk, marker_bytes):
                return False
            dropped_bytes -= marker_bytes
            if dropped_bytes == 0:
                drop_episode = False
            return True

        async def write_queued() -> None:
            nonlocal queued_bytes
            while (chunk := await queue.get()) is not None:
                queued_bytes -= len(chunk)
                if await append(chunk):
                    continue
                record_drop(len(chunk))
                closing = False
                while True:
                    try:
                        pending = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                    if pending is None:
                        closing = True
                        continue
                    queued_bytes -= len(pending)
                    record_drop(len(pending))
                if closing:
                    return
            if dropped_bytes:
                await append(b"")

        writer = asyncio.create_task(write_queued())
        try:
            while chunk := await stream.read(_READ_BYTES):
                if writer.done():
                    await writer
                if queued_bytes + len(chunk) <= _WRITE_QUEUE_BYTES:
                    queue.put_nowait(chunk)
                    queued_bytes += len(chunk)
                else:
                    record_drop(len(chunk))
        finally:
            if not writer.done():
                queue.put_nowait(None)
            try:
                await writer
            finally:
                await self.close()

    async def close(self) -> None:
        directory_fd, self._directory_fd = self._directory_fd, -1
        lock_fd, self._lock_fd = self._lock_fd, -1
        if directory_fd >= 0 or lock_fd >= 0:
            await asyncio.to_thread(_close_fds, directory_fd, lock_fd)

    def _append(self, data: bytes, dropped_bytes: int = 0) -> bool:
        try:
            with self._locked():
                marker = (
                    f"[zeta: dropped {dropped_bytes} bytes of stderr]\n".encode()
                    if dropped_bytes
                    else b""
                )
                payload = marker + data
                fd = self._open_log(0, create=True)
                try:
                    if os.fstat(fd).st_size + len(payload) > MCP_STDERR_LOG_FILE_BYTES:
                        os.close(fd)
                        fd = -1
                        self._rotate()
                        fd = self._open_log(0, create=True)
                    view = memoryview(payload)
                    while view:
                        written = os.write(fd, view)
                        view = view[written:]
                finally:
                    if fd >= 0:
                        os.close(fd)
        except _FileLockTimeout:
            return False
        return True

    def _normalize_existing(self) -> None:
        for index in range(MCP_STDERR_LOG_FILE_COUNT):
            try:
                fd = self._open_log(index)
            except FileNotFoundError:
                continue
            try:
                if os.fstat(fd).st_size > MCP_STDERR_LOG_FILE_BYTES:
                    os.ftruncate(fd, MCP_STDERR_LOG_FILE_BYTES)
            finally:
                os.close(fd)
        fd = self._open_log(0, create=True)
        try:
            should_rotate = os.fstat(fd).st_size >= MCP_STDERR_LOG_FILE_BYTES
        finally:
            os.close(fd)
        if should_rotate:
            self._rotate()

    def _rotate(self) -> None:
        for index in range(MCP_STDERR_LOG_FILE_COUNT):
            try:
                fd = self._open_log(index)
            except FileNotFoundError:
                continue
            else:
                os.close(fd)

        oldest = self._name(MCP_STDERR_LOG_FILE_COUNT - 1)
        try:
            os.unlink(oldest, dir_fd=self._directory_fd)
        except FileNotFoundError:
            pass
        for index in range(MCP_STDERR_LOG_FILE_COUNT - 2, -1, -1):
            try:
                os.replace(
                    self._name(index),
                    self._name(index + 1),
                    src_dir_fd=self._directory_fd,
                    dst_dir_fd=self._directory_fd,
                )
            except FileNotFoundError:
                continue

    def _open_log(self, index: int, *, create: bool = False) -> int:
        flags = os.O_RDWR | os.O_APPEND | (os.O_CREAT if create else 0)
        return _open_regular(self._directory_fd, self._name(index), flags)

    def _name(self, index: int) -> str:
        return self.path.name if index == 0 else f"{self.path.name}.{index}"

    def _locked(self):
        return _FileLock(self._lock_fd)


class _FileLock:
    def __init__(self, fd: int) -> None:
        self._fd = fd

    def __enter__(self) -> None:
        deadline = time.monotonic() + _LOCK_TIMEOUT_SECONDS
        while True:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except BlockingIOError as exc:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise _FileLockTimeout("timed out locking MCP stderr log") from exc
                time.sleep(min(_LOCK_RETRY_SECONDS, remaining))

    def __exit__(self, *_args: object) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_UN)


def _open_regular(directory_fd: int, name: str, flags: int) -> int:
    try:
        fd = os.open(
            name,
            flags | os.O_NOFOLLOW | os.O_NONBLOCK | os.O_CLOEXEC,
            0o600,
            dir_fd=directory_fd,
        )
    except OSError as exc:
        if exc.errno == errno.ELOOP:
            raise OSError(errno.EINVAL, "refusing symbolic link", name) from exc
        raise
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise OSError(errno.EINVAL, "not a regular file", name)
        if info.st_nlink != 1:
            raise OSError(errno.EINVAL, "refusing file with multiple links", name)
        os.fchmod(fd, 0o600)
        return fd
    except BaseException:
        os.close(fd)
        raise


def _close_fds(directory_fd: int, lock_fd: int) -> None:
    if lock_fd >= 0:
        os.close(lock_fd)
    if directory_fd >= 0:
        os.close(directory_fd)
