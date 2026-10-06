"""Crash-safe storage for complete background task output."""

from __future__ import annotations

import binascii
import fcntl
import json
import os
import struct
import threading
import weakref

from ...core.session_files import open_session_file

_ARCHIVE_NAME = "background-output.archive"
_JOURNAL_NAME = "background-output.journal"
_TRANSACTION_LOCK_NAME = "background-output.lock"
_LOG_PREFIX = "background-"
_LOG_SUFFIX = ".log"
_COPY_CHUNK = 64 * 1024
_FRAME_HEADER = struct.Struct(">4sII")
_FRAME_MAGIC = b"ZBO1"


class _BackgroundOutputArchive:
    """Own one append-only output archive and its commit journal.

    Archive writes and journal commits are serialized across processes. A generated
    task log stays advisory-locked while its writer is live, so recovery only folds
    logs left by a dead writer into the archive.
    """

    def __init__(self, directory_fd: int) -> None:
        self._directory_fd = os.dup(directory_fd)
        opened: list[int] = []
        try:
            self._archive_fd = open_session_file(
                self._directory_fd, _ARCHIVE_NAME, os.O_RDWR | os.O_CREAT
            )
            opened.append(self._archive_fd)
            self._journal_fd = open_session_file(
                self._directory_fd, _JOURNAL_NAME, os.O_RDWR | os.O_CREAT
            )
            opened.append(self._journal_fd)
            self._transaction_fd = open_session_file(
                self._directory_fd, _TRANSACTION_LOCK_NAME, os.O_RDWR | os.O_CREAT
            )
            opened.append(self._transaction_fd)
            os.fsync(self._directory_fd)
        except BaseException:
            for fd in reversed(opened):
                os.close(fd)
            os.close(self._directory_fd)
            raise
        self._release_archive = weakref.finalize(self, os.close, self._archive_fd)
        self._release_journal = weakref.finalize(self, os.close, self._journal_fd)
        self._release_transaction = weakref.finalize(
            self, os.close, self._transaction_fd
        )
        self._release_directory = weakref.finalize(self, os.close, self._directory_fd)
        self._lock = threading.RLock()
        self._closed = False
        self._committed_length = 0
        self._ranges: dict[str, tuple[int, int]] = {}
        self._journal_position = 0
        self._journal_identity: tuple[int, int] | None = None

    def recover(self) -> dict[str, int]:
        """Restore committed ranges and archive logs left by dead writers."""

        with self._lock, self._transaction():
            self._require_open()
            self._reload_commits(full=True)
            for name in sorted(os.listdir(self._directory_fd)):
                task_id = self._generated_log_task_id(name)
                if task_id is None:
                    continue
                if task_id in self._ranges:
                    self._remove_generated_log(name)
                    continue
                source_fd = open_session_file(self._directory_fd, name, os.O_RDONLY)
                try:
                    try:
                        fcntl.flock(source_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        continue
                    self._append_locked(task_id, source_fd, os.fstat(source_fd).st_size)
                finally:
                    os.close(source_fd)
            return {task_id: length for task_id, (_, length) in self._ranges.items()}

    def open_task_log(self, task_id: str) -> int:
        """Create and lock one generated log, returning its owned descriptor."""

        self._validate_task_id(task_id)
        with self._lock, self._transaction():
            self._require_open()
            fd = open_session_file(
                self._directory_fd,
                self._generated_log_name(task_id),
                os.O_RDWR | os.O_CREAT | os.O_TRUNC,
            )
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BaseException:
                os.close(fd)
                raise
            return fd

    def append(self, task_id: str, source_fd: int, length: int) -> None:
        """Durably commit exactly ``length`` source bytes for ``task_id`` once."""

        self._validate_task_id(task_id)
        if type(length) is not int or length < 0:
            raise ValueError("archive length must be a nonnegative integer")
        with self._lock, self._transaction():
            self._require_open()
            self._reload_commits()
            self._append_locked(task_id, source_fd, length)

    def pread(self, task_id: str, offset: int, limit: int) -> bytes:
        """Read a validated task-relative range from the pinned archive."""

        if type(offset) is not int or offset < 0:
            raise ValueError("archive offset must be a nonnegative integer")
        if type(limit) is not int or limit < 0:
            raise ValueError("archive limit must be a nonnegative integer")
        with self._lock:
            self._require_open()
            if task_id not in self._ranges:
                with self._transaction():
                    self._reload_commits()
            archive_range = self._ranges.get(task_id)
            if archive_range is None or limit == 0:
                return b""
            archive_offset, length = archive_range
            remaining = max(0, length - offset)
            return os.pread(
                self._archive_fd,
                min(limit, remaining),
                archive_offset + min(offset, length),
            )

    def close(self) -> None:
        """Close all pinned archive descriptors."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._release_archive()
            self._release_journal()
            self._release_transaction()
            self._release_directory()

    def _append_locked(self, task_id: str, source_fd: int, length: int) -> None:
        existing = self._ranges.get(task_id)
        if existing is not None:
            if existing[1] != length:
                raise ValueError("background task output length changed after commit")
            self._remove_generated_log(self._generated_log_name(task_id))
            return

        offset = self._committed_length
        self._copy(source_fd, offset, length)
        os.fsync(self._archive_fd)
        self._append_commit(task_id, offset, length)
        journal_stat = os.fstat(self._journal_fd)
        self._journal_position = journal_stat.st_size
        self._journal_identity = (journal_stat.st_dev, journal_stat.st_ino)
        self._ranges[task_id] = (offset, length)
        self._committed_length = offset + length
        self._remove_generated_log(self._generated_log_name(task_id))

    def _reload_commits(self, *, full: bool = False) -> None:
        archive_size = os.fstat(self._archive_fd).st_size
        journal_stat = os.fstat(self._journal_fd)
        identity = (journal_stat.st_dev, journal_stat.st_ino)
        if (
            full
            or self._journal_identity != identity
            or journal_stat.st_size < self._journal_position
        ):
            position = 0
            committed_length = 0
            ranges: dict[str, tuple[int, int]] = {}
        else:
            position = self._journal_position
            committed_length = self._committed_length
            ranges = self._ranges.copy()
        committed_length, ranges, valid_position = self._load_journal(
            archive_size, position, committed_length, ranges
        )
        if journal_stat.st_size > valid_position:
            os.ftruncate(self._journal_fd, valid_position)
            os.fsync(self._journal_fd)
        if archive_size > committed_length:
            os.ftruncate(self._archive_fd, committed_length)
            os.fsync(self._archive_fd)
        self._committed_length = committed_length
        self._ranges = ranges
        self._journal_position = valid_position
        self._journal_identity = identity

    def _load_journal(
        self,
        archive_size: int,
        position: int,
        committed_length: int,
        ranges: dict[str, tuple[int, int]],
    ) -> tuple[int, dict[str, tuple[int, int]], int]:
        size = os.fstat(self._journal_fd).st_size
        data = os.pread(self._journal_fd, size - position, position)
        data_position = 0
        while position < size:
            frame_start = position
            if size - position < _FRAME_HEADER.size:
                break
            magic, payload_length, expected_checksum = _FRAME_HEADER.unpack_from(
                data, data_position
            )
            position += _FRAME_HEADER.size
            data_position += _FRAME_HEADER.size
            frame_end = position + payload_length
            if magic != _FRAME_MAGIC:
                raise OSError("background output journal is invalid")
            if frame_end > size:
                position = frame_start
                break
            payload = data[data_position : data_position + payload_length]
            if binascii.crc32(payload) != expected_checksum:
                if frame_end == size:
                    position = frame_start
                    break
                raise OSError("background output journal checksum is invalid")
            try:
                value = json.loads(payload)
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                if frame_end == size:
                    position = frame_start
                    break
                raise OSError("background output journal record is invalid") from exc
            task_id, offset, length = self._validate_commit(value)
            if task_id in ranges or offset != committed_length:
                raise OSError("background output journal has an invalid range")
            committed_length = offset + length
            if committed_length > archive_size:
                raise OSError("background output archive is shorter than its journal")
            ranges[task_id] = (offset, length)
            position = frame_end
            data_position += payload_length
        return committed_length, ranges, position

    @staticmethod
    def _validate_commit(value: object) -> tuple[str, int, int]:
        if type(value) is not dict or set(value) != {"task_id", "offset", "length"}:
            raise OSError("background output journal record is invalid")
        task_id = value["task_id"]
        offset = value["offset"]
        length = value["length"]
        if (
            type(task_id) is not str
            or not task_id
            or "/" in task_id
            or "\x00" in task_id
            or type(offset) is not int
            or offset < 0
            or type(length) is not int
            or length < 0
        ):
            raise OSError("background output journal record is invalid")
        return task_id, offset, length

    def _append_commit(self, task_id: str, offset: int, length: int) -> None:
        payload = json.dumps(
            {"task_id": task_id, "offset": offset, "length": length},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        frame = _FRAME_HEADER.pack(
            _FRAME_MAGIC, len(payload), binascii.crc32(payload)
        ) + payload
        journal_offset = os.fstat(self._journal_fd).st_size
        self._pwrite_all(self._journal_fd, frame, journal_offset)
        os.fsync(self._journal_fd)

    def _copy(self, source_fd: int, offset: int, length: int) -> None:
        copied = 0
        while copied < length:
            chunk = os.pread(source_fd, min(_COPY_CHUNK, length - copied), copied)
            if not chunk:
                raise OSError("background log ended before all output was archived")
            self._pwrite_all(self._archive_fd, chunk, offset + copied)
            copied += len(chunk)

    @staticmethod
    def _pwrite_all(fd: int, data: bytes, offset: int) -> None:
        view = memoryview(data)
        written = 0
        while written < len(view):
            count = os.pwrite(fd, view[written:], offset + written)
            if count == 0:
                raise OSError("zero-byte write to background output storage")
            written += count

    def _remove_generated_log(self, name: str) -> None:
        try:
            os.unlink(name, dir_fd=self._directory_fd)
        except FileNotFoundError:
            return
        os.fsync(self._directory_fd)

    def _transaction(self) -> _Flock:
        return _Flock(self._transaction_fd)

    @staticmethod
    def _validate_task_id(task_id: str) -> None:
        if not task_id or "/" in task_id or "\x00" in task_id:
            raise ValueError("invalid background task id")

    @staticmethod
    def _generated_log_name(task_id: str) -> str:
        return f"{_LOG_PREFIX}{task_id}{_LOG_SUFFIX}"

    @staticmethod
    def _generated_log_task_id(name: str) -> str | None:
        if not name.startswith(_LOG_PREFIX) or not name.endswith(_LOG_SUFFIX):
            return None
        task_id = name[len(_LOG_PREFIX) : -len(_LOG_SUFFIX)]
        return task_id if task_id.startswith("task-") else None

    def _require_open(self) -> None:
        if self._closed:
            raise RuntimeError("background output archive is closed")


class _Flock:
    """Hold one process-wide advisory lock for a synchronous transaction."""

    def __init__(self, fd: int) -> None:
        self._fd = fd

    def __enter__(self) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_EX)

    def __exit__(self, *exc_info: object) -> None:
        fcntl.flock(self._fd, fcntl.LOCK_UN)
