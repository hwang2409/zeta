"""Crash-safe storage for complete background task output."""

from __future__ import annotations

import json
import os
import threading
import weakref

from ...core.session_files import (
    atomic_publish_file,
    open_session_file,
    read_session_file,
)

_ARCHIVE_NAME = "background-output.archive"
_MANIFEST_NAME = "background-output.manifest.json"
_LOG_PREFIX = "background-"
_LOG_SUFFIX = ".log"
_COPY_CHUNK = 64 * 1024


class _BackgroundOutputArchive:
    """Own one append-only archive and its crash-safe range manifest.

    A committed append has three ordered durability steps: write and sync archive
    data, atomically publish and sync the manifest, then remove the generated
    task log. Recovery truncates only data beyond the manifest's committed
    length and folds surviving generated logs into the archive idempotently.
    """

    def __init__(self, directory_fd: int) -> None:
        self._directory_fd = os.dup(directory_fd)
        try:
            self._archive_fd = open_session_file(
                self._directory_fd,
                _ARCHIVE_NAME,
                os.O_RDWR | os.O_CREAT,
            )
        except BaseException:
            os.close(self._directory_fd)
            raise
        self._release_archive = weakref.finalize(self, os.close, self._archive_fd)
        self._release_directory = weakref.finalize(self, os.close, self._directory_fd)
        self._lock = threading.RLock()
        self._closed = False
        self._committed_length = 0
        self._ranges: dict[str, tuple[int, int]] = {}

    def recover(self) -> dict[str, int]:
        """Restore committed ranges and archive any surviving generated logs."""

        with self._lock:
            self._require_open()
            committed_length, ranges = self._load_manifest()
            archive_size = os.fstat(self._archive_fd).st_size
            self._validate_manifest(committed_length, ranges, archive_size)
            if archive_size > committed_length:
                os.ftruncate(self._archive_fd, committed_length)
                os.fsync(self._archive_fd)
            self._committed_length = committed_length
            self._ranges = ranges

            for name in sorted(os.listdir(self._directory_fd)):
                task_id = self._generated_log_task_id(name)
                if task_id is None:
                    continue
                if task_id in self._ranges:
                    self._remove_generated_log(name)
                    continue
                source_fd = open_session_file(self._directory_fd, name, os.O_RDONLY)
                try:
                    self.append(task_id, source_fd, os.fstat(source_fd).st_size)
                finally:
                    os.close(source_fd)
            return {task_id: length for task_id, (_, length) in self._ranges.items()}

    def append(self, task_id: str, source_fd: int, length: int) -> None:
        """Durably commit exactly ``length`` source bytes for ``task_id`` once."""

        if not task_id or "/" in task_id or "\x00" in task_id:
            raise ValueError("invalid background task id")
        if type(length) is not int or length < 0:
            raise ValueError("archive length must be a nonnegative integer")
        with self._lock:
            self._require_open()
            existing = self._ranges.get(task_id)
            if existing is not None:
                if existing[1] != length:
                    raise ValueError("background task output length changed after commit")
                self._remove_generated_log(self._generated_log_name(task_id))
                return

            archive_size = os.fstat(self._archive_fd).st_size
            if archive_size < self._committed_length:
                raise OSError("background output archive is shorter than its manifest")
            if archive_size > self._committed_length:
                os.ftruncate(self._archive_fd, self._committed_length)

            offset = self._committed_length
            self._copy(source_fd, offset, length)
            os.fsync(self._archive_fd)

            ranges = {**self._ranges, task_id: (offset, length)}
            committed_length = offset + length
            self._persist_manifest(committed_length, ranges)
            self._ranges = ranges
            self._committed_length = committed_length
            self._remove_generated_log(self._generated_log_name(task_id))

    def pread(self, task_id: str, offset: int, limit: int) -> bytes:
        """Read a validated task-relative range from the pinned archive."""

        if type(offset) is not int or offset < 0:
            raise ValueError("archive offset must be a nonnegative integer")
        if type(limit) is not int or limit < 0:
            raise ValueError("archive limit must be a nonnegative integer")
        with self._lock:
            self._require_open()
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
        """Close the pinned archive and session directory descriptors."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._release_archive()
            self._release_directory()

    def _load_manifest(self) -> tuple[int, dict[str, tuple[int, int]]]:
        try:
            raw = read_session_file(self._directory_fd, _MANIFEST_NAME)
        except FileNotFoundError:
            return 0, {}
        try:
            value = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise OSError("background output manifest is invalid") from exc
        if type(value) is not dict or value.get("version") != 1:
            raise OSError("background output manifest is invalid")
        committed_length = value.get("committed_length")
        records = value.get("records")
        if type(committed_length) is not int or committed_length < 0 or type(records) is not dict:
            raise OSError("background output manifest is invalid")
        ranges: dict[str, tuple[int, int]] = {}
        for task_id, item in records.items():
            if type(task_id) is not str or type(item) is not dict:
                raise OSError("background output manifest is invalid")
            offset = item.get("offset")
            length = item.get("length")
            if type(offset) is not int or type(length) is not int:
                raise OSError("background output manifest is invalid")
            ranges[task_id] = (offset, length)
        return committed_length, ranges

    @staticmethod
    def _validate_manifest(
        committed_length: int,
        ranges: dict[str, tuple[int, int]],
        archive_size: int,
    ) -> None:
        if committed_length > archive_size:
            raise OSError("background output archive is shorter than its manifest")
        ordered = sorted(ranges.values())
        previous_end = 0
        for offset, length in ordered:
            if offset < 0 or length < 0 or offset < previous_end:
                raise OSError("background output manifest has an invalid range")
            end = offset + length
            if end > committed_length:
                raise OSError("background output manifest range exceeds committed data")
            previous_end = end

    def _copy(self, source_fd: int, offset: int, length: int) -> None:
        copied = 0
        while copied < length:
            chunk = os.pread(source_fd, min(_COPY_CHUNK, length - copied), copied)
            if not chunk:
                raise OSError("background log ended before all output was archived")
            written = 0
            while written < len(chunk):
                count = os.pwrite(
                    self._archive_fd,
                    chunk[written:],
                    offset + copied + written,
                )
                if count == 0:
                    raise OSError("zero-byte write to background output archive")
                written += count
            copied += len(chunk)

    def _persist_manifest(
        self, committed_length: int, ranges: dict[str, tuple[int, int]]
    ) -> None:
        records = {
            task_id: {"offset": offset, "length": length}
            for task_id, (offset, length) in ranges.items()
        }
        data = (
            json.dumps(
                {
                    "version": 1,
                    "committed_length": committed_length,
                    "records": records,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode()
        atomic_publish_file(
            self._directory_fd,
            _MANIFEST_NAME,
            data,
            sync_directory=True,
        )

    def _remove_generated_log(self, name: str) -> None:
        try:
            os.unlink(name, dir_fd=self._directory_fd)
        except FileNotFoundError:
            return
        os.fsync(self._directory_fd)

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
