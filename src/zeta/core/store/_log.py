"""Incremental append-only conversation log synchronization."""

from __future__ import annotations

import hashlib
import json
import os
import time
import warnings
from typing import TYPE_CHECKING, Any

from ...agent.receipt import encode_json
from ..checkpoints import (
    ConversationEntry,
    ConversationIntegrityError,
    _now,
    load_session_json,
)
from ..session_files import open_session_file, read_session_file
from ._validation import TASK_EXITED_NOTIFICATION_KIND

if TYPE_CHECKING:
    from ._store import ConversationStore

SCHEMA = "zeta.conversation.v1"
PREFIX_FINGERPRINT_BYTES = 64 * 1024


class ConversationLogMixin:
    """Load a log once, then synchronize only bytes appended by other writers."""

    def _load(self: ConversationStore) -> None:
        """Synchronize in-memory state with the durable log.

        Callers hold the session append flock unless the store is read-only.
        Replacements, truncations, and non-linear tails fall back to a complete
        parse; ordinary external appends read and validate only the new bytes.
        """
        try:
            fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        except FileNotFoundError:
            if self._read_only or self._must_exist:
                raise ConversationIntegrityError(
                    f"conversation file is missing: {self.path}"
                )
            self._entries = []
            self._reset_incremental_validation_state()
            header = {
                "schema": SCHEMA,
                "session_id": self.session_id,
                "cwd": self.cwd,
                "created_at": _now(),
            }
            self._write_line({"type": "header", "data": header})
            return

        try:
            stat = os.fstat(fd)
            identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
            known_identity = getattr(self, "_log_identity", None)
            known_offset = getattr(self, "_log_offset", 0)
            known_mtime = getattr(self, "_log_mtime_ns", None)
            replaced = known_identity is None or known_identity[:2] != identity[:2]
            metadata_changed = (
                known_identity != identity or stat.st_mtime_ns != known_mtime
            )
            if replaced or stat.st_size < known_offset:
                os.close(fd)
                fd = -1
                self._load_full()
                return
            if stat.st_size == known_offset:
                if metadata_changed:
                    os.close(fd)
                    fd = -1
                    self._load_full()
                return
            if not self._known_prefix_matches(fd, known_offset):
                os.close(fd)
                fd = -1
                self._load_full()
                return
            os.lseek(fd, known_offset, os.SEEK_SET)
            raw = self._read_fd(fd, stat.st_size - known_offset)
        finally:
            if fd >= 0:
                os.close(fd)

        entries, torn_at = self._parse_entry_lines(
            raw, first_row=len(self._entries) + 2
        )
        if torn_at is not None:
            if self._read_only:
                raise ConversationIntegrityError(
                    f"invalid conversation row {len(self._entries) + len(entries) + 2}: {self.path}"
                )
            self._repair_torn_tail(known_offset + torn_at)
            # Parse any complete rows before the torn suffix before recording the warning.
            if not self._accept_incremental_entries(entries):
                self._load_full()
                return
            self._log_offset = known_offset + torn_at
            self._append_torn_warning()
            return

        if not self._accept_incremental_entries(entries):
            self._load_full()
            return
        if not self._read_only and not raw.endswith(b"\n"):
            # Preserve the loader's historical recovery for a complete JSON row
            # whose writer omitted only the line terminator.
            self._write_bytes(b"\n")
        else:
            self._set_log_stat(stat)

    def _load_full(self: ConversationStore) -> None:
        raw = read_session_file(self.directory_fd, "conversation.jsonl")
        lines = raw.splitlines(keepends=True)
        valid_rows: list[dict[str, Any]] = []
        torn_offset: int | None = None
        offset = 0
        for index, line in enumerate(lines):
            try:
                row = load_session_json(line)
            except ConversationIntegrityError as exc:
                if (
                    self._read_only
                    or index != len(lines) - 1
                    or line.endswith(b"\n")
                    or not isinstance(
                        exc.__cause__, (json.JSONDecodeError, UnicodeError)
                    )
                ):
                    kind = "terminated " if line.endswith(b"\n") else ""
                    raise ConversationIntegrityError(
                        f"invalid {kind}conversation row {index + 1}: {self.path}"
                    ) from exc
                torn_offset = offset
                break
            if not isinstance(row, dict):
                raise ConversationIntegrityError(
                    f"conversation row {index + 1} is not an object: {self.path}"
                )
            valid_rows.append(row)
            offset += len(line)

        if not valid_rows:
            raise ConversationIntegrityError(f"conversation file is empty: {self.path}")
        self._validate_header(valid_rows[0])
        try:
            self._entries = [ConversationEntry.from_dict(row) for row in valid_rows[1:]]
        except ConversationIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversationIntegrityError(
                f"invalid conversation entry: {self.path}"
            ) from exc
        self._validate_entries()
        self._task_notification_ids = set()
        for entry in self._entries:
            self._validate_entry_payload(entry)
            if entry.type == "fork":
                self._validate_fork_entry(entry)
            if entry.type == "notification" and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND:
                task_id = entry.data.get("task_id")
                if type(task_id) is str and task_id:
                    self._task_notification_ids.add(task_id)
        self._rebuild_incremental_validation_state()

        if torn_offset is not None:
            self._repair_torn_tail(torn_offset)
            self._log_offset = torn_offset
            self._append_torn_warning()
        elif not self._read_only and not raw.endswith(b"\n"):
            self._write_bytes(b"\n")
        else:
            fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
            try:
                self._set_log_stat(os.fstat(fd))
            finally:
                os.close(fd)

    def _validate_header(self: ConversationStore, header: dict[str, Any]) -> None:
        header_data = header.get("data")
        if (
            type(header.get("type")) is not str
            or header.get("type") != "header"
            or type(header_data) is not dict
            or type(header_data.get("schema")) is not str
            or header_data.get("schema") != SCHEMA
        ):
            raise ConversationIntegrityError(
                f"unsupported conversation schema: {self.path}"
            )
        header_session_id = header_data.get("session_id")
        cwd = header_data.get("cwd")
        created_at = header_data.get("created_at")
        if (
            type(header_session_id) is not str
            or not header_session_id
            or type(cwd) is not str
            or not cwd
            or type(created_at) is not str
        ):
            raise ConversationIntegrityError(
                f"conversation header is incomplete: {self.path}"
            )
        self.cwd = cwd
        if header_session_id != self.session_id:
            raise ConversationIntegrityError(
                f"conversation header session id mismatch: {self.path}"
            )

    def _parse_entry_lines(
        self: ConversationStore, raw: bytes, *, first_row: int
    ) -> tuple[list[ConversationEntry], int | None]:
        lines = raw.splitlines(keepends=True)
        entries: list[ConversationEntry] = []
        offset = 0
        for index, line in enumerate(lines):
            try:
                row = load_session_json(line)
            except ConversationIntegrityError as exc:
                if (
                    index != len(lines) - 1
                    or line.endswith(b"\n")
                    or not isinstance(
                        exc.__cause__, (json.JSONDecodeError, UnicodeError)
                    )
                ):
                    kind = "terminated " if line.endswith(b"\n") else ""
                    raise ConversationIntegrityError(
                        f"invalid {kind}conversation row {first_row + index}: {self.path}"
                    ) from exc
                return entries, offset
            if not isinstance(row, dict):
                raise ConversationIntegrityError(
                    f"conversation row {first_row + index} is not an object: {self.path}"
                )
            try:
                entries.append(ConversationEntry.from_dict(row))
            except ConversationIntegrityError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ConversationIntegrityError(
                    f"invalid conversation entry: {self.path}"
                ) from exc
            offset += len(line)
        return entries, None

    @staticmethod
    def _read_fd(fd: int, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _repair_torn_tail(self: ConversationStore, offset: int) -> None:
        with os.fdopen(
            open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDWR),
            "r+b",
        ) as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
            self._set_log_stat(os.fstat(handle.fileno()))

    def _append_torn_warning(self: ConversationStore) -> None:
        self._append_row_unlocked(
            "warning", {"message": "dropped torn final conversation line"}
        )
        warnings.warn(
            f"dropped torn final conversation line from {self.path}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _write_line(self: ConversationStore, row: dict[str, Any]) -> None:
        encoded = encode_json(row) + b"\n"
        if (
            self._write_deadline is not None
            and time.monotonic() >= self._write_deadline
        ):
            from ._store import PendingPromptCommitTimeoutError

            raise PendingPromptCommitTimeoutError(
                "pending prompt commit deadline exceeded"
            )
        self._write_bytes(encoded)

    def _write_bytes(self: ConversationStore, data: bytes) -> None:
        with os.fdopen(
            open_session_file(
                self.directory_fd,
                "conversation.jsonl",
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            ),
            "ab",
        ) as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            self._set_log_stat(os.fstat(handle.fileno()))

    @staticmethod
    def _prefix_fingerprint(fd: int, offset: int) -> bytes:
        length = min(offset, PREFIX_FINGERPRINT_BYTES)
        start = offset - length
        return hashlib.blake2b(os.pread(fd, length, start), digest_size=16).digest()

    def _known_prefix_matches(self: ConversationStore, fd: int, offset: int) -> bool:
        known = getattr(self, "_log_prefix_fingerprint", None)
        return known is not None and self._prefix_fingerprint(fd, offset) == known

    def _set_log_stat(self: ConversationStore, stat: os.stat_result) -> None:
        fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        try:
            fingerprint = self._prefix_fingerprint(fd, stat.st_size)
        finally:
            os.close(fd)
        self._log_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        self._log_offset = stat.st_size
        self._log_mtime_ns = stat.st_mtime_ns
        self._log_prefix_fingerprint = fingerprint
