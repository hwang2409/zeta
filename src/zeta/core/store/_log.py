"""Incremental append-only conversation log synchronization.

The log contract permits only appends by cooperating Zeta writers under the
session flock. Mutation detection is intentionally bounded: it detects file
replacement, truncation, and rewrites in the most recent 64 KiB, but an in-place
rewrite earlier in the prefix followed by growth is not detected by a resident
store. Reopen the session to force a full load after unsupported external edits.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import time
import warnings
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class PersistedAppend:
    """Proof of one byte range durably appended by this store process."""

    start_offset: int
    end_offset: int
    digest: str
    source_device: int
    source_inode: int
    before_mtime_ns: int
    before_ctime_ns: int
    after_mtime_ns: int
    after_ctime_ns: int


SCHEMA = "zeta.conversation.v1"
PREFIX_FINGERPRINT_BYTES = 64 * 1024
MAX_PENDING_APPEND_RECEIPTS = 256


class ConversationLogMixin:
    """Load a log once, then synchronize only bytes appended by other writers."""

    def enable_persisted_append_tracking(self: ConversationStore) -> None:
        """Collect bounded append proofs for incremental transcript indexing."""
        self._collect_persisted_appends = True

    def disable_persisted_append_tracking(self: ConversationStore) -> None:
        """Stop collecting append proofs and discard pending tracking state."""
        self._collect_persisted_appends = False
        self._persisted_appends.clear()
        self._persisted_appends_unverified = False

    def _record_persisted_append(
        self: ConversationStore, receipt: PersistedAppend
    ) -> None:
        if not self._collect_persisted_appends or self._persisted_appends_unverified:
            return
        if len(self._persisted_appends) >= MAX_PENDING_APPEND_RECEIPTS:
            self._persisted_appends.clear()
            self._persisted_appends_unverified = True
            return
        self._persisted_appends.append(receipt)

    def _accept_persisted_appends(
        self: ConversationStore, receipts: tuple[PersistedAppend, ...] | None
    ) -> None:
        if not self._collect_persisted_appends:
            return
        if receipts is None:
            self._persisted_appends.clear()
            self._persisted_appends_unverified = True
            return
        for receipt in receipts:
            self._record_persisted_append(receipt)

    def take_persisted_appends(
        self: ConversationStore,
    ) -> tuple[PersistedAppend, ...] | None:
        """Transfer append proofs, or report that the pending range is unverified."""
        if self._persisted_appends_unverified:
            self._persisted_appends_unverified = False
            return None
        receipts = tuple(self._persisted_appends)
        self._persisted_appends.clear()
        return receipts

    def _log_metadata_matches(self: ConversationStore) -> bool:
        """Return whether the durable log still matches the resident indexes."""
        try:
            fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        except FileNotFoundError:
            return False
        try:
            stat = os.fstat(fd)
        finally:
            os.close(fd)

        known_identity = getattr(self, "_log_identity", None)
        return (
            known_identity == (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
            and getattr(self, "_log_offset", None) == stat.st_size
            and getattr(self, "_log_mtime_ns", None) == stat.st_mtime_ns
        )

    def _sync_log_without_waiting(self: ConversationStore) -> None:
        """Synchronize changed log metadata unless another writer owns it."""
        if self._log_metadata_matches():
            return
        with os.fdopen(
            open_session_file(self.directory_fd, ".lock", os.O_RDWR | os.O_CREAT), "r+"
        ) as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # A writer can expose a new size before releasing the lock. A
                # later query synchronizes the append after the writer exits.
                return
            try:
                self._load()
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

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
            self._rebuild_client_deliveries(self._entries)
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
            if index and index % 64 == 0:
                time.sleep(0.0001)
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
            self._entries = []
            for index, row in enumerate(valid_rows[1:]):
                if index and index % 64 == 0:
                    time.sleep(0.0001)
                self._entries.append(ConversationEntry.from_dict(row))
        except ConversationIntegrityError:
            raise
        except (KeyError, TypeError, ValueError) as exc:
            raise ConversationIntegrityError(
                f"invalid conversation entry: {self.path}"
            ) from exc
        self._validate_entries()
        self._task_notification_ids = set()
        for index, entry in enumerate(self._entries):
            if index and index % 64 == 0:
                time.sleep(0.0001)
            self._validate_entry_payload(entry)
            if entry.type == "fork":
                self._validate_fork_entry(entry)
            if entry.type == "notification" and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND:
                task_id = entry.data.get("task_id")
                if type(task_id) is str and task_id:
                    self._task_notification_ids.add(task_id)
        self._rebuild_incremental_validation_state()
        self._rebuild_client_deliveries(self._entries)

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
        entries: list[ConversationEntry] = []
        offset = 0
        index = 0
        while offset < len(raw):
            newline = raw.find(b"\n", offset)
            line_end = len(raw) if newline < 0 else newline + 1
            line = raw[offset:line_end]
            # Parsing runs in worker threads for live views. A short sleep
            # releases the GIL long enough for the owner event loop to paint.
            if index and index % 64 == 0:
                time.sleep(0.001)
            try:
                row = load_session_json(line)
            except ConversationIntegrityError as exc:
                if (
                    line_end != len(raw)
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
            offset = line_end
            index += 1
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
            before = (
                os.fstat(handle.fileno())
                if self._collect_persisted_appends
                else None
            )
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            after = os.fstat(handle.fileno())
            self._set_log_stat(after)
            if before is not None:
                self._record_persisted_append(
                    PersistedAppend(
                        start_offset=before.st_size,
                        end_offset=after.st_size,
                        digest=hashlib.sha256(data).hexdigest(),
                        source_device=after.st_dev,
                        source_inode=after.st_ino,
                        before_mtime_ns=before.st_mtime_ns,
                        before_ctime_ns=before.st_ctime_ns,
                        after_mtime_ns=after.st_mtime_ns,
                        after_ctime_ns=after.st_ctime_ns,
                    )
                )

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
