"""Durable state for automatic project-memory reconciliation.

This module owns reconciliation identity, retry scheduling, terminal receipts,
and cursor publication. The scheduler executes the work that this module makes
ready; it does not interpret or reconstruct durable retry state.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..providers.stream_diagnostics import write_stream_diagnostic

MAX_SCHEDULED_ATTEMPTS = 3
_MAX_RECENT_TERMINAL_RECEIPTS = 100
_MAX_RECEIPT_ARCHIVE_BYTES = 4 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class ReconciliationFailure:
    """The last sanitized reconciliation failure shown in session status."""

    occurred_at: str
    message: str
    seq_start: int
    seq_end: int
    terminal: bool

    def status_line(self) -> str:
        disposition = "terminal receipt" if self.terminal else "will retry"
        return (
            f"{self.occurred_at} {self.message} "
            f"(seq {self.seq_start}-{self.seq_end}; {disposition})"
        )


@dataclass(frozen=True, slots=True)
class ReconciliationWork:
    """One durable failed range that is ready for another scheduled attempt."""

    key: str
    seq_start: int
    seq_end: int
    reason: str
    attempt_count: int
    retry_after: float


@dataclass(frozen=True, slots=True)
class TerminalReceipt:
    """A content-free terminal failure receipt available for explicit retry."""

    key: str
    seq_start: int
    seq_end: int
    attempt_count: int
    validation_summary: str
    occurred_at: str


class ReconciliationState:
    """Own one session's atomic reconciliation ledger."""

    def __init__(
        self,
        path: Path,
        *,
        project_id: str,
        session_id: str,
        diagnostics_path: Path,
    ) -> None:
        self.path = path
        self.project_id = project_id
        self.session_id = session_id
        self.diagnostics_path = diagnostics_path
        self.receipts_path = path.with_name("memory-reconcile-receipts.jsonl")
        self._value = self._read()
        self._bound_loaded_receipts()

    @property
    def seq(self) -> int:
        return int(self._value["seq"])

    @property
    def transcript_bytes(self) -> int:
        return int(self._value["transcript_bytes"])

    @property
    def transcript_tokens(self) -> int:
        return int(self._value["transcript_tokens"])

    @property
    def last_failure(self) -> ReconciliationFailure | None:
        return self._parse_failure(self._value.get("last_failure"))

    def reconciliation_key(self, seq_start: int, seq_end: int) -> str:
        digest = hashlib.sha256()
        for value in (
            self.project_id,
            self.session_id,
            str(seq_start),
            str(seq_end),
        ):
            digest.update(value.encode())
            digest.update(b"\0")
        return digest.hexdigest()

    def ready_retries(self, now: float | None = None) -> tuple[ReconciliationWork, ...]:
        current = time.time() if now is None else now
        return tuple(
            self._parse_work(item)
            for item in self._value["pending_failures"]
            if float(item["retry_after"]) <= current
        )

    def next_retry_after(self) -> float | None:
        values = [
            float(item["retry_after"]) for item in self._value["pending_failures"]
        ]
        return min(values) if values else None

    def terminal_receipts(self) -> tuple[TerminalReceipt, ...]:
        """Return archived and recent receipts, newest record winning by key."""
        records = {
            str(item["key"]): item
            for item in (*self._read_receipt_archive(), *self._value["terminal_receipts"])
        }
        return tuple(self._parse_terminal(item) for item in records.values())

    def record_failure(
        self,
        *,
        seq_start: int,
        seq_end: int,
        validation_summary: str,
        reason: str,
        usage: Mapping[str, int],
        retry_backoff_seconds: float,
        now: float,
        occurred_at: str,
        end_offset: int = 0,
        end_tokens: int = 0,
        terminal: bool = False,
    ) -> ReconciliationFailure:
        """Persist one failed scheduled attempt and return its public summary."""
        summary = self._sanitize_summary(validation_summary)
        key = self.reconciliation_key(seq_start, seq_end)
        pending = self._value["pending_failures"]
        prior = next(
            (
                item
                for item in pending
                if item["key"] == key
                or (
                    int(item["seq_start"]) == seq_start
                    and int(item["seq_end"]) == seq_end
                )
            ),
            None,
        )
        if prior is not None:
            key = str(prior["key"])
        attempt_count = int(prior["attempt_count"]) + 1 if prior else 1
        if prior is not None:
            pending.remove(prior)
        aggregated_usage = self._valid_usage(usage)
        if prior is not None:
            for usage_key, value in self._valid_usage(prior.get("usage", {})).items():
                aggregated_usage[usage_key] = aggregated_usage.get(usage_key, 0) + value
        record = {
            "key": key,
            "seq_start": seq_start,
            "seq_end": seq_end,
            "attempt_count": attempt_count,
            "validation_summary": summary,
            "retry_after": now + retry_backoff_seconds * (2 ** (attempt_count - 1)),
            "reason": reason,
            "usage": aggregated_usage,
            "end_offset": end_offset,
            "end_tokens": end_tokens,
        }
        terminal = terminal or attempt_count >= MAX_SCHEDULED_ATTEMPTS
        if terminal:
            terminal_record = {
                **record,
                "retry_after": None,
                "occurred_at": occurred_at,
            }
            self._remove_archived_receipt(key)
            receipts = self._value["terminal_receipts"]
            receipts[:] = [item for item in receipts if item["key"] != key]
            receipts.append(terminal_record)
            self._archive_old_receipts()
            self._complete_unit(record)
        else:
            pending.append(record)
        failure = ReconciliationFailure(
            occurred_at=occurred_at,
            message=summary,
            seq_start=seq_start,
            seq_end=seq_end,
            terminal=terminal,
        )
        self._value["last_failure"] = self._failure_dict(failure)
        self._publish(reason)
        self._write_diagnostic(failure, key, attempt_count, aggregated_usage)
        return failure

    def record_success(
        self,
        *,
        seq_start: int,
        seq_end: int,
        reason: str,
        end_offset: int,
        end_tokens: int,
    ) -> None:
        key = self.reconciliation_key(seq_start, seq_end)

        def matches(item: Mapping[str, Any]) -> bool:
            return item["key"] == key or (
                int(item["seq_start"]) == seq_start
                and int(item["seq_end"]) == seq_end
            )

        self._value["pending_failures"] = [
            item for item in self._value["pending_failures"] if not matches(item)
        ]
        self._value["terminal_receipts"] = [
            item for item in self._value["terminal_receipts"] if not matches(item)
        ]
        self._remove_archived_receipt(
            key, seq_start=seq_start, seq_end=seq_end
        )
        self._complete_unit(
            {
                "seq_start": seq_start,
                "seq_end": seq_end,
                "end_offset": end_offset,
                "end_tokens": end_tokens,
            }
        )
        self._publish(reason)

    def retry_terminal(self, key: str, *, now: float | None = None) -> bool:
        """Re-queue one terminal receipt without deleting it before success."""
        receipt = next(
            (
                item
                for item in (
                    *self._value["terminal_receipts"],
                    *self._read_receipt_archive(),
                )
                if item["key"] == key
            ),
            None,
        )
        if receipt is None:
            return False
        pending = self._value["pending_failures"]
        pending[:] = [item for item in pending if item["key"] != key]
        pending.append(
            {
                **receipt,
                "attempt_count": 0,
                "retry_after": time.time() if now is None else now,
                "reason": "explicit-retry",
            }
        )
        self._publish("explicit-retry")
        return True

    def uncovered_ranges(self, start: int, end: int) -> tuple[tuple[int, int], ...]:
        """Return ranges not already pending, completed out of order, or terminal."""
        covered: list[tuple[int, int]] = []
        for name in ("pending_failures", "completed_ranges", "terminal_receipts"):
            covered.extend(
                (int(item["seq_start"]), int(item["seq_end"]))
                for item in self._value[name]
            )
        covered.extend(
            (int(item["seq_start"]), int(item["seq_end"]))
            for item in self._read_receipt_archive()
        )
        result: list[tuple[int, int]] = []
        cursor = max(start, self.seq + 1)
        for left, right in sorted(covered):
            if right < cursor or left > end:
                continue
            if left > cursor:
                result.append((cursor, min(end, left - 1)))
            cursor = max(cursor, right + 1)
            if cursor > end:
                break
        if cursor <= end:
            result.append((cursor, end))
        return tuple(result)

    def _complete_unit(self, item: Mapping[str, Any]) -> None:
        if int(item["seq_end"]) <= self.seq:
            return
        self._value["completed_ranges"].append(dict(item))
        self._collapse_ranges()

    def _collapse_ranges(self) -> None:
        completed = self._value["completed_ranges"]
        while True:
            item = next(
                (
                    value
                    for value in completed
                    if int(value["seq_start"]) <= self.seq + 1 <= int(value["seq_end"])
                ),
                None,
            )
            if item is None:
                break
            completed.remove(item)
            self._value["seq"] = max(self.seq, int(item["seq_end"]))
            self._value["transcript_bytes"] = max(
                self.transcript_bytes, int(item.get("end_offset", 0))
            )
            self._value["transcript_tokens"] = max(
                self.transcript_tokens, int(item.get("end_tokens", 0))
            )

    def _bound_loaded_receipts(self) -> None:
        receipts = self._value["terminal_receipts"]
        if len(receipts) <= _MAX_RECENT_TERMINAL_RECEIPTS:
            return
        overflow = receipts[:-_MAX_RECENT_TERMINAL_RECEIPTS]
        self._append_receipt_archive(overflow)
        self._value["terminal_receipts"] = receipts[-_MAX_RECENT_TERMINAL_RECEIPTS:]
        self._publish("receipt-compaction")

    def _archive_old_receipts(self) -> None:
        receipts = self._value["terminal_receipts"]
        if len(receipts) <= _MAX_RECENT_TERMINAL_RECEIPTS:
            return
        overflow = receipts[:-_MAX_RECENT_TERMINAL_RECEIPTS]
        self._append_receipt_archive(overflow)
        del receipts[:-_MAX_RECENT_TERMINAL_RECEIPTS]

    def _remove_archived_receipt(
        self,
        key: str,
        *,
        seq_start: int | None = None,
        seq_end: int | None = None,
    ) -> None:
        records = self._read_receipt_archive()
        retained = tuple(
            item
            for item in records
            if item.get("key") != key
            and not (
                seq_start is not None
                and seq_end is not None
                and item.get("seq_start") == seq_start
                and item.get("seq_end") == seq_end
            )
        )
        if len(retained) != len(records):
            self._write_receipt_archive(retained)

    def _read_receipt_archive(self) -> tuple[dict[str, Any], ...]:
        try:
            lines = self.receipts_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return ()
        records: list[dict[str, Any]] = []
        for line in lines:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict) and isinstance(value.get("key"), str):
                records.append(value)
        return tuple(records)

    def _append_receipt_archive(self, records: list[Mapping[str, Any]]) -> None:
        if not records:
            return
        self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
        payload = b"".join(
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for item in records
        )
        archive_existed = self.receipts_path.exists()
        fd = os.open(
            self.receipts_path,
            os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            0o600,
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, payload)
            os.fsync(fd)
        finally:
            os.close(fd)
        if not archive_existed:
            directory_fd = os.open(
                self.receipts_path.parent, os.O_RDONLY | os.O_DIRECTORY
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        if self.receipts_path.stat().st_size > _MAX_RECEIPT_ARCHIVE_BYTES:
            self._write_receipt_archive(self._read_receipt_archive())

    def _write_receipt_archive(self, records: tuple[Mapping[str, Any], ...]) -> None:
        deduplicated = {str(item["key"]): dict(item) for item in records}
        lines = [
            json.dumps(item, sort_keys=True, separators=(",", ":")).encode() + b"\n"
            for item in deduplicated.values()
        ]
        total = 0
        retained: list[bytes] = []
        for line in reversed(lines):
            if retained and total + len(line) > _MAX_RECEIPT_ARCHIVE_BYTES:
                break
            if len(line) > _MAX_RECEIPT_ARCHIVE_BYTES:
                continue
            retained.append(line)
            total += len(line)
        payload = b"".join(reversed(retained))
        if not payload:
            try:
                self.receipts_path.unlink()
            except FileNotFoundError:
                pass
            return
        self.receipts_path.parent.mkdir(parents=True, exist_ok=True)
        fd, temporary = tempfile.mkstemp(
            prefix=".memory-reconcile-receipts-", dir=self.receipts_path.parent
        )
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, payload)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, self.receipts_path)
            directory_fd = os.open(
                self.receipts_path.parent, os.O_RDONLY | os.O_DIRECTORY
            )
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _publish(self, reason: str) -> None:
        self._value["session_id"] = self.session_id
        self._value["project_id"] = self.project_id
        self._value["reason"] = reason
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self._value, sort_keys=True).encode()
        fd, temporary = tempfile.mkstemp(prefix=".memory-reconcile-", dir=self.path.parent)
        try:
            os.fchmod(fd, 0o600)
            os.write(fd, payload)
            os.fsync(fd)
            os.close(fd)
            fd = -1
            os.replace(temporary, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if fd >= 0:
                os.close(fd)
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _read(self) -> dict[str, Any]:
        empty: dict[str, Any] = {
            "seq": 0,
            "transcript_bytes": 0,
            "transcript_tokens": 0,
            "pending_failures": [],
            "completed_ranges": [],
            "terminal_receipts": [],
            "last_failure": None,
        }
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return empty
        if not isinstance(value, dict):
            return empty
        for name in (
            "seq",
            "transcript_bytes",
            "transcript_tokens",
        ):
            candidate = value.get(name, 0)
            if type(candidate) is not int or candidate < 0:
                return empty
            empty[name] = candidate
        for name in (
            "pending_failures",
            "completed_ranges",
            "terminal_receipts",
        ):
            candidate = value.get(name, [])
            if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
                empty[name] = candidate
        empty["last_failure"] = value.get("last_failure")
        return empty

    def _write_diagnostic(
        self,
        failure: ReconciliationFailure,
        key: str,
        attempt_count: int,
        usage: Mapping[str, int],
    ) -> None:
        write_stream_diagnostic(
            self.diagnostics_path,
            {
                "timestamp": failure.occurred_at,
                "cause": "memory_reconciliation_failure",
                "project_id": self.project_id,
                "session_id": self.session_id,
                "reconciliation_key": key,
                "seq_start": failure.seq_start,
                "seq_end": failure.seq_end,
                "attempt_count": attempt_count,
                "terminal": failure.terminal,
                "validation_summary": failure.message,
                "usage": self._valid_usage(usage),
            },
        )

    @staticmethod
    def _sanitize_summary(message: str) -> str:
        return " ".join((message or "unknown reconciliation failure").split())[:240]

    @staticmethod
    def _valid_usage(usage: object) -> dict[str, int]:
        if not isinstance(usage, Mapping):
            return {}
        return {
            key: value
            for key, value in usage.items()
            if isinstance(key, str) and type(value) is int and value >= 0
        }

    @staticmethod
    def _failure_dict(failure: ReconciliationFailure) -> dict[str, object]:
        return {
            "occurred_at": failure.occurred_at,
            "message": failure.message,
            "seq_start": failure.seq_start,
            "seq_end": failure.seq_end,
            "terminal": failure.terminal,
        }

    @staticmethod
    def _parse_failure(value: object) -> ReconciliationFailure | None:
        if not isinstance(value, Mapping):
            return None
        occurred_at = value.get("occurred_at")
        message = value.get("message")
        seq_start = value.get("seq_start")
        seq_end = value.get("seq_end")
        terminal = value.get("terminal")
        if (
            not isinstance(occurred_at, str)
            or not isinstance(message, str)
            or type(seq_start) is not int
            or type(seq_end) is not int
            or type(terminal) is not bool
        ):
            return None
        return ReconciliationFailure(
            occurred_at, message, seq_start, seq_end, terminal
        )

    @staticmethod
    def _parse_work(value: Mapping[str, Any]) -> ReconciliationWork:
        return ReconciliationWork(
            key=str(value["key"]),
            seq_start=int(value["seq_start"]),
            seq_end=int(value["seq_end"]),
            reason=str(value["reason"]),
            attempt_count=int(value["attempt_count"]),
            retry_after=float(value["retry_after"]),
        )

    @staticmethod
    def _parse_terminal(value: Mapping[str, Any]) -> TerminalReceipt:
        return TerminalReceipt(
            key=str(value["key"]),
            seq_start=int(value["seq_start"]),
            seq_end=int(value["seq_end"]),
            attempt_count=int(value["attempt_count"]),
            validation_summary=str(value["validation_summary"]),
            occurred_at=str(value["occurred_at"]),
        )
