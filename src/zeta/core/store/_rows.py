"""Resident row index and compact row facts."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass

from ...protocol.types import Message, is_passive_harness_message
from ..checkpoints import ConversationEntry, ConversationIntegrityError

_SENT_STATUS_EVENT = "project_inbox_sent_status"
_MESSAGE_ID = re.compile(r"[0-9a-f]{32}\Z")


@dataclass(frozen=True, slots=True)
class ContextCalibrationRecord:
    ratio: float | None = None
    estimate: int | None = None
    actual: int | None = None


@dataclass(frozen=True, slots=True)
class RowRef:
    seq: int
    id: str
    parent_id: str | None
    lane: str
    type: str
    facts: object | None = None


def sent_message_receipts(entry: ConversationEntry) -> tuple[tuple[str, str], ...]:
    if entry.type != "message":
        return ()
    raw = entry.data.get("message")
    if not isinstance(raw, dict) or not is_passive_harness_message(raw):
        return ()
    try:
        message = Message.from_dict(raw)
    except (KeyError, TypeError, ValueError):
        return ()
    if message.metadata.get("zeta_event") != _SENT_STATUS_EVENT:
        return ()
    statuses = message.metadata.get("sent_statuses")
    if not isinstance(statuses, list):
        return ()
    return tuple(
        (message_id, status)
        for item in statuses
        if isinstance(item, dict)
        for message_id, status in [(item.get("message_id"), item.get("status"))]
        if isinstance(message_id, str)
        and _MESSAGE_ID.fullmatch(message_id)
        and status in {"claimed", "done"}
    )


def context_calibration_record(entry: ConversationEntry) -> ContextCalibrationRecord | None:
    if entry.type != "message":
        return None
    metadata = entry.data.get("message", {}).get("metadata", {})
    if not isinstance(metadata, Mapping):
        return None
    persisted = metadata.get("context_calibration_ratio")
    if type(persisted) in {int, float}:
        return ContextCalibrationRecord(ratio=float(persisted))
    usage = metadata.get("provider_usage")
    estimate = metadata.get("context_estimate_tokens")
    if not isinstance(usage, Mapping) or type(estimate) is not int:
        return None
    values = [
        usage.get("input_tokens", usage.get("prompt_tokens")),
        usage.get("cache_read_input_tokens"),
        usage.get("cache_creation_input_tokens"),
    ]
    actual_values = [value for value in values if type(value) is int and value >= 0]
    return (
        ContextCalibrationRecord(estimate=estimate, actual=sum(actual_values))
        if actual_values
        else None
    )


class StartupFactsMixin:
    def sent_message_receipts(self) -> tuple[tuple[str, str], ...]:
        return tuple(self._sent_message_receipts)

    def context_calibration_records(self) -> tuple[ContextCalibrationRecord, ...]:
        return tuple(
            record
            for entry in self._active_branch()
            if (record := context_calibration_record(entry)) is not None
        )


def row_facts(entry: ConversationEntry) -> object | None:
    if entry.type == "message":
        return (context_calibration_record(entry), sent_message_receipts(entry))
    if entry.type == "compaction":
        data = entry.data
        return tuple(data.get(key) for key in ("source_seq_start", "source_seq_end", "replaces", "kind"))
    if entry.type == "notification":
        data = entry.data
        return tuple(data.get(key) for key in ("kind", "task_id", "child_instance_id", "display_only"))
    return None


class ResidentRows:
    """One owner for the row skeleton and its currently resident payloads."""

    def __init__(self, entries: Iterable[ConversationEntry] = ()) -> None:
        self.entries: list[ConversationEntry] = []
        self.rows: list[RowRef] = []
        self.branch: tuple[ConversationEntry, ...] = ()
        self.replace(entries)

    def append(self, entry: ConversationEntry) -> None:
        self.entries.append(entry)
        self.rows.append(RowRef(entry.seq, entry.id, entry.parent_id, entry.lane, entry.type, row_facts(entry)))
        self._rebuild_branch()

    def extend(self, entries: Iterable[ConversationEntry]) -> None:
        additions = list(entries)
        if not additions:
            return
        self.entries.extend(additions)
        self.rows.extend(
            RowRef(e.seq, e.id, e.parent_id, e.lane, e.type, row_facts(e))
            for e in additions
        )
        self._rebuild_branch()

    def truncate_to(self, length: int) -> None:
        del self.entries[length:]
        del self.rows[length:]
        self._rebuild_branch()

    def replace(self, entries: Iterable[ConversationEntry]) -> None:
        self.entries = list(entries)
        self.rows = [
            RowRef(e.seq, e.id, e.parent_id, e.lane, e.type, row_facts(e))
            for e in self.entries
        ]
        self._rebuild_branch()

    def _rebuild_branch(self) -> None:
        by_id = {e.id: e for e in self.entries}
        current = self.entries[-1] if self.entries else None
        branch: list[ConversationEntry] = []
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise ConversationIntegrityError(f"conversation parent cycle at {current.id}")
            seen.add(current.id)
            branch.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        self.branch = tuple(reversed(branch))

    def entry(self, ref: RowRef) -> ConversationEntry:
        for entry in self.entries:
            if entry.seq == ref.seq and entry.id == ref.id:
                return entry
        raise KeyError(ref.seq)

    def stream_rows(self, refs: Iterable[RowRef], *, batch: int = 256) -> Iterator[ConversationEntry]:
        del batch
        for ref in refs:
            yield self.entry(ref)
