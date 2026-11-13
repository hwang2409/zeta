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
        records: list[ContextCalibrationRecord] = []
        for ref in self._resident_rows.branch_refs():
            facts = ref.facts
            if (
                isinstance(facts, tuple)
                and facts
                and isinstance(facts[0], ContextCalibrationRecord)
            ):
                records.append(facts[0])
        return tuple(records)


def row_facts(entry: ConversationEntry) -> object | None:
    if entry.type == "message":
        calibration = context_calibration_record(entry)
        receipts = sent_message_receipts(entry)
        return (calibration, receipts) if calibration is not None or receipts else None
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
        self._by_id: dict[str, ConversationEntry] = {}
        self._rows_by_id: dict[str, RowRef] = {}
        self.branch: list[ConversationEntry] = []
        self.replace(entries)

    def append(self, entry: ConversationEntry) -> None:
        previous = self.entries[-1] if self.entries else None
        self.entries.append(entry)
        self._by_id[entry.id] = entry
        row = RowRef(entry.seq, entry.id, entry.parent_id, entry.lane, entry.type, row_facts(entry))
        self.rows.append(row)
        self._rows_by_id[entry.id] = row
        if (previous is None and entry.parent_id is None) or (
            previous is not None and entry.parent_id == previous.id
        ):
            self.branch.append(entry)
        else:
            self._rebuild_branch()

    def extend(self, entries: Iterable[ConversationEntry]) -> None:
        additions = list(entries)
        if not additions:
            return
        previous = self.entries[-1] if self.entries else None
        self.entries.extend(additions)
        self._by_id.update((entry.id, entry) for entry in additions)
        new_rows = [
            RowRef(e.seq, e.id, e.parent_id, e.lane, e.type, row_facts(e))
            for e in additions
        ]
        self.rows.extend(new_rows)
        self._rows_by_id.update((row.id, row) for row in new_rows)
        parent = previous.id if previous is not None else None
        linear = True
        for entry in additions:
            if entry.parent_id != parent:
                linear = False
                break
            parent = entry.id
        if linear:
            self.branch.extend(additions)
        else:
            self._rebuild_branch()

    def truncate_to(self, length: int) -> None:
        del self.entries[length:]
        del self.rows[length:]
        self._by_id = {entry.id: entry for entry in self.entries}
        self._rows_by_id = {row.id: row for row in self.rows}
        self._rebuild_branch()

    @classmethod
    def prepared(cls, entries: Iterable[ConversationEntry]) -> ResidentRows:
        rows = cls.__new__(cls)
        rows.entries = entries if isinstance(entries, list) else list(entries)
        rows.rows = [
            RowRef(e.seq, e.id, e.parent_id, e.lane, e.type, row_facts(e))
            for e in rows.entries
        ]
        rows._by_id = {entry.id: entry for entry in rows.entries}
        rows._rows_by_id = {row.id: row for row in rows.rows}
        rows._rebuild_branch()
        return rows

    def replace(self, entries: Iterable[ConversationEntry]) -> None:
        prepared = type(self).prepared(entries)
        self.entries = prepared.entries
        self.rows = prepared.rows
        self._by_id = prepared._by_id
        self._rows_by_id = prepared._rows_by_id
        self.branch = prepared.branch

    def _rebuild_branch(self) -> None:
        current = self.entries[-1] if self.entries else None
        branch: list[ConversationEntry] = []
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise ConversationIntegrityError(f"conversation parent cycle at {current.id}")
            seen.add(current.id)
            branch.append(current)
            current = self._by_id.get(current.parent_id) if current.parent_id else None
        self.branch = list(reversed(branch))

    def branch_refs(self) -> tuple[RowRef, ...]:
        return tuple(self._rows_by_id[entry.id] for entry in self.branch)

    def resident_refs(self) -> tuple[RowRef, ...]:
        return tuple(self.rows)

    def entry(self, ref: RowRef) -> ConversationEntry:
        entry = self._by_id.get(ref.id)
        if entry is None or entry.seq != ref.seq:
            raise KeyError(ref.seq)
        return entry

    def stream_rows(self, refs: Iterable[RowRef], *, batch: int = 256) -> Iterator[ConversationEntry]:
        del batch
        for ref in refs:
            yield self.entry(ref)
