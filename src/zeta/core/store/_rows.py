"""Resident row index and compact row facts."""

from __future__ import annotations

import hashlib
import os
import re
from collections import OrderedDict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from ...agent.receipt import encode_json
from ...protocol.types import Message, is_passive_harness_message
from ..checkpoints import (
    ConversationEntry,
    ConversationIntegrityError,
    load_session_json,
)
from ._client_delivery import (
    CLIENT_DELIVERY_METADATA,
    CLIENT_DELIVERY_PAYLOAD_DIGEST_METADATA,
    without_client_delivery_marker,
)

_SENT_STATUS_EVENT = "project_inbox_sent_status"
_MESSAGE_ID = re.compile(r"[0-9a-f]{32}\Z")


@dataclass(frozen=True, slots=True)
class ContextCalibrationRecord:
    ratio: float | None = None
    estimate: int | None = None
    actual: int | None = None


@dataclass(frozen=True, slots=True)
class MessageFacts:
    calibration: ContextCalibrationRecord | None
    receipts: tuple[tuple[str, str], ...]
    approval_request_ids: tuple[str, ...]
    context_exception: bool
    is_user: bool
    is_user_forkpoint: bool
    message_preview: str | None
    projection_digest: str


@dataclass(frozen=True, slots=True)
class TranscriptRows:
    path: Path
    refs: tuple[RowRef, ...]
    source_identity: tuple[int, int]
    relative_offsets: bool

    def read(self) -> tuple[ConversationEntry, ...]:
        fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            stat = os.fstat(fd)
            if (stat.st_dev, stat.st_ino) != self.source_identity:
                from ._history_index import StaleHistorySnapshot

                raise StaleHistorySnapshot("conversation row source changed")
            header_size = 0
            if self.relative_offsets:
                header = os.pread(fd, 4096, 0)
                header_size = header.find(b"\n") + 1
                if header_size <= 0:
                    from ._history_index import StaleHistorySnapshot

                    raise StaleHistorySnapshot("conversation header is incomplete")
            entries = []
            for ref in self.refs:
                raw = os.pread(fd, ref.length, ref.offset + header_size)
                if len(raw) != ref.length:
                    from ._history_index import StaleHistorySnapshot

                    raise StaleHistorySnapshot("conversation row offset is stale")
                entry = ConversationEntry.from_dict(load_session_json(raw))
                if entry.seq != ref.seq or entry.id != ref.id:
                    from ._history_index import StaleHistorySnapshot

                    raise StaleHistorySnapshot("conversation row identity is stale")
                entries.append(entry)
            return tuple(entries)
        finally:
            os.close(fd)


@dataclass(frozen=True, slots=True)
class RowRef:
    seq: int
    id: str
    parent_id: str | None
    lane: str
    type: str
    offset: int = 0
    length: int = 0
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
            if isinstance(facts, MessageFacts) and facts.calibration is not None:
                records.append(facts.calibration)
        return tuple(records)


def _message_facts(entry: ConversationEntry) -> MessageFacts:
    message = entry.data["message"]
    metadata = message.get("metadata", {})
    projection = message
    if isinstance(metadata, Mapping) and CLIENT_DELIVERY_METADATA in metadata:
        projection = dict(message)
        projection_metadata = dict(metadata)
        del projection_metadata[CLIENT_DELIVERY_METADATA]
        projection_metadata.pop(CLIENT_DELIVERY_PAYLOAD_DIGEST_METADATA, None)
        if projection_metadata:
            projection["metadata"] = projection_metadata
        else:
            projection.pop("metadata", None)
    projection_digest = hashlib.sha256(encode_json(projection)).hexdigest()
    is_user = message["role"] == "user"
    text = " ".join(
        block["text"].strip()
        for block in message["content"]
        if block["type"] == "text" and block["text"].strip()
    )
    metadata = message.get("metadata", {})
    context_exception = isinstance(metadata, Mapping) and bool(
        metadata.get("turn_failed")
        or metadata.get("zeta.pending_prompt_id")
        or metadata.get("zeta.step_delivery")
    )
    return MessageFacts(
        context_calibration_record(entry),
        sent_message_receipts(entry),
        tuple(request["request_id"] for request in entry.data.get("approval_requests", ())),
        context_exception,
        is_user,
        is_user and not is_passive_harness_message(message),
        " ".join(text.split())[:48] if text else None,
        projection_digest,
    )


def row_facts(entry: ConversationEntry) -> object | None:
    if entry.type == "message":
        return _message_facts(entry)
    if entry.type == "compaction":
        data = entry.data
        return tuple(data.get(key) for key in ("source_seq_start", "source_seq_end", "replaces", "kind"))
    if entry.type == "notification":
        data = entry.data
        return tuple(data.get(key) for key in ("kind", "task_id", "child_instance_id", "display_only"))
    return None


class _EntryView(Sequence[ConversationEntry]):
    """Compatibility view over row references without retaining payloads."""

    def __init__(self, owner: ResidentRows, refs: Sequence[RowRef]) -> None:
        self._owner = owner
        self._refs = refs

    def __len__(self) -> int:
        return len(self._refs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self._owner.entry(ref) for ref in self._refs[index]]
        return self._owner.entry(self._refs[index])

    def __iter__(self) -> Iterator[ConversationEntry]:
        return (self._owner.entry(ref) for ref in self._refs)


def decode_row_facts(raw: str | bytes) -> object:
    stored = load_session_json(raw.encode() if isinstance(raw, str) else raw)
    if isinstance(stored, list):
        return tuple(stored)
    if not isinstance(stored, dict):
        raise ConversationIntegrityError("warm row facts are malformed")
    kind = stored.get("kind")
    value = stored.get("value")
    if kind == "NoneType":
        return None
    if kind == "tuple" and isinstance(value, list):
        return tuple(value)
    if kind == "MessageFacts" and isinstance(value, dict):
        calibration = value.get("calibration")
        projection_digest = value.get("projection_digest")
        if not isinstance(projection_digest, str):
            raise ConversationIntegrityError("warm message facts lack projection digest")
        return MessageFacts(
            calibration=(
                ContextCalibrationRecord(**calibration)
                if isinstance(calibration, dict)
                else None
            ),
            receipts=tuple(tuple(item) for item in value.get("receipts", ())),
            approval_request_ids=tuple(value.get("approval_request_ids", ())),
            context_exception=bool(value.get("context_exception")),
            is_user=bool(value.get("is_user")),
            is_user_forkpoint=bool(value.get("is_user_forkpoint")),
            message_preview=value.get("message_preview"),
            projection_digest=projection_digest,
        )
    raise ConversationIntegrityError(f"unsupported warm row facts: {kind}")


class ResidentRows:
    _BULK_TYPES = frozenset({"message", "compaction", "notification"})

    """One owner for the row skeleton and its bounded resident payload set."""

    _RECENT_WINDOW = 64

    def __init__(self, entries: Iterable[ConversationEntry] = ()) -> None:
        self.rows: list[RowRef] = []
        self._rows_by_id: dict[str, RowRef] = {}
        self._branch_refs: list[RowRef] = []
        self._context_refs: list[RowRef] = []
        self._context_required: set[int] = set()
        self._context_source_seqs: set[int] = set()
        self._legacy_pinned_source_seqs: dict[str, int] = {}
        self.payloads: dict[int, ConversationEntry] = {}
        self._payload_sizes: dict[int, int] = {}
        self._pinned: set[int] = set()
        self._payload_lru: OrderedDict[int, int] = OrderedDict()
        self.payload_cache_bytes = int(
            os.environ.get("ZETA_STORE_PAYLOAD_CACHE_BYTES", "0")
        )
        self.resident_payload_bytes = 0
        self._source_dir_fd: int | None = None
        self._source_identity: tuple[int, int] | None = None
        self._source_generation = 0
        self._generation_reader: Callable[[], int] | None = None
        self._relative_offsets = False
        self.extend(entries)
        self._rebuild_branch()
        self._rebalance_resident()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ResidentRows):
            return False
        semantic = lambda ref: (
            ref.seq,
            ref.id,
            ref.parent_id,
            ref.lane,
            ref.type,
            ref.length,
            ref.facts,
        )
        return (
            tuple(map(semantic, self.rows)),
            tuple(map(semantic, self._branch_refs)),
        ) == (
            tuple(map(semantic, other.rows)),
            tuple(map(semantic, other._branch_refs)),
        )

    @property
    def entries(self) -> Sequence[ConversationEntry]:
        return _EntryView(self, self.rows)

    @property
    def branch(self) -> Sequence[ConversationEntry]:
        return _EntryView(self, self._branch_refs)

    def bind_source(
        self,
        directory_fd: int,
        identity: tuple[int, int, int] | None = None,
        generation: int = 0,
        generation_reader: Callable[[], int] | None = None,
    ) -> None:
        self._source_dir_fd = directory_fd
        self._source_identity = None if identity is None else identity[:2]
        self._source_generation = generation
        self._generation_reader = generation_reader
        self._rebalance_resident()

    def append(
        self,
        entry: ConversationEntry,
        *,
        offset: int = 0,
        length: int = 0,
    ) -> None:
        previous = self.rows[-1] if self.rows else None
        size = length or len(encode_json(entry.to_dict())) + 1
        if not self.rows and offset == 0:
            self._relative_offsets = True
        elif self._relative_offsets and previous is not None:
            offset = previous.offset + previous.length
        row = RowRef(
            entry.seq,
            entry.id,
            entry.parent_id,
            entry.lane,
            entry.type,
            offset,
            size,
            row_facts(entry),
        )
        self.rows.append(row)
        self._rows_by_id[row.id] = row
        self._store_payload(row, entry)
        linear = (previous is None and row.parent_id is None) or (
            previous is not None and row.parent_id == previous.id
        )
        if linear:
            self._branch_refs.append(row)
        else:
            self._rebuild_branch()
        if row.type in {"message", "compaction"}:
            self._context_source_seqs.add(row.seq)
        if row.type == "compaction" or not linear:
            self._rebalance_resident()
        else:
            self._context_refs.append(row)
            self._pinned.add(row.seq)
            self._payload_lru.pop(row.seq, None)
            self._trim_payload_cache()

    def extend(
        self,
        entries: Iterable[ConversationEntry],
        *,
        start_offset: int | None = None,
    ) -> None:
        offset = start_offset
        for entry in entries:
            length = len(encode_json(entry.to_dict())) + 1
            self.append(
                entry,
                offset=0 if offset is None else offset,
                length=length,
            )
            if offset is not None:
                offset += length

    def truncate_to(self, length: int) -> None:
        removed = self.rows[length:]
        del self.rows[length:]
        for row in removed:
            self._drop_payload(row.seq)
        self._rows_by_id = {row.id: row for row in self.rows}
        self._rebuild_branch()
        self._rebalance_resident()

    @classmethod
    def prepared(cls, entries: Iterable[ConversationEntry]) -> ResidentRows:
        return cls(entries)

    @classmethod
    def warm(
        cls,
        raw_rows: Sequence[tuple[object, ...]],
        payload_entries: Iterable[ConversationEntry],
    ) -> ResidentRows:
        rows = cls()
        for raw in raw_rows:
            seq, entry_id, parent_id, lane, entry_type, offset, length, facts = raw
            rows.rows.append(
                RowRef(
                    seq=int(seq),
                    id=str(entry_id),
                    parent_id=None if parent_id is None else str(parent_id),
                    lane=str(lane),
                    type=str(entry_type),
                    offset=int(offset),
                    length=int(length),
                    facts=decode_row_facts(facts),
                )
            )
        rows._rows_by_id = {row.id: row for row in rows.rows}
        rows._rebuild_branch()
        for entry in payload_entries:
            ref = rows._rows_by_id[entry.id]
            rows._store_payload(ref, entry)
        rows._rebalance_resident()
        return rows

    def clone(self) -> ResidentRows:
        rows = type(self).__new__(type(self))
        rows.rows = list(self.rows)
        rows._rows_by_id = dict(self._rows_by_id)
        rows._branch_refs = list(self._branch_refs)
        rows._context_refs = list(self._context_refs)
        rows._context_source_seqs = set(self._context_source_seqs)
        rows.payloads = dict(self.payloads)
        rows._payload_sizes = dict(self._payload_sizes)
        rows._pinned = set(self._pinned)
        rows._payload_lru = OrderedDict(self._payload_lru)
        rows.payload_cache_bytes = self.payload_cache_bytes
        rows.resident_payload_bytes = self.resident_payload_bytes
        rows._source_dir_fd = self._source_dir_fd
        rows._source_identity = self._source_identity
        rows._source_generation = self._source_generation
        rows._generation_reader = self._generation_reader
        rows._relative_offsets = self._relative_offsets
        return rows

    def replace(self, entries: Iterable[ConversationEntry]) -> None:
        prepared = type(self).prepared(entries)
        prepared.bind_source(
            self._source_dir_fd,
            None if self._source_identity is None else (*self._source_identity, 0),
            self._source_generation,
            self._generation_reader,
        ) if self._source_dir_fd is not None else None
        self.__dict__.update(prepared.__dict__)

    def _rebuild_branch(self) -> None:
        current = self.rows[-1] if self.rows else None
        branch: list[RowRef] = []
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise ConversationIntegrityError(
                    f"conversation parent cycle at {current.id}"
                )
            seen.add(current.id)
            branch.append(current)
            current = self._rows_by_id.get(current.parent_id) if current.parent_id else None
        self._branch_refs = list(reversed(branch))
        self._context_source_seqs = {
            ref.seq
            for ref in self._branch_refs
            if ref.type in {"message", "compaction"}
        }

    def branch_refs(self) -> tuple[RowRef, ...]:
        return tuple(self._branch_refs)

    def resident_refs(self) -> tuple[RowRef, ...]:
        return tuple(self.rows)

    def last_message_ref(self) -> RowRef | None:
        return next(
            (ref for ref in reversed(self._branch_refs) if ref.type == "message"),
            None,
        )

    def row_by_id(self, entry_id: str) -> RowRef | None:
        return self._rows_by_id.get(entry_id)

    @property
    def relative_offsets(self) -> bool:
        return self._relative_offsets

    def context_refs(self) -> tuple[RowRef, ...]:
        return tuple(self._context_refs)

    def context_source_seqs(self) -> set[int]:
        return self._context_source_seqs

    def legacy_pinned_source_seq(self, marker_id: str) -> int | None:
        return self._legacy_pinned_source_seqs.get(marker_id)

    @staticmethod
    def _compaction_values(ref: RowRef) -> tuple[object, object, object, object]:
        return ref.facts if isinstance(ref.facts, tuple) and len(ref.facts) == 4 else (None,) * 4

    def _required_payloads(self) -> set[int]:
        self._legacy_pinned_source_seqs = {}
        required: set[int] = set()
        context_required = {
            ref.seq for ref in self.rows if ref.type not in self._BULK_TYPES
        }
        for ref in self.rows:
            if isinstance(ref.facts, MessageFacts) and (
                ref.facts.approval_request_ids or ref.facts.context_exception
            ):
                required.add(ref.seq)

        markers = [ref for ref in self._branch_refs if ref.type == "compaction"]
        replaced = {
            marker_id
            for ref in markers
            for marker_id in (
                self._compaction_values(ref)[2]
                if isinstance(self._compaction_values(ref)[2], list)
                else ()
            )
        }
        live_markers = [ref for ref in markers if ref.id not in replaced]
        required.update(ref.seq for ref in live_markers)
        for marker in live_markers:
            payload = self.payloads.get(marker.seq)
            start, end, _, _ = self._compaction_values(marker)
            pinned = None if payload is None else payload.data.get("pinned_message")
            if (
                type(start) is not int
                or type(end) is not int
                or not isinstance(pinned, dict)
                or type(pinned.get("source_seq")) is int
            ):
                continue
            try:
                projection = without_client_delivery_marker(Message.from_dict(pinned))
            except (KeyError, TypeError, ValueError):
                continue
            digest = hashlib.sha256(encode_json(projection.to_dict())).hexdigest()
            match = next(
                (
                    ref.seq
                    for ref in reversed(self._branch_refs)
                    if start <= ref.seq <= end
                    and isinstance(ref.facts, MessageFacts)
                    and ref.facts.projection_digest == digest
                ),
                None,
            )
            if match is not None:
                self._legacy_pinned_source_seqs[marker.id] = match
        coverage_end = max(
            (
                value
                for ref in live_markers
                for value in [self._compaction_values(ref)[1]]
                if type(value) is int
            ),
            default=0,
        )
        required.update(ref.seq for ref in self._branch_refs if ref.seq > coverage_end)

        recent = self.rows[-self._RECENT_WINDOW :]
        required.update(
            ref.seq
            for ref in recent
            if ref.type != "compaction" or ref in live_markers
        )
        self._context_required = context_required | required
        return required

    def _rebalance_resident(self) -> None:
        required = self._required_payloads()
        self._context_refs = [
            ref for ref in self._branch_refs if ref.seq in self._context_required
        ]
        self._pinned = required
        for seq in tuple(self._payload_lru):
            if seq in required:
                self._payload_lru.pop(seq, None)
        for seq in tuple(self.payloads):
            if seq not in required and seq not in self._payload_lru:
                self._payload_lru[seq] = self._payload_sizes[seq]
        self._trim_payload_cache()

    def _store_payload(self, ref: RowRef, entry: ConversationEntry) -> None:
        old_size = self._payload_sizes.get(ref.seq, 0)
        self.payloads[ref.seq] = entry
        self._payload_sizes[ref.seq] = ref.length
        self.resident_payload_bytes += ref.length - old_size
        if ref.seq not in self._pinned:
            self._payload_lru[ref.seq] = ref.length
            self._payload_lru.move_to_end(ref.seq)

    def _drop_payload(self, seq: int) -> None:
        if self.payloads.pop(seq, None) is not None:
            self.resident_payload_bytes -= self._payload_sizes.get(seq, 0)
        self._payload_sizes.pop(seq, None)
        self._payload_lru.pop(seq, None)
        self._pinned.discard(seq)

    def _trim_payload_cache(self) -> None:
        while (
            sum(self._payload_lru.values()) > max(0, self.payload_cache_bytes)
            and self._payload_lru
        ):
            seq, _ = self._payload_lru.popitem(last=False)
            if seq not in self._pinned:
                self._drop_payload(seq)

    def _open_source(self) -> int:
        if self._source_dir_fd is None:
            raise KeyError("row source is not bound")
        fd = os.open("conversation.jsonl", os.O_RDONLY, dir_fd=self._source_dir_fd)
        info = os.fstat(fd)
        if self._source_identity is not None and (info.st_dev, info.st_ino) != self._source_identity:
            os.close(fd)
            from ._history_index import StaleHistorySnapshot
            raise StaleHistorySnapshot("conversation row source changed")
        if (
            self._generation_reader is not None
            and self._generation_reader() != self._source_generation
        ):
            os.close(fd)
            from ._history_index import StaleHistorySnapshot
            raise StaleHistorySnapshot("conversation row generation changed")
        return fd

    def _decode(self, fd: int, ref: RowRef) -> ConversationEntry:
        offset = ref.offset
        if self._relative_offsets:
            header = os.pread(fd, 4096, 0)
            header_end = header.find(b"\n") + 1
            if header_end <= 0:
                from ._history_index import StaleHistorySnapshot
                raise StaleHistorySnapshot("conversation header is incomplete")
            offset += header_end
        raw = os.pread(fd, ref.length, offset)
        if len(raw) != ref.length:
            from ._history_index import StaleHistorySnapshot
            raise StaleHistorySnapshot("conversation row offset is stale")
        value = load_session_json(raw)
        entry = ConversationEntry.from_dict(value)
        if entry.seq != ref.seq or entry.id != ref.id:
            from ._history_index import StaleHistorySnapshot
            raise StaleHistorySnapshot("conversation row identity is stale")
        return entry

    def entry(self, ref: RowRef) -> ConversationEntry:
        entry = self.payloads.get(ref.seq)
        if entry is None:
            fd = self._open_source()
            try:
                entry = self._decode(fd, ref)
            finally:
                os.close(fd)
            self._store_payload(ref, entry)
            self._trim_payload_cache()
        elif ref.seq in self._payload_lru:
            self._payload_lru.move_to_end(ref.seq)
        return entry

    def stream_rows(
        self, refs: Iterable[RowRef], *, batch: int = 256
    ) -> Iterator[ConversationEntry]:
        if batch <= 0:
            raise ValueError("batch must be positive")
        generation = self._source_generation
        fd: int | None = None
        try:
            for ref in refs:
                entry = self.payloads.get(ref.seq)
                if entry is None:
                    fd = self._open_source() if fd is None else fd
                    entry = self._decode(fd, ref)
                yield entry
            if (
                self._generation_reader is not None
                and self._generation_reader() != generation
            ):
                from ._history_index import StaleHistorySnapshot
                raise StaleHistorySnapshot("conversation row generation changed")
        finally:
            if fd is not None:
                os.close(fd)
