"""Typed project-memory state and deterministic mutations.

The module is the format-2 domain seam. It validates copied schemas, canonicalizes
complete snapshots, and applies grouped operations without storage side effects.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import json
import re
import secrets
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from typing import Any, Literal

from zeta.memory.safety import contains_secret
from zeta.project_errors import ProjectRegistryError
from zeta.project_schema import MAX_MEMORY_FILE_SIZE
from zeta.protocol.types import MessageOrigin

MAX_ENTRIES = 1_000
MAX_ENTRY_TEXT_BYTES = 4 * 1024
MAX_STATE_BYTES = 8 * 1024 * 1024
MAX_KINDS = 32
MAX_SOURCES_PER_ENTRY = 64
MAX_OPERATIONS = 128
MAX_REASON_BYTES = 1024

EntryStatus = Literal["active", "superseded", "resolved", "expired"]
ReceiptType = Literal[
    "add", "update", "supersede", "resolve", "expire", "accept", "undo", "migrate", "sync"
]
_TIMESTAMP = re.compile(
    r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z"
)
_KIND_KEY = re.compile(r"[a-z][a-z0-9_-]{0,31}")
_ENTRY_ID = re.compile(r"m_[0-9a-f]{32}")
_OPERATION_ID = re.compile(r"op_[0-9a-f]{32}")
_DIGEST = re.compile(r"[0-9a-f]{64}")
_VERSION_ID = re.compile(r"[0-9a-f]{32}")
_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}")
_ALLOWED_ORIGINS = {
    *(origin.value for origin in MessageOrigin),
    "agent",
    "tool_output",
    "harness",
    "harness_unknown",
    "harness_notification",
}


@dataclass(frozen=True, slots=True)
class _Unset:
    pass


UNSET = _Unset()
OptionalTimestamp = str | None | _Unset


@dataclass(frozen=True, slots=True)
class MemoryKind:
    key: str
    name: str
    description: str
    prompt_mode: Literal["always", "recent", "on_demand"]
    prompt_priority: int
    prompt_max_entries: int
    default_expiry_days: int | None = None


@dataclass(frozen=True, slots=True)
class MemorySchema:
    version: int
    profile: str
    kinds: tuple[MemoryKind, ...]


@dataclass(frozen=True, slots=True)
class MemorySource:
    session_id: str
    seq_start: int
    seq_end: int
    origins: tuple[str, ...]
    observed_at: str
    evidence_rank: int


@dataclass(frozen=True, slots=True)
class MigrationSource:
    source_digest: str
    source_version: str | None


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    id: str
    project_id: str
    kind: str
    text: str
    representation: Literal["entry", "legacy_document"]
    status: EntryStatus
    created_at: str
    updated_at: str
    seen_at: str
    expires_at: str | None
    valid_from: str
    valid_until: str | None
    supersedes: tuple[str, ...]
    superseded_by: tuple[str, ...]
    sources: tuple[MemorySource, ...]
    automatic: bool
    accepted_at: str | None
    accepted_by: str | None
    last_operation_id: str
    section: str | None = None
    migration_order: int | None = None
    migration_source: MigrationSource | None = None


@dataclass(frozen=True, slots=True)
class MissingEntry:
    """Content-free marker retained only while another entry links to its ID."""

    id: str


@dataclass(frozen=True, slots=True)
class MemoryState:
    format: Literal[2]
    project_id: str
    generation: int
    schema: MemorySchema
    entries: dict[str, MemoryEntry | MissingEntry]
    compacted_through_version: str | None = None


@dataclass(frozen=True, slots=True)
class AddOperation:
    kind: str
    text: str
    sources: tuple[MemorySource, ...]
    expires_at: OptionalTimestamp = UNSET
    valid_from: str | None = None
    valid_until: OptionalTimestamp = UNSET


@dataclass(frozen=True, slots=True)
class UpdateOperation:
    entry_id: str
    sources: tuple[MemorySource, ...]
    text: str | None = None
    kind: str | None = None
    expires_at: OptionalTimestamp = UNSET
    valid_from: str | None = None
    valid_until: OptionalTimestamp = UNSET


@dataclass(frozen=True, slots=True)
class SupersedeOperation:
    entry_ids: tuple[str, ...]
    kind: str
    text: str
    sources: tuple[MemorySource, ...]
    expires_at: OptionalTimestamp = UNSET
    valid_from: str | None = None
    valid_until: OptionalTimestamp = UNSET


@dataclass(frozen=True, slots=True)
class ResolveOperation:
    entry_id: str
    sources: tuple[MemorySource, ...]


@dataclass(frozen=True, slots=True)
class ExpireOperation:
    entry_id: str
    reason: str


MemoryOperation = (
    AddOperation | UpdateOperation | SupersedeOperation | ResolveOperation | ExpireOperation
)


@dataclass(frozen=True, slots=True)
class OperationReceipt:
    operation_id: str
    type: ReceiptType
    target_ids: tuple[str, ...]
    result_ids: tuple[str, ...]
    reason: str
    reconciliation_key: str | None
    automatic: bool


@dataclass(frozen=True, slots=True)
class EntryMemorySnapshot:
    state: MemoryState
    digest: str
    version: str


@dataclass(frozen=True, slots=True)
class EntryCASResult(EntryMemorySnapshot):
    published: bool
    receipts: tuple[OperationReceipt, ...]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_entry_id() -> str:
    """Return a time-sortable random ID without exposing ID creation to callers."""

    timestamp = int(time.time_ns() // 1_000_000).to_bytes(6, "big").hex()
    return f"m_{timestamp}{secrets.token_hex(10)}"


def new_operation_id() -> str:
    timestamp = int(time.time_ns() // 1_000_000).to_bytes(6, "big").hex()
    return f"op_{timestamp}{secrets.token_hex(10)}"


def _fail(message: str) -> None:
    raise ProjectRegistryError(message)


def _matches(pattern: re.Pattern[str], value: object) -> bool:
    return isinstance(value, str) and pattern.fullmatch(value) is not None


def _timestamp(value: str | None, *, optional: bool = False) -> None:
    if value is None and optional:
        return
    if not isinstance(value, str) or not _TIMESTAMP.fullmatch(value):
        _fail("invalid memory entry timestamp")
    try:
        dt.datetime.fromisoformat(value)
    except ValueError:
        _fail("invalid memory entry timestamp")


def _safe_text(value: object, *, maximum: int, label: str, empty: bool = False) -> str:
    if (
        not isinstance(value, str)
        or (not empty and not value.strip())
        or "\x00" in value
        or any(ord(character) < 32 and character not in "\n\t" for character in value)
        or len(value.encode("utf-8")) > maximum
    ):
        _fail(f"invalid memory {label}")
    return value


def validate_schema(schema: MemorySchema) -> None:
    if type(schema.version) is not int or schema.version < 1:
        _fail("invalid memory schema version")
    _safe_text(schema.profile, maximum=64, label="schema profile")
    if not schema.kinds or len(schema.kinds) > MAX_KINDS:
        _fail("invalid memory schema kinds")
    seen: set[str] = set()
    for kind in schema.kinds:
        if not _matches(_KIND_KEY, kind.key) or kind.key in seen:
            _fail("invalid memory kind key")
        seen.add(kind.key)
        _safe_text(kind.name, maximum=64, label="kind name")
        _safe_text(kind.description, maximum=1024, label="kind description")
        if kind.prompt_mode not in {"always", "recent", "on_demand"}:
            _fail("invalid memory kind prompt mode")
        if type(kind.prompt_priority) is not int or not 0 <= kind.prompt_priority <= 100:
            _fail("invalid memory kind prompt priority")
        if type(kind.prompt_max_entries) is not int or not 0 <= kind.prompt_max_entries <= 1000:
            _fail("invalid memory kind prompt bound")
        if kind.default_expiry_days is not None and (
            type(kind.default_expiry_days) is not int
            or not 1 <= kind.default_expiry_days <= 3650
        ):
            _fail("invalid memory kind expiry")


def _validate_source(source: MemorySource) -> None:
    _safe_text(source.session_id, maximum=256, label="source session")
    if (
        type(source.seq_start) is not int
        or type(source.seq_end) is not int
        or source.seq_start < 1
        or source.seq_start > source.seq_end
        or not source.origins
        or len(set(source.origins)) != len(source.origins)
        or any(origin not in _ALLOWED_ORIGINS for origin in source.origins)
        or type(source.evidence_rank) is not int
        or not 1 <= source.evidence_rank <= 6
    ):
        _fail("invalid memory entry source")
    _timestamp(source.observed_at)


def _validate_entry(entry: MemoryEntry, state: MemoryState, kinds: set[str]) -> None:
    if not _matches(_ENTRY_ID, entry.id) or entry.project_id != state.project_id:
        _fail("invalid memory entry identity")
    if (
        not isinstance(entry.kind, str)
        or entry.kind not in kinds
        or not isinstance(entry.representation, str)
        or entry.representation not in {"entry", "legacy_document"}
    ):
        _fail("invalid memory entry kind or representation")
    _safe_text(
        entry.text,
        maximum=(
            MAX_MEMORY_FILE_SIZE
            if entry.representation == "legacy_document"
            else MAX_ENTRY_TEXT_BYTES
        ),
        label="entry text",
    )
    if contains_secret(entry.text):
        _fail("memory entry text contains a secret")
    if not isinstance(entry.status, str) or entry.status not in {
        "active", "superseded", "resolved", "expired"
    }:
        _fail("invalid memory entry status")
    for value in (entry.created_at, entry.updated_at, entry.seen_at, entry.valid_from):
        _timestamp(value)
    for value in (entry.expires_at, entry.valid_until, entry.accepted_at):
        _timestamp(value, optional=True)
    if entry.section is not None:
        _safe_text(entry.section, maximum=256, label="entry section")
        if "\n" in entry.section or entry.section.startswith("#"):
            _fail("invalid memory entry section")
    if (entry.migration_order is None) != (entry.migration_source is None) or (
        entry.migration_order is not None
        and (
            type(entry.migration_order) is not int
            or not 0 <= entry.migration_order < MAX_ENTRIES
        )
    ):
        _fail("invalid memory entry migration order")
    if entry.migration_source is not None and (
        not _matches(_DIGEST, entry.migration_source.source_digest)
        or (
            entry.migration_source.source_version is not None
            and not _matches(_VERSION_ID, entry.migration_source.source_version)
        )
    ):
        _fail("invalid memory entry migration source")
    if type(entry.automatic) is not bool:
        _fail("invalid memory entry provenance")
    if (entry.accepted_at is None) != (entry.accepted_by is None):
        _fail("invalid memory entry acceptance")
    if entry.accepted_by is not None and entry.accepted_by != "user":
        _fail("invalid memory entry acceptance")
    if len(entry.sources) > MAX_SOURCES_PER_ENTRY:
        _fail("too many memory entry sources")
    for source in entry.sources:
        _validate_source(source)
    if any(
        not _matches(_ENTRY_ID, item)
        for item in (*entry.supersedes, *entry.superseded_by)
    ):
        _fail("invalid memory entry link")
    if len(set(entry.supersedes)) != len(entry.supersedes) or len(
        set(entry.superseded_by)
    ) != len(entry.superseded_by):
        _fail("duplicate memory entry link")
    if entry.id in entry.supersedes or entry.id in entry.superseded_by:
        _fail("cyclic memory entry link")
    if not _matches(_OPERATION_ID, entry.last_operation_id):
        _fail("invalid memory operation identity")


def validate_state(state: MemoryState) -> None:
    if state.format != 2 or not _matches(_PROJECT_ID, state.project_id):
        _fail("invalid format-2 memory state")
    if type(state.generation) is not int or state.generation < 0:
        _fail("invalid memory generation")
    if state.compacted_through_version is not None and not _matches(
        _VERSION_ID, state.compacted_through_version
    ):
        _fail("invalid memory compaction version")
    if not isinstance(state.schema, MemorySchema):
        _fail("invalid memory schema")
    validate_schema(state.schema)
    if not isinstance(state.entries, dict) or len(state.entries) > MAX_ENTRIES:
        _fail("too many memory entries")
    kinds = {kind.key for kind in state.schema.kinds}
    graph: dict[str, set[str]] = {}
    for entry_id, entry in state.entries.items():
        if entry_id != entry.id:
            _fail("invalid memory entry map key")
        if isinstance(entry, MissingEntry):
            if not _matches(_ENTRY_ID, entry.id):
                _fail("invalid missing memory entry")
            continue
        if not isinstance(entry, MemoryEntry):
            _fail("invalid memory entry value")
        _validate_entry(entry, state, kinds)
        links = set(entry.supersedes) | set(entry.superseded_by)
        if any(link not in state.entries for link in links):
            _fail("dangling memory entry link")
        graph[entry.id] = links
    for entry in state.entries.values():
        if not isinstance(entry, MemoryEntry):
            continue
        for target_id in entry.supersedes:
            target = state.entries[target_id]
            if isinstance(target, MemoryEntry) and entry.id not in target.superseded_by:
                _fail("non-reciprocal memory entry link")
        for replacement_id in entry.superseded_by:
            replacement = state.entries[replacement_id]
            if isinstance(replacement, MemoryEntry) and entry.id not in replacement.supersedes:
                _fail("non-reciprocal memory entry link")
    # Supersede links form a directed old-to-new graph and must be acyclic.
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(entry_id: str) -> None:
        if entry_id in visiting:
            _fail("cyclic memory entry link")
        if entry_id in visited:
            return
        visiting.add(entry_id)
        entry = state.entries[entry_id]
        if isinstance(entry, MemoryEntry):
            for replacement_id in entry.superseded_by:
                visit(replacement_id)
        visiting.remove(entry_id)
        visited.add(entry_id)

    for entry_id in graph:
        visit(entry_id)


def _source_dict(source: MemorySource) -> dict[str, object]:
    return {
        "session_id": source.session_id,
        "seq_start": source.seq_start,
        "seq_end": source.seq_end,
        "origins": list(source.origins),
        "observed_at": source.observed_at,
        "evidence_rank": source.evidence_rank,
    }


def _entry_dict(entry: MemoryEntry | MissingEntry) -> dict[str, object]:
    if isinstance(entry, MissingEntry):
        return {"id": entry.id, "missing": True}
    return {
        "id": entry.id,
        "project_id": entry.project_id,
        "kind": entry.kind,
        "text": entry.text,
        "representation": entry.representation,
        "status": entry.status,
        "created_at": entry.created_at,
        "updated_at": entry.updated_at,
        "seen_at": entry.seen_at,
        "expires_at": entry.expires_at,
        "valid_from": entry.valid_from,
        "valid_until": entry.valid_until,
        "supersedes": list(entry.supersedes),
        "superseded_by": list(entry.superseded_by),
        "sources": [_source_dict(source) for source in entry.sources],
        "automatic": entry.automatic,
        "accepted_at": entry.accepted_at,
        "accepted_by": entry.accepted_by,
        "last_operation_id": entry.last_operation_id,
        "section": entry.section,
        "migration_order": entry.migration_order,
        "migration_source": (
            None if entry.migration_source is None else dataclasses.asdict(entry.migration_source)
        ),
    }


def state_to_dict(state: MemoryState) -> dict[str, object]:
    return {
        "format": 2,
        "project_id": state.project_id,
        "generation": state.generation,
        "schema": {
            "version": state.schema.version,
            "profile": state.schema.profile,
            "kinds": [dataclasses.asdict(kind) for kind in state.schema.kinds],
        },
        "entries": {
            entry_id: _entry_dict(state.entries[entry_id])
            for entry_id in sorted(state.entries)
        },
        "compacted_through_version": state.compacted_through_version,
    }


def canonical_state_bytes(state: MemoryState) -> bytes:
    validate_state(state)
    payload = json.dumps(
        state_to_dict(state), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if len(payload) > MAX_STATE_BYTES:
        _fail("format-2 memory state is too large")
    return payload


def state_digest(state: MemoryState) -> str:
    return hashlib.sha256(canonical_state_bytes(state)).hexdigest()


def _exact_dict(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != fields:
        _fail(f"invalid {label}")
    return value


def state_from_bytes(payload: bytes) -> MemoryState:
    if len(payload) > MAX_STATE_BYTES:
        _fail("format-2 memory state is too large")
    try:
        raw = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProjectRegistryError("format-2 memory state is malformed") from exc
    root = _exact_dict(
        raw,
        {"format", "project_id", "generation", "schema", "entries", "compacted_through_version"},
        "format-2 memory state",
    )
    schema_raw = _exact_dict(root["schema"], {"version", "profile", "kinds"}, "memory schema")
    if not isinstance(schema_raw["kinds"], list):
        _fail("invalid memory schema")
    kind_fields = {
        "key", "name", "description", "prompt_mode", "prompt_priority",
        "prompt_max_entries", "default_expiry_days",
    }
    kinds = tuple(MemoryKind(**_exact_dict(item, kind_fields, "memory kind")) for item in schema_raw["kinds"])
    if not isinstance(root["entries"], dict):
        _fail("invalid memory entries")
    entries: dict[str, MemoryEntry | MissingEntry] = {}
    entry_fields = {
        "id", "project_id", "kind", "text", "representation", "status",
        "created_at", "updated_at", "seen_at", "expires_at", "valid_from",
        "valid_until", "supersedes", "superseded_by", "sources", "automatic",
        "accepted_at", "accepted_by", "last_operation_id", "section",
        "migration_order", "migration_source",
    }
    source_fields = {
        "session_id", "seq_start", "seq_end", "origins", "observed_at",
        "evidence_rank",
    }
    for entry_id, item in root["entries"].items():
        if isinstance(item, dict) and item.get("missing") is True:
            marker = _exact_dict(item, {"id", "missing"}, "missing memory entry")
            entries[entry_id] = MissingEntry(str(marker["id"]))
            continue
        value = _exact_dict(item, entry_fields, "memory entry")
        sources_raw = value["sources"]
        if (
            not isinstance(sources_raw, list)
            or not isinstance(value["supersedes"], list)
            or not isinstance(value["superseded_by"], list)
        ):
            _fail("invalid memory entry links or sources")
        sources = tuple(
            MemorySource(
                **{
                    **_exact_dict(source, source_fields, "memory entry source"),
                    "origins": tuple(source["origins"]),
                }
            )
            for source in sources_raw
            if isinstance(source, dict) and isinstance(source.get("origins"), list)
        )
        if len(sources) != len(sources_raw):
            _fail("invalid memory entry sources")
        migration_raw = value["migration_source"]
        if migration_raw is not None:
            migration_raw = _exact_dict(
                migration_raw,
                {"source_digest", "source_version"},
                "memory entry migration source",
            )
        entries[entry_id] = MemoryEntry(
            **{
                **value,
                "supersedes": tuple(value["supersedes"]),
                "superseded_by": tuple(value["superseded_by"]),
                "sources": sources,
                "migration_source": (
                    None if migration_raw is None else MigrationSource(**migration_raw)
                ),
            }
        )
    state = MemoryState(
        format=root["format"],
        project_id=root["project_id"],
        generation=root["generation"],
        schema=MemorySchema(
            version=schema_raw["version"], profile=schema_raw["profile"], kinds=kinds
        ),
        entries=entries,
        compacted_through_version=root["compacted_through_version"],
    )
    # Canonical reserialization also enforces the byte bound and all invariants.
    canonical_state_bytes(state)
    return state


def receipt_to_dict(receipt: OperationReceipt) -> dict[str, object]:
    return {
        "operation_id": receipt.operation_id,
        "type": receipt.type,
        "target_ids": list(receipt.target_ids),
        "result_ids": list(receipt.result_ids),
        "reason": receipt.reason,
        "reconciliation_key": receipt.reconciliation_key,
        "automatic": receipt.automatic,
    }


def receipt_from_dict(value: object) -> OperationReceipt:
    raw = _exact_dict(
        value,
        {"operation_id", "type", "target_ids", "result_ids", "reason", "reconciliation_key", "automatic"},
        "memory operation receipt",
    )
    if not isinstance(raw["target_ids"], list) or not isinstance(
        raw["result_ids"], list
    ):
        _fail("invalid memory operation receipt entries")
    receipt = OperationReceipt(
        operation_id=raw["operation_id"],
        type=raw["type"],
        target_ids=tuple(raw["target_ids"]),
        result_ids=tuple(raw["result_ids"]),
        reason=raw["reason"],
        reconciliation_key=raw["reconciliation_key"],
        automatic=raw["automatic"],
    )
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: OperationReceipt) -> None:
    if not _matches(_OPERATION_ID, receipt.operation_id) or not isinstance(
        receipt.type, str
    ) or receipt.type not in {
        "add", "update", "supersede", "resolve", "expire", "accept", "undo", "migrate", "sync"
    }:
        _fail("invalid memory operation receipt")
    if any(not _matches(_ENTRY_ID, item) for item in (*receipt.target_ids, *receipt.result_ids)):
        _fail("invalid memory operation receipt entry")
    _safe_text(receipt.reason, maximum=MAX_REASON_BYTES, label="operation reason")
    if receipt.reconciliation_key is not None and not _matches(_DIGEST, receipt.reconciliation_key):
        _fail("invalid memory reconciliation key")
    if type(receipt.automatic) is not bool:
        _fail("invalid memory operation provenance")


def empty_state(project_id: str, schema: MemorySchema) -> MemoryState:
    state = MemoryState(2, project_id, 0, schema, {})
    canonical_state_bytes(state)
    return state


def _merged_sources(
    existing: tuple[MemorySource, ...], added: tuple[MemorySource, ...]
) -> tuple[MemorySource, ...]:
    for source in added:
        _validate_source(source)
    values = list(existing)
    for source in added:
        if source not in values:
            values.append(source)
    if len(values) > MAX_SOURCES_PER_ENTRY:
        _fail("too many memory entry sources")
    return tuple(values)


def _check_evidence(
    sources: Iterable[MemorySource], evidence: tuple[str, int, int] | None
) -> None:
    if evidence is None:
        return
    session_id, seq_start, seq_end = evidence
    if any(
        source.session_id != session_id
        or source.seq_start < seq_start
        or source.seq_end > seq_end
        for source in sources
    ):
        _fail("memory operation source is outside supplied evidence")


def _active(entries: Mapping[str, MemoryEntry | MissingEntry], entry_id: str) -> MemoryEntry:
    entry = entries.get(entry_id)
    if not isinstance(entry, MemoryEntry):
        _fail("memory operation targets a missing entry")
    if entry.status != "active":
        _fail("memory operation targets an inactive entry")
    return entry


def _new_expiry(
    state: MemoryState,
    kind: str,
    value: OptionalTimestamp,
    *,
    automatic: bool,
    seen_at: str,
) -> str | None:
    if not isinstance(value, _Unset):
        return value
    default_days = next(
        (
            item.default_expiry_days
            for item in state.schema.kinds
            if item.key == kind
        ),
        None,
    )
    if not automatic or default_days is None:
        return None
    instant = dt.datetime.fromisoformat(seen_at) + dt.timedelta(days=default_days)
    return instant.isoformat(timespec="microseconds").replace("+00:00", "Z")


def _new_entry(
    *, project_id: str, kind: str, text: str, sources: tuple[MemorySource, ...],
    automatic: bool, now: str, operation_id: str, entry_id: str,
    expires_at: str | None, valid_from: str | None, valid_until: str | None,
    supersedes: tuple[str, ...] = (),
) -> MemoryEntry:
    seen_at = max((source.observed_at for source in sources), default=now)
    return MemoryEntry(
        id=entry_id, project_id=project_id, kind=kind, text=text,
        representation="entry", status="active", created_at=now, updated_at=now,
        seen_at=seen_at, expires_at=expires_at, valid_from=valid_from or now,
        valid_until=valid_until, supersedes=supersedes, superseded_by=(),
        sources=sources, automatic=automatic,
        accepted_at=None if automatic else now,
        accepted_by=None if automatic else "user", last_operation_id=operation_id,
    )


def apply_operations(
    state: MemoryState,
    operations: tuple[MemoryOperation, ...],
    *, reconciliation_key: str | None,
    automatic: bool,
    now: str | None = None,
    evidence: tuple[str, int, int] | None = None,
    entry_id_factory: Callable[[], str] = new_entry_id,
    operation_id_factory: Callable[[], str] = new_operation_id,
) -> tuple[MemoryState, tuple[OperationReceipt, ...]]:
    """Validate and apply a grouped transaction, returning no partial state."""

    canonical_state_bytes(state)
    if not operations or len(operations) > MAX_OPERATIONS:
        _fail("invalid memory operation group")
    if reconciliation_key is not None and not _matches(_DIGEST, reconciliation_key):
        _fail("invalid memory reconciliation key")
    now = now or utc_now()
    _timestamp(now)
    entries = dict(state.entries)
    receipts: list[OperationReceipt] = []
    touched: set[str] = set()
    for operation in operations:
        operation_sources = getattr(operation, "sources", None)
        if automatic and operation_sources == ():
            _fail("automatic memory operation requires durable sources")
        operation_id = operation_id_factory()
        if not _matches(_OPERATION_ID, operation_id):
            _fail("invalid generated memory operation identity")
        targets: tuple[str, ...] = ()
        results: tuple[str, ...] = ()
        if isinstance(operation, AddOperation):
            _check_evidence(operation.sources, evidence)
            entry_id = entry_id_factory()
            if entry_id in entries or not _matches(_ENTRY_ID, entry_id):
                _fail("invalid generated memory entry identity")
            entry = _new_entry(
                project_id=state.project_id, kind=operation.kind, text=operation.text,
                sources=operation.sources, automatic=automatic, now=now,
                operation_id=operation_id, entry_id=entry_id,
                expires_at=_new_expiry(
                    state, operation.kind, operation.expires_at,
                    automatic=automatic,
                    seen_at=max(
                        (source.observed_at for source in operation.sources),
                        default=now,
                    ),
                ),
                valid_from=operation.valid_from,
                valid_until=(
                    None if isinstance(operation.valid_until, _Unset)
                    else operation.valid_until
                ),
            )
            entries[entry_id] = entry
            results = (entry_id,)
            reason = "new memory entry"
            receipt_type: ReceiptType = "add"
        elif isinstance(operation, UpdateOperation):
            if operation.entry_id in touched:
                _fail("duplicate memory operation target")
            entry = _active(entries, operation.entry_id)
            _check_evidence(operation.sources, evidence)
            seen_at = max(
                (entry.seen_at, *(source.observed_at for source in operation.sources))
            )
            kind = entry.kind if operation.kind is None else operation.kind
            expires_at = operation.expires_at
            if isinstance(expires_at, _Unset):
                prior_default = _new_expiry(
                    state,
                    entry.kind,
                    UNSET,
                    automatic=entry.automatic,
                    seen_at=entry.seen_at,
                )
                expires_at = (
                    _new_expiry(
                        state,
                        kind,
                        UNSET,
                        automatic=automatic,
                        seen_at=seen_at,
                    )
                    if entry.expires_at == prior_default
                    else entry.expires_at
                )
            entries[entry.id] = replace(
                entry,
                text=entry.text if operation.text is None else operation.text,
                kind=kind,
                expires_at=expires_at,
                valid_from=entry.valid_from if operation.valid_from is None else operation.valid_from,
                valid_until=(
                    entry.valid_until if isinstance(operation.valid_until, _Unset)
                    else operation.valid_until
                ),
                updated_at=now,
                seen_at=seen_at,
                sources=_merged_sources(entry.sources, operation.sources),
                last_operation_id=operation_id,
            )
            targets = results = (entry.id,)
            touched.add(entry.id)
            reason = "refined memory entry"
            receipt_type = "update"
        elif isinstance(operation, SupersedeOperation):
            if not operation.entry_ids or len(set(operation.entry_ids)) != len(operation.entry_ids):
                _fail("invalid supersede targets")
            if touched.intersection(operation.entry_ids):
                _fail("duplicate memory operation target")
            old_entries = tuple(_active(entries, entry_id) for entry_id in operation.entry_ids)
            _check_evidence(operation.sources, evidence)
            entry_id = entry_id_factory()
            if entry_id in entries or not _matches(_ENTRY_ID, entry_id):
                _fail("invalid generated memory entry identity")
            replacement = _new_entry(
                project_id=state.project_id, kind=operation.kind, text=operation.text,
                sources=operation.sources, automatic=automatic, now=now,
                operation_id=operation_id, entry_id=entry_id,
                expires_at=_new_expiry(
                    state, operation.kind, operation.expires_at,
                    automatic=automatic,
                    seen_at=max(
                        (source.observed_at for source in operation.sources),
                        default=now,
                    ),
                ),
                valid_from=operation.valid_from,
                valid_until=(
                    None if isinstance(operation.valid_until, _Unset)
                    else operation.valid_until
                ),
                supersedes=operation.entry_ids,
            )
            entries[entry_id] = replacement
            for old in old_entries:
                entries[old.id] = replace(
                    old, status="superseded", updated_at=now, valid_until=now,
                    superseded_by=(*old.superseded_by, entry_id),
                    last_operation_id=operation_id,
                )
            targets, results = operation.entry_ids, (entry_id,)
            touched.update(operation.entry_ids)
            reason = "newer memory supersedes prior entries"
            receipt_type = "supersede"
        elif isinstance(operation, ResolveOperation):
            if operation.entry_id in touched:
                _fail("duplicate memory operation target")
            entry = _active(entries, operation.entry_id)
            _check_evidence(operation.sources, evidence)
            entries[entry.id] = replace(
                entry, status="resolved", updated_at=now, valid_until=now,
                sources=_merged_sources(entry.sources, operation.sources),
                last_operation_id=operation_id,
            )
            targets = (entry.id,)
            touched.add(entry.id)
            reason = "memory entry resolved"
            receipt_type = "resolve"
        elif isinstance(operation, ExpireOperation):
            if operation.entry_id in touched:
                _fail("duplicate memory operation target")
            entry = _active(entries, operation.entry_id)
            reason = _safe_text(operation.reason, maximum=MAX_REASON_BYTES, label="expiry reason")
            entries[entry.id] = replace(
                entry, status="expired", updated_at=now, valid_until=now,
                last_operation_id=operation_id,
            )
            targets = (entry.id,)
            touched.add(entry.id)
            receipt_type = "expire"
        else:
            _fail("unknown memory operation")
        receipt = OperationReceipt(
            operation_id, receipt_type, targets, results, reason,
            reconciliation_key, automatic,
        )
        validate_receipt(receipt)
        receipts.append(receipt)
    result = replace(state, generation=state.generation + 1, entries=entries)
    canonical_state_bytes(result)
    return result, tuple(receipts)


def accept_entry(
    state: MemoryState, entry_id: str, *, now: str | None = None
) -> tuple[MemoryState, OperationReceipt]:
    now = now or utc_now()
    entry = state.entries.get(entry_id)
    if not isinstance(entry, MemoryEntry):
        _fail("memory accept targets a missing entry")
    if entry.status != "active":
        _fail("memory accept targets an inactive entry")
    if not entry.automatic or entry.accepted_at is not None:
        _fail("memory accept requires an unaccepted automatic entry")
    operation_id = new_operation_id()
    entries = dict(state.entries)
    entries[entry_id] = replace(
        entry, accepted_at=now, accepted_by="user", updated_at=now,
        last_operation_id=operation_id,
    )
    result = replace(state, generation=state.generation + 1, entries=entries)
    receipt = OperationReceipt(
        operation_id, "accept", (entry_id,), (entry_id,), "accepted by user", None, False
    )
    canonical_state_bytes(result)
    validate_receipt(receipt)
    return result, receipt


def compact_inactive_entries(
    state: MemoryState,
    *, retained_operation_ids: set[str],
    compacted_through_version: str | None,
) -> MemoryState:
    """Remove inactive bodies after their transition receipts leave retention."""

    entries = dict(state.entries)
    active_links = {
        link
        for entry in entries.values()
        if isinstance(entry, MemoryEntry) and entry.status == "active"
        for link in (*entry.supersedes, *entry.superseded_by)
    }
    aged = {
        entry_id
        for entry_id, entry in entries.items()
        if isinstance(entry, MemoryEntry)
        and entry.status != "active"
        and entry.last_operation_id not in retained_operation_ids
    }
    marker_ids = aged & active_links
    deleted_ids = (aged - marker_ids) | {
        entry_id
        for entry_id, entry in entries.items()
        if isinstance(entry, MissingEntry) and entry_id not in active_links
    }
    if not marker_ids and not deleted_ids:
        return state
    for entry_id in marker_ids:
        entries[entry_id] = MissingEntry(entry_id)
    for entry_id in deleted_ids:
        del entries[entry_id]
    for entry_id, entry in tuple(entries.items()):
        if isinstance(entry, MemoryEntry) and entry.status != "active":
            entries[entry_id] = replace(
                entry,
                supersedes=tuple(
                    target for target in entry.supersedes if target not in deleted_ids
                ),
                superseded_by=tuple(
                    target for target in entry.superseded_by if target not in deleted_ids
                ),
            )
    result = replace(
        state, entries=entries, compacted_through_version=compacted_through_version
    )
    canonical_state_bytes(result)
    return result
