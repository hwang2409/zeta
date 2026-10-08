"""Deterministic entry-level project-memory synchronization.

The module is the format-2 merge seam. It knows no files, transports, locks, or
registries. Callers provide two validated snapshots and the last shared record
digests; it returns one state for each peer plus the next baseline.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from zeta.memory.entry_store import (
    MemoryEntry,
    MemorySchema,
    MemoryState,
    MissingEntry,
    canonical_state_bytes,
    validate_state,
)
from zeta.project_errors import ProjectRegistryError

_MISSING = "missing"


@dataclass(frozen=True, slots=True)
class EntryMemoryExport:
    state: MemoryState
    digest: str
    version: str
    versions: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class EntryMergeResult:
    source: MemoryState
    destination: MemoryState
    entry_baseline: dict[str, str]
    schema_baseline: str
    updated: tuple[str, ...]
    conflicts: tuple[str, ...]


def schema_digest(schema: MemorySchema) -> str:
    return hashlib.sha256(
        json.dumps(
            dataclasses.asdict(schema),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def entry_digest(entry: MemoryEntry | MissingEntry | None) -> str:
    if entry is None:
        return _MISSING
    state_value: dict[str, Any]
    if isinstance(entry, MissingEntry):
        state_value = {"id": entry.id, "missing": True}
    else:
        state_value = dataclasses.asdict(entry)
    return hashlib.sha256(
        json.dumps(
            state_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode()
    ).hexdigest()


def merge_entry_states(
    source: MemoryState,
    destination: MemoryState,
    *,
    entry_baseline: dict[str, str],
    schema_baseline: str | None,
    unresolved: set[str],
) -> EntryMergeResult:
    """Three-way merge two current entry states without semantic deduplication."""

    validate_state(source)
    validate_state(destination)
    if source.project_id != destination.project_id:
        raise ProjectRegistryError("entry sync project IDs differ")
    source_entries = dict(source.entries)
    destination_entries = dict(destination.entries)
    source_result = dict(source_entries)
    destination_result = dict(destination_entries)
    baseline = dict(entry_baseline)
    conflicts = set(unresolved)
    updated: set[str] = set()

    for entry_id in sorted(set(source_entries) | set(destination_entries) | set(baseline)):
        source_value = source_entries.get(entry_id)
        destination_value = destination_entries.get(entry_id)
        source_hash = entry_digest(source_value)
        destination_hash = entry_digest(destination_value)
        if entry_id in conflicts:
            continue
        if source_hash == destination_hash:
            baseline[entry_id] = source_hash
            continue
        old = baseline.get(entry_id, _MISSING)
        source_changed = source_hash != old
        destination_changed = destination_hash != old
        if source_changed and destination_changed:
            conflicts.add(entry_id)
            continue
        chosen = source_value if source_changed else destination_value
        if chosen is None:
            source_result.pop(entry_id, None)
            destination_result.pop(entry_id, None)
        else:
            source_result[entry_id] = chosen
            destination_result[entry_id] = chosen
        baseline[entry_id] = entry_digest(chosen)
        updated.add(entry_id)

    source_schema_digest = schema_digest(source.schema)
    destination_schema_digest = schema_digest(destination.schema)
    next_schema_digest = schema_baseline
    source_schema = source.schema
    destination_schema = destination.schema
    if "schema" not in conflicts:
        if source_schema_digest == destination_schema_digest:
            next_schema_digest = source_schema_digest
        else:
            old_schema = schema_baseline
            source_changed = old_schema is None or source_schema_digest != old_schema
            destination_changed = (
                old_schema is None or destination_schema_digest != old_schema
            )
            if source_changed and destination_changed:
                conflicts.add("schema")
            else:
                chosen_schema = source.schema if source_changed else destination.schema
                source_schema = chosen_schema
                destination_schema = chosen_schema
                next_schema_digest = schema_digest(chosen_schema)
                updated.add("schema")
    if next_schema_digest is None:
        next_schema_digest = schema_digest(destination_schema)

    source_merged = dataclasses.replace(
        source,
        generation=max(source.generation, destination.generation) + 1,
        schema=source_schema,
        entries=source_result,
    )
    destination_merged = dataclasses.replace(
        destination,
        generation=max(source.generation, destination.generation) + 1,
        schema=destination_schema,
        entries=destination_result,
    )
    # Canonical serialization is the final bounds and relationship validation.
    canonical_state_bytes(source_merged)
    canonical_state_bytes(destination_merged)
    return EntryMergeResult(
        source=source_merged,
        destination=destination_merged,
        entry_baseline=baseline,
        schema_baseline=next_schema_digest,
        updated=tuple(sorted(updated)),
        conflicts=tuple(sorted(conflicts, key=lambda item: (item != "schema", item))),
    )


def merge_version_receipts(
    first: tuple[dict[str, object], ...], second: tuple[dict[str, object], ...]
) -> tuple[dict[str, object], ...]:
    """Deduplicate retained operation/version receipts by stable version ID."""

    merged: list[dict[str, object]] = []
    seen: set[str] = set()
    for record in (*first, *second):
        version = record.get("version")
        if isinstance(version, str):
            if version in seen:
                continue
            seen.add(version)
        merged.append(dict(record))
    return tuple(merged)


__all__ = [
    "EntryMemoryExport",
    "EntryMergeResult",
    "entry_digest",
    "merge_entry_states",
    "merge_version_receipts",
    "schema_digest",
]
