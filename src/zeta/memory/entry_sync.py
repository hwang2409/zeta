"""Deterministic relationship-aware project-memory synchronization.

This is the format-2 merge seam. It owns complete-state validation, atomic
supersession-set merging, conflict recording and conflict resolution. Callers
only persist its serializable baseline/conflict records and publish the two
returned states.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

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
ConflictChoice = Literal["local", "remote"]


@dataclass(frozen=True, slots=True)
class EntryMemoryExport:
    state: MemoryState
    digest: str
    version: str
    versions: tuple[dict[str, object], ...]


@dataclass(frozen=True, slots=True)
class EntryConflict:
    """One schema conflict or one complete relationship-connected entry set."""

    kind: Literal["entries", "schema"]
    entry_ids: tuple[str, ...] = ()
    local_digest: str = ""
    remote_digest: str = ""

    def to_dict(self) -> dict[str, object]:
        value: dict[str, object] = {
            "kind": self.kind,
            "digests": sorted({self.local_digest, self.remote_digest}),
        }
        if self.kind == "entries":
            value["entry_ids"] = list(self.entry_ids)
        return value


@dataclass(frozen=True, slots=True)
class EntryMergeResult:
    local: MemoryState
    remote: MemoryState
    entry_baseline: dict[str, str]
    schema_baseline: str
    conflicts: dict[str, EntryConflict]
    local_changed: tuple[str, ...]
    remote_changed: tuple[str, ...]
    schema_changed_local: bool = False
    schema_changed_remote: bool = False

    @property
    def conflict_keys(self) -> tuple[str, ...]:
        return tuple(
            sorted(self.conflicts, key=lambda item: (item != "schema", item))
        )

    @property
    def updated(self) -> tuple[str, ...]:
        values = set(self.local_changed) | set(self.remote_changed)
        if self.schema_changed_local or self.schema_changed_remote:
            values.add("schema")
        return tuple(sorted(values, key=lambda item: (item != "schema", item)))


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
    local: MemoryState,
    remote: MemoryState,
    *,
    entry_baseline: Mapping[str, str],
    schema_baseline: str | None,
    conflicts: Mapping[str, object],
    resolutions: Mapping[str, ConflictChoice] | None = None,
) -> EntryMergeResult:
    """Merge two states, deciding schema compatibility before entry propagation."""

    validate_state(local)
    validate_state(remote)
    if local.project_id != remote.project_id:
        raise ProjectRegistryError("entry sync project IDs differ")
    baseline = _validate_baseline(entry_baseline)
    existing = _parse_conflicts(conflicts)
    choices = dict(resolutions or {})
    if set(choices) - set(existing):
        raise ProjectRegistryError("entry sync resolution targets no recorded conflict")
    if any(choice not in {"local", "remote"} for choice in choices.values()):
        raise ProjectRegistryError("entry sync resolution choice is invalid")

    next_conflicts: dict[str, EntryConflict] = {}
    local_schema = local.schema
    remote_schema = remote.schema
    local_schema_changed = False
    remote_schema_changed = False
    schema_blocked = False
    schema_choice = choices.get("schema")
    existing_schema = existing.get("schema")
    local_schema_hash = schema_digest(local.schema)
    remote_schema_hash = schema_digest(remote.schema)

    if existing_schema is not None and schema_choice is None:
        schema_blocked = True
        next_conflicts["schema"] = _schema_conflict(local.schema, remote.schema)
        next_schema_digest = schema_baseline or local_schema_hash
    elif existing_schema is not None:
        chosen_schema = local.schema if schema_choice == "local" else remote.schema
        local_schema_changed = local.schema != chosen_schema
        remote_schema_changed = remote.schema != chosen_schema
        local_schema = remote_schema = chosen_schema
        next_schema_digest = schema_digest(chosen_schema)
    elif local_schema_hash == remote_schema_hash:
        next_schema_digest = local_schema_hash
    else:
        local_side_changed = (
            schema_baseline is None or local_schema_hash != schema_baseline
        )
        remote_side_changed = (
            schema_baseline is None or remote_schema_hash != schema_baseline
        )
        if local_side_changed and remote_side_changed:
            schema_blocked = True
            next_conflicts["schema"] = _schema_conflict(local.schema, remote.schema)
            next_schema_digest = schema_baseline or local_schema_hash
        else:
            chosen_schema = local.schema if local_side_changed else remote.schema
            local_schema_changed = local.schema != chosen_schema
            remote_schema_changed = remote.schema != chosen_schema
            local_schema = remote_schema = chosen_schema
            next_schema_digest = schema_digest(chosen_schema)

    local_entries = dict(local.entries)
    remote_entries = dict(remote.entries)
    if not schema_blocked and local_schema == remote_schema:
        local_entries = _entries_valid_for_schema(local_entries, local_schema)
        remote_entries = _entries_valid_for_schema(remote_entries, remote_schema)

    local_result = dict(local_entries)
    remote_result = dict(remote_entries)
    local_changed = {
        entry_id for entry_id in local.entries if entry_id not in local_entries
    }
    remote_changed = {
        entry_id for entry_id in remote.entries if entry_id not in remote_entries
    }

    entry_conflicts = {
        key: conflict
        for key, conflict in existing.items()
        if conflict.kind == "entries"
    }
    all_ids = (
        set(local_entries)
        | set(remote_entries)
        | {
            entry_id
            for conflict in entry_conflicts.values()
            for entry_id in conflict.entry_ids
        }
    )
    for component in _relationship_components(all_ids, local_entries, remote_entries):
        conflict_keys = [
            key
            for key, conflict in entry_conflicts.items()
            if component.intersection(conflict.entry_ids)
        ]
        component_choices = {choices[key] for key in conflict_keys if key in choices}
        if len(component_choices) > 1:
            raise ProjectRegistryError(
                "entry sync relationship component has conflicting resolutions"
            )
        choice = next(iter(component_choices), None)

        if schema_blocked and not _component_valid_under_both_schemas(
            component, local_entries, remote_entries, local.schema, remote.schema
        ):
            if conflict_keys:
                key = min(component)
                next_conflicts[key] = _entry_conflict(
                    component, local_entries, remote_entries
                )
            continue

        if conflict_keys and choice is None:
            key = min(component)
            next_conflicts[key] = _entry_conflict(
                component, local_entries, remote_entries
            )
            continue
        if choice is not None:
            chosen = local_entries if choice == "local" else remote_entries
            _propagate_component(
                component,
                chosen,
                local_result,
                remote_result,
                baseline,
                local_changed,
                remote_changed,
            )
            continue

        local_hashes = {
            entry_id: entry_digest(local_entries.get(entry_id))
            for entry_id in component
        }
        remote_hashes = {
            entry_id: entry_digest(remote_entries.get(entry_id))
            for entry_id in component
        }
        if local_hashes == remote_hashes:
            baseline.update(local_hashes)
            continue
        local_side_changed = any(
            digest != baseline.get(entry_id, _MISSING)
            for entry_id, digest in local_hashes.items()
        )
        remote_side_changed = any(
            digest != baseline.get(entry_id, _MISSING)
            for entry_id, digest in remote_hashes.items()
        )
        if local_side_changed and remote_side_changed:
            key = min(component)
            next_conflicts[key] = _entry_conflict(
                component, local_entries, remote_entries
            )
            continue
        chosen = local_entries if local_side_changed else remote_entries
        _propagate_component(
            component,
            chosen,
            local_result,
            remote_result,
            baseline,
            local_changed,
            remote_changed,
        )

    retained_ids = set(local_result) | set(remote_result)
    retained_ids.update(
        entry_id
        for conflict in next_conflicts.values()
        for entry_id in conflict.entry_ids
    )
    baseline = {
        entry_id: digest
        for entry_id, digest in baseline.items()
        if entry_id in retained_ids
    }

    generation = max(local.generation, remote.generation) + 1
    local_merged = dataclasses.replace(
        local,
        generation=(
            generation if local_changed or local_schema_changed else local.generation
        ),
        schema=local_schema,
        entries=local_result,
    )
    remote_merged = dataclasses.replace(
        remote,
        generation=(
            generation if remote_changed or remote_schema_changed else remote.generation
        ),
        schema=remote_schema,
        entries=remote_result,
    )
    canonical_state_bytes(local_merged)
    canonical_state_bytes(remote_merged)
    return EntryMergeResult(
        local=local_merged,
        remote=remote_merged,
        entry_baseline=baseline,
        schema_baseline=next_schema_digest,
        conflicts=next_conflicts,
        local_changed=tuple(sorted(local_changed)),
        remote_changed=tuple(sorted(remote_changed)),
        schema_changed_local=local_schema_changed,
        schema_changed_remote=remote_schema_changed,
    )


def _entries_valid_for_schema(
    entries: Mapping[str, MemoryEntry | MissingEntry], schema: MemorySchema
) -> dict[str, MemoryEntry | MissingEntry]:
    allowed = {kind.key for kind in schema.kinds}
    result: dict[str, MemoryEntry | MissingEntry] = {}
    for component in _relationship_components(set(entries), entries, {}):
        if all(
            isinstance(entries[entry_id], MissingEntry)
            or entries[entry_id].kind in allowed
            for entry_id in component
        ):
            result.update((entry_id, entries[entry_id]) for entry_id in component)
    return result


def _component_valid_under_both_schemas(
    component: set[str],
    local_entries: Mapping[str, MemoryEntry | MissingEntry],
    remote_entries: Mapping[str, MemoryEntry | MissingEntry],
    local_schema: MemorySchema,
    remote_schema: MemorySchema,
) -> bool:
    local_kinds = {kind.key for kind in local_schema.kinds}
    remote_kinds = {kind.key for kind in remote_schema.kinds}
    return all(
        isinstance(entry, MissingEntry)
        or entry.kind in local_kinds
        and entry.kind in remote_kinds
        for entry_id in component
        for entry in (local_entries.get(entry_id), remote_entries.get(entry_id))
        if entry is not None
    )


def _propagate_component(
    component: set[str],
    chosen: Mapping[str, MemoryEntry | MissingEntry],
    local_result: dict[str, MemoryEntry | MissingEntry],
    remote_result: dict[str, MemoryEntry | MissingEntry],
    baseline: dict[str, str],
    local_changed: set[str],
    remote_changed: set[str],
) -> None:
    for entry_id in component:
        value = chosen.get(entry_id)
        if local_result.get(entry_id) != value:
            _assign(local_result, entry_id, value)
            local_changed.add(entry_id)
        if remote_result.get(entry_id) != value:
            _assign(remote_result, entry_id, value)
            remote_changed.add(entry_id)
        baseline[entry_id] = entry_digest(value)


def _validate_baseline(value: Mapping[str, str]) -> dict[str, str]:
    baseline = dict(value)
    if any(
        not isinstance(entry_id, str)
        or not entry_id.startswith("m_")
        or not _valid_digest(digest)
        for entry_id, digest in baseline.items()
    ):
        raise ProjectRegistryError("entry sync baseline is invalid")
    return baseline


def _parse_conflicts(values: Mapping[str, object]) -> dict[str, EntryConflict]:
    parsed: dict[str, EntryConflict] = {}
    occupied: set[str] = set()
    for key, raw in values.items():
        if not isinstance(key, str) or not isinstance(raw, Mapping):
            raise ProjectRegistryError("entry sync conflict record is invalid")
        kind = raw.get("kind")
        digests = raw.get("digests")
        if (
            kind not in {"entries", "schema"}
            or not isinstance(digests, list)
            or len(digests) not in {1, 2}
            or digests != sorted(set(digests))
            or any(not _valid_digest(value) for value in digests)
        ):
            raise ProjectRegistryError("entry sync conflict record is invalid")
        local_digest = str(digests[0])
        remote_digest = str(digests[-1])
        if kind == "schema":
            if key != "schema" or set(raw) != {"kind", "digests"}:
                raise ProjectRegistryError("entry sync conflict record is invalid")
            parsed[key] = EntryConflict("schema", (), local_digest, remote_digest)
            continue
        entry_ids = raw.get("entry_ids")
        if (
            set(raw) != {"kind", "entry_ids", "digests"}
            or not isinstance(entry_ids, list)
            or not entry_ids
            or entry_ids != sorted(set(entry_ids))
            or any(not isinstance(item, str) or not item.startswith("m_") for item in entry_ids)
            or key != min(entry_ids)
            or occupied.intersection(entry_ids)
        ):
            raise ProjectRegistryError("entry sync conflict record is invalid")
        occupied.update(entry_ids)
        parsed[key] = EntryConflict(
            "entries", tuple(entry_ids), local_digest, remote_digest
        )
    return parsed


def _relationship_components(
    entry_ids: set[str],
    local: Mapping[str, MemoryEntry | MissingEntry],
    remote: Mapping[str, MemoryEntry | MissingEntry],
) -> tuple[set[str], ...]:
    adjacency = {entry_id: set() for entry_id in entry_ids}
    for entries in (local, remote):
        for entry_id, entry in entries.items():
            if not isinstance(entry, MemoryEntry):
                continue
            for related in (*entry.supersedes, *entry.superseded_by):
                adjacency.setdefault(entry_id, set()).add(related)
                adjacency.setdefault(related, set()).add(entry_id)
    components: list[set[str]] = []
    remaining = set(adjacency)
    while remaining:
        root = min(remaining)
        component: set[str] = set()
        pending = [root]
        while pending:
            entry_id = pending.pop()
            if entry_id in component:
                continue
            component.add(entry_id)
            pending.extend(adjacency[entry_id] - component)
        remaining -= component
        components.append(component)
    return tuple(components)


def _entry_conflict(
    entry_ids: set[str],
    local: Mapping[str, MemoryEntry | MissingEntry],
    remote: Mapping[str, MemoryEntry | MissingEntry],
) -> EntryConflict:
    ordered = tuple(sorted(entry_ids))
    return EntryConflict(
        "entries",
        ordered,
        _entry_set_digest(ordered, local),
        _entry_set_digest(ordered, remote),
    )


def _schema_conflict(local: MemorySchema, remote: MemorySchema) -> EntryConflict:
    return EntryConflict("schema", (), schema_digest(local), schema_digest(remote))


def _entry_set_digest(
    entry_ids: tuple[str, ...], entries: Mapping[str, MemoryEntry | MissingEntry]
) -> str:
    return hashlib.sha256(
        json.dumps(
            {entry_id: entry_digest(entries.get(entry_id)) for entry_id in entry_ids},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _assign(
    entries: dict[str, MemoryEntry | MissingEntry],
    entry_id: str,
    value: MemoryEntry | MissingEntry | None,
) -> None:
    if value is None:
        entries.pop(entry_id, None)
    else:
        entries[entry_id] = value


def _valid_digest(value: object) -> bool:
    return value == _MISSING or (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
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
    "ConflictChoice",
    "EntryConflict",
    "EntryMemoryExport",
    "EntryMergeResult",
    "entry_digest",
    "merge_entry_states",
    "merge_version_receipts",
    "schema_digest",
]
