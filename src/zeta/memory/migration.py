"""Pure format-1 to format-2 project-memory migration.

This module has no registry, command, CLI, import, or project-discovery dependency.
Callers provide an already locked snapshot and an explicit clock value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zeta.memory.entry_store import (
    MAX_ENTRY_TEXT_BYTES,
    MemoryEntry,
    MemoryState,
    MigrationSource,
    OperationReceipt,
    canonical_state_bytes,
    validate_state,
)
from zeta.memory.profiles import memory_profile
from zeta.memory_migration_plan import (
    LEGACY_MEMORY_FILES,
    build_migration_entries,
    legacy_memory_digest,
    migration_operation_id,
    normalize_migration_text,
)
from zeta.project_errors import ProjectRegistryError

LEGACY_FILES = LEGACY_MEMORY_FILES


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """A deterministic migration result and its exact format-1 reversal data."""

    state: MemoryState
    receipt: OperationReceipt
    source_digest: str
    source_version: str | None
    source_contents: dict[str, str]
    automatic_files: frozenset[str]
    rendered_mirrors: dict[str, str]
    migrated_at: str


def _entry_from_wire(value: Mapping[str, object]) -> MemoryEntry:
    migration = value["migration_source"]
    assert isinstance(migration, dict)
    return MemoryEntry(
        **{
            **value,
            "supersedes": tuple(value["supersedes"]),
            "superseded_by": tuple(value["superseded_by"]),
            "sources": (),
            "migration_source": MigrationSource(**migration),
        }
    )


def _render_entries(entries: list[MemoryEntry]) -> str:
    blocks: list[str] = []
    section: str | None = None
    for entry in entries:
        if entry.section != section:
            if entry.section is None:
                raise ProjectRegistryError("migration entries have invalid section order")
            blocks.append(f"## {entry.section}")
            section = entry.section
        blocks.append(entry.text)
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def migrate_format_one(
    *,
    project_id: str,
    contents: Mapping[str, str],
    source_digest: str,
    source_version: str | None,
    migrated_at: str,
    automatic_files: frozenset[str] = frozenset(),
) -> MigrationPlan:
    """Build one deterministic, structurally split migration plan."""

    if set(contents) != set(LEGACY_FILES):
        raise ProjectRegistryError("migration requires the complete format-1 snapshot")
    if not automatic_files <= set(LEGACY_FILES):
        raise ProjectRegistryError("migration has invalid automatic files")
    copied = {name: contents[name] for name in LEGACY_FILES}
    expected_digest = legacy_memory_digest(copied)
    if source_digest != expected_digest:
        raise ProjectRegistryError("migration source digest does not match its snapshot")
    wire_entries = build_migration_entries(
        project_id=project_id,
        contents=copied,
        source_digest=source_digest,
        source_version=source_version,
        migrated_at=migrated_at,
        automatic_files=automatic_files,
        max_entry_bytes=MAX_ENTRY_TEXT_BYTES,
    )
    entries = {
        entry_id: _entry_from_wire(value) for entry_id, value in wire_entries.items()
    }
    state = MemoryState(2, project_id, 1, memory_profile("zeta"), entries)
    validate_state(state)
    canonical_state_bytes(state)
    operation_id = migration_operation_id(project_id, source_digest)
    receipt = OperationReceipt(
        operation_id=operation_id,
        type="migrate",
        target_ids=(),
        result_ids=tuple(entries),
        reason="structurally migrated format-1 memory",
        reconciliation_key=source_digest,
        automatic=False,
    )
    rendered = {
        name: _render_entries(
            [entry for entry in entries.values() if entry.kind == name.removesuffix(".md")]
        )
        for name in LEGACY_FILES
    }
    if any(
        normalize_migration_text(rendered[name])
        != normalize_migration_text(copied[name])
        for name in LEGACY_FILES
    ):
        raise ProjectRegistryError("migration structural round-trip failed")
    return MigrationPlan(
        state=state,
        receipt=receipt,
        source_digest=source_digest,
        source_version=source_version,
        source_contents=copied,
        automatic_files=automatic_files,
        rendered_mirrors=rendered,
        migrated_at=migrated_at,
    )


def reverse_migration(plan: MigrationPlan) -> dict[str, str]:
    """Return the exact format-1 snapshot stored by a migration plan."""

    if not isinstance(plan, MigrationPlan):
        raise ProjectRegistryError("invalid memory migration plan")
    if legacy_memory_digest(plan.source_contents) != plan.source_digest:
        raise ProjectRegistryError("memory migration reversal data is corrupt")
    return dict(plan.source_contents)


__all__ = ["LEGACY_FILES", "MigrationPlan", "migrate_format_one", "reverse_migration"]
