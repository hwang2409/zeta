"""Pure format-1 to format-2 project-memory migration.

This module has no registry, command, CLI, import, or project-discovery dependency.
Callers provide an already locked snapshot and an explicit clock value.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from zeta.memory.entry_store import (
    MAX_ENTRY_TEXT_BYTES,
    MemoryState,
    OperationReceipt,
    state_from_bytes,
)
from zeta.memory_migration_plan import (
    LEGACY_MEMORY_FILES,
    build_migration_plan,
    legacy_memory_digest,
    migration_operation_id,
)
from zeta.project_errors import ProjectRegistryError

LEGACY_FILES = LEGACY_MEMORY_FILES


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """A deterministic migration result and its exact format-1 reversal data."""

    state: MemoryState
    canonical_state: bytes
    receipt: OperationReceipt
    source_digest: str
    source_version: str | None
    source_contents: dict[str, str]
    automatic_files: frozenset[str]
    rendered_mirrors: dict[str, str]
    migrated_at: str


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
    try:
        canonical = build_migration_plan(
        project_id=project_id,
        contents=copied,
        source_digest=source_digest,
        source_version=source_version,
        migrated_at=migrated_at,
        automatic_files=automatic_files,
        max_entry_bytes=MAX_ENTRY_TEXT_BYTES,
    )
        state = state_from_bytes(canonical.canonical_state)
    except ValueError as exc:
        raise ProjectRegistryError(str(exc)) from exc
    operation_id = migration_operation_id(project_id, source_digest)
    receipt = OperationReceipt(
        operation_id=operation_id,
        type="migrate",
        target_ids=(),
        result_ids=tuple(canonical.entries),
        reason="structurally migrated format-1 memory",
        reconciliation_key=source_digest,
        automatic=False,
    )
    return MigrationPlan(
        state=state,
        canonical_state=canonical.canonical_state,
        receipt=receipt,
        source_digest=source_digest,
        source_version=source_version,
        source_contents=copied,
        automatic_files=automatic_files,
        rendered_mirrors=canonical.rendered_mirrors,
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
