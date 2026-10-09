"""Pure format-1 to format-2 project-memory migration.

This module has no registry, command, CLI, import, or project-discovery dependency.
Callers provide an already locked snapshot and an explicit clock value.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass

from zeta.memory.entry_store import (
    MemoryEntry,
    MemoryState,
    OperationReceipt,
    canonical_state_bytes,
    validate_state,
)
from zeta.memory.entry_views import render_all_kinds
from zeta.memory.profiles import memory_profile
from zeta.project_errors import ProjectRegistryError

LEGACY_FILES = ("brief.md", "state.md", "backlog.md", "changelog.md", "decisions.md")


@dataclass(frozen=True, slots=True)
class MigrationPlan:
    """A deterministic migration result and its exact format-1 reversal data."""

    state: MemoryState
    receipt: OperationReceipt
    source_digest: str
    source_version: str | None
    source_contents: dict[str, str]
    rendered_mirrors: dict[str, str]
    migrated_at: str


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


def migrate_format_one(
    *,
    project_id: str,
    contents: Mapping[str, str],
    source_digest: str,
    source_version: str | None,
    migrated_at: str,
) -> MigrationPlan:
    """Build one deterministic, byte-preserving migration plan.

    Idempotence is keyed by the authoritative source digest. The supplied time is
    data, not an implicit clock, so fixture runs are reproducible.
    """

    if set(contents) != set(LEGACY_FILES):
        raise ProjectRegistryError("migration requires the complete format-1 snapshot")
    copied = {name: contents[name] for name in LEGACY_FILES}
    expected_digest = _legacy_digest(copied)
    if source_digest != expected_digest:
        raise ProjectRegistryError("migration source digest does not match its snapshot")
    schema = memory_profile("zeta")
    operation_id = _stable_id("op", project_id, source_digest, "migrate")
    entries: dict[str, MemoryEntry] = {}
    result_ids: list[str] = []
    for name in LEGACY_FILES:
        text = copied[name]
        if not text:
            continue
        kind = name.removesuffix(".md")
        entry_id = _stable_id("m", project_id, source_digest, kind)
        entries[entry_id] = MemoryEntry(
            id=entry_id,
            project_id=project_id,
            kind=kind,
            text=text,
            representation="legacy_document",
            status="active",
            created_at=migrated_at,
            updated_at=migrated_at,
            seen_at=migrated_at,
            expires_at=None,
            valid_from=migrated_at,
            valid_until=None,
            supersedes=(),
            superseded_by=(),
            sources=(),
            automatic=False,
            accepted_at=None,
            accepted_by=None,
            last_operation_id=operation_id,
        )
        result_ids.append(entry_id)
    state = MemoryState(2, project_id, 1, schema, entries)
    validate_state(state)
    canonical_state_bytes(state)
    receipt = OperationReceipt(
        operation_id=operation_id,
        type="migrate",
        target_ids=(),
        result_ids=tuple(result_ids),
        reason="migrated exact format-1 documents",
        reconciliation_key=source_digest,
        automatic=False,
    )
    rendered = {
        f"{kind}.md": body for kind, body in render_all_kinds(state).items()
    }
    if rendered != copied:
        raise ProjectRegistryError("migration mirror comparison failed")
    return MigrationPlan(
        state=state,
        receipt=receipt,
        source_digest=source_digest,
        source_version=source_version,
        source_contents=copied,
        rendered_mirrors=rendered,
        migrated_at=migrated_at,
    )


def reverse_migration(plan: MigrationPlan) -> dict[str, str]:
    """Return the exact format-1 snapshot stored by a migration plan."""

    if not isinstance(plan, MigrationPlan):
        raise ProjectRegistryError("invalid memory migration plan")
    if _legacy_digest(plan.source_contents) != plan.source_digest:
        raise ProjectRegistryError("memory migration reversal data is corrupt")
    return dict(plan.source_contents)


def _legacy_digest(contents: Mapping[str, str]) -> str:
    digest = hashlib.sha256()
    for name in LEGACY_FILES:
        digest.update(name.encode())
        digest.update(b"\0")
        digest.update(contents[name].encode())
        digest.update(b"\0")
    return digest.hexdigest()


__all__ = ["MigrationPlan", "migrate_format_one", "reverse_migration"]
