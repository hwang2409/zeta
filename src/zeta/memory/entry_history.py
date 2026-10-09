"""Storage adapter and activation controls for format-2 project memory."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping

from zeta.core.session_files import atomic_publish_file
from zeta.memory.entry_access import EntryMemoryViewMixin
from zeta.memory.entry_store import (
    AddOperation,
    EntryCASResult,
    EntryMemorySnapshot,
    MemoryEntry,
    MemoryOperation,
    MemorySchema,
    OperationReceipt,
    SupersedeOperation,
    accept_entry,
    apply_operations,
    canonical_state_bytes,
    compact_inactive_entries,
    empty_state,
    new_operation_id,
    receipt_from_dict,
    receipt_to_dict,
    state_digest,
    state_from_bytes,
    utc_now,
)
from zeta.memory.entry_store import (
    MemoryState as EntryMemoryState,
)
from zeta.memory.entry_sync import EntryMemoryExport
from zeta.memory.entry_undo import plan_entry_transaction_undo
from zeta.memory.migration import MigrationPlan, migrate_format_one
from zeta.memory.profiles import memory_profile
from zeta.memory.version_store import (
    PreparedVersion,
    PublicationContext,
    publish_version,
    require_memory_format,
)
from zeta.project_errors import ProjectRegistryError

_CURRENT = "memory-current.json"
_MAX_STATE_BYTES = 8 * 1024 * 1024
FORMAT_TWO_CAPABILITIES = frozenset(
    {"updater", "prompt_projection", "commands_api", "sync"}
)


def _format_two_capabilities() -> frozenset[str]:
    from zeta import project_memory_commands
    from zeta.core import project_context
    from zeta.remote_sync import memory as remote_memory

    from . import entry_reconciler

    consumers = {
        "updater": entry_reconciler,
        "prompt_projection": project_context,
        "commands_api": project_memory_commands,
        "sync": remote_memory,
    }
    return frozenset(
        name
        for name, consumer in consumers.items()
        if 2 in consumer.SUPPORTED_MEMORY_FORMATS
    )


def _require_format_two_capabilities(capabilities: frozenset[str]) -> None:
    missing = sorted(FORMAT_TWO_CAPABILITIES - capabilities)
    if missing:
        raise ProjectRegistryError(
            "missing format-2 capabilities: " + ", ".join(missing)
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _FormatTwoPayloadAdapter:
    """Prepare one canonical entry-state payload for the shared publisher."""

    state: EntryMemoryState
    before: EntryMemoryState
    kind: str
    receipts: tuple[OperationReceipt, ...]
    reconciliation_key: str | None
    target_version: str | None
    rejected_groups: tuple[str, ...] = ()
    source_history: tuple[dict[str, object], ...] = ()
    provenance: Mapping[str, object] | None = None
    migration_version: str | None = None

    def prepare(self, context: PublicationContext) -> PreparedVersion[EntryMemoryState]:
        retained_operation_ids = {receipt.operation_id for receipt in self.receipts}
        for record in self.source_history:
            operations = record.get("operations", ())
            if isinstance(operations, list):
                retained_operation_ids.update(
                    receipt_from_dict(value).operation_id for value in operations
                )
        retained_versions = set(context.retained_history)
        for version, record in zip(
            context.old_history, context.old_manifests, strict=True
        ):
            if version in retained_versions and record.get("format") == 2:
                retained_operation_ids.update(
                    receipt.operation_id
                    for receipt in EntryMemoryHistoryMixin._entry_receipts(record)
                )
                imported = record.get("source_history", ())
                if isinstance(imported, list):
                    for source_record in imported:
                        if isinstance(source_record, dict):
                            retained_operation_ids.update(
                                receipt.operation_id
                                for receipt in EntryMemoryHistoryMixin._entry_receipts(
                                    source_record
                                )
                            )
        compacted_through = (
            context.dropped_history[-1]
            if context.dropped_history
            else self.state.compacted_through_version
        )
        state = compact_inactive_entries(
            self.state,
            retained_operation_ids=retained_operation_ids,
            compacted_through_version=compacted_through,
        )
        before = compact_inactive_entries(
            self.before,
            retained_operation_ids=retained_operation_ids,
            compacted_through_version=compacted_through,
        )
        fields: dict[str, object] = {
            "format": 2,
            "kind": self.kind,
            "created_at": utc_now(),
            "operations": [receipt_to_dict(receipt) for receipt in self.receipts],
        }
        if self.reconciliation_key is not None:
            fields["reconciliation_key"] = self.reconciliation_key
        if self.target_version is not None:
            fields["entry_target_version"] = self.target_version
        if self.rejected_groups:
            fields["rejected_groups"] = list(self.rejected_groups)
        if self.source_history:
            fields["source_history"] = [dict(item) for item in self.source_history]
        if self.provenance is not None:
            fields["provenance"] = dict(self.provenance)
        if self.migration_version is not None:
            fields["migration_version"] = self.migration_version
        return PreparedVersion(
            {"state": canonical_state_bytes(state)},
            {"state": canonical_state_bytes(before)},
            fields,
            state,
            scalar_payload=True,
        )


@dataclasses.dataclass(frozen=True, slots=True)
class _MigrationPayloadAdapter:
    plan: MigrationPlan

    def prepare(self, context: PublicationContext) -> PreparedVersion[EntryMemoryState]:
        return PreparedVersion(
            {"state": canonical_state_bytes(self.plan.state)},
            {name: content.encode() for name, content in self.plan.source_contents.items()},
            {
                "format": 2,
                "kind": "migrate",
                "created_at": self.plan.migrated_at,
                "operations": [receipt_to_dict(self.plan.receipt)],
                "source_digest": self.plan.source_digest,
                "source_version": self.plan.source_version,
                "source_automatic_files": sorted(self.plan.automatic_files),
                "target_version": self.plan.source_version,
                "migration_version": context.version,
            },
            self.plan.state,
            scalar_payload=False,
        )


class EntryMemoryHistoryMixin(EntryMemoryViewMixin):
    """Adapt the format-2 domain module to the existing version protocol."""

    @staticmethod
    def _ensure_entry_memory_capabilities() -> None:
        _require_format_two_capabilities(_format_two_capabilities())

    @staticmethod
    def _entry_blob(blobs_fd: int, digest: object) -> tuple[EntryMemoryState, str]:
        if isinstance(digest, dict) and set(digest) == {"state"}:
            digest = digest["state"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ProjectRegistryError("format-2 memory version is malformed")
        try:
            fd = os.open(digest, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=blobs_fd)
        except OSError as exc:
            raise ProjectRegistryError("format-2 memory snapshot is incomplete") from exc
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_STATE_BYTES:
                raise ProjectRegistryError("format-2 memory snapshot is unsafe")
            payload = os.read(fd, info.st_size + 1)
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ProjectRegistryError("format-2 memory snapshot is corrupt")
            return state_from_bytes(payload), digest
        except OSError as exc:
            raise ProjectRegistryError("format-2 memory snapshot is unreadable") from exc
        finally:
            os.close(fd)

    def _entry_snapshot_locked(self, directory_fd: int) -> EntryMemorySnapshot:
        require_memory_format(
            directory_fd,
            2,
            pointer_reader=self._pointer,
            version_handles=lambda fd, create: self._version_handles(fd, create=create),
            manifest_reader=self._manifest,
        )
        pointer = self._pointer(directory_fd)
        if pointer is None:
            raise ProjectRegistryError("format-2 memory is not initialized")
        root, blobs_fd, versions_fd = self._version_handles(directory_fd, create=False)
        try:
            version = str(pointer["current"])
            manifest = self._manifest(versions_fd, version)
            state, digest = self._entry_blob(blobs_fd, manifest.get("snapshot"))
            return EntryMemorySnapshot(state, digest, version)
        finally:
            os.close(versions_fd)
            os.close(blobs_fd)
            os.close(root)

    @staticmethod
    def _entry_receipts(record: Mapping[str, object]) -> tuple[OperationReceipt, ...]:
        values = record.get("operations", [])
        if not isinstance(values, list):
            raise ProjectRegistryError("format-2 memory receipts are malformed")
        return tuple(receipt_from_dict(value) for value in values)

    def _publish_entry_version(
        self,
        directory_fd: int,
        *,
        state: EntryMemoryState,
        before: EntryMemoryState,
        kind: str,
        receipts: tuple[OperationReceipt, ...],
        reconciliation_key: str | None = None,
        target_version: str | None = None,
        reset_history: bool = False,
        rejected_groups: tuple[str, ...] = (),
        source_history: tuple[dict[str, object], ...] = (),
        provenance: Mapping[str, object] | None = None,
    ) -> EntryMemorySnapshot:
        migration_version = None
        if kind != "migration-finalize":
            pointer = self._pointer(directory_fd)
            if pointer is not None:
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    current_version = str(pointer["current"])
                    current_manifest = self._manifest(versions_fd, current_version)
                    candidate = current_manifest.get("migration_version")
                    if current_manifest.get("kind") == "migrate":
                        migration_version = current_version
                    elif isinstance(candidate, str):
                        migration_version = candidate
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
        adapter = _FormatTwoPayloadAdapter(
            state=state,
            before=before,
            kind=kind,
            receipts=receipts,
            reconciliation_key=reconciliation_key,
            target_version=target_version,
            rejected_groups=rejected_groups,
            source_history=source_history,
            provenance=provenance,
            migration_version=migration_version,
        )
        published = publish_version(
            directory_fd,
            adapter=adapter,
            retention_limit=self._entry_retention_limit(),
            reset_history=reset_history,
            pointer_reader=self._pointer,
            version_handles=lambda fd, create: self._version_handles(fd, create=create),
            manifest_reader=self._manifest,
            transaction_step=self._memory_transaction_step,
            prune_versions=self._prune_versions,
        )
        if not isinstance(published.snapshot, str):
            raise TypeError("format-2 publication returned a non-scalar snapshot")
        snapshot = EntryMemorySnapshot(
            published.value, published.snapshot, published.version
        )
        self._refresh_entry_memory_mirror(
            directory_fd,
            snapshot.state,
            mirror_path=self.root / snapshot.state.project_id / "memory",
        )
        return snapshot

    @staticmethod
    def _logical_entry_history(
        records: list[dict[str, object]],
    ) -> tuple[dict[str, object], ...]:
        fields = (
            "version",
            "kind",
            "created_at",
            "operations",
            "reconciliation_key",
            "source_digest",
            "provenance",
        )
        logical: list[dict[str, object]] = []
        seen: set[str] = set()
        for record in records:
            imported = record.get("source_history")
            candidates = [*imported, record] if isinstance(imported, list) else [record]
            for candidate in candidates:
                if not isinstance(candidate, dict):
                    continue
                version = candidate.get("version")
                if isinstance(version, str) and version in seen:
                    continue
                summary = {key: candidate[key] for key in fields if key in candidate}
                if isinstance(version, str):
                    seen.add(version)
                logical.append(summary)
        return tuple(logical)

    def _export_entry_memory(self, project_id: str) -> EntryMemoryExport:
        """Export one format-2 fixture without exposing storage layout."""
        def read(root_fd: int) -> EntryMemoryExport:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                snapshot = self._entry_snapshot_locked(directory_fd)
                pointer = self._pointer(directory_fd)
                assert pointer is not None
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    records = [
                        self._manifest(versions_fd, version)
                        for version in pointer["history"]
                    ]
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
                return EntryMemoryExport(
                    snapshot.state,
                    snapshot.digest,
                    snapshot.version,
                    self._logical_entry_history(records),
                )
            finally:
                os.close(directory_fd)

        return self._read(read)

    def _import_entry_memory(
        self,
        project_id: str,
        exported: EntryMemoryExport,
        *,
        expected_digest: str,
        changed_entry_ids: tuple[str, ...],
        provenance: Mapping[str, object],
    ) -> EntryMemorySnapshot:
        """CAS-import one validated format-2 snapshot."""
        if (
            not isinstance(exported, EntryMemoryExport)
            or exported.state.project_id != project_id
            or exported.digest != state_digest(exported.state)
            or not re.fullmatch(r"[0-9a-f]{64}", expected_digest)
        ):
            raise ProjectRegistryError("invalid format-2 memory export")
        try:
            encoded_history = json.dumps(exported.versions, sort_keys=True).encode()
        except (TypeError, ValueError) as exc:
            raise ProjectRegistryError("invalid format-2 memory export") from exc
        if len(encoded_history) > 512 * 1024 or any(
            not isinstance(record, dict) for record in exported.versions
        ):
            raise ProjectRegistryError("format-2 memory export history is too large")
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                current = self._entry_snapshot_locked(directory_fd)
                if current.digest != expected_digest:
                    raise ProjectRegistryError("project memory digest mismatch")
                actual_changed = tuple(
                    sorted(
                        entry_id
                        for entry_id in (
                            current.state.entries.keys() | exported.state.entries.keys()
                        )
                        if current.state.entries.get(entry_id)
                        != exported.state.entries.get(entry_id)
                    )
                )
                if actual_changed != tuple(sorted(set(changed_entry_ids))):
                    raise ProjectRegistryError(
                        "format-2 memory import changed-entry receipt is invalid"
                    )
                if current.digest == exported.digest:
                    self._refresh_entry_memory_mirror(
                        directory_fd,
                        current.state,
                        mirror_path=self.root / project_id / "memory",
                    )
                    return current
                receipt = OperationReceipt(
                    operation_id=new_operation_id(),
                    type="sync",
                    target_ids=tuple(
                        entry_id
                        for entry_id in actual_changed
                        if entry_id in current.state.entries
                    ),
                    result_ids=tuple(
                        entry_id
                        for entry_id in actual_changed
                        if entry_id in exported.state.entries
                    ),
                    reason="synchronized memory entries",
                    reconciliation_key=None,
                    automatic=False,
                )
                return self._publish_entry_version(
                    directory_fd,
                    state=exported.state,
                    before=current.state,
                    kind="entry-import",
                    receipts=(receipt,),
                    source_history=exported.versions,
                    provenance=provenance,
                )
            finally:
                os.close(directory_fd)

    def _activate_entry_memory_locked(
        self, directory_fd: int, project_id: str, profile: str
    ) -> EntryMemorySnapshot:
        try:
            schema = memory_profile(profile)
        except ValueError as exc:
            raise ProjectRegistryError(str(exc)) from exc
        if self._pointer(directory_fd) is not None:
            raise ProjectRegistryError(
                "existing project memory must be activated with migrate"
            )
        expected = {
            "brief.md": "# Brief\n",
            "state.md": "# Current state\n",
            "backlog.md": "# Backlog\n",
            "changelog.md": "# Changelog\n",
            "decisions.md": "# Decisions\n",
        }
        if self._legacy_contents(directory_fd) != expected:
            raise ProjectRegistryError(
                "existing project memory must be activated with migrate"
            )
        state = empty_state(project_id, schema)
        return self._publish_entry_version(
            directory_fd,
            state=state,
            before=state,
            kind="entry-activate",
            receipts=(),
            reset_history=True,
        )

    def activate_entry_memory(
        self,
        project_id: str,
        profile: str = "zeta",
    ) -> EntryMemorySnapshot:
        """Activate an empty real project when every format-2 consumer is ready."""
        self._ensure_entry_memory_capabilities()
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                return self._activate_entry_memory_locked(
                    directory_fd, project_id, profile
                )
            finally:
                os.close(directory_fd)

    def set_memory_profile(
        self,
        project_id: str,
        profile: str,
        *,
        kind_mappings: Mapping[str, str] | None = None,
        resolve_kinds: frozenset[str] = frozenset(),
    ) -> EntryMemorySnapshot:
        """Replace the copied schema and explicitly handle removed populated kinds."""
        try:
            selected = memory_profile(profile)
        except ValueError as exc:
            raise ProjectRegistryError(str(exc)) from exc
        mappings = dict(kind_mappings or {})
        if set(mappings) & set(resolve_kinds):
            raise ProjectRegistryError("a memory kind cannot be mapped and resolved")
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                current = self._entry_snapshot_locked(directory_fd)
                target_keys = {kind.key for kind in selected.kinds}
                source_keys = {kind.key for kind in current.state.schema.kinds}
                if set(mappings) - source_keys or set(resolve_kinds) - source_keys:
                    raise ProjectRegistryError("profile change names an unknown source kind")
                if any(target not in target_keys for target in mappings.values()):
                    raise ProjectRegistryError("profile mapping targets an unknown kind")
                removed = source_keys - target_keys
                if (set(mappings) | set(resolve_kinds)) - removed:
                    raise ProjectRegistryError("profile mapping applies only to removed kinds")
                populated = {
                    entry.kind
                    for entry in current.state.entries.values()
                    if isinstance(entry, MemoryEntry) and entry.kind in removed
                }
                missing = sorted(populated - set(mappings) - set(resolve_kinds))
                if missing:
                    raise ProjectRegistryError(
                        "profile change requires --map-kind or --resolve-kind for: "
                        + ", ".join(missing)
                    )
                operation_id = new_operation_id()
                entries = {}
                touched: list[str] = []
                for entry_id, entry in current.state.entries.items():
                    if not isinstance(entry, MemoryEntry) or entry.kind not in removed:
                        entries[entry_id] = entry
                    elif entry.kind in mappings:
                        entries[entry_id] = dataclasses.replace(
                            entry,
                            kind=mappings[entry.kind],
                            last_operation_id=operation_id,
                        )
                        touched.append(entry_id)
                    elif entry.kind in resolve_kinds:
                        touched.append(entry_id)
                schema = dataclasses.replace(
                    selected, version=current.state.schema.version + 1
                )
                state = dataclasses.replace(
                    current.state,
                    generation=current.state.generation + 1,
                    schema=schema,
                    entries=entries,
                )
                canonical_state_bytes(state)
                receipt = OperationReceipt(
                    operation_id,
                    "sync",
                    tuple(touched),
                    tuple(entry_id for entry_id in touched if entry_id in entries),
                    f"set memory profile to {profile}",
                    None,
                    False,
                )
                return self._publish_entry_version(
                    directory_fd,
                    state=state,
                    before=current.state,
                    kind="schema-profile",
                    receipts=(receipt,),
                )
            finally:
                os.close(directory_fd)

    def migrate_memory(
        self,
        project_id: str,
        *,
        migrated_at: str | None = None,
    ) -> MigrationPlan:
        self._ensure_entry_memory_capabilities()
        return self._migrate_memory_for_test(
            project_id, migrated_at=migrated_at or utc_now()
        )

    def rollback_memory_migration(self, project_id: str) -> None:
        """Restore the protected format-1 migration target."""
        self._rollback_memory_migration_for_test(project_id)

    def finalize_memory_migration(self, project_id: str) -> EntryMemorySnapshot:
        """End special rollback protection while retaining normal history."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                current = self._entry_snapshot_locked(directory_fd)
                pointer = self._pointer(directory_fd)
                assert pointer is not None
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    records = [
                        self._manifest(versions_fd, version)
                        for version in pointer["history"]
                    ]
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
                if any(record.get("kind") == "migration-finalize" for record in records):
                    raise ProjectRegistryError("memory migration is already finalized")
                current_manifest = records[-1]
                if not any(record.get("kind") == "migrate" for record in records) and not isinstance(
                    current_manifest.get("migration_version"), str
                ):
                    raise ProjectRegistryError("memory migration is not initialized")
                return self._publish_entry_version(
                    directory_fd,
                    state=current.state,
                    before=current.state,
                    kind="migration-finalize",
                    receipts=(),
                )
            finally:
                os.close(directory_fd)

    def _migrate_memory_for_test(
        self, project_id: str, *, migrated_at: str
    ) -> MigrationPlan:
        """Run the dormant migration only through an explicit fixture seam."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                pointer = self._pointer(directory_fd)
                if pointer is not None:
                    root, blobs_fd, versions_fd = self._version_handles(
                        directory_fd, create=False
                    )
                    try:
                        current_manifest = self._manifest(
                            versions_fd, str(pointer["current"])
                        )
                        if current_manifest.get("format", 1) == 2:
                            if current_manifest.get("kind") != "migrate":
                                raise ProjectRegistryError(
                                    "format-2 memory is not a migration fixture"
                                )
                            source = self._contents_from_manifest(
                                blobs_fd, current_manifest, key="before_snapshot"
                            )
                            return migrate_format_one(
                                project_id=project_id,
                                contents=source,
                                source_digest=str(current_manifest["source_digest"]),
                                source_version=(
                                    str(current_manifest["source_version"])
                                    if current_manifest.get("source_version") is not None
                                    else None
                                ),
                                migrated_at=str(current_manifest["created_at"]),
                                automatic_files=frozenset(
                                    current_manifest.get("source_automatic_files", ())
                                ),
                            )
                    finally:
                        os.close(versions_fd)
                        os.close(blobs_fd)
                        os.close(root)
                before = self._snapshot_locked(directory_fd)
                source_version = None if pointer is None else str(pointer["current"])
                automatic_files = self._automatic_files_from_records(
                    self._records_locked(directory_fd)
                )
                plan = migrate_format_one(
                    project_id=project_id,
                    contents=before.contents,
                    source_digest=before.digest,
                    source_version=source_version,
                    migrated_at=migrated_at,
                    automatic_files=frozenset(automatic_files),
                )
                published = publish_version(
                    directory_fd,
                    adapter=_MigrationPayloadAdapter(plan),
                    retention_limit=self._entry_retention_limit(),
                    reset_history=False,
                    pointer_reader=self._pointer,
                    version_handles=lambda fd, create: self._version_handles(
                        fd, create=create
                    ),
                    manifest_reader=self._manifest,
                    transaction_step=self._memory_transaction_step,
                    prune_versions=self._prune_versions,
                )
                if published.value != plan.state:
                    raise AssertionError("migration publication changed its state")
                self._refresh_entry_memory_mirror(
                    directory_fd,
                    plan.state,
                    mirror_path=self.root / project_id / "memory",
                )
                return plan
            finally:
                os.close(directory_fd)

    def _rollback_memory_migration_for_test(self, project_id: str) -> None:
        """Restore the protected format-1 pointer for a fixture migration."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                pointer = self._pointer(directory_fd)
                if pointer is None:
                    raise ProjectRegistryError("memory migration is not initialized")
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    migration = None
                    for version in reversed(pointer["history"]):
                        candidate = self._manifest(versions_fd, str(version))
                        if candidate.get("kind") == "migration-finalize":
                            raise ProjectRegistryError("memory migration is finalized")
                        if candidate.get("kind") == "migrate":
                            migration = candidate
                            break
                    if migration is None:
                        current = self._manifest(versions_fd, str(pointer["current"]))
                        migration_version = current.get("migration_version")
                        if isinstance(migration_version, str):
                            migration = self._manifest(versions_fd, migration_version)
                    if migration is None:
                        raise ProjectRegistryError("memory is not a reversible migration")
                    source_version = migration.get("source_version")
                    if source_version is None:
                        source_contents = self._contents_from_manifest(
                            blobs_fd, migration, key="before_snapshot"
                        )
                        restored: list[str] = []
                    elif isinstance(source_version, str):
                        source_manifest = self._manifest(versions_fd, source_version)
                        source_contents = self._contents_from_manifest(
                            blobs_fd, source_manifest
                        )
                        history = list(pointer["history"])
                        restored = (
                            history[: history.index(source_version) + 1]
                            if source_version in history
                            else [source_version]
                        )
                    else:
                        raise ProjectRegistryError("memory is not a reversible migration")
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
                if source_version is None:
                    os.unlink(_CURRENT, dir_fd=directory_fd)
                    os.fsync(directory_fd)
                else:
                    atomic_publish_file(
                        directory_fd,
                        _CURRENT,
                        json.dumps(
                            {"current": source_version, "history": restored},
                            sort_keys=True,
                        ).encode(),
                        sync_directory=True,
                    )
                if source_version is None:
                    memory_fd = self._memory_fd(directory_fd)
                    try:
                        for name, content in source_contents.items():
                            self._publish_mirror_file(
                                memory_fd, name, content.encode("utf-8")
                            )
                        os.fsync(memory_fd)
                    finally:
                        os.close(memory_fd)
                else:
                    self._refresh_memory_mirror(
                        directory_fd,
                        source_contents,
                        mirror_path=self.root / project_id / "memory",
                    )
            finally:
                os.close(directory_fd)

    def _replace_entry_state_for_test(
        self,
        project_id: str,
        state: EntryMemoryState,
        *,
        expected_digest: str,
    ) -> EntryMemorySnapshot:
        """Publish a complete dormant fixture state through CAS."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                current = self._entry_snapshot_locked(directory_fd)
                if current.digest != expected_digest:
                    raise ProjectRegistryError("project memory digest mismatch")
                return self._publish_entry_version(
                    directory_fd,
                    state=state,
                    before=current.state,
                    kind="entry-fixture-replace",
                    receipts=(),
                )
            finally:
                os.close(directory_fd)

    def _create_entry_memory_for_test(
        self, project_id: str, schema: MemorySchema
    ) -> EntryMemorySnapshot:
        """Create dormant format-2 state for isolated tests only."""
        state = empty_state(project_id, schema)
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                return self._publish_entry_version(
                    directory_fd,
                    state=state,
                    before=state,
                    kind="entry-fixture",
                    receipts=(),
                    reset_history=True,
                )
            finally:
                os.close(directory_fd)

    def _compare_and_swap_entries(
        self,
        project_id: str,
        *,
        expected_digest: str,
        operations: tuple[MemoryOperation, ...],
        reconciliation_key: str | None,
        automatic: bool = True,
        evidence: tuple[str, int, int] | None = None,
        rejected_groups: tuple[str, ...] = (),
        now: str | None = None,
    ) -> EntryCASResult:
        """Apply one format-2 transaction."""
        if not isinstance(expected_digest, str) or not re.fullmatch(
            r"[0-9a-f]{64}", expected_digest
        ):
            raise ProjectRegistryError("invalid memory digest")
        if len(rejected_groups) > 64 or any(
            not isinstance(reason, str)
            or not reason
            or len(reason.encode("utf-8")) > 512
            for reason in rejected_groups
        ):
            raise ProjectRegistryError("invalid rejected memory operation groups")
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                pointer = self._pointer(directory_fd)
                if pointer is None:
                    raise ProjectRegistryError("format-2 memory is not initialized")
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    records = [
                        self._manifest(versions_fd, version)
                        for version in pointer["history"]
                    ]
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
                if reconciliation_key is not None:
                    duplicate = next(
                        (
                            record
                            for record in reversed(records)
                            if record.get("format") == 2
                            and record.get("reconciliation_key") == reconciliation_key
                        ),
                        None,
                    )
                    if duplicate is not None:
                        current = self._entry_snapshot_locked(directory_fd)
                        return EntryCASResult(
                            current.state,
                            current.digest,
                            str(duplicate["version"]),
                            False,
                            self._entry_receipts(duplicate),
                        )
                current = self._entry_snapshot_locked(directory_fd)
                if current.digest != expected_digest:
                    raise ProjectRegistryError("project memory digest mismatch")
                if operations:
                    state, receipts = apply_operations(
                        current.state,
                        operations,
                        reconciliation_key=reconciliation_key,
                        automatic=automatic,
                        evidence=evidence,
                        now=now,
                    )
                elif rejected_groups:
                    state, receipts = current.state, ()
                else:
                    raise ProjectRegistryError("empty format-2 memory transaction")
                published = self._publish_entry_version(
                    directory_fd,
                    state=state,
                    before=current.state,
                    kind="entry-update",
                    receipts=receipts,
                    reconciliation_key=reconciliation_key,
                    rejected_groups=rejected_groups,
                )
                return EntryCASResult(
                    published.state,
                    published.digest,
                    published.version,
                    True,
                    receipts,
                )
            finally:
                os.close(directory_fd)

    def _accept_memory_entry(self, project_id: str, entry_id: str) -> EntryCASResult:
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                current = self._entry_snapshot_locked(directory_fd)
                state, receipt = accept_entry(current.state, entry_id)
                published = self._publish_entry_version(
                    directory_fd,
                    state=state,
                    before=current.state,
                    kind="entry-accept",
                    receipts=(receipt,),
                )
                return EntryCASResult(
                    published.state,
                    published.digest,
                    published.version,
                    True,
                    (receipt,),
                )
            finally:
                os.close(directory_fd)

    def _replace_entry_kind(
        self, project_id: str, kind: str, content: str
    ) -> EntryCASResult:
        """Replace one kind with an accepted legacy-document entry."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                current = self._entry_snapshot_locked(directory_fd)
                active_ids = tuple(
                    entry.id
                    for entry in current.state.entries.values()
                    if isinstance(entry, MemoryEntry)
                    and entry.kind == kind
                    and entry.status == "active"
                )
                operation: MemoryOperation = (
                    SupersedeOperation(active_ids, kind, content, ())
                    if active_ids
                    else AddOperation(kind, content, ())
                )
                state, receipts = apply_operations(
                    current.state,
                    (operation,),
                    reconciliation_key=None,
                    automatic=False,
                )
                created_id = receipts[0].result_ids[0]
                created = state.entries[created_id]
                assert isinstance(created, MemoryEntry)
                state = dataclasses.replace(
                    state,
                    entries={
                        **state.entries,
                        created_id: dataclasses.replace(
                            created, representation="legacy_document"
                        ),
                    },
                )
                published = self._publish_entry_version(
                    directory_fd,
                    state=state,
                    before=current.state,
                    kind="entry-set-kind",
                    receipts=receipts,
                )
                return EntryCASResult(
                    published.state,
                    published.digest,
                    published.version,
                    True,
                    receipts,
                )
            finally:
                os.close(directory_fd)

    def _undo_memory_entry(self, project_id: str, entry_id: str) -> EntryCASResult:
        return self._undo_entry_transaction(project_id, entry_id)

    def _undo_entry_transaction(
        self, project_id: str, target_id: str | None = None
    ) -> EntryCASResult:
        """Undo the latest, entry-matching, or exact retained transaction."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                current = self._entry_snapshot_locked(directory_fd)
                pointer = self._pointer(directory_fd)
                assert pointer is not None
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    history = list(pointer["history"])
                    target: dict[str, object] | None = None
                    target_index = -1
                    for index in range(len(history) - 1, -1, -1):
                        version = history[index]
                        record = self._manifest(versions_fd, version)
                        receipts = (
                            self._entry_receipts(record)
                            if record.get("format") == 2
                            else ()
                        )
                        matches = target_id is None and bool(receipts)
                        if target_id == version:
                            matches = bool(receipts)
                        elif target_id is not None and target_id.startswith("m_"):
                            matches = any(
                                target_id in (*item.target_ids, *item.result_ids)
                                for item in receipts
                            )
                        if matches:
                            target = record
                            target_index = index
                            break
                    if target is None:
                        raise ProjectRegistryError(
                            "no retained memory transaction matches that target"
                        )
                    before, _ = self._entry_blob(
                        blobs_fd, target.get("before_snapshot")
                    )
                    after, _ = self._entry_blob(blobs_fd, target.get("snapshot"))
                    later_receipts = tuple(
                        receipt
                        for version in history[target_index + 1 :]
                        for record in (self._manifest(versions_fd, version),)
                        if record.get("format") == 2
                        for receipt in self._entry_receipts(record)
                    )
                    target_receipts = self._entry_receipts(target)
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
                plan = plan_entry_transaction_undo(
                    current=current.state,
                    before=before,
                    after=after,
                    target_receipts=target_receipts,
                    later_receipts=later_receipts,
                )
                ids = (
                    (target_id,)
                    if target_id is not None and target_id.startswith("m_")
                    else plan.target_ids
                )
                receipt = OperationReceipt(
                    new_operation_id(),
                    "undo",
                    ids,
                    ids,
                    "restored retained entry transaction",
                    None,
                    False,
                )
                published = self._publish_entry_version(
                    directory_fd,
                    state=plan.state,
                    before=current.state,
                    kind="entry-undo",
                    receipts=(receipt,),
                    target_version=str(target["version"]),
                )
                return EntryCASResult(
                    published.state,
                    published.digest,
                    published.version,
                    True,
                    (receipt,),
                )
            finally:
                os.close(directory_fd)
