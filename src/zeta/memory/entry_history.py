"""Private storage adapter for dormant format-2 project memory."""

from __future__ import annotations

import dataclasses
import hashlib
import os
import re
import stat
from collections.abc import Mapping

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
    state_from_bytes,
    utc_now,
)
from zeta.memory.entry_store import (
    MemoryState as EntryMemoryState,
)
from zeta.memory.entry_undo import plan_entry_transaction_undo
from zeta.memory.version_store import (
    PreparedVersion,
    PublicationContext,
    publish_version,
    require_memory_format,
)
from zeta.project_errors import ProjectRegistryError

_CURRENT = "memory-current.json"
_MAX_STATE_BYTES = 8 * 1024 * 1024


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

    def prepare(self, context: PublicationContext) -> PreparedVersion[EntryMemoryState]:
        retained_operation_ids = {receipt.operation_id for receipt in self.receipts}
        retained_versions = set(context.retained_history)
        for version, record in zip(
            context.old_history, context.old_manifests, strict=True
        ):
            if version in retained_versions and record.get("format") == 2:
                retained_operation_ids.update(
                    receipt.operation_id
                    for receipt in EntryMemoryHistoryMixin._entry_receipts(record)
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
        return PreparedVersion(
            {"state": canonical_state_bytes(state)},
            {"state": canonical_state_bytes(before)},
            fields,
            state,
            scalar_payload=True,
        )


class EntryMemoryHistoryMixin(EntryMemoryViewMixin):
    """Adapt the format-2 domain module to the existing version protocol."""

    @staticmethod
    def _entry_blob(blobs_fd: int, digest: object) -> tuple[EntryMemoryState, str]:
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
    ) -> EntryMemorySnapshot:
        adapter = _FormatTwoPayloadAdapter(
            state=state,
            before=before,
            kind=kind,
            receipts=receipts,
            reconciliation_key=reconciliation_key,
            target_version=target_version,
            rejected_groups=rejected_groups,
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
        """Apply one dormant format-2 transaction."""
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
