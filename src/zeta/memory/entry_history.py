"""Private storage adapter for dormant format-2 project memory."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping

from zeta.core.session_files import atomic_publish_file
from zeta.memory.entry_store import (
    EntryCASResult,
    EntryMemorySnapshot,
    MemoryOperation,
    MemorySchema,
    OperationReceipt,
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
from zeta.project_errors import ProjectRegistryError

_CURRENT = "memory-current.json"
_MAX_STATE_BYTES = 8 * 1024 * 1024


class EntryMemoryHistoryMixin:
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
        pointer = self._pointer(directory_fd)
        if pointer is None:
            raise ProjectRegistryError("format-2 memory is not initialized")
        root, blobs_fd, versions_fd = self._version_handles(directory_fd, create=False)
        try:
            version = str(pointer["current"])
            manifest = self._manifest(versions_fd, version)
            if manifest.get("format") != 2:
                raise ProjectRegistryError("project memory is not format 2")
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
    ) -> EntryMemorySnapshot:
        root, blobs_fd, versions_fd = self._version_handles(directory_fd, create=True)
        try:
            pointer = self._pointer(directory_fd)
            old_history = (
                [] if reset_history or pointer is None else list(pointer["history"])
            )
            version = uuid.uuid4().hex
            prospective_history = [*old_history, version][-self._entry_retention_limit():]
            dropped = old_history[: max(0, len(old_history) + 1 - self._entry_retention_limit())]
            retained_operation_ids = {receipt.operation_id for receipt in receipts}
            for retained_version in prospective_history:
                if retained_version == version:
                    continue
                record = self._manifest(versions_fd, retained_version)
                if record.get("format") == 2:
                    retained_operation_ids.update(
                        receipt.operation_id for receipt in self._entry_receipts(record)
                    )
            compacted_through = (
                dropped[-1] if dropped else state.compacted_through_version
            )
            state = compact_inactive_entries(
                state,
                retained_operation_ids=retained_operation_ids,
                compacted_through_version=compacted_through,
            )
            before = compact_inactive_entries(
                before,
                retained_operation_ids=retained_operation_ids,
                compacted_through_version=compacted_through,
            )

            def publish_blob(value: EntryMemoryState) -> str:
                payload = canonical_state_bytes(value)
                digest = hashlib.sha256(payload).hexdigest()
                try:
                    os.stat(digest, dir_fd=blobs_fd, follow_symlinks=False)
                except FileNotFoundError:
                    atomic_publish_file(blobs_fd, digest, payload)
                return digest

            snapshot_digest = publish_blob(state)
            before_digest = publish_blob(before)
            os.fsync(blobs_fd)
            self._memory_transaction_step("snapshot")
            manifest: dict[str, object] = {
                "version": version,
                "format": 2,
                "kind": kind,
                "created_at": utc_now(),
                "snapshot": snapshot_digest,
                "before_snapshot": before_digest,
                "operations": [receipt_to_dict(receipt) for receipt in receipts],
            }
            if reconciliation_key is not None:
                manifest["reconciliation_key"] = reconciliation_key
            if target_version is not None:
                # Use a format-specific name so bounded format-2 history does not
                # inherit format 1's indefinitely followed undo target.
                manifest["entry_target_version"] = target_version
            atomic_publish_file(
                versions_fd,
                f"{version}.json",
                json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
            )
            os.fsync(versions_fd)
            self._memory_transaction_step("manifest")
            atomic_publish_file(
                directory_fd,
                _CURRENT,
                json.dumps(
                    {"current": version, "history": prospective_history},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                sync_directory=True,
            )
            self._memory_transaction_step("publish")
            self._prune_versions(blobs_fd, versions_fd, set(prospective_history))
            return EntryMemorySnapshot(state, snapshot_digest, version)
        finally:
            os.close(versions_fd)
            os.close(blobs_fd)
            os.close(root)

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

    def _entry_memory_state(self, project_id: str) -> EntryMemorySnapshot:
        """Read dormant format-2 state through the private storage seam."""
        def read(root_fd: int) -> EntryMemorySnapshot:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                return self._entry_snapshot_locked(directory_fd)
            finally:
                os.close(directory_fd)

        return self._read(read)

    def _compare_and_swap_entries(
        self,
        project_id: str,
        *,
        expected_digest: str,
        operations: tuple[MemoryOperation, ...],
        reconciliation_key: str | None,
        automatic: bool = True,
        evidence: tuple[str, int, int] | None = None,
    ) -> EntryCASResult:
        """Apply one dormant format-2 transaction."""
        if not re.fullmatch(r"[0-9a-f]{64}", expected_digest):
            raise ProjectRegistryError("invalid memory digest")
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
                state, receipts = apply_operations(
                    current.state,
                    operations,
                    reconciliation_key=reconciliation_key,
                    automatic=automatic,
                    evidence=evidence,
                )
                published = self._publish_entry_version(
                    directory_fd,
                    state=state,
                    before=current.state,
                    kind="entry-update",
                    receipts=receipts,
                    reconciliation_key=reconciliation_key,
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

    def _undo_memory_entry(self, project_id: str, entry_id: str) -> EntryCASResult:
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
                    target: dict[str, object] | None = None
                    for version in reversed(pointer["history"]):
                        record = self._manifest(versions_fd, version)
                        if record.get("format") != 2:
                            continue
                        if any(
                            entry_id in (*receipt.target_ids, *receipt.result_ids)
                            for receipt in self._entry_receipts(record)
                        ):
                            target = record
                            break
                    if target is None:
                        raise ProjectRegistryError(
                            "no retained memory operation touches that entry"
                        )
                    restored, _ = self._entry_blob(
                        blobs_fd, target.get("before_snapshot")
                    )
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
                operation_id = new_operation_id()
                receipt = OperationReceipt(
                    operation_id,
                    "undo",
                    (entry_id,),
                    (entry_id,),
                    "restored retained entry transaction",
                    None,
                    False,
                )
                restored = dataclasses.replace(
                    restored, generation=current.state.generation + 1
                )
                published = self._publish_entry_version(
                    directory_fd,
                    state=restored,
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

