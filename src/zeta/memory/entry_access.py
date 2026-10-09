"""Read models and derived mirrors for format-2 project memory."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from zeta.memory.entry_store import EntryMemorySnapshot, receipt_to_dict
from zeta.memory.entry_store import MemoryState as EntryMemoryState
from zeta.memory.entry_views import (
    GENERATED_MIRROR_HEADER,
    inspect_value,
    receipt_touches,
    render_all_kinds,
)
from zeta.project_errors import ProjectRegistryError

_LOG = logging.getLogger(__name__)


def _json_size(value: object) -> int:
    return len(json.dumps(value, separators=(",", ":"), ensure_ascii=False).encode())


class EntryMemoryViewMixin:
    def _refresh_entry_memory_mirror(
        self,
        directory_fd: int,
        state: EntryMemoryState,
        *,
        mirror_path: Path,
    ) -> None:
        """Best-effort publication of per-kind derived Markdown files."""
        try:
            rendered = render_all_kinds(state)
            memory_fd = self._memory_fd(directory_fd)
            try:
                expected_names = {f"{key}.md" for key in rendered}
                changed = False
                for key, content in rendered.items():
                    name = f"{key}.md"
                    payload = (GENERATED_MIRROR_HEADER + content).encode("utf-8")
                    if self._mirror_file_matches(memory_fd, name, payload):
                        continue
                    self._publish_mirror_file(memory_fd, name, payload)
                    changed = True
                for name in os.listdir(memory_fd):
                    if name.endswith(".md") and name not in expected_names:
                        os.unlink(name, dir_fd=memory_fd)
                        changed = True
                if changed:
                    os.fsync(memory_fd)
            finally:
                os.close(memory_fd)
        except (OSError, ProjectRegistryError, ValueError) as exc:
            _LOG.warning(
                "could not refresh project memory mirror at %s: %s", mirror_path, exc
            )

    def _entry_memory_state_for_context(self, project_id: str) -> EntryMemorySnapshot:
        """Load format-2 context and best-effort repair its derived mirrors."""
        with self._locked(write=True) as root_fd:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                snapshot = self._entry_snapshot_locked(directory_fd)
                self._refresh_entry_memory_mirror(
                    directory_fd,
                    snapshot.state,
                    mirror_path=self.root / project_id / "memory",
                )
                return snapshot
            finally:
                os.close(directory_fd)

    def _entry_memory_state(self, project_id: str) -> EntryMemorySnapshot:
        """Read format-2 state through the storage seam."""
        def read(root_fd: int) -> EntryMemorySnapshot:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                return self._entry_snapshot_locked(directory_fd)
            finally:
                os.close(directory_fd)

        return self._read(read)

    def memory_format(self, project_id: str) -> int:
        """Return the authoritative format without activating either format."""
        def read(root_fd: int) -> int:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                pointer = self._pointer(directory_fd)
                if pointer is None:
                    return 1
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    record = self._manifest(versions_fd, str(pointer["current"]))
                    return int(record.get("format", 1))
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
            finally:
                os.close(directory_fd)

        value = self._read(read)
        if value not in {1, 2}:
            raise ProjectRegistryError("unsupported project memory format")
        return value

    def _entry_memory_view(
        self, project_id: str, *, byte_cap: int | None = None
    ) -> dict[str, object]:
        snapshot = self._entry_memory_state(project_id)
        value = {
            **inspect_value(
                snapshot.state,
                byte_cap=None if byte_cap is None else max(1024, byte_cap - 256),
            ),
            "digest": snapshot.digest,
            "version_id": snapshot.version,
        }
        entries = value["entries"]
        assert isinstance(entries, list)
        while byte_cap is not None and _json_size(value) > byte_cap and entries:
            entries.pop()
            value["entries_truncated"] = True
        if byte_cap is not None and _json_size(value) > byte_cap:
            raise ProjectRegistryError("memory view metadata exceeds byte cap")
        return value

    def _entry_memory_mirrors(self, project_id: str) -> dict[str, str]:
        snapshot = self._entry_memory_state_for_context(project_id)
        return render_all_kinds(snapshot.state)

    def _entry_memory_log(
        self,
        project_id: str,
        *,
        entry_id: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """Read retained format-2 operation receipts, oldest to newest."""
        if type(limit) is not int or not 1 <= limit <= 10_000:
            raise ProjectRegistryError("invalid memory log limit")

        def read(root_fd: int) -> list[dict[str, object]]:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                current = self._entry_snapshot_locked(directory_fd)
                if kind is not None and kind not in {
                    item.key for item in current.state.schema.kinds
                }:
                    raise ProjectRegistryError(f"unknown memory kind: {kind}")
                pointer = self._pointer(directory_fd)
                assert pointer is not None
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    records: list[dict[str, object]] = []
                    for version in pointer["history"]:
                        record = self._manifest(versions_fd, version)
                        if record.get("format") != 2:
                            continue
                        receipts = self._entry_receipts(record)
                        operations = [receipt_to_dict(item) for item in receipts]
                        if kind is not None:
                            after, _ = self._entry_blob(
                                blobs_fd, record.get("snapshot")
                            )
                            before, _ = self._entry_blob(
                                blobs_fd, record.get("before_snapshot")
                            )
                            kind_ids = {
                                entry.id
                                for state in (before, after)
                                for entry in state.entries.values()
                                if getattr(entry, "kind", None) == kind
                            }
                            operations = [
                                item
                                for item in operations
                                if kind_ids.intersection(
                                    (*item["target_ids"], *item["result_ids"])
                                )
                            ]
                        if entry_id is not None:
                            operations = [
                                item
                                for item in operations
                                if entry_id
                                in (
                                    *item["target_ids"],
                                    *item["result_ids"],
                                )
                            ]
                        value = {
                            "version": str(record["version"]),
                            "created_at": record.get("created_at"),
                            "kind": record.get("kind"),
                            "operations": operations,
                        }
                        if not operations:
                            continue
                        if entry_id is None or receipt_touches(value, entry_id):
                            records.append(value)
                    return records[-limit:]
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
            finally:
                os.close(directory_fd)

        return self._read(read)

    def _entry_memory_version(
        self, project_id: str, version: str
    ) -> tuple[dict[str, object], EntryMemoryState, EntryMemoryState]:
        """Read one retained receipt and its structured before/after states."""
        def read(root_fd: int) -> tuple[dict[str, object], EntryMemoryState, EntryMemoryState]:
            directory_fd = self._project_dir(root_fd, project_id)
            try:
                self._entry_snapshot_locked(directory_fd)
                pointer = self._pointer(directory_fd)
                assert pointer is not None
                if version not in pointer["history"]:
                    raise ProjectRegistryError(f"memory version not found: {version}")
                root, blobs_fd, versions_fd = self._version_handles(
                    directory_fd, create=False
                )
                try:
                    record = self._manifest(versions_fd, version)
                    if record.get("format") != 2:
                        raise ProjectRegistryError("memory version is not format 2")
                    after, _ = self._entry_blob(blobs_fd, record.get("snapshot"))
                    before, _ = self._entry_blob(
                        blobs_fd, record.get("before_snapshot")
                    )
                    public = {
                        "version": version,
                        "created_at": record.get("created_at"),
                        "kind": record.get("kind"),
                        "operations": [
                            receipt_to_dict(item)
                            for item in self._entry_receipts(record)
                        ],
                    }
                    return public, before, after
                finally:
                    os.close(versions_fd)
                    os.close(blobs_fd)
                    os.close(root)
            finally:
                os.close(directory_fd)

        return self._read(read)
