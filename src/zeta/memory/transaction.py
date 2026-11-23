"""One read owner for project memory."""

from __future__ import annotations

import os
import re
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..project_errors import ProjectRegistryError
from ..project_memory_history import (
    _CURRENT,
    PROJECT_MEMORY_FILES,
    MemoryContextEntry,
    MemorySnapshot,
)
from ..project_schema import MAX_RECORD_SIZE
from ..state_snapshot import BODY_BYTES, clip, timestamp_ns
from .entry_store import EntryMemorySnapshot, MemoryEntry
from .entry_views import active_entries, entry_is_accepted, inspect_value, render_all_kinds

if TYPE_CHECKING:
    from ..project_registry import ProjectRegistry


STATE_MEMORY_KINDS = ("brief", "decision", "state", "backlog", "changelog", "legacy")


def _state_kind(entry: MemoryEntry) -> str:
    if entry.representation == "legacy_document":
        return "legacy"
    if entry.kind == "decisions":
        return "decision"
    return entry.kind if entry.kind in STATE_MEMORY_KINDS else "legacy"


@dataclass(frozen=True, slots=True)
class HealthyMemoryProjection:
    project_id: str
    format: Literal[1, 2]
    generation: str | None
    digest: str
    snapshot: MemorySnapshot | EntryMemorySnapshot
    memory: tuple[MemoryContextEntry, ...]

    def view(self, *, byte_cap: int | None = None) -> dict[str, object] | dict[str, str]:
        if byte_cap is not None and (
            type(byte_cap) is not int or byte_cap < 0 or byte_cap > MAX_RECORD_SIZE
        ):
            raise ProjectRegistryError("invalid memory byte cap")
        if self.format == 1:
            assert isinstance(self.snapshot, MemorySnapshot)
            remaining = byte_cap
            result: dict[str, str] = {}
            for name, content in self.snapshot.contents.items():
                size = len(content.encode())
                if remaining is None or size <= remaining:
                    result[name] = content
                    if remaining is not None:
                        remaining -= size
            return result
        assert isinstance(self.snapshot, EntryMemorySnapshot)
        value = {
            **inspect_value(
                self.snapshot.state,
                byte_cap=None if byte_cap is None else max(1024, byte_cap - 256),
            ),
            "digest": self.digest,
            "version_id": self.generation,
        }
        entries = value["entries"]
        assert isinstance(entries, list)
        while byte_cap is not None and len(str(value).encode()) > byte_cap and entries:
            entries.pop()
            value["entries_truncated"] = True
        return value

    def mirror_files(self) -> dict[str, str]:
        if self.format == 1:
            assert isinstance(self.snapshot, MemorySnapshot)
            return dict(self.snapshot.contents)
        assert isinstance(self.snapshot, EntryMemorySnapshot)
        return render_all_kinds(self.snapshot.state)


class MemoryTransactions:
    """Read project memory while holding p06's memory-domain capability."""

    def __init__(self, registry: ProjectRegistry):
        self._registry = weakref.proxy(registry)

    def read(self, project_id: str) -> HealthyMemoryProjection:
        projection, _ = self._read(project_id, tolerate_malformed_files=False)
        return projection

    def read_for_context(
        self, project_id: str
    ) -> tuple[HealthyMemoryProjection, str | None]:
        """Keep valid legacy files when another optional file is malformed."""

        return self._read(project_id, tolerate_malformed_files=True)

    def state_generation(self, project_id: str) -> object:
        from ..state_snapshot import stamps

        project = self._registry.root / project_id
        return stamps([project, project / "project.json", project / _CURRENT,
                       *(project / "memory" / name for name in PROJECT_MEMORY_FILES)])

    def state_rows(
        self, project_id: str, *, kind: str | None = None, memory_id: str | None = None,
    ) -> list[dict[str, object]] | dict[str, object] | None:
        """Project the current owner snapshot without taking a memory lock."""
        registry = self._registry
        with registry.state_project_fd(project_id) as fd:
            legacy = registry._memory_format_locked(fd, project_id) == 1
            if not legacy:
                snapshot = registry._entry_snapshot_locked(fd)
                entries = (
                    (e.id, _state_kind(e),
                     e.text, e.automatic, entry_is_accepted(e),
                     timestamp_ns(e.created_at), timestamp_ns(e.updated_at),
                     e.sources[0].session_id if e.sources else None)
                    for e in active_entries(snapshot.state)
                )
            else:
                project = registry._decode(registry._read_fd(fd, "project.json"), project_id)
                snapshot = registry._snapshot_locked(fd)
                entries = (
                    (name, "legacy", text, False, True, timestamp_ns(project.created_at),
                     timestamp_ns(project.updated_at), None)
                    for name, text in snapshot.contents.items() if text
                )
            rows = []
            for ident, entry_kind, text, automatic, accepted, created, updated, source in entries:
                if (kind is not None and entry_kind != kind) or (memory_id is not None and ident != memory_id):
                    continue
                row = {
                    "memory_id": ident, "project_id": project_id, "kind": entry_kind,
                    "automatic": automatic, "accepted": accepted,
                    "summary": clip(text.partition("\n")[0], 512)[0],
                    "created_ns": created, "updated_ns": updated,
                    "source_session_id": source,
                    "has_body": bool(text) and not (
                        legacy and re.fullmatch(r"#{1,6}(?:[ \t]+[^\n]*)?", text.strip())
                    ),
                }
                if memory_id is not None:
                    body, truncated = clip(text, BODY_BYTES)
                    return {"memory": row, "body": body, "truncated": truncated}
                rows.append(row)
            return rows if memory_id is None else None

    def _read(
        self, project_id: str, *, tolerate_malformed_files: bool
    ) -> tuple[HealthyMemoryProjection, str | None]:
        registry = self._registry
        with registry._project_lock(project_id, "memory", exclusive=False) as project_fd:
            directory_fd = os.dup(project_fd)
            try:
                fmt = registry._memory_format_locked(project_fd, project_id)
                if fmt == 2:
                    snapshot = registry._entry_snapshot_locked(directory_fd)
                    memory = tuple(
                        MemoryContextEntry(f"{kind}.md", rendered, False)
                        for kind, rendered in render_all_kinds(snapshot.state).items()
                    )
                    return (
                        HealthyMemoryProjection(
                            project_id, 2, snapshot.version, snapshot.digest, snapshot, memory
                        ),
                        None,
                    )
                malformed_source = None
                pointer = registry._pointer(directory_fd)
                if pointer is not None:
                    snapshot = registry._snapshot_locked(directory_fd)
                    contents = snapshot.contents
                else:
                    memory_fd = registry._memory_fd(directory_fd)
                    try:
                        contents = {}
                        for name in PROJECT_MEMORY_FILES:
                            try:
                                content = registry._read_memory_file(memory_fd, name)
                                if content.startswith("<!-- zeta-memory-mirror:v1"):
                                    raise ProjectRegistryError(
                                        "memory mirror is not authoritative"
                                    )
                                contents[name] = content
                            except FileNotFoundError:
                                continue
                            except (OSError, ProjectRegistryError):
                                if not tolerate_malformed_files:
                                    raise
                                malformed_source = malformed_source or f"memory/{name}"
                    finally:
                        os.close(memory_fd)
                    snapshot = MemorySnapshot(
                        contents, registry._memory_digest_value(contents)
                    )
                try:
                    automatic = registry._automatic_files_from_records(
                        registry._records_locked(directory_fd)
                    )
                except (OSError, ProjectRegistryError, KeyError, TypeError, ValueError):
                    automatic = set()
                memory = tuple(
                    MemoryContextEntry(name, content, name in automatic)
                    for name, content in contents.items()
                    if name in PROJECT_MEMORY_FILES
                )
                return (
                    HealthyMemoryProjection(
                        project_id, 1, None, snapshot.digest, snapshot, memory
                    ),
                    malformed_source,
                )
            finally:
                os.close(directory_fd)


__all__ = ["HealthyMemoryProjection", "MemoryTransactions"]
