"""One read owner for project memory."""

from __future__ import annotations

import os
import weakref
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..project_errors import ProjectRegistryError
from ..project_memory_history import (
    PROJECT_MEMORY_FILES,
    MemoryContextEntry,
    MemorySnapshot,
)
from ..project_schema import MAX_RECORD_SIZE
from .entry_store import EntryMemorySnapshot
from .entry_views import inspect_value, render_all_kinds

if TYPE_CHECKING:
    from ..project_registry import ProjectRegistry


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
