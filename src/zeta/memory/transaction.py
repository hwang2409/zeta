"""One read owner for project memory."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..project_memory_history import (
    PROJECT_MEMORY_FILES,
    MemoryContextEntry,
    MemorySnapshot,
)
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
        if self.format == 1:
            assert isinstance(self.snapshot, MemorySnapshot)
            return dict(self.snapshot.contents)
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


@dataclass(frozen=True, slots=True)
class BrokenMemoryProjection:
    project_id: str
    format: Literal[1, 2] | None
    message: str

MemoryProjection = HealthyMemoryProjection | BrokenMemoryProjection


class MemoryTransactions:
    """Read project memory while holding p06's memory-domain capability."""

    def __init__(self, registry: ProjectRegistry):
        self._registry = registry

    def read(self, project_id: str, *, byte_cap: int = 64 * 1024) -> HealthyMemoryProjection:
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
                    return HealthyMemoryProjection(
                        project_id, 2, snapshot.version, snapshot.digest, snapshot, memory
                    )
                snapshot = registry._snapshot_locked(directory_fd)
                contents = snapshot.contents
                memory = tuple(
                    MemoryContextEntry(name, content, False)
                    for name, content in contents.items()
                    if name in PROJECT_MEMORY_FILES
                )
                return HealthyMemoryProjection(project_id, 1, None, snapshot.digest, snapshot, memory)
            finally:
                os.close(directory_fd)


__all__ = ["BrokenMemoryProjection", "HealthyMemoryProjection", "MemoryProjection", "MemoryTransactions"]
