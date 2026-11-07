"""Tolerant, read-only project context projection."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

from ..memory.entry_store import EntryMemorySnapshot
from ..project_errors import ProjectRegistryError, UnsupportedMemoryFormatError
from ..project_memory_history import PROJECT_MEMORY_FILES, MemoryContextEntry
from ..project_schema import MAX_PROJECTS, PROJECT_ID_PATTERN

if TYPE_CHECKING:
    from ..project_registry import Project

_T = TypeVar("_T")


@dataclass(frozen=True, slots=True)
class ProjectContextHealth:
    code: Literal[
        "project_not_found",
        "malformed_project",
        "malformed_memory",
        "project_registry_unavailable",
    ]
    project_id: str | None
    source: str | None
    message: str

    @property
    def summary(self) -> str:
        details = "/".join(
            value for value in (self.project_id, self.source) if value is not None
        )
        return f"{self.code}{f' ({details})' if details else ''}: {self.message}"


@dataclass(frozen=True, slots=True)
class ProjectContextProjection:
    session_project: Project | None
    projects: tuple[Project, ...]
    memory: tuple[MemoryContextEntry, ...]
    entry_memory: EntryMemorySnapshot | None
    health: ProjectContextHealth | None


_HEALTH_MESSAGES = {
    "project_not_found": "saved project was not found",
    "malformed_project": "project metadata is malformed",
    "malformed_memory": "project memory is malformed",
    "project_registry_unavailable": (
        "project registry is temporarily unavailable; try again later"
    ),
}


def _health(
    code: Literal[
        "project_not_found",
        "malformed_project",
        "malformed_memory",
        "project_registry_unavailable",
    ],
    project_id: str | None = None,
    source: str | None = None,
) -> ProjectContextHealth:
    return ProjectContextHealth(code, project_id, source, _HEALTH_MESSAGES[code])


class ProjectContextProjectionMixin:
    """Read safe prompt inputs under one registry-lock acquisition."""

    def _read_context_projection(
        self, operation: Callable[[int], _T]
    ) -> _T:
        """Own the lock seam that p06 can replace in one place."""

        return self._read(operation)

    def _read_project_identities(
        self, root_fd: int, names: list[str]
    ) -> tuple[list[Project], list[ProjectContextHealth]]:
        projects: list[Project] = []
        malformed: list[ProjectContextHealth] = []
        for name in names:
            try:
                directory_fd = self._project_dir(root_fd, name)
                try:
                    record = self._read_fd(directory_fd, "project.json")
                    projects.append(self._decode(record, name))
                finally:
                    os.close(directory_fd)
            except (OSError, ProjectRegistryError):
                malformed.append(_health("malformed_project", name, "project.json"))
        return projects, malformed

    def _read_selected_memory(
        self, root_fd: int, selected: Project | None
    ) -> tuple[
        list[MemoryContextEntry],
        EntryMemorySnapshot | None,
        ProjectContextHealth | None,
    ]:
        memory: list[MemoryContextEntry] = []
        entry_memory: EntryMemorySnapshot | None = None
        memory_health: ProjectContextHealth | None = None
        if selected is None:
            return memory, entry_memory, memory_health
        directory_fd = self._project_dir(root_fd, selected.project_id)
        try:
            pointer: dict[str, object] | None = None
            try:
                pointer = self._pointer(directory_fd)
                if pointer is None:
                    memory_fd = self._memory_fd(directory_fd)
                    try:
                        for name in PROJECT_MEMORY_FILES:
                            try:
                                content = self._read_memory_file(memory_fd, name)
                                if content.startswith("<!-- zeta-memory-mirror:v1"):
                                    raise ProjectRegistryError(
                                        "memory mirror is not authoritative"
                                    )
                            except FileNotFoundError:
                                continue
                            except (OSError, ProjectRegistryError):
                                if memory_health is None:
                                    memory_health = _health(
                                        "malformed_memory",
                                        selected.project_id,
                                        f"memory/{name}",
                                    )
                                continue
                            memory.append(MemoryContextEntry(name, content, False))
                    finally:
                        os.close(memory_fd)
                else:
                    try:
                        snapshot = self._snapshot_locked(directory_fd)
                    except UnsupportedMemoryFormatError:
                        entry_memory = self._entry_snapshot_locked(directory_fd)
                    else:
                        automatic = self._automatic_files_from_records(
                            self._records_locked(directory_fd)
                        )
                        memory.extend(
                            MemoryContextEntry(name, content, name in automatic)
                            for name, content in snapshot.contents.items()
                        )
            except (OSError, ProjectRegistryError, ValueError):
                source = "memory-current.json"
                if pointer is not None and isinstance(pointer.get("current"), str):
                    source = f"memory-versions/versions/{pointer['current']}.json"
                memory.clear()
                memory_health = _health(
                    "malformed_memory", selected.project_id, source
                )
        finally:
            os.close(directory_fd)
        return memory, entry_memory, memory_health

    def project_context_projection(
        self, session_project_id: str | None
    ) -> ProjectContextProjection:
        """Read safe context while omitting only damaged project-owned data."""

        def read(root_fd: int) -> ProjectContextProjection:
            names = sorted(
                name for name in os.listdir(root_fd) if PROJECT_ID_PATTERN.fullmatch(name)
            )
            if len(names) > MAX_PROJECTS:
                raise ProjectRegistryError("project limit exceeded")
            projects, malformed = self._read_project_identities(root_fd, names)
            selected = next(
                (item for item in projects if item.project_id == session_project_id),
                None,
            )
            memory, entry_memory, memory_health = self._read_selected_memory(
                root_fd, selected
            )

            if session_project_id is not None and selected is None:
                finding = next(
                    (
                        item
                        for item in malformed
                        if item.project_id == session_project_id
                    ),
                    None,
                ) or _health("project_not_found", session_project_id)
            else:
                finding = memory_health or (malformed[0] if malformed else None)
            return ProjectContextProjection(
                session_project=selected,
                projects=tuple(
                    sorted(projects, key=lambda item: (item.name, item.project_id))
                ),
                memory=tuple(memory),
                entry_memory=entry_memory,
                health=finding,
            )

        try:
            return self._read_context_projection(read)
        except (OSError, ProjectRegistryError):
            return ProjectContextProjection(
                session_project=None,
                projects=(),
                memory=(),
                entry_memory=None,
                health=_health("project_registry_unavailable"),
            )


__all__ = [
    "ProjectContextHealth",
    "ProjectContextProjection",
    "ProjectContextProjectionMixin",
]
