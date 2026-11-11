"""Tolerant, read-only project context projection."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeVar

from ..memory.entry_store import EntryMemorySnapshot
from ..project_errors import ProjectRegistryError
from ..project_memory_history import MemoryContextEntry
from ..project_schema import MAX_PROJECTS, PROJECT_ID_PATTERN

if TYPE_CHECKING:
    from ..project_registry import Project, ProjectRegistry

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


def compose_with_stable_projection[T](
    registry: ProjectRegistry | None,
    project_id: str | None,
    projection: ProjectContextProjection | None,
    compose: Callable[[ProjectContextProjection | None], T],
) -> tuple[T, ProjectContextProjection | None]:
    """Compose only after two consecutive project projections match."""

    while True:
        composition = compose(projection)
        if registry is None or projection is None:
            return composition, projection
        fresh = registry.project_context_projection(project_id)
        if fresh == projection:
            return composition, projection
        projection = fresh


class ProjectContextProjectionMixin:
    """Read project identities, then selected memory under its domain lock."""

    def _read_context_projection(self, operation: Callable[[int], _T]) -> _T:
        """Read the identity projection under identity authority."""

        return self._identity_read(operation)

    def _read_project_identities(
        self, root_fd: int, names: list[str]
    ) -> tuple[list[Project], list[ProjectContextHealth]]:
        projects: list[Project] = []
        malformed: list[ProjectContextHealth] = []
        for name in names:
            try:
                directory_fd = self._project_dir(root_fd, name)
                try:
                    try:
                        record = self._read_fd(directory_fd, "project.json")
                    except ProjectRegistryError as exc:
                        if isinstance(exc.__cause__, FileNotFoundError):
                            continue
                        raise
                    projects.append(self._decode(record, name))
                finally:
                    os.close(directory_fd)
            except ProjectRegistryError:
                malformed.append(_health("malformed_project", name, "project.json"))
            except OSError:
                malformed.append(_health("malformed_project", name, "project.json"))
        return projects, malformed

    def project_context_projection(
        self, session_project_id: str | None
    ) -> ProjectContextProjection:
        """Read safe context while omitting only damaged project-owned data."""

        def read_identities(
            root_fd: int,
        ) -> tuple[list[Project], list[ProjectContextHealth], Project | None]:
            names = sorted(
                name
                for name in os.listdir(root_fd)
                if PROJECT_ID_PATTERN.fullmatch(name)
            )
            if len(names) > MAX_PROJECTS:
                raise ProjectRegistryError("project limit exceeded")
            projects, malformed = self._read_project_identities(root_fd, names)
            selected = next(
                (item for item in projects if item.project_id == session_project_id),
                None,
            )
            return projects, malformed, selected

        try:
            projects, malformed, selected = self._read_context_projection(
                read_identities
            )
            if selected is None:
                memory, entry_memory, memory_health = [], None, None
            else:
                try:
                    projection, malformed_source = (
                        self.memory_transactions.read_for_context(selected.project_id)
                    )
                    memory = list(projection.memory)
                    entry_memory = (
                        projection.snapshot if projection.format == 2 else None
                    )
                    memory_health = (
                        _health(
                            "malformed_memory",
                            selected.project_id,
                            malformed_source,
                        )
                        if malformed_source is not None
                        else None
                    )
                except (OSError, ProjectRegistryError, ValueError):
                    memory, entry_memory = [], None
                    source = "memory"
                    pointer_path = self.root / selected.project_id / "memory-current.json"
                    try:
                        import json

                        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
                        current = pointer.get("current")
                        if isinstance(current, str):
                            source = f"memory-versions/versions/{current}.json"
                    except (OSError, ValueError, TypeError):
                        source = "memory-current.json"
                    memory_health = _health(
                        "malformed_memory", selected.project_id, source
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
            elif session_project_id is not None:
                finding = memory_health
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
    "compose_with_stable_projection",
]
