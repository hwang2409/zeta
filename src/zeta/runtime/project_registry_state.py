"""Process-scoped project registry availability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..project_registry import ProjectRegistry
from ..storage_format import WriterLease


@dataclass(frozen=True)
class ProjectRegistryUnavailable:
    path: Path
    reason: str

    @property
    def warning(self) -> str:
        return "warning: project registry is temporarily unavailable; try again later"


@dataclass(frozen=True)
class ProjectRegistryState:
    registry: ProjectRegistry | None
    unavailable: ProjectRegistryUnavailable | None = None


def project_registry(
    root: Path | str, *, _storage_lease: WriterLease | None = None
) -> ProjectRegistry:
    """Construct a registry without probing or changing it."""

    return ProjectRegistry(root, _storage_lease=_storage_lease)


def available_project_registry(
    root: Path | str, *, _storage_lease: WriterLease | None = None
) -> ProjectRegistryState:
    """Construct an unprobed capability when no startup probe is needed."""

    return ProjectRegistryState(project_registry(root, _storage_lease=_storage_lease))


def load_project_registry(root: Path | str) -> ProjectRegistryState:
    """Probe an existing registry once without creating or repairing it."""

    state = available_project_registry(root)
    registry = state.registry
    assert registry is not None
    try:
        registry.root.lstat()
    except FileNotFoundError:
        return state
    except OSError as exc:
        return ProjectRegistryState(
            registry, ProjectRegistryUnavailable(registry.root, str(exc))
        )
    projection = registry.project_context_projection(None)
    if (
        projection.health is not None
        and projection.health.code == "project_registry_unavailable"
    ):
        return ProjectRegistryState(
            registry,
            ProjectRegistryUnavailable(registry.root, projection.health.message),
        )
    return state
