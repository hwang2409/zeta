"""Process-scoped project registry availability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from ..project_errors import ProjectRegistryError
from ..project_registry import ProjectRegistry


@dataclass(frozen=True)
class ProjectRegistryUnavailable:
    path: Path
    reason: str

    @property
    def warning(self) -> str:
        return (
            f"warning: project registry unavailable at {self.path}: {self.reason}; "
            "project features are disabled"
        )


@dataclass(frozen=True)
class ProjectRegistryState:
    registry: ProjectRegistry | None
    unavailable: ProjectRegistryUnavailable | None = None


def project_registry(root: Path | str) -> ProjectRegistry:
    """Construct a registry without probing or changing it."""

    return ProjectRegistry(root)


def available_project_registry(root: Path | str) -> ProjectRegistryState:
    """Construct an unprobed capability when no startup probe is needed."""

    return ProjectRegistryState(project_registry(root))


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
            None, ProjectRegistryUnavailable(registry.root, str(exc))
        )
    try:
        registry.list_projects()
    except (OSError, ProjectRegistryError) as exc:
        return ProjectRegistryState(
            None, ProjectRegistryUnavailable(registry.root, str(exc))
        )
    return state
