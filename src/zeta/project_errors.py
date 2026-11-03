"""Shared project-registry exceptions."""


class ProjectRegistryError(ValueError):
    """A registry operation was rejected or stored state is unsafe."""


class ProjectRegistryBusyError(ProjectRegistryError):
    """Another process held the registry lock until the wait deadline."""

    code = "project_registry_busy"
    retryable = True

    def __init__(self) -> None:
        super().__init__("another process holds the project registry lock; retry")


class UnsupportedMemoryFormatError(ProjectRegistryError):
    """Project memory is valid but this code cannot interpret its format."""


class ProjectNotFoundError(ProjectRegistryError):
    """The requested project does not exist in a valid registry."""
