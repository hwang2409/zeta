"""Shared project-registry exceptions."""


class ProjectRegistryError(ValueError):
    """A registry operation was rejected or stored state is unsafe."""


class UnsupportedMemoryFormatError(ProjectRegistryError):
    """Project memory is valid but this code cannot interpret its format."""


class ProjectNotFoundError(ProjectRegistryError):
    """The requested project does not exist in a valid registry."""
