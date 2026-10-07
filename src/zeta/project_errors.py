"""Shared project-registry exceptions."""


class ProjectRegistryError(ValueError):
    """A registry operation was rejected or stored state is unsafe."""


class ProjectNotFoundError(ProjectRegistryError):
    """The requested project does not exist in a valid registry."""
