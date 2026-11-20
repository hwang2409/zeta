"""Shared project-registry exceptions."""


class ProjectRegistryError(ValueError):
    """A registry operation was rejected or stored state is unsafe."""


class ProjectRegistryBusyError(ProjectRegistryError):
    """Another process held the registry lock until the wait deadline."""

    code = "project_registry_busy"
    retryable = True

    def __init__(
        self, *, holder_pid: int | None = None, holder_operation: str | None = None
    ) -> None:
        detail = ""
        if holder_pid is not None or holder_operation is not None:
            detail = f" (holder pid={holder_pid}, operation={holder_operation or 'unknown'})"
        super().__init__(
            f"another process holds the project registry lock{detail}; retry"
        )


class MemorySecretError(ProjectRegistryError):
    """A memory write was refused because its text contains a secret."""


class UnsupportedMemoryFormatError(ProjectRegistryError):
    """Project memory is valid but this code cannot interpret its format."""


class ProjectNotFoundError(ProjectRegistryError):
    """The requested project does not exist in a valid registry."""


class MemoryDigestMismatchError(ProjectRegistryError):
    """Project memory changed since the caller read its digest."""

    def __init__(self) -> None:
        super().__init__("project memory digest mismatch")


def require_memory_digest(actual: str, expected: str | None) -> None:
    """Compare under the memory lock; ``None`` keeps an unconditional write."""
    if expected is not None and actual != expected:
        raise MemoryDigestMismatchError
