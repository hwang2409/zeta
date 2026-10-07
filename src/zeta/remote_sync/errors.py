"""Shared remote-transfer errors."""


class RemoteSyncError(ValueError):
    """A transfer was unsafe or conflicted with newer state."""


__all__ = ["RemoteSyncError"]
