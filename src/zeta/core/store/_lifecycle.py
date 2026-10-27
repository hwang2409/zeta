"""Conversation-store lifetime and refresh operations."""

from __future__ import annotations

from contextlib import nullcontext
from typing import Self

from ..checkpoints import ConversationIntegrityError


class StoreLifecycleMixin:
    """Close, refresh, and context-manage one conversation store."""

    def close(self) -> None:
        """Drain durable writers, then release this store's activity leases."""
        if self._drain_durable_writes_for_close():
            if self._runtime_lease is not None:
                self._runtime_lease.close()
            self.directory_fd = -1
            self._release_lease()

    def refresh(self) -> None:
        """Reload durable state without acquiring ownership of the session."""
        if self._closed:
            raise ConversationIntegrityError("conversation store is closed")
        with nullcontext() if self._read_only else self._append_lock():
            self._load()
            self._load_session_state()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()
