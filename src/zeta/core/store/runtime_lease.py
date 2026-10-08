"""Epoch-bound session runtime leases for prompt resume coordination."""

from __future__ import annotations

import fcntl
import os
from collections.abc import Iterator
from contextlib import contextmanager

from ..session_files import open_session_file

RUNTIME_LEASE_NAME = "runtime.lease"


class SessionRuntimeLease:
    """Own one runtime's advisory lease for exactly its store lifetime.

    A newly opened runtime starts unlocked so prompt resume can probe for an
    existing runtime with a non-blocking exclusive lock. After that decision,
    the lease remains shared until ``close``. The kernel releases all modes if
    the process exits.
    """

    def __init__(self, directory_fd: int) -> None:
        self._fd = open_session_file(
            directory_fd, RUNTIME_LEASE_NAME, os.O_RDWR | os.O_CREAT
        )
        self._shared = False

    def activate(self) -> None:
        """Hold a shared lease for a runtime that does not need resume work."""

        if self._fd < 0:
            raise RuntimeError("session runtime lease is closed")
        if not self._shared:
            fcntl.flock(self._fd, fcntl.LOCK_SH)
            self._shared = True

    @contextmanager
    def resume(self) -> Iterator[bool]:
        """Yield whether this is the only live runtime, then remain shared.

        The caller must hold the session metadata lock for this complete
        context. This serializes the exclusive probe, prompt persistence, and
        downgrade so a concurrent runtime cannot observe a publication gap.
        """

        if self._fd < 0:
            raise RuntimeError("session runtime lease is closed")
        if self._shared:
            raise RuntimeError("session runtime lease is already active")
        try:
            fcntl.flock(self._fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            compose = True
        except BlockingIOError:
            fcntl.flock(self._fd, fcntl.LOCK_SH)
            compose = False
        try:
            yield compose
        finally:
            fcntl.flock(self._fd, fcntl.LOCK_SH)
            self._shared = True

    def close(self) -> None:
        """Release this exact runtime epoch; repeated closes are harmless."""

        if self._fd >= 0:
            fd, self._fd = self._fd, -1
            self._shared = False
            os.close(fd)
