"""Lease release that is safe to start from a garbage-collector finalizer.

The collector can run a finalizer at any allocation, including inside a
``with`` block that holds a lease registry lock in the same thread. A
finalizer that took that lock there would deadlock (``Lock``) or change the
registry under its holder (``RLock``). So finalizers only queue the release
with ``release_later``; the queue runs at once when this thread holds no
registry lock, otherwise when the thread lets go of its last one.

Release callbacks may take only ``RegistryLock`` locks.
"""

from __future__ import annotations

import sys
import threading
from collections import deque
from collections.abc import Callable
from typing import Any


class _ThreadState(threading.local):
    held = 0
    draining = False


_STATE = _ThreadState()
_PENDING: deque[tuple[Callable[..., None], tuple[Any, ...]]] = deque()


def release_later(close: Callable[..., None], *args: Any) -> None:
    """Finalizer callback: queue ``close(*args)`` and run it when no registry lock is held."""

    _PENDING.append((close, args))
    if not _STATE.held:
        _drain()


def _drain() -> None:
    if _STATE.draining:
        return
    _STATE.draining = True
    try:
        while True:
            try:
                close, args = _PENDING.popleft()
            except IndexError:
                return
            try:
                close(*args)
            except Exception:  # noqa: BLE001 - a failed release must not fail the unrelated lock holder
                sys.excepthook(*sys.exc_info())
    finally:
        _STATE.draining = False


class RegistryLock:
    """A non-reentrant lock whose holder runs queued releases after it lets go."""

    def __init__(self) -> None:
        self._lock = threading.Lock()

    def __enter__(self) -> None:
        # Count before acquiring and uncount after releasing, so a finalizer
        # that fires in between queues instead of waiting on this lock.
        _STATE.held += 1
        try:
            self._lock.acquire()
        except BaseException:
            _STATE.held -= 1
            raise

    def __exit__(self, *_exc: object) -> None:
        self._lock.release()
        _STATE.held -= 1
        if not _STATE.held and _PENDING:
            _drain()
