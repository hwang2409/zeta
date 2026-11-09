"""Private lock-order assertions for cross-domain filesystem locks."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from enum import IntEnum


class _Rank(IntEnum):
    FORMAT = 1
    SOCKET = 2
    EXECUTION = 3
    IDENTITY = 4
    DOMAIN = 5
    SHAPE = 6


_state = threading.local()


def _stack() -> list[tuple[_Rank, tuple[str, str] | None]]:
    stack = getattr(_state, "stack", None)
    if stack is None:
        stack = []
        _state.stack = stack
    return stack


@contextmanager
def ranked_lock(
    rank: _Rank, *, domain: tuple[str, str] | None = None
) -> Iterator[None]:
    """Reject inversions before a physical lock is attempted."""
    stack = _stack()
    if stack:
        previous_rank, previous_domain = stack[-1]
        if rank < previous_rank:
            raise RuntimeError("filesystem lock order inversion")
        if rank == _Rank.DOMAIN and previous_rank == _Rank.DOMAIN:
            assert domain is not None and previous_domain is not None
            if domain <= previous_domain:
                raise RuntimeError("project domain locks must be acquired in sorted order")
    if rank == _Rank.DOMAIN and domain is None:
        raise AssertionError("project domain rank requires a domain key")
    stack.append((rank, domain))
    try:
        yield
    finally:
        if not stack or stack[-1] != (rank, domain):
            raise RuntimeError("filesystem lock rank released on another thread or out of order")
        stack.pop()


def format_lock() -> AbstractContextManager[None]:
    return ranked_lock(_Rank.FORMAT)


def socket_lock() -> AbstractContextManager[None]:
    return ranked_lock(_Rank.SOCKET)


def execution_lock() -> AbstractContextManager[None]:
    return ranked_lock(_Rank.EXECUTION)


def identity_lock() -> AbstractContextManager[None]:
    return ranked_lock(_Rank.IDENTITY)


def domain_lock(project_id: str, domain: str) -> AbstractContextManager[None]:
    return ranked_lock(_Rank.DOMAIN, domain=(project_id, domain))


def shape_lock() -> AbstractContextManager[None]:
    return ranked_lock(_Rank.SHAPE)
