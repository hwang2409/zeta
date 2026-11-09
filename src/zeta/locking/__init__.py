"""Private lock-order assertions for cross-domain filesystem locks."""

from __future__ import annotations

import fcntl
import os
import threading
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from enum import IntEnum
from pathlib import Path

# sorts before every valid project id, so retention authorities are acquired
# with project domains in one global order.
RETENTION_HOME_ID = "\x00zeta-retention"


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


@contextmanager
def domain_authority(
    home: Path, project_id: str, domain: str, *, exclusive: bool
) -> Iterator[None]:
    """Hold one ranked domain authority and its process-shared lock file."""
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_dir = home / ".domain-locks"
    lock_dir.mkdir(mode=0o700, exist_ok=True)
    lock_name = f"{project_id.encode('utf-8').hex()}-{domain.encode('utf-8').hex()}.lock"
    fd = os.open(lock_dir / lock_name, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        with domain_lock(project_id, domain):
            fcntl.flock(fd, fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH)
            try:
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
