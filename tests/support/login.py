"""Synchronization helpers for login timing tests."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable

import pytest

from zeta.core import login_flow


def block_loop_until_publication_decision_expires(
    monkeypatch: pytest.MonkeyPatch,
    loop: asyncio.AbstractEventLoop,
) -> tuple[Callable[[], None], threading.Event]:
    """Keep the loop blocked until the worker observes decision expiry."""

    decision_expired = threading.Event()
    loop_blocked = threading.Event()
    real_is_set = login_flow._PublicationArbiter.is_set

    def observed_is_set(arbiter: login_flow._PublicationArbiter) -> bool:
        result = real_is_set(arbiter)
        if arbiter.decision_failed:
            decision_expired.set()
        return result

    def block_loop() -> None:
        loop_blocked.set()
        assert decision_expired.wait(timeout=2)

    monkeypatch.setattr(login_flow._PublicationArbiter, "is_set", observed_is_set)
    return block_loop, loop_blocked
