"""Hold, observe, or fail one named ConversationStore write in a test."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from zeta.core.store import ConversationStore


@dataclass
class StoreGate:
    name: str
    hold: bool = False
    error: BaseException | None = None
    then: Callable[[ConversationStore], object] | None = None
    entered: asyncio.Event = field(default_factory=asyncio.Event)
    done: asyncio.Event = field(default_factory=asyncio.Event)
    released: asyncio.Event = field(default_factory=asyncio.Event)
    calls: list[ConversationStore] = field(default_factory=list)

    def release(self) -> None:
        self.released.set()


def gate_store(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    *,
    when: Callable[[ConversationStore, tuple, dict], bool] = lambda *_: True,
    hold: bool = False,
    error: BaseException | None = None,
    then: Callable[[ConversationStore], object] | None = None,
    once: bool = False,
) -> StoreGate:
    """Wrap ``ConversationStore.<name>`` for calls that match ``when``.

    A matching call is recorded, then (async writes only) held until
    ``release()``, then raises ``error`` instead of writing, or writes and runs
    ``then`` in the same tick. ``once`` limits the hold and error to the first match.
    """
    gate = StoreGate(name, hold=hold, error=error, then=then)
    original = getattr(ConversationStore, name)

    def matches(store: ConversationStore, args: tuple, kwargs: dict) -> bool:
        if not when(store, args, kwargs):
            return False
        gate.calls.append(store)
        gate.entered.set()
        return not once or len(gate.calls) == 1

    def finish(store: ConversationStore) -> None:
        gate.done.set()
        if gate.then is not None:
            gate.then(store)

    if inspect.iscoroutinefunction(original):
        async def wrapped(store, *args, **kwargs):
            if not matches(store, args, kwargs):
                return await original(store, *args, **kwargs)
            if gate.hold:
                # wait_for, not await_or_fail: a canceled writer must not leave the wait behind.
                await asyncio.wait_for(gate.released.wait(), 10)
            if gate.error is not None:
                raise gate.error
            result = await original(store, *args, **kwargs)
            finish(store)
            return result
    else:
        if hold:
            raise ValueError(f"{name} is synchronous; it cannot be held")

        def wrapped(store, *args, **kwargs):
            if not matches(store, args, kwargs):
                return original(store, *args, **kwargs)
            if gate.error is not None:
                raise gate.error
            result = original(store, *args, **kwargs)
            finish(store)
            return result

    # test-contract: seam hold or fail one named durable write at the store boundary.
    monkeypatch.setattr(ConversationStore, name, wrapped)
    return gate
