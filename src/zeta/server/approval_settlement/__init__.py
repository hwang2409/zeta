"""Durable approval settlement and cancellation recovery."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from ...core.store import ConversationStore


@dataclass(frozen=True, slots=True)
class SettlementResult:
    resolution: dict[str, str] | None
    cancelled: bool = False

    @property
    def should_retry(self) -> bool:
        """True when a cancelled settlement never committed a durable decision."""
        return self.cancelled and self.resolution is None


async def finish_despite_cancellation[T](
    operation: Callable[[], Awaitable[T]],
) -> T:
    """Keep one settlement owner alive through cancellation."""
    while True:
        try:
            return await operation()
        except asyncio.CancelledError:
            continue


async def settlement_outcome[T](
    admitted: asyncio.Future[T], settlement: asyncio.Task[T]
) -> T:
    """Return admission, unless the settlement owner reports a real failure first."""
    done, _ = await asyncio.wait(
        (admitted, settlement), return_when=asyncio.FIRST_COMPLETED
    )
    return (admitted if admitted in done else settlement).result()


async def persist_durably(
    store: ConversationStore,
    request_id: str,
    decision: str,
    reason: str | None = None,
) -> SettlementResult:
    """Commit a decision and reconcile deferred cancellation with durable state."""
    kwargs: dict[str, Any] = {}
    if decision == "abort" and reason:
        kwargs["reason"] = reason
    try:
        await store._to_thread_durable(
            store.resolve_approval, request_id, decision, **kwargs
        )
    except asyncio.CancelledError:
        resolution = store.approval_resolution_data(request_id)
        return SettlementResult(resolution, cancelled=True)
    except OSError:
        resolution = store.approval_resolution_data(request_id)
        if resolution is not None:
            return SettlementResult(resolution)
        try:
            await store._to_thread_durable(
                store.resolve_approval,
                request_id,
                "abort",
                reason=reason or "turn_canceled",
            )
        except OSError:
            return SettlementResult(None)
    return SettlementResult(store.approval_resolution_data(request_id))
