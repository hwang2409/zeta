"""Durable approval settlement and cancellation recovery."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from ...core.store import ConversationStore


@dataclass(frozen=True, slots=True)
class SettlementResult:
    resolution: dict[str, str] | None
    cancelled: bool = False

    @property
    def should_retry(self) -> bool:
        """True when a cancelled settlement never committed a durable decision."""
        return self.cancelled and self.resolution is None


async def persist_durably(
    store: ConversationStore,
    request_id: str,
    decision: str,
    reason: str | None = None,
) -> SettlementResult:
    """Commit a decision and reconcile deferred cancellation with durable state."""
    try:
        await store.resolve_approval_async(
            request_id,
            decision,
            reason=reason if decision == "abort" else None,
        )
    except asyncio.CancelledError:
        resolution = store.approval_resolution_data(request_id)
        return SettlementResult(resolution, cancelled=True)
    except OSError:
        resolution = store.approval_resolution_data(request_id)
        if resolution is not None:
            return SettlementResult(resolution)
        try:
            await store.resolve_approval_async(
                request_id,
                "abort",
                reason=reason or "turn_canceled",
            )
        except OSError:
            return SettlementResult(None)
    return SettlementResult(store.approval_resolution_data(request_id))
