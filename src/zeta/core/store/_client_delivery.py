"""Durable idempotency records for client-submitted turns and steering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ...protocol.types import Message

if TYPE_CHECKING:
    from ._store import ConversationStore

CLIENT_DELIVERY_METADATA = "zeta_client_delivery_id"
MAX_RECENT_CLIENT_DELIVERIES = 1_000

DeliveryMethod = Literal["send", "steer"]
DeliveryStatus = Literal["queued", "delivered"]


@dataclass(frozen=True, slots=True)
class ClientDelivery:
    delivery_id: str
    method: DeliveryMethod
    status: DeliveryStatus
    outcome: dict[str, object]


class ClientDeliveryMixin:
    """Persist and query a bounded recent set of accepted client deliveries."""

    def client_delivery(self: ConversationStore, delivery_id: str) -> ClientDelivery | None:
        recent = 0
        accepted = None
        for entry in reversed(self._entries):
            if entry.type != "client_delivery":
                continue
            recent += 1
            if recent > MAX_RECENT_CLIENT_DELIVERIES:
                break
            if entry.data["delivery_id"] == delivery_id:
                accepted = entry
                break
        if accepted is None:
            return None

        status: DeliveryStatus = accepted.data["status"]
        if status == "queued":
            for entry in self._entries:
                if entry.seq <= accepted.seq or entry.type != "message":
                    continue
                metadata = entry.data["message"].get("metadata", {})
                if metadata.get(CLIENT_DELIVERY_METADATA) == delivery_id:
                    status = "delivered"
                    break
        return ClientDelivery(
            delivery_id=delivery_id,
            method=accepted.data["method"],
            status=status,
            outcome=dict(accepted.data["outcome"]),
        )

    def append_client_delivery(
        self: ConversationStore,
        delivery_id: str,
        method: DeliveryMethod,
        status: DeliveryStatus,
        outcome: dict[str, object],
        *,
        message: Message | None = None,
    ) -> ClientDelivery:
        """Atomically append one acceptance and its optional user message."""

        with self._append_lock():
            self._load()
            if self.client_delivery(delivery_id) is not None:
                raise ValueError(f"client delivery already exists: {delivery_id}")
            rows: list[tuple[str, dict[str, object]]] = [
                (
                    "client_delivery",
                    {
                        "delivery_id": delivery_id,
                        "method": method,
                        "status": status,
                        "outcome": dict(outcome),
                    },
                )
            ]
            if message is not None:
                rows.append(
                    (
                        "message",
                        {"message": tag_client_delivery(message, delivery_id).to_dict()},
                    )
                )
            self._append_many_unlocked(rows)
        return ClientDelivery(delivery_id, method, status, dict(outcome))

    async def append_client_delivery_async(
        self: ConversationStore,
        delivery_id: str,
        method: DeliveryMethod,
        status: DeliveryStatus,
        outcome: dict[str, object],
        *,
        message: Message | None = None,
        on_persisted: Callable[[], None] | None = None,
    ) -> ClientDelivery:
        """Append off-loop, then run the effect callback after durable publication."""

        return await self._to_thread_durable(
            self.append_client_delivery,
            delivery_id,
            method,
            status,
            outcome,
            message=message,
            _on_persisted=on_persisted,
        )


def tag_client_delivery(message: Message, delivery_id: str) -> Message:
    """Attach a private durable-delivery marker without changing message content."""

    return Message(
        message.role,
        message.content,
        tool_result=message.tool_result,
        metadata={**message.metadata, CLIENT_DELIVERY_METADATA: delivery_id},
    )
