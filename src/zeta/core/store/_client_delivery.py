"""Durable lifecycle for idempotent client delivery and live steering."""

from __future__ import annotations

import re
from collections import OrderedDict, deque
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, TypeGuard

from ...protocol.types import Message, MessageRole, require_new_message_origin

if TYPE_CHECKING:
    from ..checkpoints import ConversationEntry
    from ._store import ConversationStore

CLIENT_DELIVERY_METADATA = "zeta_client_delivery_id"
MAX_RECENT_CLIENT_DELIVERIES = 1_000
DELIVERY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")

DeliveryMethod = Literal["send", "steer"]
DeliveryStatus = Literal["queued", "delivered", "dropped"]
DeliveryDropReason = Literal["restart", "abort", "clear", "disconnect", "turn_end"]


@dataclass(frozen=True, slots=True)
class ClientDelivery:
    delivery_id: str
    method: DeliveryMethod
    status: DeliveryStatus
    outcome: dict[str, object]
    reason: DeliveryDropReason | None = None


@dataclass(frozen=True, slots=True)
class _QueuedSteering:
    message: Message
    delivery_id: str | None


def valid_client_delivery_id(value: object) -> TypeGuard[str]:
    return isinstance(value, str) and DELIVERY_ID_PATTERN.fullmatch(value) is not None


def validate_client_delivery_data(
    data: Mapping[str, object], session_id: str
) -> None:
    """Validate one acceptance or terminal transition record."""

    delivery_id = data.get("delivery_id")
    method = data.get("method")
    status = data.get("status")
    if not valid_client_delivery_id(delivery_id) or method not in {"send", "steer"}:
        raise ValueError("invalid client delivery")

    if "outcome" in data:
        outcome = data.get("outcome")
        valid_acceptance = (
            method == "send"
            and status == "delivered"
            and outcome == {"accepted": True, "session_id": session_id}
        ) or (
            method == "steer"
            and status == "queued"
            and outcome == {"accepted": True}
        )
        if set(data) != {"delivery_id", "method", "status", "outcome"} or not valid_acceptance:
            raise ValueError("invalid client delivery")
        return

    reason = data.get("reason")
    valid_transition = (
        method == "steer"
        and (
            (status == "delivered" and reason is None and set(data) == {"delivery_id", "method", "status"})
            or (
                status == "dropped"
                and reason in {"restart", "abort", "clear", "disconnect", "turn_end"}
                and set(data) == {"delivery_id", "method", "status", "reason"}
            )
        )
    )
    if not valid_transition:
        raise ValueError("invalid client delivery")


class ClientDeliveryMixin:
    """Own acceptance, queue state, transitions, and bounded O(1) lookup."""

    def _initialize_client_delivery_lifecycle(self: ConversationStore) -> None:
        self._client_deliveries: OrderedDict[str, ClientDelivery] = OrderedDict()
        self._client_steering: deque[_QueuedSteering] = deque()

    def _rebuild_client_deliveries(
        self: ConversationStore, entries: list[ConversationEntry]
    ) -> None:
        self._client_deliveries = OrderedDict()
        self._record_client_delivery_entries(entries)

    def _record_client_delivery_entries(
        self: ConversationStore, entries: list[ConversationEntry]
    ) -> None:
        for entry in entries:
            if entry.type != "client_delivery":
                continue
            data = entry.data
            delivery_id = data["delivery_id"]
            if "outcome" in data:
                self._client_deliveries[delivery_id] = ClientDelivery(
                    delivery_id=delivery_id,
                    method=data["method"],
                    status=data["status"],
                    outcome=dict(data["outcome"]),
                )
                self._client_deliveries.move_to_end(delivery_id)
                while len(self._client_deliveries) > MAX_RECENT_CLIENT_DELIVERIES:
                    self._client_deliveries.popitem(last=False)
                continue
            accepted = self._client_deliveries.get(delivery_id)
            if accepted is None:
                continue
            self._client_deliveries[delivery_id] = ClientDelivery(
                delivery_id=delivery_id,
                method=accepted.method,
                status=data["status"],
                outcome=accepted.outcome,
                reason=data.get("reason"),
            )

    def recover_client_deliveries(self: ConversationStore) -> int:
        """Mark accepted steering with no live owner as dropped after restart."""

        if not any(
            delivery.method == "steer" and delivery.status == "queued"
            for delivery in self._client_deliveries.values()
        ):
            return 0
        with self._append_lock():
            self._load()
            ids = [
                delivery.delivery_id
                for delivery in self._client_deliveries.values()
                if delivery.method == "steer" and delivery.status == "queued"
            ]
            if ids:
                self._append_many_unlocked(
                    [
                        _transition_row(delivery_id, "dropped", "restart")
                        for delivery_id in ids
                    ]
                )
        return len(ids)

    def client_delivery(self: ConversationStore, delivery_id: str) -> ClientDelivery | None:
        """Return one retained delivery without reading the conversation log."""

        return self._client_deliveries.get(delivery_id)

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
        delivery = self.client_delivery(delivery_id)
        assert delivery is not None
        return delivery

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
        """Append off-loop, then run the effect callback after publication."""

        return await self._to_thread_durable(
            self.append_client_delivery,
            delivery_id,
            method,
            status,
            outcome,
            message=message,
            _on_persisted=on_persisted,
        )

    def queue_client_steering(
        self: ConversationStore, message: Message, delivery_id: str | None = None
    ) -> None:
        if message.role is not MessageRole.USER:
            raise ValueError("steering message must have the user role")
        marker = message.metadata.get(CLIENT_DELIVERY_METADATA)
        if delivery_id is None and valid_client_delivery_id(marker):
            delivery_id = marker
        self._client_steering.append(
            _QueuedSteering(require_new_message_origin(message), delivery_id)
        )

    @property
    def has_pending_client_steering(self: ConversationStore) -> bool:
        return bool(self._client_steering)

    async def deliver_next_client_steering(
        self: ConversationStore,
        append_message: Callable[[Message], Awaitable[object]],
    ) -> None:
        queued = self._client_steering[0]
        if queued.delivery_id is None:
            await append_message(queued.message)
        else:
            await self._to_thread_durable(
                self._deliver_client_steering,
                queued.delivery_id,
                queued.message,
            )
        self._client_steering.popleft()

    def _deliver_client_steering(
        self: ConversationStore, delivery_id: str, message: Message
    ) -> None:
        with self._append_lock():
            self._load()
            delivery = self.client_delivery(delivery_id)
            if delivery is None or delivery.status != "queued":
                raise ValueError(f"client delivery is not queued: {delivery_id}")
            self._append_many_unlocked(
                [
                    ("message", {"message": tag_client_delivery(message, delivery_id).to_dict()}),
                    _transition_row(delivery_id, "delivered"),
                ]
            )

    def drop_client_steering(self: ConversationStore, reason: DeliveryDropReason) -> int:
        """Durably drop every queued delivery before releasing queue ownership."""

        queued = tuple(self._client_steering)
        delivery_ids = [item.delivery_id for item in queued if item.delivery_id is not None]
        if delivery_ids:
            with self._append_lock():
                self._load()
                rows = [
                    _transition_row(delivery_id, "dropped", reason)
                    for delivery_id in delivery_ids
                    if (delivery := self.client_delivery(delivery_id)) is not None
                    and delivery.status == "queued"
                ]
                if rows:
                    self._append_many_unlocked(rows)
        self._client_steering.clear()
        return len(queued)


def _transition_row(
    delivery_id: str,
    status: Literal["delivered", "dropped"],
    reason: DeliveryDropReason | None = None,
) -> tuple[str, dict[str, object]]:
    data: dict[str, object] = {
        "delivery_id": delivery_id,
        "method": "steer",
        "status": status,
    }
    if reason is not None:
        data["reason"] = reason
    return "client_delivery", data


def tag_client_delivery(message: Message, delivery_id: str) -> Message:
    """Attach a private durable-delivery marker without changing message content."""

    return Message(
        message.role,
        message.content,
        tool_result=message.tool_result,
        metadata={**message.metadata, CLIENT_DELIVERY_METADATA: delivery_id},
    )


def without_client_delivery_marker(message: Message) -> Message:
    """Remove protocol bookkeeping before exposing a message outside the store."""

    if CLIENT_DELIVERY_METADATA not in message.metadata:
        return message
    metadata = dict(message.metadata)
    del metadata[CLIENT_DELIVERY_METADATA]
    return Message(
        message.role,
        message.content,
        tool_result=message.tool_result,
        metadata=metadata,
    )
