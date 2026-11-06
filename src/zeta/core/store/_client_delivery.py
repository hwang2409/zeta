"""Durable lifecycle for idempotent client delivery and live steering."""

from __future__ import annotations

import base64
import binascii
import hashlib
import re
from collections import OrderedDict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal, TypeGuard

from ...protocol.types import (
    ImageContent,
    Message,
    MessageRole,
    TextContent,
    require_new_message_origin,
)

if TYPE_CHECKING:
    from ..checkpoints import ConversationEntry
    from ._store import ConversationStore

CLIENT_DELIVERY_METADATA = "zeta_client_delivery_id"
MAX_RECENT_CLIENT_DELIVERIES = 1_000
DELIVERY_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")

DeliveryMethod = Literal["send", "steer"]
DeliveryStatus = Literal["queued", "delivered", "dropped"]
DeliveryDropReason = Literal["restart", "abort", "clear", "disconnect", "turn_end", "failed"]

_OUTCOME_BYTES: dict[DeliveryStatus, bytes] = {
    "queued": b"q",
    "delivered": b"d",
    "dropped": b"x",
}
_BYTE_OUTCOMES = {value: key for key, value in _OUTCOME_BYTES.items()}


@dataclass(frozen=True, slots=True)
class ClientDelivery:
    delivery_id: str
    method: DeliveryMethod
    status: DeliveryStatus
    outcome: dict[str, object]
    payload_digest: str | None = None
    reason: DeliveryDropReason | None = None


@dataclass(frozen=True, slots=True)
class _QueuedSteering:
    message: Message
    delivery_id: str | None


@dataclass(frozen=True, slots=True)
class ClientSteeringBatch:
    """One immutable provider-request snapshot of queued steering."""

    _queued: tuple[_QueuedSteering, ...]

    @property
    def messages(self) -> tuple[Message, ...]:
        return tuple(
            without_client_delivery_marker(item.message) for item in self._queued
        )


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
        accepted_message = data.get("accepted_message")
        if accepted_message is not None:
            if not (method == "steer" and status == "queued" and isinstance(accepted_message, dict)):
                raise ValueError("invalid accepted steering message")
            require_new_message_origin(Message.from_dict(accepted_message))
        allowed = {"delivery_id", "method", "status", "outcome", "accepted_message"}
        if set(data) - allowed or not valid_acceptance:
            raise ValueError("invalid client delivery")
        return

    reason = data.get("reason")
    valid_transition = (
        method == "steer"
        and (
            (status == "delivered" and reason is None and set(data) == {"delivery_id", "method", "status"})
            or (
                status == "dropped"
                and reason in {"restart", "abort", "clear", "disconnect", "turn_end", "failed"}
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
        self._client_delivery_outcomes: dict[bytes, bytes] = {}
        self._client_steering: deque[_QueuedSteering] = deque()

    def _rebuild_client_deliveries(
        self: ConversationStore, entries: list[ConversationEntry]
    ) -> None:
        self._client_deliveries = OrderedDict()
        self._client_delivery_outcomes = {}
        self._record_client_delivery_entries(entries)

    def _record_client_delivery_entries(
        self: ConversationStore, entries: list[ConversationEntry]
    ) -> None:
        payload_digests: dict[str, str] = {}
        for entry in entries:
            if entry.type != "message":
                continue
            message = Message.from_dict(entry.data["message"])
            marker = message.metadata.get(CLIENT_DELIVERY_METADATA)
            digest = _message_image_payload_digest(message)
            if valid_client_delivery_id(marker) and digest is not None:
                payload_digests[marker] = digest
        for entry in entries:
            if entry.type != "client_delivery":
                continue
            data = entry.data
            delivery_id = data["delivery_id"]
            if "outcome" in data:
                self._client_delivery_outcomes[_delivery_digest(delivery_id)] = (
                    _OUTCOME_BYTES[data["status"]]
                )
                self._client_deliveries[delivery_id] = ClientDelivery(
                    delivery_id=delivery_id,
                    method=data["method"],
                    status=data["status"],
                    outcome=dict(data["outcome"]),
                    payload_digest=payload_digests.get(delivery_id),
                )
                self._client_deliveries.move_to_end(delivery_id)
                while len(self._client_deliveries) > MAX_RECENT_CLIENT_DELIVERIES:
                    self._client_deliveries.popitem(last=False)
                continue
            digest = _delivery_digest(delivery_id)
            if digest not in self._client_delivery_outcomes:
                continue
            self._client_delivery_outcomes[digest] = _OUTCOME_BYTES[data["status"]]
            accepted = self._client_deliveries.get(delivery_id)
            if accepted is not None:
                self._client_deliveries[delivery_id] = ClientDelivery(
                    delivery_id=delivery_id,
                    method=accepted.method,
                    status=data["status"],
                    outcome=accepted.outcome,
                    reason=data.get("reason"),
                    payload_digest=accepted.payload_digest,
                )

    def recover_client_deliveries(self: ConversationStore) -> int:
        """Mark accepted steering with no live owner as dropped after restart."""

        ids = [
            entry.data["delivery_id"]
            for entry in self._entries
            if entry.type == "client_delivery"
            and "outcome" in entry.data
            and entry.data["method"] == "steer"
            and self.client_delivery_outcome(entry.data["delivery_id"]) == "queued"
        ]
        if not ids:
            return 0
        with self._append_lock():
            ids = [
                delivery_id
                for delivery_id in ids
                if self.client_delivery_outcome(delivery_id) == "queued"
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
        """Return one detail-retained delivery without reading the conversation log."""

        return self._client_deliveries.get(delivery_id)

    def client_delivery_outcome(
        self: ConversationStore, delivery_id: str
    ) -> DeliveryStatus | None:
        """Return the compact lifetime outcome without retaining the delivery ID."""

        value = self._client_delivery_outcomes.get(_delivery_digest(delivery_id))
        return _BYTE_OUTCOMES.get(value) if value is not None else None

    def append_client_delivery(
        self: ConversationStore,
        delivery_id: str,
        method: DeliveryMethod,
        status: DeliveryStatus,
        outcome: dict[str, object],
        *,
        message: Message | None = None,
        accepted_message: Message | None = None,
    ) -> ClientDelivery:
        """Atomically append one acceptance and its optional user message."""

        if message is not None and accepted_message is not None:
            raise ValueError("message and accepted_message are mutually exclusive")
        if accepted_message is not None and (
            method != "steer" or status != "queued" or accepted_message.role is not MessageRole.USER
        ):
            raise ValueError("invalid accepted steering message")
        with self._append_lock():
            if self.client_delivery_outcome(delivery_id) is not None:
                raise ValueError(f"client delivery already exists: {delivery_id}")
            rows: list[tuple[str, dict[str, object]]] = [
                (
                    "client_delivery",
                    {
                        "delivery_id": delivery_id,
                        "method": method,
                        "status": status,
                        "outcome": dict(outcome),
                        **({"accepted_message": accepted_message.to_dict()} if accepted_message is not None else {}),
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
        accepted_message: Message | None = None,
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
            accepted_message=accepted_message,
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

    def pending_client_steering(self: ConversationStore) -> ClientSteeringBatch | None:
        """Snapshot steering to include in one provider request."""

        if not self._client_steering:
            return None
        return ClientSteeringBatch(tuple(self._client_steering))

    async def deliver_client_steering(
        self: ConversationStore, batch: ClientSteeringBatch
    ) -> None:
        """Publish a dispatched steering batch as one durable append."""

        await self._to_thread_durable(self._deliver_client_steering, batch)
        for expected in batch._queued:
            if not self._client_steering or self._client_steering[0] != expected:
                raise RuntimeError("client steering queue changed during delivery")
            self._client_steering.popleft()

    def _deliver_client_steering(
        self: ConversationStore, batch: ClientSteeringBatch
    ) -> None:
        with self._append_lock():
            rows: list[tuple[str, dict[str, object]]] = []
            for queued in batch._queued:
                delivery_id = queued.delivery_id
                if (
                    delivery_id is not None
                    and self.client_delivery_outcome(delivery_id) != "queued"
                ):
                    raise ValueError(f"client delivery is not queued: {delivery_id}")
                rows.append(
                    (
                        "message",
                        {
                            "message": (
                                tag_client_delivery(queued.message, delivery_id)
                                if delivery_id is not None
                                else queued.message
                            ).to_dict()
                        },
                    )
                )
                if delivery_id is not None:
                    rows.append(_transition_row(delivery_id, "delivered"))
            if rows:
                self._append_many_unlocked(rows)

    def drop_client_steering(self: ConversationStore, reason: DeliveryDropReason) -> int:
        """Durably drop every queued delivery before releasing queue ownership."""

        queued = tuple(self._client_steering)
        self._drop_client_steering(queued, reason)
        self._client_steering.clear()
        return len(queued)

    def _drop_client_steering(
        self: ConversationStore,
        queued: tuple[_QueuedSteering, ...],
        reason: DeliveryDropReason,
    ) -> list[ConversationEntry]:
        delivery_ids = [
            item.delivery_id for item in queued if item.delivery_id is not None
        ]
        if not delivery_ids:
            return []
        with self._append_lock():
            rows = [
                _transition_row(delivery_id, "dropped", reason)
                for delivery_id in delivery_ids
                if self.client_delivery_outcome(delivery_id) == "queued"
            ]
            if not rows:
                return []
            return [
                self._snapshot_entry(entry)
                for entry in self._append_many_unlocked(rows)
            ]

    async def drop_client_steering_async(
        self: ConversationStore, reason: DeliveryDropReason
    ) -> tuple[tuple[ConversationEntry, Message], ...]:
        """Commit queued drops off-loop, then release their queue ownership."""

        queued = tuple(self._client_steering)
        entries = await self._to_thread_durable(
            self._drop_client_steering, queued, reason
        )
        by_id = {entry.data["delivery_id"]: entry for entry in entries}
        committed = tuple(
            (by_id[item.delivery_id], item.message)
            for item in queued
            if item.delivery_id in by_id
        )
        for expected in queued:
            if not self._client_steering or self._client_steering[0] != expected:
                raise RuntimeError("client steering queue changed during drop")
            self._client_steering.popleft()
        return committed


def image_payload_digest(
    text: str, images: list[tuple[str, str, bytes]]
) -> str:
    """Fingerprint the semantic payload of one ``send_images`` request."""

    digest = hashlib.sha256()
    values = [text.encode()]
    for name, media_type, data in images:
        values.extend((name.encode(), media_type.encode(), data))
    for value in values:
        digest.update(len(value).to_bytes(8, "big"))
        digest.update(value)
    return digest.hexdigest()


def _message_image_payload_digest(message: Message) -> str | None:
    if (
        message.role is not MessageRole.USER
        or not message.content
        or not isinstance(message.content[0], TextContent)
    ):
        return None
    images: list[tuple[str, str, bytes]] = []
    for block in message.content[1:]:
        if not isinstance(block, ImageContent) or block.path is None:
            return None
        try:
            data = base64.b64decode(block.data, validate=True)
        except (ValueError, binascii.Error):
            return None
        images.append((Path(block.path).name, block.mime_type, data))
    if not images:
        return None
    return image_payload_digest(message.content[0].text, images)


def _delivery_digest(delivery_id: str) -> bytes:
    return hashlib.sha256(delivery_id.encode("ascii")).digest()[:16]


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
