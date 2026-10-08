"""Negotiated idempotent delivery for serve send and steer requests."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from ..core.store._client_delivery import (
    tag_client_delivery,
    valid_client_delivery_id,
)
from ..protocol.types import (
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    user_message_for_turn,
    with_message_origin,
)
from .protocol import ProtocolError

if TYPE_CHECKING:
    from .server import _Client

class DeliveryRequests:
    """Own feature gating, persistence, deduplication, and status queries."""

    def __init__(self, client: _Client) -> None:
        self._client = client

    async def send(self, params: dict[str, Any]) -> dict[str, object]:
        client = self._client
        runtime = client.server.runtime
        if runtime.loop is None or runtime.opened is None:
            raise ProtocolError(-32003, "no active session")
        delivery_id = self._delivery_id(params)
        duplicate = self._duplicate(delivery_id)
        if duplicate is not None:
            return duplicate
        if client._turn_busy():
            raise ProtocolError(-32004, "a turn is already running")
        value = client._model_inputs.resolve(
            runtime.session_id,
            params,
            enabled="model_input_ids" in client.features,
        )
        await client._user_message(value.display_text, "send")
        if delivery_id is None:
            client._turn_task = asyncio.create_task(
                client._run_turn(
                    value.text,
                    origin=value.origin,
                    user_message=value.message,
                )
            )
            return {"accepted": True, "session_id": runtime.session_id}

        message = tag_client_delivery(
            user_message_for_turn(
                value.text,
                origin=value.origin,
                message=value.message,
            ),
            delivery_id,
        )
        outcome: dict[str, object] = {
            "accepted": True,
            "session_id": runtime.session_id,
        }

        def start_turn() -> None:
            client._turn_task = asyncio.create_task(
                client._run_turn(
                    value.text,
                    origin=value.origin,
                    user_message=message,
                    persist_user_message=False,
                )
            )

        await runtime.opened.store.append_client_delivery_async(
            delivery_id,
            "send",
            "delivered",
            outcome,
            message=message,
            on_persisted=start_turn,
        )
        return outcome

    async def steer(self, params: dict[str, Any]) -> dict[str, object]:
        client = self._client
        runtime = client.server.runtime
        loop = runtime.loop
        if loop is None or runtime.opened is None:
            raise ProtocolError(-32003, "no active session")
        delivery_id = self._delivery_id(params)
        duplicate = self._duplicate(delivery_id)
        if duplicate is not None:
            return duplicate
        if not client._turn_busy():
            raise ProtocolError(-32005, "no turn is running")
        text = _required_string(params, "text")
        message = with_message_origin(
            Message(MessageRole.USER, [TextContent(text)]), MessageOrigin.USER
        )
        await client._user_message(text, "steer")
        if delivery_id is None:
            loop.steer(message)
            return {"accepted": True}

        outcome: dict[str, object] = {"accepted": True}
        tagged = tag_client_delivery(message, delivery_id)
        await runtime.opened.store.append_client_delivery_async(
            delivery_id,
            "steer",
            "queued",
            outcome,
            on_persisted=lambda: loop.steer(tagged),
        )
        return outcome

    def status(self, params: dict[str, Any]) -> dict[str, object]:
        client = self._client
        client._require_feature("delivery_id", "delivery_status")
        delivery_id = _required_delivery_id(params)
        runtime = client.server.runtime
        if runtime.opened is None:
            raise ProtocolError(-32003, "no active session")
        delivery = runtime.opened.store.client_delivery(delivery_id)
        return {
            "delivery_id": delivery_id,
            "status": delivery.status if delivery is not None else "unknown",
        }

    def _delivery_id(self, params: dict[str, Any]) -> str | None:
        if "delivery_id" not in params:
            return None
        self._client._require_feature("delivery_id", "delivery_id")
        return _required_delivery_id(params)

    def _duplicate(self, delivery_id: str | None) -> dict[str, object] | None:
        runtime = self._client.server.runtime
        if delivery_id is None or runtime.opened is None:
            return None
        delivery = runtime.opened.store.client_delivery(delivery_id)
        if delivery is None:
            return None
        if delivery.status == "dropped":
            return {"accepted": False, "duplicate": True, "status": "dropped"}
        return {**delivery.outcome, "duplicate": True}


def _required_string(params: dict[str, Any], name: str) -> str:
    value = params.get(name)
    if not isinstance(value, str) or not value:
        raise ProtocolError(-32602, f"{name} must be a nonempty string")
    return value


def _required_delivery_id(params: dict[str, Any]) -> str:
    value = params.get("delivery_id")
    if not valid_client_delivery_id(value):
        raise ProtocolError(
            -32602,
            "delivery_id must be 1-128 ASCII letters, digits, '.', '_', ':', or '-'",
        )
    return value
