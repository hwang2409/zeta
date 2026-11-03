"""Negotiated idempotent delivery for serve send and steer requests."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ..core.store._client_delivery import (
    tag_client_delivery,
    valid_client_delivery_id,
)
from ..model_input import ModelInputEnvelope
from ..protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    user_message_for_turn,
    with_message_origin,
)
from .protocol import ProtocolError

if TYPE_CHECKING:
    from .control import ServeControl
    from .runtime import SessionState


class TurnOutcome(StrEnum):
    COMPLETED = "completed"
    FAILED = "failed"
    FOREGROUND_CANCELED = "foreground-canceled"
    SESSION_CANCELED = "session-canceled"
    DISCONNECTED = "disconnected"


class DeliveryRequests:
    """Own feature gating, persistence, deduplication, and status queries."""

    def __init__(self, control: ServeControl) -> None:
        self._control = control

    async def send(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        control = self._control
        runtime = control.server.runtime
        if runtime.loop is None or runtime.opened is None:
            raise ProtocolError(-32003, "no active session")
        delivery_id = self._delivery_id(params, features)
        duplicate = self._duplicate(delivery_id)
        if duplicate is not None:
            return duplicate
        if control.turn_busy():
            raise ProtocolError(-32004, "a turn is already running")
        value = control._model_inputs.resolve(
            runtime.session_id,
            params,
            enabled="model_input_ids" in features,
        )
        await control.user_message(
            value.display_text, "send", features, delivery_id=delivery_id
        )
        if delivery_id is None:
            control.start_turn(
                value.text,
                origin=value.origin,
                user_message=value.message,
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
            control.start_turn(
                value.text,
                origin=value.origin,
                user_message=message,
                persist_user_message=False,
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

    async def steer(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        control = self._control
        runtime = control.server.runtime
        loop = runtime.loop
        if loop is None or runtime.opened is None:
            raise ProtocolError(-32003, "no active session")
        delivery_id = self._delivery_id(params, features)
        duplicate = self._duplicate(delivery_id)
        if duplicate is not None:
            return duplicate
        if not control.turn_busy():
            raise ProtocolError(-32005, "no turn is running")
        text = _required_string(params, "text")
        message = with_message_origin(
            Message(MessageRole.USER, [TextContent(text)]), MessageOrigin.USER
        )
        await control.user_message(text, "steer", features, delivery_id=delivery_id)
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

    def finalize_turn(
        self,
        session_id: str,
        state: SessionState,
        outcome: TurnOutcome,
        *,
        schedule_wake: bool,
    ) -> None:
        """Apply all lifecycle effects for one explicit terminal outcome."""

        control = self._control
        reason = {
            TurnOutcome.COMPLETED: "turn_end",
            TurnOutcome.FAILED: "failed",
            TurnOutcome.SESSION_CANCELED: "abort",
            TurnOutcome.DISCONNECTED: "disconnect",
        }.get(outcome)
        if reason is not None:
            state.loop.drop_pending_steering(reason)
        control._approvals.prune_ended()
        if control.server.runtime.state is state:
            state.turn_finished()
        if control._turn_task is asyncio.current_task():
            control._turn_task = None
            control._turn_cancel_outcome = None
        if (
            schedule_wake
            and outcome is TurnOutcome.COMPLETED
            and control.server.runtime.state is state
            and state.loop.notification_system_message() is not None
        ):
            control._schedule_background_wake(session_id)

    def status(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        control = self._control
        _require_feature(features, "delivery_id", "delivery_status")
        delivery_id = _required_delivery_id(params)
        runtime = control.server.runtime
        if runtime.opened is None:
            raise ProtocolError(-32003, "no active session")
        store = runtime.opened.store
        delivery = store.client_delivery(delivery_id)
        if delivery is not None:
            return {"delivery_id": delivery_id, "status": delivery.status}
        outcome = store.client_delivery_outcome(delivery_id)
        if outcome is not None:
            return {
                "delivery_id": delivery_id,
                "status": "evicted",
                "outcome": outcome,
            }
        return {"delivery_id": delivery_id, "status": "unknown"}

    def _delivery_id(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> str | None:
        if "delivery_id" not in params:
            return None
        _require_feature(features, "delivery_id", "delivery_id")
        return _required_delivery_id(params)

    def _duplicate(self, delivery_id: str | None) -> dict[str, object] | None:
        runtime = self._control.server.runtime
        if delivery_id is None or runtime.opened is None:
            return None
        store = runtime.opened.store
        delivery = store.client_delivery(delivery_id)
        if delivery is not None:
            if delivery.status == "dropped":
                return {"accepted": False, "duplicate": True, "status": "dropped"}
            return {**delivery.outcome, "duplicate": True}
        outcome = store.client_delivery_outcome(delivery_id)
        if outcome is not None:
            raise ProtocolError(
                -32010,
                "delivery id already used",
                {"outcome": outcome},
            )
        return None


def _require_feature(
    features: frozenset[str], feature: str, param: str
) -> None:
    if feature not in features:
        raise ProtocolError(-32602, f"{param} requires negotiated feature '{feature}'")


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


@dataclass(frozen=True, slots=True)
class ResolvedModelInput:
    text: str
    display_text: str
    origin: MessageOrigin
    message: Message | None

class PendingModelInputs:
    """Issue and consume single-use IDs for server-generated envelopes."""
    def __init__(self, *, limit: int = 32) -> None:
        self._limit = limit
        self._inputs: dict[str, tuple[str, ModelInputEnvelope]] = {}
    def register(self, session_id: str, envelope: ModelInputEnvelope) -> str:
        while len(self._inputs) >= self._limit:
            del self._inputs[next(iter(self._inputs))]
        input_id = secrets.token_urlsafe(24)
        self._inputs[input_id] = (session_id, envelope)
        return input_id
    def registrar(self, session_id: str, *, enabled: bool) -> Callable[[ModelInputEnvelope], str] | None:
        if not enabled:
            return None
        return lambda envelope: self.register(session_id, envelope)
    def resolve(self, session_id: str, params: Mapping[str, object], *, enabled: bool) -> ResolvedModelInput:
        has_text, has_input_id = "text" in params, "input_id" in params
        if has_text == has_input_id:
            raise ValueError("send requires exactly one of text or input_id")
        if has_text:
            text = params["text"]
            if not isinstance(text, str) or not text.strip():
                raise ValueError("text must be a nonempty string")
            return ResolvedModelInput(text, text, MessageOrigin.USER, None)
        if not enabled:
            raise ValueError("input_id requires the negotiated model_input_ids feature")
        input_id = params["input_id"]
        if not isinstance(input_id, str) or not input_id:
            raise ValueError("input_id must be a nonempty string")
        pending = self._inputs.pop(input_id, None)
        if pending is None or pending[0] != session_id:
            raise ValueError("input_id is invalid or expired")
        envelope = pending[1]
        message = Message(MessageRole.USER, [TextContent(envelope.text)], metadata={MESSAGE_ORIGIN_METADATA: envelope.origin.value, "zeta.user_display_text": envelope.display_text})
        return ResolvedModelInput(envelope.text, envelope.display_text, envelope.origin, message)
