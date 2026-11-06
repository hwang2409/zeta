"""Negotiated idempotent delivery for serve send and steer requests."""

from __future__ import annotations

import asyncio
import secrets
from collections.abc import Callable, Coroutine, Mapping
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
from . import event_contract
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
        control.reject_client_turn_if_detached()
        control._reject_if_unhealthy()
        if control.turn_busy():
            raise ProtocolError(-32004, "a turn is already running")
        value = control._model_inputs.resolve(
            runtime.session_id,
            params,
            enabled="model_input_ids" in features,
        )
        if delivery_id is None:
            await control.user_message(value.display_text, "send", features)
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

        canceled_after_commit = False
        try:
            await runtime.opened.store.append_client_delivery_async(
                delivery_id,
                "send",
                "delivered",
                outcome,
                message=message,
            )
        except asyncio.CancelledError:
            if runtime.opened.store.client_delivery(delivery_id) is None:
                raise
            canceled_after_commit = True

        await _deliver_after_commit(
            lambda: control.user_message(
                value.display_text, "send", features, delivery_id=delivery_id
            ),
            lambda: control.start_turn(
                value.text,
                origin=value.origin,
                user_message=message,
                persist_user_message=False,
            ),
            canceled=canceled_after_commit,
        )
        return outcome

    async def send_images(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        from . import ergonomics

        control = self._control
        runtime = control.server.runtime
        if runtime.loop is None or runtime.opened is None:
            raise ProtocolError(-32003, "no active session")
        delivery_id = self._delivery_id(
            params, features, feature="send_images_delivery_id"
        )
        payload_digest = (
            ergonomics.image_payload_digest(params) if delivery_id is not None else None
        )
        duplicate = self._duplicate(delivery_id, payload_digest=payload_digest)
        if duplicate is not None:
            return duplicate
        control.reject_client_turn_if_detached()
        control._reject_if_unhealthy()
        if control.turn_busy():
            raise ProtocolError(-32004, "a turn is already running")
        ergonomics.require_mutable(runtime, control.has_pending_approvals())
        message = await asyncio.to_thread(ergonomics.image_message, runtime, params)
        text = params.get("text", "")
        assert isinstance(text, str)
        attachments = ergonomics.attachment_summary(message)
        if delivery_id is None:
            await control.user_message(text, "send", features, attachments)
            control.start_turn(text, message)
            return {"accepted": True, "session_id": runtime.session_id}

        tagged = tag_client_delivery(message, delivery_id)
        outcome: dict[str, object] = {
            "accepted": True,
            "session_id": runtime.session_id,
        }
        canceled_after_commit = False
        try:
            await runtime.opened.store.append_client_delivery_async(
                delivery_id,
                "send",
                "delivered",
                outcome,
                message=tagged,
            )
        except asyncio.CancelledError:
            if runtime.opened.store.client_delivery(delivery_id) is None:
                raise
            canceled_after_commit = True

        await _deliver_after_commit(
            lambda: control.user_message(
                text,
                "send",
                features,
                attachments,
                delivery_id=delivery_id,
            ),
            lambda: control.start_turn(
                text,
                tagged,
                persist_user_message=False,
            ),
            canceled=canceled_after_commit,
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
        requested_delivery_id = self._delivery_id(params, features)
        duplicate = self._duplicate(requested_delivery_id)
        if duplicate is not None:
            return duplicate
        control._reject_if_unhealthy()
        if not control.turn_busy():
            raise ProtocolError(-32005, "no turn is running")
        text = _required_string(params, "text")
        message = with_message_origin(
            Message(MessageRole.USER, [TextContent(text)]), MessageOrigin.USER
        )
        delivery_id = requested_delivery_id or secrets.token_hex(16)
        outcome: dict[str, object] = {"accepted": True}
        tagged = tag_client_delivery(message, delivery_id)
        canceled_after_commit = False
        try:
            await runtime.opened.store.append_client_delivery_async(
                delivery_id,
                "steer",
                "queued",
                outcome,
                accepted_message=message,
            )
        except asyncio.CancelledError:
            if runtime.opened.store.client_delivery(delivery_id) is None:
                raise
            canceled_after_commit = True

        await _deliver_after_commit(
            lambda: control.user_message(
                text, "steer", features, delivery_id=requested_delivery_id
            ),
            lambda: loop.steer(tagged),
            canceled=canceled_after_commit,
        )
        return outcome

    async def finalize_turn(
        self,
        session_id: str,
        state: SessionState,
        outcome: TurnOutcome,
        *,
        schedule_wake: bool,
    ) -> None:
        """Finish durable lifecycle work despite cancellation of the turn task."""

        owner_task = asyncio.current_task()
        task = asyncio.create_task(
            self._finalize_turn(
                session_id,
                state,
                outcome,
                schedule_wake=schedule_wake,
                owner_task=owner_task,
            )
        )
        while not task.done():
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                pass
        task.result()

    async def _finalize_turn(
        self,
        session_id: str,
        state: SessionState,
        outcome: TurnOutcome,
        *,
        schedule_wake: bool,
        owner_task: asyncio.Task[Any] | None,
    ) -> None:
        control = self._control
        reason = {
            TurnOutcome.COMPLETED: "turn_end",
            TurnOutcome.FAILED: "failed",
            TurnOutcome.SESSION_CANCELED: "abort",
            TurnOutcome.DISCONNECTED: "disconnect",
        }.get(outcome)
        if reason is not None:
            dropped = await state.opened.store.drop_client_steering_async(reason)
            connections = control.server.connections
            if connections is not None:
                for entry, message in dropped:
                    row = event_contract.delivery_status_row(entry, message)
                    await connections.publish_controller(
                        "delivery_status",
                        session_id,
                        required_feature="session_history_fidelity",
                        **{key: value for key, value in row.items() if key != "event"},
                    )
        turn_reason = {
            TurnOutcome.COMPLETED: "completed",
            TurnOutcome.FOREGROUND_CANCELED: "user_stop",
            TurnOutcome.SESSION_CANCELED: "user_stop",
            TurnOutcome.FAILED: "error",
            TurnOutcome.DISCONNECTED: "disconnect",
        }[outcome]
        await state.opened.store.append_turn_end_async(turn_reason)
        control._approvals.prune_ended()
        if control.server.runtime.state is state:
            state.turn_finished()
        if control._turn_task is owner_task:
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
        self,
        params: dict[str, Any],
        features: frozenset[str],
        *,
        feature: str = "delivery_id",
    ) -> str | None:
        if "delivery_id" not in params:
            return None
        _require_feature(features, feature, "delivery_id")
        _require_feature(features, "delivery_id", "delivery_id")
        return _required_delivery_id(params)

    def _duplicate(
        self, delivery_id: str | None, *, payload_digest: str | None = None
    ) -> dict[str, object] | None:
        runtime = self._control.server.runtime
        if delivery_id is None or runtime.opened is None:
            return None
        store = runtime.opened.store
        delivery = store.client_delivery(delivery_id)
        if delivery is not None:
            if (
                payload_digest is not None
                and delivery.payload_digest != payload_digest
            ):
                raise ProtocolError(
                    -32010,
                    "delivery id already used for a different payload",
                    {"outcome": delivery.status},
                )
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


async def _deliver_after_commit(
    echo: Callable[[], Coroutine[Any, Any, None]],
    work: Callable[[], object],
    *,
    canceled: bool,
) -> None:
    """Run the best-effort echo, then the committed work exactly once.

    The work runs in this frame's ``finally``, never inside the echo task, so no
    cancel or echo failure can skip it. A caller already canceled at commit gets
    no echo. The first cancel during the echo reaches the echo task once, so the
    bounded frame-write abort runs; later cancels wait for it. A caller cancel is
    re-raised after the work; otherwise an echo error is.
    """
    error: BaseException | None = None
    try:
        if not canceled:
            task = asyncio.create_task(echo())
            while not task.done():
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    if not canceled:
                        canceled = True
                        task.cancel()
            if not task.cancelled():
                error = task.exception()
    finally:
        work()
    if canceled:
        raise asyncio.CancelledError
    if error is not None:
        raise error


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
