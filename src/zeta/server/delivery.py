"""Negotiated idempotent delivery for serve send and steer requests."""

from __future__ import annotations

import asyncio
import logging
import secrets
from collections.abc import Callable, Coroutine, Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ..core.store._client_delivery import (
    DeliveryDropReason,
    tag_client_delivery,
    valid_client_delivery_id,
)
from ..model_input import ModelInputEnvelope
from ..protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    StreamEventType,
    TextContent,
    user_message_for_turn,
    with_message_origin,
)
from . import event_contract
from .handlers import validation
from .protocol import ErrorDescriptor, ProtocolError
from .session_replay.rows import RowFacts

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .control import ServeControl
    from .runtime import SessionState
    from .session_controllers import SessionUse


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
        # orders every busy check and turn start (send, steer acceptance, and
        # the end-of-turn decision in continue_after_turn) so only one wins.
        self._turn_guard = asyncio.Lock()

    # Each request runs as one completion that owns the request's session use:
    # the commit, its echo, and its turn start finish even after the request is
    # cancelled, and the record cannot close before they do.
    async def send(
        self, use: SessionUse, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        return await use.run(self._guarded(self._send, params, features))

    async def send_images(
        self, use: SessionUse, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        return await use.run(self._guarded(self._send_images, params, features))

    async def steer(
        self, use: SessionUse, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        return await use.run(self._steer(params, features))

    async def _guarded(
        self,
        work: Callable[[dict[str, Any], frozenset[str]], Coroutine[Any, Any, dict[str, object]]],
        params: dict[str, Any],
        features: frozenset[str],
    ) -> dict[str, object]:
        async with self._turn_guard:
            return await work(params, features)

    async def _send(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        control = self._control
        runtime = control.runtime
        if runtime.loop is None or runtime.opened is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        delivery_id = self._delivery_id(params, features)
        duplicate = self._duplicate(delivery_id)
        if duplicate is not None:
            return duplicate
        control.reject_client_turn_if_detached()
        control._reject_if_unhealthy()
        if control.turn_busy():
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY, "a turn is already running"
            )
        value = control._model_inputs.resolve(
            runtime.session_id,
            params,
            enabled="model_input_ids" in features,
        )
        if delivery_id is None:
            await control.user_message(
                value.display_text, "send", features, origin=value.origin
            )
            control.start_turn(
                value.text,
                origin=value.origin,
                user_message=value.message,
                allow_assistant_reset="assistant_reset" in features,
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
                allow_assistant_reset="assistant_reset" in features,
            ),
            canceled=canceled_after_commit,
        )
        return outcome

    async def _send_images(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        from .handlers import send_images

        control = self._control
        runtime = control.runtime
        if runtime.loop is None or runtime.opened is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        delivery_id = self._delivery_id(
            params, features, feature="send_images_delivery_id"
        )
        control.reject_client_turn_if_detached()
        control._reject_if_unhealthy()
        if control.turn_busy():
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY, "a turn is already running"
            )
        validation.require_mutable(runtime, control.has_pending_approvals())
        worker = asyncio.create_task(
            asyncio.to_thread(send_images.prepare_image_delivery, runtime, params)
        )
        try:
            # test-contract: seam request cancellation owns and cleans up this worker
            prepared = await asyncio.shield(worker)
        except asyncio.CancelledError:
            await _cleanup_canceled_image_delivery(worker, runtime)
            raise
        try:
            duplicate = self._duplicate(
                delivery_id, payload_digest=prepared.payload_digest
            )
        except ProtocolError:
            await asyncio.to_thread(
                send_images.cleanup_image_delivery, runtime, prepared
            )
            raise
        if duplicate is not None:
            await asyncio.to_thread(
                send_images.cleanup_image_delivery, runtime, prepared
            )
            return duplicate
        message = prepared.message
        text = params.get("text", "")
        assert isinstance(text, str)
        if delivery_id is None:
            try:
                await control.user_message(text, "send", features, prepared.attachments)
            except asyncio.CancelledError:
                await asyncio.to_thread(
                    send_images.cleanup_image_delivery, runtime, prepared
                )
                raise
            control.start_turn(
                text, message, allow_assistant_reset="assistant_reset" in features
            )
            return {"accepted": True, "session_id": runtime.session_id}

        tagged = tag_client_delivery(message, delivery_id, prepared.payload_digest)
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
                await asyncio.to_thread(
                    send_images.cleanup_image_delivery, runtime, prepared
                )
                raise
            canceled_after_commit = True

        await _deliver_after_commit(
            lambda: control.user_message(
                text,
                "send",
                features,
                prepared.attachments,
                delivery_id=delivery_id,
            ),
            lambda: control.start_turn(
                text,
                tagged,
                persist_user_message=False,
                allow_assistant_reset="assistant_reset" in features,
            ),
            canceled=canceled_after_commit,
        )
        return outcome

    async def _steer(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        control = self._control
        runtime = control.runtime
        loop = runtime.loop
        if loop is None or runtime.opened is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        requested_delivery_id = self._delivery_id(params, features)
        duplicate = self._duplicate(requested_delivery_id)
        if duplicate is not None:
            return duplicate
        control._reject_if_unhealthy()
        text = validation.required_string(params, "text")
        message = with_message_origin(
            Message(MessageRole.USER, [TextContent(text)]), MessageOrigin.USER
        )
        delivery_id = requested_delivery_id or secrets.token_hex(16)
        outcome: dict[str, object] = {"accepted": True}
        tagged = tag_client_delivery(message, delivery_id)
        canceled_after_commit = False
        async with self._turn_guard:
            if not control.turn_busy():
                raise ProtocolError(
                    ErrorDescriptor.STATE_UNAVAILABLE, "no turn is running"
                )
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
            loop.steer(tagged)

        await _deliver_after_commit(
            lambda: control.user_message(
                text, "steer", features, delivery_id=requested_delivery_id
            ),
            lambda: None,
            canceled=canceled_after_commit,
        )
        return outcome

    async def finalize_turn(
        self,
        session_id: str,
        state: SessionState,
        turn_id: str,
        outcome: TurnOutcome,
    ) -> None:
        """Finish durable lifecycle work despite cancellation of the turn task."""

        owner_task = asyncio.current_task()
        task = asyncio.create_task(
            self._finalize_turn(
                session_id,
                state,
                turn_id,
                outcome,
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
        turn_id: str,
        outcome: TurnOutcome,
        *,
        owner_task: asyncio.Task[Any] | None,
    ) -> None:
        # a completed turn keeps its unconsumed steering: continue_after_turn
        # starts the next turn with it once turn_end is published.
        reason = {
            TurnOutcome.FAILED: "failed",
            TurnOutcome.SESSION_CANCELED: "abort",
            TurnOutcome.DISCONNECTED: "disconnect",
        }.get(outcome)
        if reason is not None:
            await self._drop_steering(session_id, state, reason)
        turn_reason = {
            TurnOutcome.COMPLETED: "completed",
            TurnOutcome.FOREGROUND_CANCELED: "user_stop",
            TurnOutcome.SESSION_CANCELED: "user_stop",
            TurnOutcome.FAILED: "error",
            TurnOutcome.DISCONNECTED: "disconnect",
        }[outcome]
        await state.opened.store.append_turn_end_async(turn_id, turn_reason)
        self._release_turn(state, owner_task)

    async def finalize_or_fail(
        self,
        session_id: str,
        state: SessionState,
        turn_id: str,
        outcome: TurnOutcome,
    ) -> TurnOutcome:
        """Finalize the turn; if that raises, end it as failed so the client
        still gets an error, turn_end, and agent_end, and no steer stays queued."""
        try:
            await self.finalize_turn(session_id, state, turn_id, outcome)
            return outcome
        except Exception as exc:  # a raise here would wedge the session
            logger.exception(
                "serve turn finalization failed: session=%s turn=%s",
                session_id,
                turn_id,
            )
            error = {"code": "server_error", "message": str(exc)}
        try:
            await self._drop_steering(session_id, state, "failed")
        except Exception:  # the turn must still be released
            logger.exception(
                "dropping steering after failed finalization failed: session=%s",
                session_id,
            )
        finally:
            self._release_turn(state, asyncio.current_task())
        await self._control._notify(
            event_contract.STREAM_EVENT_WIRE_EVENTS[StreamEventType.ERROR],
            session_id,
            error=error,
            data={},
        )
        return TurnOutcome.FAILED

    def _release_turn(
        self, state: SessionState, owner_task: asyncio.Task[Any] | None
    ) -> None:
        control = self._control
        control._approvals.prune_ended()
        if control.runtime.state is state:
            state.turn_finished()
        if control._turn_task is owner_task:
            control._turn_handoff_task = owner_task
            control._turn_end_published = False
            control._turn_task = None
            control._turn_cancel_outcome = None

    async def continue_after_turn(
        self,
        session_id: str,
        state: SessionState,
        *,
        allow_assistant_reset: bool = True,
    ) -> None:
        """Answer steering a completed turn did not consume, else wake for notifications.

        Call last in the finished turn task, after its turn_end is published.
        The guard orders this decision against send and steer: a steer that
        passed its busy check is queued before this check, a later steer sees
        no running turn, and a send either starts the turn this check finds or
        sees the turn this decision starts and gets OPERATION_BUSY.
        """

        control = self._control
        async with self._turn_guard:
            running = control._turn_task
            if (
                running is not None
                and not running.done()
                and running is not asyncio.current_task()
            ):
                # a send started that turn under the guard; its first provider
                # request reads the queued steering, and its end wakes as needed.
                return
            steering = state.opened.store.pending_client_steering()
            if steering is None:
                control._schedule_background_wake(session_id)
                return
            if (
                not control.server.controllers.is_closing(control)
                and control.controller_present()
                and not control.client_turn_detached()
                and control.runtime.state is state
            ):
                # the next turn's first provider request drains the queue,
                # so each steer becomes delivered exactly as a mid-turn steer.
                control.start_turn(
                    "\n\n".join(
                        block.text
                        for message in steering.messages
                        for block in message.content
                        if isinstance(block, TextContent)
                    ),
                    persist_user_message=False,
                    allow_assistant_reset=allow_assistant_reset,
                )
                return
        # no client can see this turn: the server closes, the bindings left, or
        # the controller is gone with no reconnect grace.
        await self._drop_steering(session_id, state, "disconnect")

    async def _drop_steering(
        self, session_id: str, state: SessionState, reason: DeliveryDropReason
    ) -> None:
        async with self._turn_guard:
            dropped = await state.opened.store.drop_client_steering_async(reason)
        for entry, accepted_message in dropped:
            row = event_contract.delivery_status_row(entry, accepted_message)
            await self._control._notify(
                event_contract.AUXILIARY_WIRE_EVENTS[
                    event_contract.AuxiliaryEvent.DELIVERY_STATUS
                ],
                session_id,
                replay_row=RowFacts(seq=entry.seq),
                **{key: value for key, value in row.items() if key != "event"},
            )

    def status(
        self, params: dict[str, Any], features: frozenset[str]
    ) -> dict[str, object]:
        control = self._control
        _require_feature(features, "delivery_id", "delivery_status")
        delivery_id = _required_delivery_id(params)
        runtime = control.runtime
        if runtime.opened is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
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
        runtime = self._control.runtime
        if delivery_id is None or runtime.opened is None:
            return None
        store = runtime.opened.store
        delivery = store.client_delivery(delivery_id)
        if delivery is not None:
            if payload_digest is not None and delivery.payload_digest != payload_digest:
                raise ProtocolError(
                    ErrorDescriptor.DELIVERY_ID_EVICTED,
                    "delivery id already used for a different payload",
                    {"outcome": delivery.status},
                )
            if delivery.status == "dropped":
                return {"accepted": False, "duplicate": True, "status": "dropped"}
            return {**delivery.outcome, "duplicate": True}
        outcome = store.client_delivery_outcome(delivery_id)
        if outcome is not None:
            raise ProtocolError(
                ErrorDescriptor.DELIVERY_ID_EVICTED,
                "delivery id already used",
                {"outcome": outcome},
            )
        return None


async def _cleanup_canceled_image_delivery(
    worker: asyncio.Task[Any], runtime: Any
) -> None:
    from .handlers import send_images

    settled = asyncio.Event()
    worker.add_done_callback(lambda _: settled.set())
    while not settled.is_set():
        try:
            await asyncio.shield(settled.wait())
        except asyncio.CancelledError:
            continue
    if worker.cancelled() or worker.exception() is not None:
        return
    cleanup = asyncio.create_task(
        asyncio.to_thread(
            send_images.cleanup_image_delivery, runtime, worker.result()
        )
    )
    while not cleanup.done():
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            continue
    cleanup.result()


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


def _require_feature(features: frozenset[str], feature: str, param: str) -> None:
    if feature not in features:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            f"{param} requires negotiated feature '{feature}'",
        )


def _required_delivery_id(params: dict[str, Any]) -> str:
    value = params.get("delivery_id")
    if not valid_client_delivery_id(value):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
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

    def registrar(
        self, session_id: str, *, enabled: bool
    ) -> Callable[[ModelInputEnvelope], str] | None:
        if not enabled:
            return None
        return lambda envelope: self.register(session_id, envelope)

    def resolve(
        self, session_id: str, params: Mapping[str, object], *, enabled: bool
    ) -> ResolvedModelInput:
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
        message = Message(
            MessageRole.USER,
            [TextContent(envelope.text)],
            metadata={
                MESSAGE_ORIGIN_METADATA: envelope.origin.value,
                "zeta.user_display_text": envelope.display_text,
            },
        )
        return ResolvedModelInput(
            envelope.text, envelope.display_text, envelope.origin, message
        )
