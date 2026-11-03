"""Logical serve turn and approval lifetime."""

from __future__ import annotations

import asyncio
import contextlib
import functools
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ..core.approval import ApprovalPolicy, ApprovalRequest
from ..core.store import ConversationStore
from ..protocol.types import (
    Message,
    MessageOrigin,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
)
from . import abort_scope, event_contract, model_selection
from .approval_lifecycle import (
    ApprovalKey,
    ApprovalLifecycle,
    SessionHealth,
    approval_display_fields,
)
from .delivery import DeliveryRequests, PendingModelInputs, TurnOutcome
from .protocol import FrameWriteCancelled, ProtocolError, bounded

if TYPE_CHECKING:
    from .server import ZetaServer

USER_MESSAGE_MAX_BYTES = 262_144

NotificationSender = Callable[..., Awaitable[bool]]


@dataclass(frozen=True, slots=True)
class _ApprovalTarget:
    key: ApprovalKey
    store: ConversationStore
    store_request_id: str
    request: ApprovalRequest | None = None
    history_store: ConversationStore | None = None


@dataclass(slots=True)
class _ApprovalRecord:
    target: _ApprovalTarget
    wire_id: str
    generation: int
    token: int
    state: Literal["starting", "pending", "reserved"] = "starting"
    disposition: str = "active"
    request: ApprovalRequest | None = None
    history_published: bool = False
    published: bool = False
    settlement: asyncio.Task[object] | None = None

class NotificationPresentation:
    """Publish each durable notification at most once per serve session."""

    def __init__(self) -> None:
        self._presented: set[tuple[str, str]] = set()

    async def publish(
        self,
        session_id: str | None,
        notification: Mapping[str, object],
        protocol_version: str,
        send: NotificationSender,
    ) -> None:
        notification_id = notification.get("notification_id")
        key = (
            (session_id, notification_id)
            if isinstance(session_id, str) and isinstance(notification_id, str)
            else None
        )
        if key is not None and key in self._presented:
            return
        if key is not None:
            self._presented.add(key)
        event_name = event_contract.notification_wire_event(
            notification.get("kind", "agent_completion"), protocol_version
        )
        try:
            written = await send(event_name, session_id, data=dict(notification))
        except FrameWriteCancelled as cancelled:
            if not cancelled.written and key is not None:
                self._presented.discard(key)
            raise
        except BaseException:
            if key is not None:
                self._presented.discard(key)
            raise
        if not written and key is not None:
            self._presented.discard(key)

class ServeControl:
    """Own the active semantic lifetime independently of physical clients."""

    def __init__(self, server: ZetaServer) -> None:
        self.server = server
        self._turn_task: asyncio.Task[None] | None = None
        self._turn_cancel_outcome: TurnOutcome | None = None
        self._model_inputs = PendingModelInputs()
        self.deliveries = DeliveryRequests(self)
        self._approvals = ApprovalLifecycle()
        self._approval_records: OrderedDict[ApprovalKey, _ApprovalRecord] = (
            OrderedDict()
        )
        self._approval_generation = 0
        self._approval_token = 0
        self.closing = False

    def start_turn(
        self,
        text: str,
        user_message: Message | None = None,
        *,
        origin: MessageOrigin = MessageOrigin.USER,
        persist_user_message: bool = True,
    ) -> None:
        self._turn_task = asyncio.create_task(
            self._run_turn(
                text,
                user_message,
                origin=origin,
                persist_user_message=persist_user_message,
            )
        )

    def unhealthy(self) -> bool:
        return self._approvals.state.health is not None

    def _reject_if_unhealthy(self) -> None:
        health = self._approvals.state.health
        if health is not None:
            raise ProtocolError(-32000, "approval persistence failed", {
                "code": health.code,
            })

    def request_turn_failure(self) -> None:
        """Request whole-session cancellation without joining teardown."""
        loop = self.server.runtime.loop
        if loop is not None:
            loop.abort(steering_drop_reason=None)
        task = self._turn_task
        if task is not None and not task.done():
            self._turn_cancel_outcome = TurnOutcome.SESSION_CANCELED

    def turn_busy(self) -> bool:
        loop = self.server.runtime.loop
        return (self._turn_task is not None and not self._turn_task.done()) or (
            loop is not None and loop.notification_turn_state != "idle"
        )

    async def require_idle(self) -> None:
        if self.turn_busy():
            raise ProtocolError(-32004, "a turn is already running")

    def model_input_registrar(self, features: frozenset[str]):
        return self._model_inputs.registrar(
            self.server.runtime.session_id,
            enabled="model_input_ids" in features,
        )

    def bind_policy(self, policy: ApprovalPolicy) -> None:
        """Bind one serve policy to this control for its complete lifetime."""

        policy.bind_serve_control(self)

    def pending_approvals_now(self) -> list[dict[str, object]]:
        """Return the loop-owned visible projection without touching a store."""

        return [
            {
                **self._approvals.observe(record.request),
                **approval_display_fields(record.request),
            }
            for record in self._approval_records.values()
            if record.state == "pending" and record.request is not None
        ]

    def pending_approval_labels(self) -> tuple[str, ...]:
        return tuple(
            f"{request.key} ({request.label or request.tool_call.name})"
            for record in self._approval_records.values()
            if record.state == "pending" and (request := record.request) is not None
        )

    def has_pending_approvals(self) -> bool:
        return any(
            record.state in {"starting", "pending", "reserved"}
            for record in self._approval_records.values()
        )

    async def pending_approvals(self) -> list[dict[str, object]]:
        """Build the legacy strict projection off-loop with unchanged fields."""

        return await self._pending_approvals_projection()

    async def _pending_approvals_projection(self) -> list[dict[str, object]]:
        policy = self.server.runtime.policy
        if policy is None or self.unhealthy():
            return []
        foreground = policy.store
        delegated = tuple(
            record.target
            for record in self._approval_records.values()
            if isinstance(record.target.key, tuple)
        )

        def strict() -> tuple[
            dict[str, tuple[ToolCall, str | None]],
            dict[ConversationStore, dict[str, tuple[ToolCall, str | None]]],
        ]:
            child_stores = {target.store for target in delegated}
            return foreground.approval_states(), {
                store: store.approval_states() for store in child_stores
            }

        foreground_states, child_states = await self._run_blocking(strict)
        requests = [
            policy.request_from_store(request_id, tool_call)
            for request_id, (tool_call, decision) in foreground_states.items()
            if decision is None
        ]
        for target in delegated:
            state = child_states[target.store].get(target.store_request_id)
            if state is not None and state[1] is None and target.request is not None:
                requests.append(target.request)
        return [
            {**self._approvals.observe(item), **approval_display_fields(item)}
            for item in requests
        ]

    def clear_approvals(self) -> None:
        self._approvals.clear()
        self._approval_records.clear()

    def session_replaced(self) -> None:
        self._approval_generation += 1
        self.clear_approvals()
        self._approvals.transition("replace")

    async def wait_for_approval_settlement(self) -> None:
        while tasks := [
            record.settlement
            for record in self._approval_records.values()
            if record.settlement is not None and not record.settlement.done()
        ]:
            await asyncio.gather(*tasks, return_exceptions=True)
    async def controller_claimed(self) -> None:
        await self._attach_pending_notifications()
    async def controller_released(self) -> None:
        await self._terminate_pending_approvals(suppress_write_errors=True)
        if self._turn_task is not None and not self._turn_task.done():
            loop = self.server.runtime.loop
            if loop is not None:
                loop.abort(steering_drop_reason=None)
            self._turn_cancel_outcome = TurnOutcome.DISCONNECTED
            self._turn_task.cancel()
            await asyncio.gather(self._turn_task, return_exceptions=True)
        self._turn_task = None
        self._approvals.prune_ended()
    async def close(self) -> None:
        self.closing = True
        await self.controller_released()
    async def user_message(
        self,
        text: str,
        mode: str,
        features: frozenset[str],
        attachments: list[dict[str, object]] | None = None,
        delivery_id: str | None = None,
    ) -> None:
        """Echo an accepted user message so observers can rebuild the transcript."""
        if "user_message_event" in features:
            fields: dict[str, object] = {
                "text": bounded(text, USER_MESSAGE_MAX_BYTES),
                "mode": mode,
                "attachments": attachments or [],
            }
            if delivery_id is not None and {
                "delivery_id", "user_message_delivery_id"
            } <= features:
                fields["delivery_id"] = delivery_id
            await self._notify(
                event_contract.AUXILIARY_WIRE_EVENTS[
                    event_contract.AuxiliaryEvent.USER_MESSAGE
                ],
                self.server.runtime.session_id,
                **fields,
            )
    async def _run_blocking(self, job: Callable[[], object]) -> object:
        if self.closing:
            return await asyncio.to_thread(job)
        registry = self.server.connections
        if registry is None:
            raise RuntimeError("serve control worker is unavailable")
        return await registry.run_blocking(job)
    def _approval_transition(self, event: str, record: _ApprovalRecord, disposition: str) -> None:
        self._approvals.transition(event, record=record, disposition=disposition)
    def _new_record(self, target: _ApprovalTarget) -> _ApprovalRecord:
        self._approval_token += 1
        record = _ApprovalRecord(
            target=target,
            wire_id=self._approvals.wire_id(target.key),
            generation=self._approval_generation,
            token=self._approval_token,
            request=target.request,
            disposition="abort" if self.closing else "active",
        )
        self._approval_records[target.key] = record
        return record
    def register_delegated(
        self,
        request: ApprovalRequest,
        store: ConversationStore,
        child_instance_id: str,
    ) -> None:
        key: ApprovalKey = (child_instance_id, request.request_id)
        exact = ApprovalRequest(
            request.request_id,
            request.tool_call,
            label=request.label,
            child_instance_id=child_instance_id,
            effective_cwd=request.effective_cwd,
            resolved_path=request.resolved_path,
            action=request.action,
        )
        history_store = (
            self.server.runtime.opened.store
            if self.server.runtime.opened is not None
            else None
        )
        target = _ApprovalTarget(key, store, request.request_id, exact, history_store)
        record = self._approval_records.get(key)
        if record is None:
            record = self._new_record(target)
        elif record.state == "starting":
            record.target = target
            record.request = exact
        else:
            return
        if record.disposition == "external":
            record.settlement = asyncio.create_task(self._finish_external(record))
        elif record.disposition != "active" or self.closing:
            reason = (
                "turn_canceled" if record.disposition == "abort" else record.disposition
            )
            self._approval_transition("reserve", record, reason)
            record.settlement = asyncio.create_task(
                self._settle_record(record, "abort", reason)
            )
    def cleanup_delegated(self, child_instance_id: str) -> Awaitable[object] | None:
        settlements: list[asyncio.Task[object]] = []
        for key, record in tuple(self._approval_records.items()):
            if not isinstance(key, tuple) or key[0] != child_instance_id:
                continue
            if self._reserve(record, "child_cleanup"):
                if record.state == "starting":
                    if record.settlement is not None:
                        settlements.append(record.settlement)
                    continue
                record.settlement = asyncio.create_task(
                    self._settle_record(record, "abort", "child_cleanup")
                )
                settlements.append(record.settlement)
        if not settlements:
            return None
        return asyncio.gather(*settlements, return_exceptions=True)

    def _reserve(self, record: _ApprovalRecord, decision: str) -> bool:
        if record.state == "starting":
            if record.disposition != "active":
                return False
            self._approval_transition("reserve_starting", record, decision)
            return True
        if record.state != "pending":
            return False
        self._approval_token += 1
        record.token = self._approval_token
        self._approval_transition("reserve", record, decision)
        return True

    async def resolve_approval(
        self, method: str, request_id: str, scope: str = "once"
    ) -> dict[str, object]:
        policy = self.server.runtime.policy
        loop = self.server.runtime.loop
        if policy is None or loop is None:
            raise ProtocolError(-32003, "no active session")
        self._reject_if_unhealthy()
        core_key = self._approvals.core_key(request_id)
        record = await self._record_for_decision(core_key, policy)
        if record is None or record.state != "pending" or record.request is None:
            raise ProtocolError(
                -32006, f"approval request not found or already resolved: {request_id}"
            )
        pending = record.request
        if scope == "always_tool" and pending.child_instance_id is not None:
            raise ProtocolError(
                -32602, "scope 'always_tool' is unavailable for delegated approvals"
            )
        if scope == "always_tool":
            try:
                pending.always_allow_rule()
            except ValueError as exc:
                raise ProtocolError(
                    -32602, f"cannot always allow approval request: {exc}"
                ) from exc
        decision = "allow" if method == "approve" else "deny"
        if not self._reserve(record, decision):
            raise ProtocolError(
                -32006, f"approval request not found or already resolved: {request_id}"
            )
        record.settlement = asyncio.create_task(
            self._settle_record(
                record,
                decision,
                grant=pending if scope == "always_tool" else None,
            )
        )
        try:
            resolution = await record.settlement
        except OSError as exc:
            raise ProtocolError(-32000, f"approval persistence failed: {exc}") from exc
        active = self.turn_busy()
        if (
            not active
            and isinstance(core_key, str)
            and loop.prepare_resume_pending_tool(core_key)
        ):
            if self.server.runtime.state is not None:
                self.server.runtime.state.tool_started()
            task = asyncio.create_task(self._resume_tool(core_key))
            self._turn_task = task
        confirmed = resolution["decision"]
        result: dict[str, object] = {
            "accepted": True,
            "request_id": request_id,
            "decision": "approve" if confirmed == "allow" else confirmed,
        }
        if scope != "once" and confirmed == decision:
            result["scope"] = scope
        if reason := resolution.get("reason"):
            result["reason"] = reason
        return result

    async def _record_for_decision(
        self, core_key: ApprovalKey, policy: ApprovalPolicy
    ) -> _ApprovalRecord | None:
        record = self._approval_records.get(core_key)
        if record is not None or not isinstance(core_key, str):
            return record
        record = self._new_record(_ApprovalTarget(core_key, policy.store, core_key))
        token = record.token
        generation = record.generation
        state = await self._run_blocking(
            lambda: record.target.store.approval_states().get(core_key)
        )
        current = self._approval_records.get(core_key)
        if (
            current is not record
            or token != record.token
            or generation != self._approval_generation
            or state is None
            or state[1] is not None
        ):
            self._approval_records.pop(core_key, None)
            return None
        if record.disposition != "active":
            self._approval_transition("reserve", record, record.disposition)
            await self._settle_record(record, "abort", record.disposition)
            return None
        record.request = policy.request_from_store(core_key, state[0])
        self._approval_transition("mark_pending", record, "active")
        record.published = True
        self._approvals.observe(record.request)
        return record

    async def _settle_record(
        self,
        record: _ApprovalRecord,
        decision: str,
        reason: str | None = None,
        *,
        grant: ApprovalRequest | None = None,
    ) -> dict[str, str]:
        token = record.token
        generation = record.generation
        target = record.target

        def resolve(outcome: str, outcome_reason: str | None) -> None:
            target.store.resolve_approval(
                target.store_request_id,
                outcome,
                **(
                    {"reason": outcome_reason}
                    if outcome == "abort" and outcome_reason
                    else {}
                ),
            )

        def persist() -> dict[str, str] | None:
            write_failed = False
            try:
                resolve(decision, reason)
            except OSError:
                write_failed = True
            try:
                resolution = target.store.approval_resolution_data(
                    target.store_request_id
                )
            except OSError:
                write_failed = True
                resolution = None
            if resolution is not None:
                return resolution
            if not write_failed:
                return None
            try:
                resolve("abort", reason or "turn_canceled")
                return target.store.approval_resolution_data(target.store_request_id)
            except OSError:
                return None

        resolution = await self._run_blocking(persist)
        current = self._approval_records.get(target.key)
        owns_record = (
            current is record
            and generation == self._approval_generation
            and token == record.token
        )
        if resolution is None:
            if owns_record:
                self._approvals.transition(
                    "fail",
                    health=SessionHealth(
                        "approval_persistence_failed",
                        "approval decision could not be persisted",
                    ),
                )
                with contextlib.suppress(ConnectionError, OSError):
                    await self._notify(
                        event_contract.STREAM_EVENT_WIRE_EVENTS[StreamEventType.ERROR],
                        self.server.runtime.session_id
                        if self.server.runtime.opened
                        else None,
                        error={
                            "code": "server_error",
                            "message": "approval decision could not be persisted",
                        },
                        data={"request_id": record.wire_id},
                    )
                self.request_turn_failure()
                self._approval_transition("health_abort", record, reason or "turn_canceled")
                record.settlement = None
            raise OSError("approval decision could not be persisted")
        if not owns_record:
            return resolution
        if grant is not None and resolution["decision"] == "allow":
            policy = self.server.runtime.policy
            if policy is not None:
                policy.remember_allow(grant)
        await self._finish_record(record, resolution)
        return resolution

    def request_approval_abort(
        self,
        request_id: str,
        *,
        child_instance_id: str | None = None,
        reason: str = "turn_canceled",
    ) -> asyncio.Task[dict[str, str] | None]:
        """Route a serve-bound cancellation through the settlement owner."""
        key: ApprovalKey = (
            (child_instance_id, request_id)
            if child_instance_id is not None
            else request_id
        )
        return asyncio.create_task(self._settle_canceled_approval(key, reason))
    async def _settle_canceled_approval(
        self, key: ApprovalKey, reason: str
    ) -> dict[str, str] | None:
        record = self._approval_records.get(key)
        if record is None:
            return None
        if (
            record.state == "reserved"
            and self._approvals.state.health is not None
            and (record.settlement is None or record.settlement.done())
        ):
            self._approval_transition("health_abort", record, reason)
        elif not self._reserve(record, reason):
            settlement = record.settlement
            if settlement is not None and settlement is not asyncio.current_task():
                with contextlib.suppress(OSError):
                    return await settlement
            return await self._run_blocking(
                lambda: record.target.store.approval_resolution_data(
                    record.target.store_request_id
                )
            )
        if record.state == "starting":
            settlement = record.settlement
            if settlement is not None and settlement is not asyncio.current_task():
                with contextlib.suppress(OSError):
                    await settlement
            return await self._run_blocking(
                lambda: record.target.store.approval_resolution_data(
                    record.target.store_request_id
                )
            )
        record.settlement = asyncio.create_task(
            self._settle_record(record, "abort", reason)
        )
        return await record.settlement
    async def _finish_external(self, record: _ApprovalRecord) -> None:
        resolution = await self._run_blocking(
            lambda: record.target.store.approval_resolution_data(
                record.target.store_request_id
            )
        )
        if resolution is not None:
            await self._finish_record(record, resolution)
        else:
            self._approval_records.pop(record.target.key, None)
    async def _finish_record(
        self, record: _ApprovalRecord, resolution: dict[str, str]
    ) -> None:
        data: dict[str, object] = {
            key: value for key, value in resolution.items() if key != "request_id"
        }
        payload = self._approvals.end(record.target.key, data=data)
        try:
            if payload is not None and record.history_published:
                assert record.target.history_store is not None
                await record.target.history_store.append_approval_history_async(
                    {"event": "approval_end", **payload}
                )
            if payload is not None and record.published:
                await self._notify(
                    event_contract.STREAM_EVENT_WIRE_EVENTS[
                        StreamEventType.TOOL_APPROVAL_END
                    ],
                    self.server.runtime.session_id
                    if self.server.runtime.opened
                    else None,
                    **payload,
                )
        finally:
            self._approval_records.pop(record.target.key, None)

    async def abort(
        self, scope: abort_scope.AbortScope = "session"
    ) -> dict[str, object]:
        session_id = self.server.runtime.session_id
        outcome = (
            TurnOutcome.FOREGROUND_CANCELED
            if scope == "foreground"
            else TurnOutcome.SESSION_CANCELED
        )
        turn_task = self._turn_task

        def mark_canceled() -> None:
            if turn_task is not None and not turn_task.done():
                self._turn_cancel_outcome = outcome

        aborted = await abort_scope.abort_active_turn(
            turn_task,
            scope=scope,
            loop=self.server.runtime.loop,
            terminate_approvals=functools.partial(
                self._terminate_pending_approvals, reason="turn_canceled"
            ),
            before_cancel=mark_canceled,
        )
        if aborted:
            await self._notify(
                event_contract.AUXILIARY_WIRE_EVENTS[
                    event_contract.AuxiliaryEvent.TURN_ABORTED
                ],
                session_id,
            )
        return {"aborted": aborted}

    async def _run_turn(
        self,
        text: str,
        user_message: Message | None = None,
        *,
        origin: MessageOrigin = MessageOrigin.USER,
        persist_user_message: bool = True,
    ) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None:
            return
        session_id = state.session_id
        state.turn_started()
        outcome = TurnOutcome.COMPLETED
        agent_end: StreamEvent | None = None
        try:
            async for event in loop.run_turn(
                text,
                origin=origin,
                user_message=user_message,
                persist_user_message=persist_user_message,
            ):
                if event.type is StreamEventType.ERROR:
                    outcome = TurnOutcome.FAILED
                if event.type is StreamEventType.AGENT_END:
                    agent_end = event
                else:
                    await self._event(event, session_id=session_id)
        except asyncio.CancelledError:
            outcome = self._turn_cancel_outcome or TurnOutcome.SESSION_CANCELED
            raise
        except Exception as exc:  # noqa: BLE001 - serialize all turn failures
            outcome = TurnOutcome.FAILED
            await self._notify(
                event_contract.STREAM_EVENT_WIRE_EVENTS[StreamEventType.ERROR],
                session_id,
                error={"code": "server_error", "message": str(exc)},
                data={},
            )
        finally:
            self.deliveries.finalize_turn(
                session_id, state, outcome, schedule_wake=agent_end is None
            )
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id)
            if outcome is TurnOutcome.COMPLETED:
                self._schedule_background_wake(session_id)

    def _schedule_background_wake(self, session_id: str) -> None:
        if self.closing:
            return
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None or state.session_id != session_id:
            return
        if not loop.schedule_notification_turn():
            return
        self._turn_task = asyncio.create_task(self._run_notification_turn(session_id))

    async def _run_notification_turn(self, session_id: str) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None or state.session_id != session_id:
            return
        outcome = TurnOutcome.FAILED
        started = False
        agent_end: StreamEvent | None = None
        try:
            await asyncio.sleep(0.01)
            state.turn_started()
            started = True
            success, agent_end = await self.server.turn_contexts.run_notification_turn(
                session_id, loop, self._event
            )
            outcome = TurnOutcome.COMPLETED if success else TurnOutcome.FAILED
        except asyncio.CancelledError:
            outcome = self._turn_cancel_outcome or TurnOutcome.SESSION_CANCELED
            raise
        except Exception as exc:  # noqa: BLE001 - serialize all turn failures
            outcome = TurnOutcome.FAILED
            await self._notify(
                event_contract.STREAM_EVENT_WIRE_EVENTS[StreamEventType.ERROR],
                session_id,
                error={"code": "server_error", "message": str(exc)},
                data={},
            )
        finally:
            if not started and loop.notification_turn_state == "scheduled":
                loop.cancel_scheduled_notification_turn()
            if (
                outcome is TurnOutcome.FOREGROUND_CANCELED
                and loop.notification_system_message() is not None
            ):
                self._schedule_background_wake(session_id)
            self.deliveries.finalize_turn(
                session_id, state, outcome, schedule_wake=agent_end is None
            )
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id)
            if outcome is TurnOutcome.COMPLETED:
                self._schedule_background_wake(session_id)

    async def _attach_pending_notifications(self) -> None:
        runtime = self.server.runtime
        loop = runtime.loop
        if loop is None or runtime.state is None:
            return
        if loop.notification_system_message() is not None:
            self._schedule_background_wake(runtime.session_id)

    async def _resume_tool(self, request_id: str) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None:
            return
        session_id = state.session_id
        event_tasks: list[asyncio.Task[None]] = []
        agent_end: StreamEvent | None = None

        def emit(event: StreamEvent) -> None:
            nonlocal agent_end
            if event.type is StreamEventType.AGENT_END:
                agent_end = event
            else:
                event_tasks.append(
                    asyncio.create_task(self._event(event, session_id=session_id))
                )

        outcome = TurnOutcome.COMPLETED
        try:
            await loop.resume_pending_tool(
                request_id,
                prepared=True,
                event_sink=emit,
            )
        except asyncio.CancelledError:
            outcome = self._turn_cancel_outcome or TurnOutcome.SESSION_CANCELED
            raise
        except BaseException:
            outcome = TurnOutcome.FAILED
            raise
        finally:
            if event_tasks:
                await asyncio.gather(*event_tasks, return_exceptions=True)
            self.deliveries.finalize_turn(
                session_id, state, outcome, schedule_wake=agent_end is None
            )
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id)
            self._schedule_background_wake(session_id)

    async def _event(
        self,
        event: StreamEvent,
        *,
        session_id: str | None = None,
        background: bool = False,
    ) -> None:
        state = self.server.runtime.state
        if session_id is None and not background and state is not None:
            session_id = state.session_id
        foreground = (
            not background
            and state is not None
            and session_id is not None
            and state.session_id == session_id
        )
        kind = event.type
        wire_event = event_contract.STREAM_EVENT_WIRE_EVENTS[kind]
        if kind not in {
            StreamEventType.TOOL_APPROVAL_START,
            StreamEventType.TOOL_APPROVAL_END,
        }:
            await self._end_stale_approvals(session_id)
        if kind is StreamEventType.MESSAGE_UPDATE:
            delta = event.delta
            stream_kind = "assistant"
            if isinstance(event.content, TextContent):
                delta = event.content.text
            elif event.content is not None:
                delta = getattr(event.content, "text", None)
                stream_kind = str(getattr(event.content, "type", "content"))
            if delta:
                await self._notify(
                    wire_event,
                    session_id,
                    delta=bounded(delta),
                    kind=stream_kind,
                )
            return
        if kind is StreamEventType.MESSAGE_END:
            if foreground and not event.data.get("truncated"):
                await model_selection.confirm(self.server.runtime)
            usage = event.data.get("usage")
            if foreground and isinstance(usage, Mapping) and usage:
                state.usage.update(usage)
                await self._notify(
                    event_contract.AUXILIARY_WIRE_EVENTS[
                        event_contract.AuxiliaryEvent.USAGE
                    ],
                    session_id,
                    usage=dict(usage),
                )
            if event.message is not None:
                await self._notify(
                    wire_event, session_id, **event_contract.assistant_message(event)
                )
            return
        if kind is StreamEventType.TOOL_EXECUTION_UPDATE:
            await self._notify(
                wire_event,
                session_id,
                **event_contract.tool_output(event),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_START:
            if foreground:
                state.tool_started()
            await self._notify(
                wire_event,
                session_id,
                **event_contract.tool_start(event),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_END:
            if foreground:
                state.tool_finished()
            await self._notify(
                wire_event,
                session_id,
                **event_contract.tool_end(event),
            )
            return
        if kind is StreamEventType.TOOL_APPROVAL_END:
            await self._end_approval(
                _approval_key(event), session_id, data=dict(event.data)
            )
            return
        if kind is StreamEventType.TOOL_APPROVAL_START:
            if foreground:
                state.tool_started()
            await self._start_approval(event, session_id, wire_event)
            return
        if kind is StreamEventType.AGENT_NOTIFICATION:
            await self.server.notification_presentations.publish(
                session_id, event.data, self._protocol_version(), self._notify
            )
            return
        if kind is StreamEventType.ERROR:
            error = (
                {"code": event.error.code, "message": event.error.message}
                if event.error
                else {"code": "unknown", "message": "unknown error"}
            )
            login_provider = None
            if foreground and event.error is not None:
                failed_provider = self.server.runtime.metadata.provider
                offers_login = event.error.provider_error and (
                    event.error.code == "auth_error" or event.error.status_code == 401
                )
                recovered = await model_selection.recover_async(
                    self.server.runtime, event.error
                )
                if self._protocol_version() != "1.0":
                    error = recovered
                    if offers_login:
                        login_provider = failed_provider
            await self._notify(
                wire_event,
                session_id,
                error=error,
                data={
                    **event.data,
                    **({"login_provider": login_provider} if login_provider else {}),
                },
            )
            return
        if kind is StreamEventType.COMPACTION_START:
            await self._notify(wire_event, session_id, data=dict(event.data))
            return
        if kind is StreamEventType.COMPACTION_END:
            await self._notify(wire_event, session_id, data=dict(event.data))
            return
        if foreground and kind is StreamEventType.AGENT_END:
            state.turn_finished()
        await self._notify(wire_event, session_id, data=dict(event.data))

    async def _start_approval(
        self, event: StreamEvent, session_id: str | None, wire_event: str
    ) -> None:
        raw_request_id = event.tool_call.id if event.tool_call else ""
        child_id = event.data.get("agent_instance_id")
        key: ApprovalKey = (
            (child_id, raw_request_id)
            if isinstance(child_id, str) and raw_request_id
            else raw_request_id
        )
        policy = self.server.runtime.policy
        if policy is None:
            return
        record = self._approval_records.get(key)
        if record is None:
            record = self._new_record(
                _ApprovalTarget(key, policy.store, raw_request_id)
            )
        token = record.token
        generation = record.generation
        record.settlement = asyncio.current_task()
        target = record.target
        def lookup() -> tuple[ToolCall, str | None] | None:
            return target.store.approval_states().get(target.store_request_id)

        try:
            state = await self._run_blocking(lookup)
        except OSError:
            await self._settle_failed_start(record)
            return
        current = self._approval_records.get(key)
        if (
            current is not record
            or token != record.token
            or generation != self._approval_generation
        ):
            return
        await self._complete_approval_start(
            record, state, policy, token, generation, session_id, wire_event
        )

    async def _complete_approval_start(
        self,
        record: _ApprovalRecord,
        state: tuple[ToolCall, str | None] | None,
        policy: ApprovalPolicy,
        token: int,
        generation: int,
        session_id: str | None,
        wire_event: str,
    ) -> None:
        target = record.target
        if state is None:
            self._approval_records.pop(target.key, None)
            await self._notify(
                event_contract.STREAM_EVENT_WIRE_EVENTS[StreamEventType.ERROR],
                session_id,
                error={
                    "code": "approval_context_missing",
                    "message": "approval request is unavailable; denying live approval",
                },
                data={"request_id": target.store_request_id},
            )
            return
        if state[1] is not None:
            self._approval_records.pop(target.key, None)
            return
        request = target.request or policy.request_from_store(
            target.store_request_id, state[0]
        )
        record.request = request
        payload = self._approvals.observe(
            request, fields=approval_display_fields(request)
        )
        if not await self._write_approval_start_history(record, payload):
            return
        if (
            self._approval_records.get(target.key) is not record
            or token != record.token
            or generation != self._approval_generation
        ):
            return
        if record.disposition == "external":
            resolution = await self._run_blocking(
                lambda: target.store.approval_resolution_data(target.store_request_id)
            )
            if resolution is not None:
                await self._finish_record(record, resolution)
            else:
                self._approval_records.pop(target.key, None)
            return
        if record.disposition != "active":
            self._approval_transition("reserve", record, record.disposition)
            record.settlement = asyncio.create_task(
                self._settle_record(record, "abort", record.disposition)
            )
            await record.settlement
            return
        self._approval_transition("mark_pending", record, "active")
        record.settlement, record.published = None, True
        await self._notify(wire_event, session_id, **payload)

    async def _write_approval_start_history(
        self, record: _ApprovalRecord, payload: dict[str, object]
    ) -> bool:
        target = record.target
        if not isinstance(target.key, tuple) or target.history_store is None:
            return True
        try:
            await target.history_store.append_approval_history_async(
                {"event": "approval_request", **payload}
            )
        except OSError:
            await self._settle_failed_start(record)
            return False
        record.history_published = True
        return True

    async def _settle_failed_start(self, record: _ApprovalRecord) -> None:
        if self._approval_records.get(record.target.key) is not record:
            return
        reason = (
            record.disposition
            if record.disposition not in {"active", "external"}
            else "turn_canceled"
        )
        self._approval_transition("reserve", record, reason)
        record.settlement = asyncio.create_task(
            self._settle_record(record, "abort", reason)
        )
        await record.settlement

    def _protocol_version(self) -> str:
        connections = self.server.connections
        controller = (
            connections.current_controller() if connections is not None else None
        )
        return controller.protocol_version if controller is not None else "1.0"

    def _publish_background_event(self, session_id: str, event: StreamEvent) -> None:
        """Schedule child events from the loop's synchronous sink."""
        if not self.closing:
            asyncio.create_task(
                self._event(event, session_id=session_id, background=True)
            )

    def _publish_memory_notice(self, session_id: str, message: str) -> None:
        """Publish updates only when the client negotiated the optional feature."""
        controller = (
            self.server.connections.current_controller()
            if self.server.connections
            else None
        )
        if (
            not self.closing
            and controller is not None
            and "memory_updated" in controller.features
        ):
            asyncio.create_task(
                self._notify(
                    event_contract.AUXILIARY_WIRE_EVENTS[
                        event_contract.AuxiliaryEvent.MEMORY_UPDATED
                    ],
                    session_id,
                    message=message,
                )
            )

    async def _notify(
        self, event: str, session_id: str | None = None, **fields: object
    ) -> bool:
        if session_id is None and self.server.runtime.opened is not None:
            session_id = self.server.runtime.session_id
        connections = self.server.connections
        if connections is None:
            return False
        return await connections.publish_controller(event, session_id, **fields)

    async def _end_approval(
        self,
        key: ApprovalKey,
        session_id: str | None,
        *,
        data: dict[str, object] | None = None,
        suppress_write_errors: bool = False,
    ) -> None:
        del session_id, data, suppress_write_errors
        record = self._approval_records.get(key)
        if record is None:
            return
        if record.state == "starting":
            if record.disposition == "active":
                self._approval_transition("externalize", record, "external")
            return
        if record.state != "pending":
            return
        resolution = await self._run_blocking(
            lambda: record.target.store.approval_resolution_data(
                record.target.store_request_id
            )
        )
        if resolution is not None:
            await self._finish_record(record, resolution)

    async def _end_stale_approvals(self, session_id: str | None) -> None:
        del session_id
        for record in tuple(self._approval_records.values()):
            if record.state != "pending":
                continue
            resolution = await self._run_blocking(
                lambda record=record: record.target.store.approval_resolution_data(
                    record.target.store_request_id
                )
            )
            if resolution is not None:
                await self._finish_record(record, resolution)

    async def _terminate_pending_approvals(
        self,
        *,
        foreground_only: bool = False,
        suppress_write_errors: bool = False,
        reason: str = "controller_disconnected",
    ) -> None:
        await self._terminate_approval_records(
            foreground_only=foreground_only,
            suppress_write_errors=suppress_write_errors,
            reason=reason,
        )

    async def _terminate_approval_records(
        self,
        *,
        foreground_only: bool,
        suppress_write_errors: bool,
        reason: str,
    ) -> None:
        settlements = [
            settlement
            for key, record in tuple(self._approval_records.items())
            if not foreground_only or isinstance(key, str)
            if (settlement := self._reserve_termination(record, reason)) is not None
        ]
        if settlements:
            results = await asyncio.gather(*settlements, return_exceptions=True)
            error = next(
                (result for result in results if isinstance(result, BaseException)),
                None,
            )
            if error is not None and not suppress_write_errors:
                raise error
        await self.wait_for_approval_settlement()

    def _reserve_termination(
        self, record: _ApprovalRecord, reason: str
    ) -> asyncio.Task[object] | None:
        if not self._reserve(record, reason) or record.state == "starting":
            return None
        record.settlement = asyncio.create_task(
            self._settle_record(record, "abort", reason)
        )
        return record.settlement

def _approval_key(event: StreamEvent) -> ApprovalKey:
    raw_request_id = event.tool_call.id if event.tool_call is not None else ""
    child_id = event.data.get("agent_instance_id")
    if isinstance(child_id, str) and raw_request_id:
        return child_id, raw_request_id
    return raw_request_id
