"""Logical serve turn and approval lifetime."""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import TYPE_CHECKING

from ..core.approval import ApprovalPolicy, ApprovalRequest
from ..core.store import ConversationStore
from ..protocol.types import (
    ErrorInfo,
    Message,
    MessageOrigin,
    StreamEvent,
    StreamEventType,
    TextContent,
)
from . import abort_scope, event_contract, model_selection
from .approval_lifecycle import (
    ApprovalKey,
    ApprovalLifecycle,
    BeginGrace,
    Cancel,
    Decide,
    Expire,
    ExternalEnd,
    Ready,
    Reclaim,
    Start,
    Terminate,
)
from .delivery import DeliveryRequests, PendingModelInputs, TurnOutcome
from .host_commands import SessionHostCommands
from .protocol import FrameWriteCancelled, ProtocolError, bounded

if TYPE_CHECKING:
    from .server import ZetaServer

USER_MESSAGE_MAX_BYTES = 262_144

NotificationSender = Callable[..., Awaitable[bool]]


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
            written = await send(
                event_name,
                session_id,
                data=event_contract.notification_payload(notification),
            )
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
        self._turn_handoff_task: asyncio.Task[None] | None = None
        self._session_swap_active = False
        self._session_swap_id: str | None = None
        self._turn_cancel_outcome: TurnOutcome | None = None
        self._grace_task: asyncio.Task[None] | None = None
        self._grace_expiring = False
        self._model_inputs = PendingModelInputs()
        self.deliveries = DeliveryRequests(self)
        self.closing = False
        self._active_host_commands: SessionHostCommands | None = None
        self._host_binding_required = False
        self._approvals = ApprovalLifecycle(
            publish=lambda *args, **kwargs: self._notify(*args, **kwargs),
            request_turn_failure=self.request_turn_failure,
            policy=lambda: self.server.runtime.policy,
            session_id=lambda: (
                self.server.runtime.session_id
                if self.server.runtime.opened is not None
                else None
            ),
            parent_store=lambda: (
                self.server.runtime.opened.store
                if self.server.runtime.opened is not None
                else None
            ),
            offload=lambda job: self._run_blocking(job),
            is_closing=lambda: self.closing,
        )

    def start_turn(
        self,
        text: str,
        user_message: Message | None = None,
        *,
        origin: MessageOrigin = MessageOrigin.USER,
        persist_user_message: bool = True,
    ) -> None:
        host_commands = self._active_host_commands
        self._turn_task = asyncio.create_task(
            self._run_turn(
                text,
                user_message,
                origin=origin,
                persist_user_message=persist_user_message,
                host_commands=host_commands,
                turn_id=uuid.uuid4().hex,
            )
        )

    def unhealthy(self) -> bool:
        return self._approvals.state.health is not None

    def _reject_if_unhealthy(self) -> None:
        self._approvals.reject_if_unhealthy()

    def reject_client_turn_if_detached(self) -> None:
        if self._host_binding_required and self._active_host_commands is None:
            raise ProtocolError(
                -32003,
                "session is not attached",
                {"code": "host_binding_required"},
            )

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
        return (
            self._session_swap_active
            or (self._turn_task is not None and not self._turn_task.done())
            or (
                self._turn_handoff_task is not None
                and not self._turn_handoff_task.done()
            )
            or (loop is not None and loop.notification_turn_state != "idle")
        )

    async def require_idle(self) -> None:
        if self.turn_busy():
            raise ProtocolError(-32004, "a turn is already running")

    @contextmanager
    def reserve_session_swap(self) -> Iterator[None]:
        if self.turn_busy():
            raise ProtocolError(-32004, "a turn is already running")
        state = self.server.runtime.state
        session_id = None if state is None else state.session_id
        self._session_swap_active = True
        self._session_swap_id = session_id
        try:
            yield
        finally:
            self._session_swap_active = False
            self._session_swap_id = None
            if session_id is not None:
                self._schedule_background_wake(session_id)

    def model_input_registrar(self, features: frozenset[str]):
        return self._model_inputs.registrar(
            self.server.runtime.session_id,
            enabled="model_input_ids" in features,
        )

    def bind_policy(self, policy: ApprovalPolicy) -> None:
        """Bind one serve policy to this control for its complete lifetime."""

        policy.bind_serve_control(self)

    def pending_approvals_now(
        self,
        *,
        reconnect: bool = False,
        include_truncated: bool = False,
    ) -> list[dict[str, object]] | tuple[list[dict[str, object]], bool]:
        return self._approvals.pending_approvals_now(
            reconnect=reconnect, include_truncated=include_truncated
        )

    def pending_approval_labels(self) -> tuple[str, ...]:
        return self._approvals.pending_approval_labels()

    def has_pending_approvals(self) -> bool:
        return self._approvals.has_pending_approvals()

    async def pending_approvals(self) -> list[dict[str, object]]:
        return await self._approvals.pending_approvals()

    def clear_approvals(self) -> None:
        self._approvals.clear_approvals()

    def session_replaced(self) -> None:
        self._approvals.session_replaced()

    async def wait_for_approval_settlement(self) -> None:
        await self._approvals.wait_for_approval_settlement()

    async def controller_claimed(
        self, *, reconnect: bool = False, require_host_binding: bool = False
    ) -> None:
        self._host_binding_required |= require_host_binding
        task = self._grace_task
        if task is None:
            return
        if self._grace_expiring:
            await asyncio.shield(task)
            return
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._grace_task = None
        if reconnect:
            await self._approvals.transition(Reclaim())
        else:
            await self._approvals.transition(Expire())
            await self._disconnect_turn()

    async def controller_ready(self, *, reconnect: bool = False) -> None:
        if reconnect:
            await self._approvals.transition(Ready())
        await self._attach_pending_notifications()

    async def controller_released(self, *, reconnect: bool = False) -> None:
        self._active_host_commands = None
        if reconnect and self._grace_task is not None:
            await asyncio.shield(self._grace_task)
            return
        if reconnect and self._approvals.has_pending_approvals():
            await self._approvals.transition(BeginGrace())
            self._grace_task = asyncio.create_task(self._expire_grace())
            if self.server.approval_reconnect_grace == 0:
                await self._grace_task
            return
        await self._terminate_pending_approvals(suppress_write_errors=True)
        await self._disconnect_turn()

    async def _expire_grace(self) -> None:
        try:
            await asyncio.sleep(self.server.approval_reconnect_grace)
            self._grace_expiring = True
            expiry = asyncio.create_task(self._approvals.transition(Expire()))
            try:
                await asyncio.shield(expiry)
            except asyncio.CancelledError:
                await expiry
                raise
            await self._disconnect_turn()
        finally:
            self._grace_expiring = False
            if self._grace_task is asyncio.current_task():
                self._grace_task = None

    async def _disconnect_turn(self) -> None:
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
        task = self._grace_task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._grace_task = None
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
            if (
                delivery_id is not None
                and {"delivery_id", "user_message_delivery_id"} <= features
            ):
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

    def register_delegated(
        self, request: ApprovalRequest, store: ConversationStore, child_instance_id: str
    ) -> None:
        self._approvals.register_delegated(request, store, child_instance_id)

    def cleanup_delegated(self, child_instance_id: str) -> Awaitable[object] | None:
        return self._approvals.cleanup_delegated(child_instance_id)

    async def resolve_approval(
        self, method: str, request_id: str, scope: str = "once"
    ) -> dict[str, object]:
        loop = self.server.runtime.loop
        if loop is None:
            raise ProtocolError(-32003, "no active session")
        result, core_key = await self._approvals.transition(
            Decide(method, request_id, scope)
        )
        assert isinstance(result, dict)
        if (
            not self.turn_busy()
            and isinstance(core_key, str)
            and loop.prepare_resume_pending_tool(core_key)
        ):
            host_commands = self._active_host_commands
            if self.server.runtime.state is not None:
                self.server.runtime.state.tool_started()
            self._turn_task = asyncio.create_task(
                self._resume_tool(core_key, host_commands, turn_id=uuid.uuid4().hex)
            )
        return result

    def request_approval_abort(
        self,
        request_id: str,
        *,
        child_instance_id: str | None = None,
        reason: str = "turn_canceled",
    ) -> Awaitable[object]:
        return asyncio.create_task(
            self._approvals.transition(Cancel(request_id, child_instance_id, reason))
        )

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
            cancel_task=not any(
                isinstance(key, str) for key in self._approvals.active_keys()
            ),
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
        host_commands: SessionHostCommands | None = None,
        turn_id: str,
    ) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None:
            return
        session_id = state.session_id
        state.turn_started()
        outcome = TurnOutcome.COMPLETED
        terminal_events: list[StreamEvent] = []
        pending_turn_end: StreamEvent | None = None
        terminal_error: dict[str, object] | None = None
        agent_end: StreamEvent | None = None
        try:
            async for event in loop.run_turn(
                text,
                origin=origin,
                user_message=user_message,
                persist_user_message=persist_user_message,
                host_environment=None
                if host_commands is None
                else host_commands.environment,
                host_instruction=""
                if host_commands is None
                else host_commands.instruction,
            ):
                if pending_turn_end is not None:
                    if event.type is StreamEventType.AGENT_END:
                        terminal_events.append(pending_turn_end)
                    else:
                        await self._event(
                            pending_turn_end,
                            session_id=session_id,
                            turn_id=turn_id,
                        )
                    pending_turn_end = None
                if event.type is StreamEventType.TURN_END:
                    pending_turn_end = event
                elif event.type is StreamEventType.ERROR:
                    outcome = TurnOutcome.FAILED
                    terminal_events.append(event)
                elif event.type is StreamEventType.AGENT_END:
                    agent_end = event
                else:
                    await self._event(event, session_id=session_id, turn_id=turn_id)
        except asyncio.CancelledError:
            outcome = self._turn_cancel_outcome or TurnOutcome.SESSION_CANCELED
            raise
        except Exception as exc:  # noqa: BLE001 - serialize all turn failures
            outcome = TurnOutcome.FAILED
            terminal_error = {"code": "server_error", "message": str(exc)}
        finally:
            if self._turn_cancel_outcome is not None:
                outcome = self._turn_cancel_outcome
                terminal_events.clear()
                terminal_error = None
                agent_end = None
            await self.deliveries.finalize_turn(
                session_id,
                state,
                turn_id,
                outcome,
                schedule_wake=agent_end is None,
            )
        if terminal_error is not None:
            await self._notify(
                event_contract.STREAM_EVENT_WIRE_EVENTS[StreamEventType.ERROR],
                session_id,
                error=terminal_error,
                data={},
            )
        for event in terminal_events:
            await self._event(event, session_id=session_id, turn_id=turn_id)
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id, turn_id=turn_id)
            if outcome is TurnOutcome.COMPLETED:
                self._schedule_background_wake(session_id)

    def _schedule_background_wake(self, session_id: str) -> None:
        if self.closing or self._session_swap_id == session_id:
            return
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None or state.session_id != session_id:
            return
        host_commands = self._active_host_commands
        if (
            self._host_binding_required and host_commands is None
        ) or not loop.schedule_notification_turn():
            return
        self._turn_task = asyncio.create_task(
            self._run_notification_turn(
                session_id, host_commands, turn_id=uuid.uuid4().hex
            )
        )

    async def _run_notification_turn(
        self,
        session_id: str,
        host_commands: SessionHostCommands | None = None,
        *,
        turn_id: str,
    ) -> None:
        loop = self.server.runtime.loop
        state = self.server.runtime.state
        if loop is None or state is None or state.session_id != session_id:
            return
        outcome = TurnOutcome.FAILED
        started = False
        agent_end: StreamEvent | None = None
        terminal_events: list[StreamEvent] = []

        async def publish(
            event: StreamEvent,
            *,
            session_id: str | None = None,
            background: bool = False,
        ) -> None:
            if event.type in {StreamEventType.TURN_END, StreamEventType.ERROR}:
                terminal_events.append(event)
            else:
                await self._event(
                    event,
                    session_id=session_id,
                    background=background,
                    turn_id=turn_id,
                )

        try:
            await asyncio.sleep(0.01)
            state.turn_started()
            started = True
            success, agent_end = await self.server.turn_contexts.run_notification_turn(
                session_id,
                loop,
                publish,
                host_environment=None
                if host_commands is None
                else host_commands.environment,
                host_instruction=""
                if host_commands is None
                else host_commands.instruction,
            )
            outcome = TurnOutcome.COMPLETED if success else TurnOutcome.FAILED
        except asyncio.CancelledError:
            outcome = self._turn_cancel_outcome or TurnOutcome.SESSION_CANCELED
            raise
        except Exception as exc:  # noqa: BLE001 - serialize all turn failures
            outcome = TurnOutcome.FAILED
            terminal_events.append(
                StreamEvent(
                    StreamEventType.ERROR,
                    error=ErrorInfo("server_error", str(exc)),
                )
            )
        finally:
            if not started and loop.notification_turn_state == "scheduled":
                loop.cancel_scheduled_notification_turn()
            if (
                outcome is TurnOutcome.FOREGROUND_CANCELED
                and loop.notification_system_message() is not None
            ):
                self._schedule_background_wake(session_id)
            await self.deliveries.finalize_turn(
                session_id,
                state,
                turn_id,
                outcome,
                schedule_wake=agent_end is None,
            )
        for event in terminal_events:
            await self._event(event, session_id=session_id, turn_id=turn_id)
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id, turn_id=turn_id)
            if outcome is TurnOutcome.COMPLETED:
                self._schedule_background_wake(session_id)

    async def _attach_pending_notifications(self) -> None:
        runtime = self.server.runtime
        loop = runtime.loop
        if loop is None or runtime.state is None:
            return
        if loop.notification_system_message() is not None:
            self._schedule_background_wake(runtime.session_id)

    async def _resume_tool(
        self,
        request_id: str,
        host_commands: SessionHostCommands | None = None,
        *,
        turn_id: str,
    ) -> None:
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
                    asyncio.create_task(
                        self._event(event, session_id=session_id, turn_id=turn_id)
                    )
                )

        outcome = TurnOutcome.COMPLETED
        try:
            await loop.resume_pending_tool(
                request_id,
                prepared=True,
                event_sink=emit,
                host_environment=None
                if host_commands is None
                else host_commands.environment,
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
            await self.deliveries.finalize_turn(
                session_id,
                state,
                turn_id,
                outcome,
                schedule_wake=agent_end is None,
            )
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id, turn_id=turn_id)
            self._schedule_background_wake(session_id)

    async def _event(
        self,
        event: StreamEvent,
        *,
        session_id: str | None = None,
        background: bool = False,
        turn_id: str | None = None,
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
        connections = getattr(self.server, "connections", None)
        controller = (
            connections.controller_metadata() if connections is not None else None
        )
        fidelity = (
            controller is not None and "session_history_fidelity" in controller.features
        )
        if kind is StreamEventType.TURN_START:
            data = dict(event.data)
            if (
                controller is not None
                and "served_turn_end" in controller.features
                and turn_id is not None
            ):
                data["turn_id"] = turn_id
            await self._notify(wire_event, session_id, data=data)
            return
        if kind is StreamEventType.TOOL_EXECUTION_UPDATE:
            await self._notify(
                wire_event,
                session_id,
                **event_contract.tool_output(event, fidelity=fidelity),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_START:
            if foreground:
                state.tool_started()
            await self._notify(
                wire_event,
                session_id,
                **event_contract.tool_start(event, fidelity=fidelity),
            )
            return
        if kind is StreamEventType.TOOL_EXECUTION_END:
            if foreground:
                state.tool_finished()
            await self._notify(
                wire_event,
                session_id,
                **event_contract.tool_end(event, fidelity=fidelity),
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
        await self._approvals.transition(Start(event, session_id, wire_event))

    def _protocol_version(self) -> str:
        connections = self.server.connections
        controller = (
            connections.controller_metadata() if connections is not None else None
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
            self.server.connections.controller_metadata()
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
        data: dict[str, object],
        suppress_write_errors: bool = False,
    ) -> None:
        await self._approvals.transition(
            ExternalEnd(key, session_id, data, suppress_write_errors)
        )

    async def _end_stale_approvals(self, session_id: str | None) -> None:
        await self._approvals._end_stale_approvals(session_id)

    async def _terminate_pending_approvals(
        self,
        *,
        foreground_only: bool = False,
        suppress_write_errors: bool = False,
        reason: str = "controller_disconnected",
    ) -> None:
        await self._approvals.transition(
            Terminate(foreground_only, suppress_write_errors, reason)
        )


def _approval_key(event: StreamEvent) -> ApprovalKey:
    raw_request_id = event.tool_call.id if event.tool_call is not None else ""
    child_id = event.data.get("agent_instance_id")
    if isinstance(child_id, str) and raw_request_id:
        return child_id, raw_request_id
    return raw_request_id
