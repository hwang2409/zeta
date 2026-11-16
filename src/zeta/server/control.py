"""Logical serve turn and approval lifetime."""

from __future__ import annotations

import asyncio
import functools
import uuid
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import aclosing, contextmanager
from typing import TYPE_CHECKING, Any

from ..core.approval import (
    ApprovalPolicy,
    ApprovalRequest,
    PermissionSnapshot,
    SessionGrantChange,
    SessionGrantNotFound,
)
from ..core.commands.slash_registry import SlashLock
from ..core.permission_grants import (
    permission_snapshot,
    revoke_session_grants,
    session_grant_change,
)
from ..core.store import ConversationStore
from ..protocol.types import (
    ErrorInfo,
    Message,
    MessageOrigin,
    StreamEvent,
    StreamEventType,
    TextContent,
)
from ..runtime.foreground_events import ForegroundEventPublisher
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
from .protocol import (
    ApprovalDecision,
    ErrorDescriptor,
    EventDescriptor,
    FrameWriteCancelled,
    ProtocolError,
    bound_reconnect_approvals,
    bounded,
)
from .session_replay.in_flight import InFlightReply
from .session_replay.rows import RowFacts

if TYPE_CHECKING:
    from ..mcp import MCPServerConfig
    from .runtime import ServerRuntime
    from .server import ZetaServer

USER_MESSAGE_MAX_BYTES = 262_144
CONTROLLER_HANDOFF_DEADLINE_SECONDS = 10.0

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
        *,
        row_seq: int | None = None,
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
                replay_row=RowFacts(seq=row_seq),
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

    def __init__(
        self,
        server: ZetaServer,
        runtime: ServerRuntime,
        foreground_event_publisher: ForegroundEventPublisher,
    ) -> None:
        self.server = server
        self.runtime = runtime
        # this control's unfinished reply; a join reads the attached control's
        self.in_flight = InFlightReply()
        self.foreground_event_publisher = foreground_event_publisher
        self._turn_task: asyncio.Task[None] | None = None
        # The scheduled notification turn; a send may preempt it before it starts.
        self._notification_task: asyncio.Task[None] | None = None
        self._turn_handoff_task: asyncio.Task[None] | None = None
        self._turn_end_published = False
        self._notification_turn_started = False
        # a wake past this gate may be in setup before its turn_start; state
        # changes then count it as busy, and a running state change holds the gate.
        self._wake_admitted = False
        self._state_changes = 0
        self._no_state_change = asyncio.Event()
        self._no_state_change.set()
        self._session_swap_active = False
        self._session_swap_id: str | None = None
        self._turn_cancel_outcome: TurnOutcome | None = None
        self._grace_task: asyncio.Task[None] | None = None
        # the grace deadline; under a turn grace it also bounds held bound MCP
        # calls, so it outlives a hello reclaim until the bindings return.
        self._grace_deadline_task: asyncio.Task[None] | None = None
        self._controller_release_task: asyncio.Task[None] | None = None
        self._grace_expiring = False
        # a controller holds the slot; the end-of-turn step starts no turn without one.
        self._controller_attached = False
        self._model_inputs = PendingModelInputs()
        self.deliveries = DeliveryRequests(self)
        # Serve reads requests serially per connection and has one controller,
        # so this lock never contends today; status projects its holder.
        self.slash_lock = SlashLock(offload=self._run_blocking)
        self.closing = False
        self._active_host_commands: SessionHostCommands | None = None
        self._active_mcp_bindings: dict[str, dict[str, str]] = {}
        self._mcp_binding_servers: frozenset[str] = frozenset()
        self._mcp_binding_required = False
        self._host_binding_required = False
        self._approvals = ApprovalLifecycle(
            admit=lambda *args, **kwargs: self._admit_controller(*args, **kwargs),
            request_turn_failure=self.request_turn_failure,
            policy=lambda: self.runtime.policy,
            session_id=lambda: self.runtime.session_id
            if self.runtime.opened is not None
            else None,
            parent_store=lambda: self.runtime.opened.store
            if self.runtime.opened is not None
            else None,
            offload=lambda job: self._run_blocking(job),
            is_closing=lambda: self.closing,
            grant_published=self._publish_session_grant,
        )

    def start_turn(
        self,
        text: str,
        user_message: Message | None = None,
        *,
        origin: MessageOrigin = MessageOrigin.USER,
        persist_user_message: bool = True,
        allow_assistant_reset: bool = True,
    ) -> None:
        host_commands = self._active_host_commands
        loop = self.runtime.loop
        if (
            self._turn_end_published
            and loop is not None
            and loop.notification_turn_state != "idle"
        ):
            notification_task = self._notification_task
            loop.cancel_scheduled_notification_turn()
            if notification_task is not None and not notification_task.done():
                notification_task.cancel()
        self._turn_task = asyncio.create_task(
            self._run_turn(
                text,
                user_message,
                origin=origin,
                persist_user_message=persist_user_message,
                host_commands=host_commands,
                turn_id=uuid.uuid4().hex,
                allow_assistant_reset=allow_assistant_reset,
            )
        )
        self._turn_task.add_done_callback(self._turn_done)
        connections = self.server.connections
        session_id = self.runtime.session_id
        if connections is not None and session_id is not None:
            connections.close_deferred_replay_when_done(session_id, self._turn_task)

    def unhealthy(self) -> bool:
        return self._approvals.state.health is not None

    @property
    def attached(self) -> bool:
        """This pair is the client's session; a parked pair sends no frame."""
        return self.server.control is self

    def pair_state(self) -> str:
        state = self.runtime.state
        return (
            "idle"
            if self.unhealthy() or state is None
            else "running" if self.state_change_busy() else state.status
        )

    def parked_row(self) -> dict[str, object]:
        """This pair's row in ``status.parked_sessions``."""
        return {
            "session_id": self.runtime.session_id,
            "state": self.pair_state(),
            "pending_approvals": len(self.pending_approvals_now()),
        }

    def holds_work(self) -> bool:
        """Closing this pair now would cut work: a turn, a wake, an approval, a
        background child or task, or a task exit not yet delivered."""
        loop = self.runtime.loop
        if loop is None:
            return False
        return (
            any(
                task is not None and not task.done()
                for task in (self._turn_task, self._turn_handoff_task)
            )
            or loop.notification_turn_state != "idle"
            or self.has_pending_approvals()
            or loop.has_running_work
            # not ``receipt_pending``: it stays set after a failed delivery, so the
            # next open recovers the exit; closing the pair is the way to that retry
            or loop.tool_registry.background_tasks.exit_delivery_pending
        )

    def _turn_done(self, _task: asyncio.Task[None]) -> None:
        self._maybe_idle()

    def _maybe_idle(self) -> None:
        """Ask the server to report this parked pair's state and close it once
        idle; loop thread only."""
        self.server.parking.changed(self)

    def work_changed(self) -> None:
        """``_maybe_idle`` from any thread."""
        connections = self.server.connections
        if connections is None:
            return
        try:
            connections.loop.call_soon_threadsafe(self._maybe_idle)
        except RuntimeError:
            if not connections.loop.is_closed():
                raise

    def _reject_if_unhealthy(self) -> None:
        self._approvals.reject_if_unhealthy()

    def controller_present(self) -> bool:
        """A controller is attached, or one may still reclaim within the grace."""
        return self._controller_attached or (
            self._grace_task is not None and not self._grace_expiring
        )

    def client_turn_detached(self) -> bool:
        return (
            (self._host_binding_required and self._active_host_commands is None)
            or (self._mcp_binding_required and not self._active_mcp_bindings)
        )

    def reject_client_turn_if_detached(self) -> None:
        if self.client_turn_detached():
            raise ProtocolError(
                ErrorDescriptor.HOST_BINDING_REQUIRED,
                "session is not attached",
            )

    def request_turn_failure(self) -> None:
        """Request whole-session cancellation without joining teardown."""
        loop = self.runtime.loop
        if loop is not None:
            loop.abort(steering_drop_reason=None)
        task = self._turn_task
        if task is not None and not task.done():
            self._turn_cancel_outcome = TurnOutcome.SESSION_CANCELED

    def turn_busy(self) -> bool:
        loop = self.runtime.loop
        return (
            self._session_swap_active
            or (
                self._turn_task is not None
                and not self._turn_task.done()
                and (
                    self._turn_task is not self._notification_task
                    or loop is None
                    or loop.notification_turn_state == "idle"
                    or self._notification_turn_started
                    or not self._turn_end_published
                )
            )
            or (
                self._turn_handoff_task is not None
                and not self._turn_handoff_task.done()
                and not self._turn_end_published
            )
        )

    def state_change_busy(self) -> bool:
        """Report a turn a state-changing request must wait for."""
        task = self._turn_task
        return self.turn_busy() or (
            self._wake_admitted
            and task is not None
            and task is self._notification_task
            and not task.done()
        )

    @contextmanager
    def hold_wake_start(self) -> Iterator[None]:
        """Keep a scheduled wake from starting until this state change ends."""
        self._state_changes += 1
        self._no_state_change.clear()
        try:
            yield
        finally:
            self._state_changes -= 1
            if not self._state_changes:
                self._no_state_change.set()

    async def require_idle(self) -> None:
        if self.state_change_busy():
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY, "a turn is already running"
            )

    @contextmanager
    def reserve_session_swap(self) -> Iterator[None]:
        if self.state_change_busy() or (
            self._turn_handoff_task is not None
            and not self._turn_handoff_task.done()
        ):
            raise ProtocolError(
                ErrorDescriptor.OPERATION_BUSY, "a turn is already running"
            )
        state = self.runtime.state
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
            self.runtime.session_id,
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

    async def reconnect_baseline(self) -> tuple[list[dict[str, object]], bool]:
        """Pending approvals a reconnecting client sees: the loop-owned live records
        merged with the durable store projection, keyed by request_id so a live row
        wins on overlap (keeping its resolution_pending). The durable projection covers
        a resumed idle session with no in-memory records yet, and also any approval that
        is only durable while another is settling live. Bounds the merged rows the same
        way the live projection does."""

        live, _ = self._approvals.pending_approvals_now(
            reconnect=True, include_truncated=True
        )
        durable = await self._approvals.pending_approvals()
        seen = {row["request_id"] for row in live}
        merged = live + [row for row in durable if row["request_id"] not in seen]
        return bound_reconnect_approvals(merged)

    def clear_approvals(self) -> None:
        self._approvals.clear_approvals()

    def session_replaced(self) -> None:
        self._approvals.session_replaced()

    async def wait_for_approval_settlement(self, timeout: float | None = None) -> None:
        if timeout is None:
            await self._approvals.wait_for_approval_settlement()
        else:
            await self._approvals.wait_for_approval_settlement(timeout)

    def begin_controller_release(
        self, *, reconnect: bool = False, keep_turn: bool = False
    ) -> asyncio.Task[None]:
        previous = self._controller_release_task
        # Establish approval grace before the physical controller slot is released.
        operation = (
            None
            if previous is not None and not previous.done()
            else self._controller_release(reconnect, keep_turn=keep_turn)
        )

        async def release() -> None:
            if previous is not None:
                await asyncio.shield(previous)
            if operation is None:
                await self.controller_released(reconnect=reconnect)
            else:
                await operation

        task = asyncio.create_task(release())
        self._controller_release_task = task
        return task

    async def controller_claimed(
        self, *, reconnect: bool = False, require_host_binding: bool = False
    ) -> bool:
        release = self._controller_release_task
        if release is not None:
            try:
                await asyncio.wait_for(
                    asyncio.shield(release), CONTROLLER_HANDOFF_DEADLINE_SECONDS
                )
            except TimeoutError:
                return False
            if self._controller_release_task is release:
                self._controller_release_task = None
        self._host_binding_required |= require_host_binding
        self._controller_attached = True
        task = self._grace_task
        if task is None:
            return True
        if self._grace_expiring:
            await asyncio.shield(task)
            return True
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        self._grace_task = None
        if reconnect:
            await self._approvals.transition(Reclaim())
        else:
            self._end_grace_deadline()
            await self._approvals.transition(Expire())
            await self._disconnect_turn()
        return True

    async def reattach_mcp_bindings(
        self,
        headers: dict[str, dict[str, str]],
        configs: dict[str, MCPServerConfig],
    ) -> None:
        """Re-bind the active session's released MCP headers for a returning client."""
        loop = self.runtime.loop
        assert loop is not None
        await loop.rebind_mcp(headers, configs)
        self._end_grace_deadline()
        await loop._ensure_mcp_servers()
        self._active_mcp_bindings = headers

    async def controller_ready(self, *, reconnect: bool = False) -> None:
        if reconnect:
            await self._approvals.transition(Ready())
        await self._attach_pending_notifications()

    async def controller_released(self, *, reconnect: bool = False) -> None:
        await self._controller_release(reconnect)

    def _controller_release(
        self, reconnect: bool, *, keep_turn: bool = False
    ) -> Awaitable[None]:
        self._controller_attached = False
        self._active_host_commands = None
        had_mcp_bindings = bool(self._active_mcp_bindings)
        self._active_mcp_bindings = {}
        grace_running = reconnect and self._grace_task is not None
        begin_grace = not grace_running and reconnect and (
            self._approvals.has_pending_approvals()
            or (keep_turn and bool(self._grace_protected(self._notification_task)))
        )
        if begin_grace:
            self._end_grace_deadline()
            self._grace_deadline_task = asyncio.create_task(self._grace_deadline())
        # under a turn grace a bound MCP call waits for the client's return.
        hold = (
            self._grace_deadline_task
            if keep_turn and (begin_grace or grace_running)
            else None
        )

        async def clear_mcp() -> None:
            loop = self.runtime.loop
            if loop is not None:
                await loop.clear_mcp_bindings(hold=hold)
        clear_task = asyncio.create_task(clear_mcp()) if had_mcp_bindings else None
        if grace_running:

            async def wait_for_grace() -> None:
                if clear_task is not None:
                    await asyncio.shield(clear_task)
                assert self._grace_task is not None
                await asyncio.shield(self._grace_task)

            return wait_for_grace()
        if begin_grace:
            self._approvals.transition(BeginGrace())
            self._grace_task = asyncio.create_task(self._expire_grace())

            async def start_grace() -> None:
                if clear_task is not None:
                    await asyncio.shield(clear_task)
                if self.server.approval_reconnect_grace == 0:
                    assert self._grace_task is not None
                    await self._grace_task

            return start_grace()

        async def disconnect() -> None:
            if clear_task is not None:
                await asyncio.shield(clear_task)
            await self._terminate_pending_approvals(suppress_write_errors=True)
            await self._disconnect_turn()

        return disconnect()

    def _grace_protected(
        self, wake: asyncio.Task[None] | None = None
    ) -> set[asyncio.Task[None]]:
        """The client's running turn or its end-of-turn step, which may start the next.

        A background wake turn is protected only when it ran at the disconnect
        (``wake``); one that starts later is not the gone client's work.
        """
        return {
            task
            for task in (self._turn_task, self._turn_handoff_task)
            if task is not None
            and not task.done()
            and (task is not self._notification_task or task is wake)
        }

    async def _grace_deadline(self) -> None:
        await asyncio.sleep(self.server.approval_reconnect_grace)

    def _end_grace_deadline(self) -> None:
        """End the deadline; held bound MCP calls get the detached result."""
        task = self._grace_deadline_task
        self._grace_deadline_task = None
        if task is not None:
            task.cancel()

    async def _expire_grace(self) -> None:
        deadline = self._grace_deadline_task
        assert deadline is not None
        wake = self._notification_task
        try:
            # the grace ends early once no client turn runs and no approval
            # waits, so a later wake is not cancelled by this old deadline.
            while not deadline.done():
                protected = self._grace_protected(wake)
                if not protected and not self._approvals.has_pending_approvals():
                    self._end_grace_deadline()
                    return
                await asyncio.wait(
                    {deadline, *protected}, return_when=asyncio.FIRST_COMPLETED
                )
            self._grace_expiring = True
            expiry = asyncio.create_task(self._approvals.transition(Expire()))
            try:
                await asyncio.shield(expiry)
            except asyncio.CancelledError:
                await expiry
                raise
            await self._disconnect_turn()
            if self._grace_deadline_task is deadline:
                self._grace_deadline_task = None
        finally:
            self._grace_expiring = False
            if self._grace_task is asyncio.current_task():
                self._grace_task = None

    async def _disconnect_turn(self) -> None:
        if self._turn_task is not None and not self._turn_task.done():
            loop = self.runtime.loop
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
        self._end_grace_deadline()
        await self.controller_released()
        await self._finish_turn_handoff()

    async def _finish_turn_handoff(self) -> None:
        # the finished turn's end-of-turn step may still drop steering; it must
        # write before the runtime closes.
        task = self._turn_handoff_task
        if task is not None and task is not asyncio.current_task():
            await asyncio.gather(task, return_exceptions=True)

    async def user_message(
        self,
        text: str,
        mode: str,
        features: frozenset[str],
        attachments: list[dict[str, object]] | None = None,
        delivery_id: str | None = None,
        origin: MessageOrigin = MessageOrigin.USER,
    ) -> None:
        """Echo an accepted user message so observers can rebuild the transcript.

        Without a delivery id the row installs later in the turn, so ``origin``
        (what the caller passes to ``start_turn``) and the mark find it.
        """
        fields: dict[str, object] = {
            "text": bounded(text, USER_MESSAGE_MAX_BYTES),
            "mode": mode,
            "attachments": attachments or [],
        }
        if delivery_id is not None:
            fields["delivery_id"] = delivery_id
        await self._notify(
            event_contract.AUXILIARY_WIRE_EVENTS[
                event_contract.AuxiliaryEvent.USER_MESSAGE
            ],
            self.runtime.session_id,
            replay_row=RowFacts(delivery_id=delivery_id)
            if delivery_id is not None
            else RowFacts(seq=self._row_mark(), origin=origin),
            **fields,
        )

    def _row_mark(self) -> int | None:
        """The newest resident row seq; a publish-first row installs after it."""
        opened = self.runtime.opened
        return None if opened is None else opened.store.active_branch_head_sequence()

    def _close_deferred_replay(self, session_id: str | None) -> None:
        """At a turn's end, admit deferred replay copies; rows not found never install.

        The turn's unfinished reply ends too: a cancel publishes no agent_end.
        """
        self.in_flight.end_run()
        connections = self.server.connections
        if connections is not None and session_id is not None and self.attached:
            connections.release_deferred_replay(session_id, close=True)

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
        self, method: ApprovalDecision, request_id: str, scope: str = "once"
    ) -> dict[str, object]:
        loop = self.runtime.loop
        policy = self.runtime.policy
        if loop is None or policy is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        result, core_key = await self._approvals.transition(
            Decide(ApprovalDecision(method), request_id, scope)
        )
        assert isinstance(result, dict)
        # The Decide just wrote on this thread, so the memoized `*_now` seam has
        # the resolution without an await (and without the B1 stale-read risk of
        # an off-loop read here). Resume only a request the Decide resolved.
        state = (
            loop.store.approval_state_now(core_key)
            if isinstance(core_key, str)
            else None
        )
        if (
            (self._turn_task is None or self._turn_task.done())
            and state is not None
            and state[1] in ("allow", "deny")
            and loop.prepare_resume_pending_tool(state[0])
        ):
            host_commands = self._active_host_commands
            if self.runtime.state is not None:
                self.runtime.state.tool_started()
            self._turn_task = asyncio.create_task(
                self._resume_tool(core_key, host_commands, turn_id=uuid.uuid4().hex)
            )
            self._turn_task.add_done_callback(self._turn_done)
        return result

    async def _publish_session_grant(self, tool: str) -> None:
        policy = self.runtime.policy
        if policy is not None:
            await self.publish_session_grant_change(
                session_grant_change(policy, "grant", (tool,))
            )

    async def publish_session_grant_change(self, change: SessionGrantChange) -> None:
        connections = self.server.connections
        session_id = self.runtime.session_id
        if connections is None or session_id is None or not self.attached:
            return
        await connections.publish_feature_event(
            FEATURE, EVENT, session_id, data=event_data(change, session_id)
        )

    async def publish_session_grant_reset(self) -> None:
        policy = self.runtime.policy
        if policy is not None:
            await self.publish_session_grant_change(
                session_grant_change(policy, "reset")
            )

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

    def request_approval_abort_gate(
        self,
        request_id: str,
        *,
        child_instance_id: str | None = None,
        reason: str = "turn_canceled",
    ) -> Awaitable[object]:
        return self._approvals.request_approval_abort(
            request_id,
            child_instance_id=child_instance_id,
            reason=reason,
        )

    async def abort(
        self, scope: abort_scope.AbortScope = "session"
    ) -> dict[str, object]:
        session_id = self.runtime.session_id
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
            loop=self.runtime.loop,
            terminate_approvals=functools.partial(
                self._terminate_pending_approvals, reason="turn_canceled"
            ),
            before_cancel=mark_canceled,
            # a wake still at the start gate has nothing to keep; cancel it, as
            # waiting for it would hold this request's gate forever.
            cancel_task=not any(
                isinstance(key, str) for key in self._approvals.active_keys()
            )
            or (turn_task is self._notification_task and not self._wake_admitted),
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
        allow_assistant_reset: bool = True,
    ) -> None:
        loop = self.runtime.loop
        state = self.runtime.state
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
            async with aclosing(
                self.foreground_event_publisher.events(
                    loop.run_turn(
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
                        allow_assistant_reset=allow_assistant_reset,
                    )
                )
            ) as event_stream:
                async for event in event_stream:
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
            self._close_deferred_replay(session_id)
            await self.deliveries.finalize_turn(
                session_id,
                state,
                turn_id,
                outcome,
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
            await self.deliveries.continue_after_turn(
                session_id, state, allow_assistant_reset=allow_assistant_reset
            )

    def _schedule_background_wake(self, session_id: str) -> None:
        if self.closing or self._session_swap_id == session_id:
            return
        loop = self.runtime.loop
        state = self.runtime.state
        if loop is None or state is None or state.session_id != session_id:
            return
        host_commands = self._active_host_commands
        if (
            self._host_binding_required and host_commands is None
        ) or not loop.schedule_notification_turn():
            return
        self._notification_turn_started = False
        self._turn_task = self._notification_task = asyncio.create_task(
            self._run_notification_turn(
                session_id, host_commands, turn_id=uuid.uuid4().hex
            )
        )
        self._turn_task.add_done_callback(self._turn_done)

    async def _run_notification_turn(
        self,
        session_id: str,
        host_commands: SessionHostCommands | None = None,
        *,
        turn_id: str,
    ) -> None:
        loop = self.runtime.loop
        state = self.runtime.state
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
            nonlocal started
            if event.type in {StreamEventType.TURN_END, StreamEventType.ERROR}:
                terminal_events.append(event)
            else:
                await self._event(
                    event,
                    session_id=session_id,
                    background=background,
                    turn_id=turn_id,
                )
                if event.type is StreamEventType.TURN_START:
                    # After _event publishes, update busy state before the caller settles.
                    self._notification_turn_started = True
                    started = True
                    state.turn_started()

        try:
            # a second change can take the hold before this wake resumes
            while self._state_changes:
                await self._no_state_change.wait()
            if self.runtime.loop is not loop:
                return  # a session change ran while this wake waited
            self._wake_admitted = True
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
            self._notification_turn_started = False
            self._wake_admitted = False
            if started:
                self._close_deferred_replay(session_id)
                await self.deliveries.finalize_turn(
                    session_id,
                    state,
                    turn_id,
                    outcome,
                )
        for event in terminal_events:
            await self._event(event, session_id=session_id, turn_id=turn_id)
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id, turn_id=turn_id)
        if outcome is TurnOutcome.COMPLETED:
            await self.deliveries.continue_after_turn(session_id, state)

    async def _attach_pending_notifications(self) -> None:
        runtime = self.runtime
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
        loop = self.runtime.loop
        state = self.runtime.state
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
            self._close_deferred_replay(session_id)
            await self.deliveries.finalize_turn(
                session_id,
                state,
                turn_id,
                outcome,
            )
        if agent_end is not None:
            await self._event(agent_end, session_id=session_id, turn_id=turn_id)
        if outcome is TurnOutcome.COMPLETED:
            await self.deliveries.continue_after_turn(session_id, state)

    async def _publish_config_changed(
        self, event: StreamEvent, session_id: str | None
    ) -> None:
        if self.server.connections is None:
            return
        self._end_stale_approvals(session_id)
        await self._notify(
            event_contract.STREAM_EVENT_WIRE_EVENTS[event.type],
            session_id,
            **event_contract.config_changed_wire_fields(event),
        )

    async def _event(
        self,
        event: StreamEvent,
        *,
        session_id: str | None = None,
        background: bool = False,
        turn_id: str | None = None,
    ) -> None:
        state = self.runtime.state
        if session_id is None and not background and state is not None:
            session_id = state.session_id
        foreground = (
            not background
            and state is not None
            and session_id is not None
            and state.session_id == session_id
        )
        kind = event.type
        if kind is StreamEventType.CONFIG_CHANGED:
            await self._publish_config_changed(event, session_id)
            return
        if foreground:
            state.usage_tracker.record(kind, state.model)
        wire_event = event_contract.STREAM_EVENT_WIRE_EVENTS[kind]
        connections = getattr(self.server, "connections", None)
        # updated in the loop step that publishes, so a join's snapshot is exact
        in_flight = self.in_flight if foreground else None
        if in_flight is not None:
            if kind is StreamEventType.AGENT_START:
                in_flight.start_run(session_id)
            elif kind in {StreamEventType.MESSAGE_START, StreamEventType.ASSISTANT_RESET}:
                in_flight.end_message()
        if kind not in {
            StreamEventType.TOOL_APPROVAL_START,
            StreamEventType.TOOL_APPROVAL_END,
        }:
            self._end_stale_approvals(session_id)
        if kind is StreamEventType.MESSAGE_UPDATE:
            delta = event.delta
            stream_kind = "assistant"
            if isinstance(event.content, TextContent):
                delta = event.content.text
            elif event.content is not None:
                delta = getattr(event.content, "text", None)
                stream_kind = str(getattr(event.content, "type", "content"))
            if delta:
                delta = bounded(delta)
                if in_flight is not None:
                    in_flight.delta(session_id, delta, stream_kind)
                await self._notify(
                    wire_event,
                    session_id,
                    delta=delta,
                    kind=stream_kind,
                )
            return
        if kind is StreamEventType.MESSAGE_END:
            if foreground and not event.data.get("truncated"):
                await model_selection.confirm(self.runtime)
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
            if in_flight is not None:
                in_flight.end_message()
            if event.message is not None:
                await self._notify(
                    wire_event,
                    session_id,
                    replay_row=RowFacts(seq=self._row_mark()),
                    **event_contract.assistant_message(event),
                )
            return
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
            fields = event_contract.tool_output(event, fidelity=fidelity)
            if in_flight is not None:
                in_flight.tool_output(session_id, fields)
            await self._notify(wire_event, session_id, **fields)
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
            fields = event_contract.tool_end(event, fidelity=fidelity)
            if in_flight is not None:
                in_flight.end_tool(fields)
            if not foreground and self._approvals.release_or_hold_terminal(
                _approval_key(event), session_id, fields
            ):
                await self._notify(wire_event, session_id, _admitted=True, **fields)
                return
            await self._notify(
                wire_event, session_id, replay_row=RowFacts(seq=event.row_seq), **fields
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
                session_id,
                event.data,
                self._protocol_version(),
                self._notify,
                row_seq=event.row_seq,
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
                failed_provider = self.runtime.metadata.provider
                offers_login = event.error.provider_error and (
                    event.error.code == "auth_error" or event.error.status_code == 401
                )
                recovered = await model_selection.recover_async(
                    self.runtime, event.error
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
        if kind is StreamEventType.TURN_END:
            self._turn_end_published = True

    async def _start_approval(
        self, event: StreamEvent, session_id: str | None, wire_event: EventDescriptor
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
        self._maybe_idle()
        if not self.closing:
            asyncio.create_task(
                self._event(event, session_id=session_id, background=True)
            )

    def _publish_run_work_event(self, session_id: str, event: EventDescriptor, fields: dict[str, object]) -> None:
        connections = self.server.connections
        if connections is None:
            return
        def publish() -> None:
            # the sink may run on a worker thread; check idleness on the loop
            self._maybe_idle()
            controller = connections.controller_metadata()
            if not self.closing and controller is not None and "run_work" in controller.features:
                if event is EventDescriptor.BACKGROUND_TASK_UPDATED:
                    awaitable = self._notify(EventDescriptor.BACKGROUND_TASK_UPDATED, session_id, **fields)
                elif event is EventDescriptor.TODO_UPDATED:
                    awaitable = self._notify(EventDescriptor.TODO_UPDATED, session_id, **fields)
                else:
                    raise ValueError(f"unsupported run-work event: {event.value}")
                asyncio.create_task(awaitable)
        try:
            connections.loop.call_soon_threadsafe(publish)
        except RuntimeError:
            if not connections.loop.is_closed():
                raise

    def _publish_memory_notice(self, session_id: str, message: str) -> None:
        """Publish updates only when the client negotiated the optional feature."""
        if not self.closing and self.server.connections is not None:
            asyncio.create_task(
                self._notify(
                    event_contract.AUXILIARY_WIRE_EVENTS[
                        event_contract.AuxiliaryEvent.MEMORY_UPDATED
                    ],
                    session_id,
                    message=message,
                )
            )

    def _admit_controller(
        self, event: EventDescriptor, session_id: str | None = None, **fields: object
    ) -> object:
        connections = getattr(self.server, "connections", None)
        if connections is None or not self.attached:
            if connections is not None:
                self._maybe_idle()  # a dropped frame of a parked pair may change its row
            from .connections import Admission

            return Admission("no_controller")
        return connections.try_admit_controller(event, session_id, **fields)

    async def _notify(
        self, event: EventDescriptor, session_id: str | None = None, **fields: object
    ) -> bool:
        if fields.pop("_admitted", False):
            return True
        if session_id is None and self.runtime.opened is not None:
            session_id = self.runtime.session_id
        connections = self.server.connections
        if connections is None or not self.attached:
            if connections is not None:
                self._maybe_idle()  # a dropped frame of a parked pair may change its row
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

    def _end_stale_approvals(self, session_id: str | None) -> None:
        self._approvals._end_stale_approvals(session_id)

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


FEATURE = "session_grants"
READ = "session_grants"
REVOKE = "revoke_session_grants"
EVENT = event_contract.AUXILIARY_WIRE_EVENTS[
    event_contract.AuxiliaryEvent.SESSION_GRANTS_CHANGED
]


def require_feature(features: frozenset[str]) -> None:
    if FEATURE not in features:
        raise ProtocolError(
            ErrorDescriptor.METHOD_NOT_FOUND,
            "session grants require the negotiated session_grants feature",
        )


def read(
    policy: ApprovalPolicy | None, session_id: str | None, params: dict[str, Any]
) -> dict[str, object]:
    if params:
        raise ProtocolError(
            ErrorDescriptor.INVALID_SESSION_GRANTS_PARAMS,
            "session_grants takes no params",
        )
    if policy is None or session_id is None:
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
    return snapshot_result(permission_snapshot(policy), session_id)


def revoke(
    policy: ApprovalPolicy | None,
    session_id: str | None,
    params: dict[str, Any],
) -> tuple[dict[str, object], SessionGrantChange]:
    if policy is None or session_id is None:
        raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
    valid_grant = (
        set(params) == {"grant"}
        and isinstance(params.get("grant"), str)
        and bool(params["grant"])
    )
    valid_clear = set(params) == {"clear"} and params.get("clear") is True
    if not (valid_grant or valid_clear):
        raise ProtocolError(
            ErrorDescriptor.INVALID_SESSION_GRANTS_REVOKE,
            "invalid revoke_session_grants params",
        )
    try:
        change = (
            revoke_session_grants(policy, exact=params["grant"])
            if valid_grant
            else revoke_session_grants(policy, clear=True)
        )
    except SessionGrantNotFound as exc:
        raise ProtocolError(
            ErrorDescriptor.SESSION_GRANT_NOT_FOUND,
            "session grant not found",
            {"grant": exc.grant},
        ) from exc
    return {
        "removed": list(change.changed),
        "grants": snapshot_result(change.snapshot, session_id),
    }, change


def event_data(change: SessionGrantChange, session_id: str) -> dict[str, object]:
    return {
        "change": change.change,
        "changed": list(change.changed),
        "grants": snapshot_result(change.snapshot, session_id),
    }


def snapshot_result(snapshot: PermissionSnapshot, session_id: str) -> dict[str, object]:
    return {
        "session_id": session_id,
        "default": snapshot.default.value,
        "precedence": list(snapshot.precedence),
        "rules": {
            "deny": list(snapshot.deny),
            "ask": list(snapshot.ask),
            "allow": list(snapshot.allow),
        },
        "session_grants": list(snapshot.session_grants),
    }
