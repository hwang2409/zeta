"""Server-owned wire identity and terminal state for approvals."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from uuid import uuid4

from ..core.approval import ApprovalPolicy, ApprovalRequest
from ..core.store import ConversationStore
from ..protocol.types import StreamEvent, StreamEventType, ToolCall
from . import approval_settlement, event_contract
from .approval_settlement.approval_grants import (
    ApprovalRecord,
    GrantPublication,
    _ApprovalTarget,
    finish_grant,
)
from .approval_settlement.owner import TaskOwner
from .connections import Admission
from .protocol import (
    FRAME_CANCEL_GRACE,
    ApprovalDecision,
    ErrorDescriptor,
    EventDescriptor,
    ProtocolError,
    bound_reconnect_approvals,
)
from .session_replay.rows import DELEGATED, WRITE_FAILED, RowFacts

Admit = Callable[..., Admission]
Offload = Callable[[Callable[[], object]], Awaitable[object]]

logger = logging.getLogger(__name__)

APPROVAL_SETTLEMENT_CLOSE_TIMEOUT = FRAME_CANCEL_GRACE


from .approval_machine import (
    ApprovalEvent,
    ApprovalKey,
    ApprovalMachineState,
    BeginGrace,
    Cancel,
    CleanupChild,
    Close,
    Decide,
    Expire,
    ExternalEnd,
    Ready,
    Reclaim,
    Register,
    Replace,
    SessionHealth,
    Start,
    Terminate,
    _Completed,
    approval_display_fields,
)


class ApprovalLifecycle:
    """Own stable approval IDs and make terminal events exactly once."""

    def __init__(
        self,
        *,
        request_turn_failure: Callable[[], None],
        policy: Callable[[], ApprovalPolicy | None],
        session_id: Callable[[], str | None],
        parent_store: Callable[[], ConversationStore | None],
        offload: Offload,
        is_closing: Callable[[], bool],
        admit: Admit | None = None,
        grant_published: Callable[[str], Awaitable[None]] | None = None,
    ) -> None:
        self._admit = admit or (lambda *_args, **_kwargs: Admission("admitted"))
        self._request_turn_failure = request_turn_failure
        self._policy = policy
        self._session_id = session_id
        self._parent_store = parent_store
        self._offload = offload
        self._is_closing = is_closing
        self._grant_published = grant_published
        self._state = ApprovalMachineState()
        self._task_owner = TaskOwner()

    @property
    def _settlement_tasks(self) -> dict[asyncio.Task[object], ApprovalKey]:
        return self._task_owner.tasks

    @property
    def state(self) -> ApprovalMachineState:
        return self._state

    def _swap(
        self,
        records: Mapping[ApprovalKey, ApprovalRecord],
        *,
        health: SessionHealth | None | object = ...,
        generation: int | None = None,
    ) -> None:
        next_health = self._state.health if health is ... else health
        assert next_health is None or isinstance(next_health, SessionHealth)
        self._state = ApprovalMachineState(
            self._state.generation if generation is None else generation,
            MappingProxyType(dict(records)),
            next_health,
        )

    def _put(self, record: ApprovalRecord) -> ApprovalRecord:
        records = dict(self._state.records)
        records[record.target.key] = record
        self._swap(records)
        return record

    def _next_token(self) -> int:
        return max((item.token for item in self._state.records.values()), default=0) + 1

    def _remove(self, key: ApprovalKey) -> None:
        records = dict(self._state.records)
        records.pop(key, None)
        self._swap(records)

    def _change(self, record: ApprovalRecord, **changes: object) -> ApprovalRecord:
        current = self._state.records.get(record.target.key)
        if current is None or current.token != record.token:
            return record
        return self._put(replace(current, **changes))

    def _owns(self, record: ApprovalRecord) -> bool:
        current = self._state.records.get(record.target.key)
        return current is not None and (
            current.generation == record.generation and current.token == record.token
        )

    def _reserve_record(
        self,
        record: ApprovalRecord,
        disposition: str,
        *,
        keep_starting: bool = False,
    ) -> ApprovalRecord:
        if record.phase not in {"starting", "pending"}:
            raise ValueError(f"cannot reserve approval {record.phase}")
        if keep_starting:
            if record.phase != "starting" or record.disposition != "active":
                raise ValueError(f"cannot reserve approval {record.phase}")
            return self._change(record, disposition=disposition)
        return self._change(record, phase="settling", disposition=disposition)

    def _mark_pending(self, record: ApprovalRecord) -> ApprovalRecord:
        if record.phase != "starting" or record.disposition not in {
            "active",
            "grace",
            "reclaiming",
        }:
            raise ValueError(f"cannot mark approval {record.phase} as pending")
        return self._change(record, phase="pending")

    def _externalize(self, record: ApprovalRecord) -> ApprovalRecord:
        if record.phase != "starting" or record.disposition != "active":
            raise ValueError(f"cannot externalize approval {record.phase}")
        return self._change(record, disposition="external")

    def _health_abort(self, record: ApprovalRecord, disposition: str) -> ApprovalRecord:
        if record.phase != "settling":
            raise ValueError(f"cannot update health abort from {record.phase}")
        return self._change(record, disposition=disposition)

    def _fail(self, health: SessionHealth) -> None:
        self._swap(self._state.records, health=health)

    async def _report_persistence_failure(
        self, record: ApprovalRecord, message: str
    ) -> None:
        self._fail(SessionHealth("approval_persistence_failed", message))
        with contextlib.suppress(ConnectionError, OSError):
            self._admit(
                EventDescriptor.ERROR,
                self._session_id(),
                error={"code": "server_error", "message": message},
                data={"request_id": record.wire_id},
            )
        self._request_turn_failure()

    def _replace_state(self) -> None:
        self._swap({}, health=None, generation=self._state.generation + 1)

    def transition(self, event: ApprovalEvent) -> Awaitable[object]:
        """Run one typed lifecycle event, including its durable operation."""

        async def close() -> object:
            await self._terminate_approval_records(
                foreground_only=False,
                suppress_write_errors=True,
                reason="session_closed",
            )
            self._swap(self._state.records)
            return None

        match event:
            case Register(request, store, child_instance_id):
                self._register_delegated(request, store, child_instance_id)
                return _Completed()
            case Start(stream_event, session_id, wire_event):
                return self._start_approval(stream_event, session_id, wire_event)
            case Decide(method, request_id, scope):
                return self.resolve_approval(method, request_id, scope)
            case ExternalEnd(key, session_id, data, suppress_write_errors):
                return self._end_approval(
                    key,
                    session_id,
                    data=dict(data),
                    suppress_write_errors=suppress_write_errors,
                )
            case Cancel(request_id, child_instance_id, reason):
                key: ApprovalKey = (
                    (child_instance_id, request_id)
                    if child_instance_id is not None
                    else request_id
                )
                return self._settle_canceled_approval(key, reason)
            case CleanupChild(child_instance_id):
                cleanup = self._cleanup_delegated(child_instance_id)
                return cleanup if cleanup is not None else _Completed()
            case Terminate(foreground_only, suppress_write_errors, reason):
                return self._terminate_approval_records(
                    foreground_only=foreground_only,
                    suppress_write_errors=suppress_write_errors,
                    reason=reason,
                )
            case BeginGrace():
                self._set_grace("grace")
                return _Completed()
            case Reclaim():
                self._set_grace("reclaiming")
                return _Completed()
            case Ready():
                return self._ready_reclaimed()
            case Expire():
                return self._expire_grace()
            case Replace():
                self._replace_state()
                return _Completed()
            case Close():
                return close()
            case _ as unreachable:
                raise AssertionError(
                    f"unhandled approval lifecycle event: {unreachable!r}"
                )

    def observe(
        self,
        request: ApprovalRequest,
        *,
        fields: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Return the live wire projection for one tracked request."""
        record = self._state.records.get(request.key)
        request_id = record.wire_id if record is not None else self.wire_id(request.key)
        return {
            "request_id": request_id,
            "tool_call": request.tool_call.to_dict(),
            "delegated": request.child_instance_id is not None,
            **(
                {"agent_instance_id": request.child_instance_id}
                if request.child_instance_id is not None
                else {}
            ),
            **({"label": request.label} if request.label is not None else {}),
            **(fields or {}),
        }

    def core_key(self, request_id: str) -> ApprovalKey:
        for key, record in self._state.records.items():
            if record.wire_id == request_id:
                return key
        return request_id

    def wire_id(self, key: ApprovalKey) -> str:
        record = self._state.records.get(key)
        if record is not None:
            return record.wire_id
        return key if isinstance(key, str) else f"approval-{uuid4().hex}"

    def end(
        self, key: ApprovalKey, *, data: dict[str, object] | None = None
    ) -> dict[str, object] | None:
        record = self._state.records.get(key)
        if record is None or record.request is None:
            return None
        terminal = data or {}
        return {
            "request_id": record.wire_id,
            "tool_call": record.request.tool_call.to_dict(),
            "data": terminal,
            **({"reason": terminal["reason"]} if "reason" in terminal else {}),
        }

    def active_keys(self) -> tuple[ApprovalKey, ...]:
        return tuple(self._state.records)

    def prune_ended(self) -> None:
        return

    def pending_approvals_now(
        self,
        *,
        reconnect: bool = False,
        include_truncated: bool = False,
    ) -> list[dict[str, object]] | tuple[list[dict[str, object]], bool]:
        """Return the loop-owned visible projection without touching a store."""

        visible = [
            {
                **self.observe(record.request),
                **approval_display_fields(record.request),
                **(
                    {"resolution_pending": self._resolution_pending(record)}
                    if reconnect and record.phase == "settling"
                    else {}
                ),
            }
            for record in self._state.records.values()
            if record.phase in ({"pending", "settling"} if reconnect else {"pending"})
            and record.request is not None
        ]
        if not reconnect:
            return visible
        bounded, truncated = bound_reconnect_approvals(visible)
        return (bounded, truncated) if include_truncated else bounded

    @staticmethod
    def _resolution_pending(record: ApprovalRecord) -> str:
        return {
            "allow": "approve",
            "deny": "deny",
            "grace_expired": "expire",
        }.get(record.disposition, "abort")

    def _set_grace(self, disposition: str) -> None:
        for record in tuple(self._state.records.values()):
            if record.phase in {"starting", "pending"} and record.disposition in {
                "active",
                "grace",
            }:
                self._change(record, disposition=disposition)

    async def _ready_reclaimed(self) -> None:
        replay: list[tuple[str, str | None, dict[str, object]]] = []
        for record in tuple(self._state.records.values()):
            if record.disposition != "reclaiming":
                continue
            record = self._change(record, disposition="active")
            if record.phase == "pending" and record.request is not None:
                replay.append(
                    (
                        event_contract.STREAM_EVENT_WIRE_EVENTS[
                            StreamEventType.TOOL_APPROVAL_START
                        ],
                        self._session_id(),
                        self.observe(
                            record.request,
                            fields=approval_display_fields(record.request),
                        ),
                    )
                )
        for event, session_id, payload in replay:
            self._admit(event, session_id, **payload)

    async def _expire_grace(self) -> None:
        for record in tuple(self._state.records.values()):
            if record.disposition in {"grace", "reclaiming"}:
                self._change(record, disposition="active")
        await self._terminate_approval_records(
            foreground_only=False,
            suppress_write_errors=True,
            reason="grace_expired",
        )

    def pending_approval_labels(self) -> tuple[str, ...]:
        return tuple(
            f"{request.key} ({request.label or request.tool_call.name})"
            for record in self._state.records.values()
            if record.phase == "pending" and (request := record.request) is not None
        )

    def has_pending_approvals(self) -> bool:
        return any(
            record.phase in {"starting", "pending", "settling"}
            for record in self._state.records.values()
        )

    async def pending_approvals(self) -> list[dict[str, object]]:
        """Build the legacy strict projection off-loop with unchanged fields."""

        return await self._pending_approvals_projection()

    async def _pending_approvals_projection(self) -> list[dict[str, object]]:
        policy = self._policy()
        if policy is None or self._state.health is not None:
            return []
        foreground = policy.store
        delegated = tuple(
            record.target
            for record in self._state.records.values()
            if isinstance(record.target.key, tuple)
        )

        # Read the resident approval index, not the durable log: a status call
        # must not reparse stored rows. The resident page already holds every
        # pending foreground request, and delegated requests come from records.
        pending: list[tuple[str, ToolCall]] = []
        after_seq = 0
        while True:
            page = foreground.pending_approval_page_now(after_seq)
            pending.extend((k, ToolCall.from_dict(dict(v))) for k, v in page.items)
            if page.next_seq is None:
                break
            after_seq = page.next_seq
        requests = [
            policy.request_from_store(
                request_id, tool_call, label=await policy.display_label(tool_call)
            )
            for request_id, tool_call in pending
        ]
        for target in delegated:
            state = target.store.approval_state_now(target.store_request_id)
            if state is not None and state[1] is None and target.request is not None:
                requests.append(target.request)
        return [
            {**self.observe(item), **approval_display_fields(item)} for item in requests
        ]

    def clear_approvals(self) -> None:
        self.transition(Replace())

    def session_replaced(self) -> None:
        self.transition(Replace())

    async def wait_for_approval_settlement(
        self, timeout: float = APPROVAL_SETTLEMENT_CLOSE_TIMEOUT
    ) -> None:
        await self._task_owner.drain(timeout)

    def _start_settlement(
        self,
        record: ApprovalRecord | ApprovalKey,
        operation: Awaitable[object],
        kind: str = "settlement",
    ) -> asyncio.Task[object]:
        key = record.target.key if isinstance(record, ApprovalRecord) else record
        return self._task_owner.start(key, operation, kind)

    def _new_record(self, target: _ApprovalTarget) -> ApprovalRecord:
        token = self._next_token()
        return self._put(
            ApprovalRecord(
                target=target,
                wire_id=self.wire_id(target.key),
                generation=self._state.generation,
                token=token,
                request=target.request,
                admitted=asyncio.Future(),
                disposition="abort" if self._is_closing() else "active",
            )
        )

    def register_delegated(
        self,
        request: ApprovalRequest,
        store: ConversationStore,
        child_instance_id: str,
    ) -> None:
        self.transition(Register(request, store, child_instance_id))

    def _register_delegated(
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
        history_store = self._parent_store()
        target = _ApprovalTarget(key, store, request.request_id, exact, history_store)
        record = self._state.records.get(key)
        if record is None:
            record = self._new_record(target)
        elif record.phase == "starting":
            record = self._change(record, target=target, request=exact)
        else:
            return
        if record.disposition == "external":
            task = self._start_settlement(record, self._finish_external(record))
            self._change(record, settlement=task)
        elif record.disposition != "active" or self._is_closing():
            reason = (
                "turn_canceled" if record.disposition == "abort" else record.disposition
            )
            record = self._reserve_record(record, reason)
            task = self._start_settlement(record, self._settle_record(record, "abort", reason))
            self._change(record, settlement=task)

    def cleanup_delegated(self, child_instance_id: str) -> Awaitable[object] | None:
        cleanup = self.transition(CleanupChild(child_instance_id))
        return None if isinstance(cleanup, _Completed) else cleanup

    def _cleanup_delegated(self, child_instance_id: str) -> Awaitable[object] | None:
        settlements: list[asyncio.Task[object]] = []
        for key, record in tuple(self._state.records.items()):
            if not isinstance(key, tuple) or key[0] != child_instance_id:
                continue
            reserved = self._reserve(record, "child_cleanup")
            if reserved is not None:
                if reserved.phase == "starting":
                    if reserved.settlement is not None:
                        settlements.append(reserved.settlement)
                    continue
                task = self._start_settlement(
                    reserved, self._settle_record(reserved, "abort", "child_cleanup")
                )
                self._change(reserved, settlement=task)
                settlements.append(task)
        if not settlements:
            return None
        return asyncio.gather(*settlements, return_exceptions=True)

    def _reserve(self, record: ApprovalRecord, decision: str) -> ApprovalRecord | None:
        if record.phase == "starting":
            if record.disposition != "active":
                return None
            return self._reserve_record(record, decision, keep_starting=True)
        if record.phase != "pending":
            return None
        record = self._change(record, token=self._next_token())
        return self._reserve_record(record, decision)

    async def resolve_approval(
        self, method: str, request_id: str, scope: str = "once"
    ) -> tuple[dict[str, object], ApprovalKey]:
        policy = self._policy()
        if policy is None:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "no active session")
        self.reject_if_unhealthy()
        core_key = self.core_key(request_id)
        record = await self._record_for_decision(core_key, policy)
        if record is None or record.phase != "pending" or record.request is None:
            raise ProtocolError(
                ErrorDescriptor.APPROVAL_UNAVAILABLE,
                f"approval request not found or already resolved: {request_id}",
            )
        pending = record.request
        if scope == "always_tool" and pending.child_instance_id is not None:
            raise ProtocolError(
                ErrorDescriptor.INVALID_PARAMS,
                "scope 'always_tool' is unavailable for delegated approvals",
            )
        if scope == "always_tool":
            try:
                pending.always_allow_rule()
            except ValueError as exc:
                raise ProtocolError(
                    ErrorDescriptor.INVALID_PARAMS,
                    f"cannot always allow approval request: {exc}",
                ) from exc
        decision = "allow" if method is ApprovalDecision.APPROVE else "deny"
        record = self._reserve(record, decision)
        if record is None:
            raise ProtocolError(
                ErrorDescriptor.APPROVAL_UNAVAILABLE,
                f"approval request not found or already resolved: {request_id}",
            )
        record = (
            self._change(record, grant=pending) if scope == "always_tool" else record
        )
        settlement = self._start_settlement(record, self._settle_record(record, decision))
        self._change(record, settlement=settlement)
        try:
            resolution = await approval_settlement.settlement_outcome(record.admitted, settlement)
        except OSError as exc:
            raise ProtocolError(
                ErrorDescriptor.INTERNAL_ERROR, f"approval persistence failed: {exc}"
            ) from exc
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
        return result, core_key

    def reject_if_unhealthy(self) -> None:
        health = self._state.health
        if health is not None:
            raise ProtocolError(
                ErrorDescriptor.APPROVAL_PERSISTENCE_FAILED,
                "approval persistence failed",
            )

    async def _record_for_decision(
        self, core_key: ApprovalKey, policy: ApprovalPolicy
    ) -> ApprovalRecord | None:
        record = self._state.records.get(core_key)
        if record is not None or not isinstance(core_key, str):
            return record
        record = self._new_record(_ApprovalTarget(core_key, policy.store, core_key))
        token = record.token
        generation = record.generation
        state = await record.target.store.approval_state(core_key)
        label = None if state is None else await policy.display_label(state[0])
        current = self._state.records.get(core_key)
        if (
            current is None
            or current.token != token
            or generation != self._state.generation
            or state is None
            or state[1] is not None
        ):
            self._remove(core_key)
            return None
        record = current
        if record.disposition != "active":
            record = self._reserve_record(record, record.disposition)
            await self._settle_record(record, "abort", record.disposition)
            return None
        request = policy.request_from_store(core_key, state[0], label=label)
        record = self._change(record, request=request)
        record = self._mark_pending(record)
        record = self._change(record, published=True)
        self.observe(request)
        return record

    async def _settle_record(
        self,
        record: ApprovalRecord,
        decision: str,
        reason: str | None = None,
    ) -> dict[str, str]:
        token = record.token
        generation = record.generation
        target = record.target
        result = await approval_settlement.finish_despite_cancellation(
            lambda: approval_settlement.persist_durably(
                target.store, target.store_request_id, decision, reason
            )
        )
        resolution = result.resolution
        current = self._state.records.get(target.key)
        owns_record = (
            current is not None
            and generation == self._state.generation
            and token == current.token
        )
        if resolution is None:
            if owns_record and decision == "abort" and self.state.health is not None:
                synthetic = {
                    "request_id": target.store_request_id,
                    "decision": "abort",
                    "reason": reason or "turn_canceled",
                }
                # no resolution row was written: the end is live-only (WRITE_FAILED)
                finish = self._start_settlement(
                    record, self._finish_record(record, synthetic, WRITE_FAILED)
                )
                try:
                    await approval_settlement.finish_despite_cancellation(
                        lambda: asyncio.shield(finish)
                    )
                except BaseException as exc:
                    self._release_gate(record, error=exc)
                    raise
                return synthetic
            if owns_record:
                await self._report_persistence_failure(
                    record, "approval decision could not be persisted"
                )
                record = self._health_abort(record, reason or "turn_canceled")
                self._change(record, settlement=None)
            error = OSError("approval decision could not be persisted")
            if not owns_record:
                self._release_gate(record, error=error)
            raise error
        if not owns_record:
            self._release_gate(record, resolution=resolution)
            return resolution
        try:
            await approval_settlement.finish_despite_cancellation(
                lambda: self._finish_record(record, resolution)
            )
        except BaseException as exc:
            self._release_gate(record, error=exc)
            if self._owns(record):
                self._remove(target.key)
            raise
        return resolution

    def request_approval_abort(
        self,
        request_id: str,
        *,
        child_instance_id: str | None = None,
        reason: str = "turn_canceled",
    ) -> Awaitable[dict[str, str] | None]:
        """Route a serve-bound cancellation through the settlement owner."""
        key: ApprovalKey = (
            (child_instance_id, request_id)
            if child_instance_id is not None
            else request_id
        )
        admitted = self._begin_cancel(key, reason)
        if admitted is not None:
            return admitted
        settlement = self.transition(Cancel(request_id, child_instance_id, reason))
        assert not isinstance(settlement, _Completed)
        return self._start_settlement(key, settlement)

    def _release_gate(
        self,
        record: ApprovalRecord,
        resolution: dict[str, str] | None = None,
        error: BaseException | None = None,
    ) -> None:
        if record.admitted.done():
            return
        if error is not None:
            record.admitted.set_exception(error)
        else:
            assert resolution is not None
            record.admitted.set_result(resolution)

    def _begin_cancel(
        self, key: ApprovalKey, reason: str
    ) -> asyncio.Future[dict[str, str]] | None:
        record = self._state.records.get(key)
        if record is None:
            return None
        if (
            record.phase == "settling"
            and self.state.health is not None
            and (record.settlement is None or record.settlement.done())
        ):
            record = self._health_abort(record, reason)
        else:
            reserved = self._reserve(record, reason)
            if reserved is None:
                return record.admitted
            record = reserved
        if record.phase == "starting":
            return record.admitted
        settlement = self._start_settlement(record, self._settle_record(record, "abort", reason))
        self._change(record, settlement=settlement)
        return record.admitted

    async def _settle_canceled_approval(
        self, key: ApprovalKey, reason: str
    ) -> dict[str, str] | None:
        admitted = self._begin_cancel(key, reason)
        if admitted is None:
            return None
        return await asyncio.shield(admitted)

    async def _finish_external(self, record: ApprovalRecord) -> None:
        resolution = await record.target.store.approval_resolution(
            record.target.store_request_id
        )
        if resolution is not None:
            await self._finish_record(record, resolution)
        else:
            self._remove(record.target.key)

    async def _append_delegated_history(
        self, record: ApprovalRecord, payload: dict[str, object] | None
    ) -> OSError | asyncio.CancelledError | None:
        if payload is None or not record.history_published:
            return None
        assert record.target.history_store is not None
        try:
            await record.target.history_store.append_approval_history_async(
                {"event": "approval_end", **payload}
            )
        except OSError as exc:
            logger.warning("delegated approval history write failed", exc_info=True)
            return exc
        except asyncio.CancelledError as exc:
            logger.warning("delegated approval history write cancelled", exc_info=True)
            return exc
        return None

    def _release_terminal(self, record: ApprovalRecord) -> Admission | None:
        terminal = record.terminal
        if terminal is None or record.terminal_result is not None:
            return record.terminal_result
        _, session_id, fields = terminal
        result = self._admit(EventDescriptor.TOOL_END, session_id, **fields)
        self._change(record, terminal_result=result, terminal=None)
        if record.terminal_watchdog is not None:
            record.terminal_watchdog.cancel()
        if self._owns(record) and record.live_end is not None:
            self._remove(record.target.key)
        return result

    async def _terminal_bound(self, key: ApprovalKey, token: int) -> None:
        try:
            await asyncio.wait_for(asyncio.Event().wait(), timeout=30)
        except TimeoutError:
            record = self._state.records.get(key)
            if record is None or record.token != token or record.terminal is None:
                return
            logger.warning(
                "approval terminal order bound exceeded", extra={"approval_key": key}
            )
            self._release_terminal(record)
        except asyncio.CancelledError:
            return

    def release_or_hold_terminal(
        self,
        key: ApprovalKey,
        session_id: str | None,
        fields: dict[str, object],
    ) -> bool:
        """Own the matching delegated terminal and admit it exactly once."""
        record = self._state.records.get(key)
        if record is None or not isinstance(key, tuple):
            return False
        if record.terminal_result is not None:
            return True
        if record.live_end is not None:
            result = self._admit(EventDescriptor.TOOL_END, session_id, **fields)
            self._change(record, terminal_result=result, terminal=None)
            if self._owns(record):
                self._remove(key)
            return True
        record = self._change(
            record,
            terminal=(EventDescriptor.TOOL_END, session_id, fields),
        )
        if record.terminal_watchdog is None:
            task = self._start_settlement(
                key, self._terminal_bound(key, record.token), "terminal watchdog"
            )
            self._change(record, terminal_watchdog=task)
        return True

    async def _finish_record(
        self, record: ApprovalRecord, resolution: dict[str, str], row: RowFacts | None = None
    ) -> None:
        current = self._state.records.get(record.target.key)
        if (
            current is None
            or current.token != record.token
            or current.live_end is not None
        ):
            self._release_gate(record, resolution=resolution)
            return
        owner = asyncio.current_task()
        assert owner is not None
        if (
            current.publication_owner is not None
            and current.publication_owner is not owner
        ):
            return
        record = self._change(current, publication_owner=owner)
        grant_cancelled = False
        if resolution["decision"] == "allow" and record.grant is not None:
            policy = self._policy()
            if policy is not None:
                publication = GrantPublication(policy, self._grant_published)
                committed, published, grant_cancelled = await finish_grant(
                    publication,
                    record.grant,
                    record.grant_committed,
                    record.grant_event_published,
                    start_task=lambda operation, kind: self._start_settlement(
                        record, operation, kind
                    ),
                )
                record = self._change(
                    record,
                    grant_committed=committed,
                    grant_event_published=published,
                )
        data: dict[str, object] = {
            key: value for key, value in resolution.items() if key != "request_id"
        }
        payload = self.end(record.target.key, data=data)
        history_outcome: OSError | asyncio.CancelledError | None = None
        try:
            history_outcome = await self._append_delegated_history(record, payload)
            current = self._state.records.get(record.target.key)
            if current is not None and current.token == record.token:
                record = current
            if payload is not None and record.published:
                if isinstance(record.target.key, tuple):
                    terminal = record.terminal
                    session_id = self._session_id()
                    written = history_outcome is None and record.history_published
                    end_result = self._admit(
                        EventDescriptor.APPROVAL_END, session_id,
                        replay_row=DELEGATED if written else WRITE_FAILED, **payload,
                    )
                    terminal_result = (
                        self._admit(EventDescriptor.TOOL_END, session_id, **terminal[2])
                        if terminal is not None
                        else None
                    )
                    record = self._change(record, live_end=end_result)
                    self._release_gate(record, resolution=resolution)
                    if terminal is not None:
                        record = self._change(
                            record, terminal_result=terminal_result, terminal=None
                        )
                else:
                    admission = self._admit(
                        event_contract.STREAM_EVENT_WIRE_EVENTS[
                            StreamEventType.TOOL_APPROVAL_END
                        ],
                        session_id=self._session_id(),
                        replay_row=row,
                        **payload,
                    )
                    record = self._change(record, live_end=admission)
                    self._release_gate(record, resolution=resolution)
            elif isinstance(record.target.key, tuple):
                self._release_terminal(record)
                self._release_gate(record, resolution=resolution)
            if isinstance(history_outcome, OSError):
                await self._report_persistence_failure(
                    record, "approval history could not be persisted"
                )
        finally:
            if self._owns(record):
                current = self._state.records.get(record.target.key)
                if current is not None:
                    if current.terminal_watchdog is not None:
                        current.terminal_watchdog.cancel()
                    self._remove(record.target.key)
        # approval_end is already published after a cancelled history await; do not
        # retry it. Only a deferred grant-publication cancellation is propagated.
        if grant_cancelled:
            raise asyncio.CancelledError

    async def _start_approval(
        self, event: StreamEvent, session_id: str | None, wire_event: EventDescriptor
    ) -> None:
        raw_request_id = event.tool_call.id if event.tool_call else ""
        child_id = event.data.get("agent_instance_id")
        key: ApprovalKey = (
            (child_id, raw_request_id)
            if isinstance(child_id, str) and raw_request_id
            else raw_request_id
        )
        policy = self._policy()
        if policy is None:
            return
        record = self._state.records.get(key)
        if record is None:
            record = self._new_record(
                _ApprovalTarget(key, policy.store, raw_request_id)
            )
        record = self._change(record, settlement=asyncio.current_task())
        token = record.token
        generation = record.generation
        target = record.target

        # The admission reads the request's durable state. approval_state (async)
        # is p08b's replacement for main's removed approval_states_snapshot() map.
        durable_state = await target.store.approval_state(target.store_request_id)
        # Look the label up before the ownership check, so a stale record is dropped.
        label = (
            None
            if durable_state is None
            else await policy.display_label(durable_state[0])
        )
        current = self._state.records.get(key)
        if (
            current is None
            or current.token != token
            or generation != self._state.generation
        ):
            return
        await self._complete_approval_start(
            current,
            durable_state,
            label,
            policy,
            token,
            generation,
            session_id,
            wire_event,
        )

    async def _complete_approval_start(
        self,
        record: ApprovalRecord,
        durable_state: tuple[ToolCall, str | None] | None,
        label: str | None,
        policy: ApprovalPolicy,
        token: int,
        generation: int,
        session_id: str | None,
        wire_event: EventDescriptor,
    ) -> None:
        target = record.target
        if durable_state is None:
            self._remove(target.key)
            self._admit(
                EventDescriptor.ERROR,
                session_id,
                error={
                    "code": "approval_context_missing",
                    "message": "approval request is unavailable; denying live approval",
                },
                data={"request_id": target.store_request_id},
            )
            return
        if durable_state[1] is not None:
            self._remove(target.key)
            return
        request = target.request or policy.request_from_store(
            target.store_request_id, durable_state[0], label=label
        )
        record = self._change(record, request=request)
        payload = self.observe(request, fields=approval_display_fields(request))
        if isinstance(target.key, tuple) and target.history_store is not None:
            try:
                await target.history_store.append_approval_history_async(
                    {"event": "approval_request", **payload}
                )
            except OSError:
                await self._settle_failed_start(record)
                return
            record = self._change(record, history_published=True)
        current = self._state.records.get(target.key)
        if (
            current is None
            or current.token != token
            or generation != self._state.generation
        ):
            return
        record = current
        if record.disposition == "external":
            resolution = await target.store.approval_resolution(
                target.store_request_id
            )
            if resolution is not None:
                await asyncio.shield(self._finish_record(record, resolution))
            else:
                self._remove(target.key)
            return
        if record.disposition in {"grace", "reclaiming"}:
            record = self._mark_pending(record)
            self._change(record, settlement=None)
            return
        if record.disposition != "active":
            record = self._reserve_record(record, record.disposition)
            settlement = self._start_settlement(
                record, self._settle_record(record, "abort", record.disposition)
            )
            self._change(record, settlement=settlement)
            await settlement
            return
        record = self._mark_pending(record)
        self._change(record, settlement=None, published=True)
        self._admit(wire_event, session_id, **payload)

    async def _settle_failed_start(self, record: ApprovalRecord) -> None:
        current = self._state.records.get(record.target.key)
        if current is None or current.token != record.token:
            return
        record = current
        reason = (
            record.disposition
            if record.disposition not in {"active", "external"}
            else "turn_canceled"
        )
        record = self._reserve_record(record, reason)
        settlement = self._start_settlement(record, self._settle_record(record, "abort", reason))
        self._change(record, settlement=settlement)
        await settlement

    async def _end_approval(
        self,
        key: ApprovalKey,
        session_id: str | None,
        *,
        data: dict[str, object] | None = None,
        suppress_write_errors: bool = False,
    ) -> None:
        del session_id, data, suppress_write_errors
        record = self._state.records.get(key)
        if record is None:
            return
        if record.phase == "starting":
            if record.disposition == "active":
                self._externalize(record)
            return
        if record.phase != "pending" and record.phase != "settling":
            return
        resolution = await record.target.store.approval_resolution(
            record.target.store_request_id
        )
        if resolution is not None:
            await asyncio.shield(self._finish_record(record, resolution))

    def _end_stale_approvals(self, session_id: str | None) -> None:
        del session_id
        for stale in tuple(self._state.records.values()):
            if stale.phase not in {"pending", "settling"} or stale.live_end is not None:
                continue
            data = stale.target.store.approval_resolution_now(
                stale.target.store_request_id
            )
            record = self._state.records.get(stale.target.key)
            if data is None or record is None or record.token != stale.token:
                continue
            if record.live_end is not None:
                continue
            owner = record.publication_owner
            if owner is not None and owner.done():
                continue
            task = self._start_settlement(record, self._finish_record(record, data))
            if owner is None:
                self._change(record, publication_owner=task)

    async def _terminate_approval_records(
        self,
        *,
        foreground_only: bool,
        suppress_write_errors: bool,
        reason: str,
    ) -> None:
        settlements = [
            settlement
            for key, record in tuple(self._state.records.items())
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
        self, record: ApprovalRecord, reason: str
    ) -> asyncio.Task[object] | None:
        record = self._reserve(record, reason)
        if record is None or record.phase == "starting":
            return None
        settlement = self._start_settlement(record, self._settle_record(record, "abort", reason))
        self._change(record, settlement=settlement)
        return settlement
