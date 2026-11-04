"""Server-owned wire identity and terminal state for approvals."""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Literal
from uuid import uuid4

from ..core.approval import ApprovalPolicy, ApprovalRequest
from ..core.store import ConversationStore
from ..protocol.types import StreamEvent, StreamEventType, ToolCall
from . import event_contract
from .protocol import ProtocolError

Publish = Callable[..., Awaitable[bool]]
Offload = Callable[[Callable[[], object]], Awaitable[object]]

ApprovalKey = str | tuple[str, str]


@dataclass(frozen=True, slots=True)
class _ApprovalTarget:
    key: ApprovalKey
    store: ConversationStore
    store_request_id: str
    request: ApprovalRequest | None = None
    history_store: ConversationStore | None = None


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    target: _ApprovalTarget
    wire_id: str
    generation: int
    token: int
    phase: Literal["starting", "pending", "settling"] = "starting"
    disposition: str = "active"
    request: ApprovalRequest | None = None
    history_published: bool = False
    published: bool = False
    settlement: asyncio.Task[object] | None = None


@dataclass(frozen=True, slots=True)
class SessionHealth:
    code: str
    message: str


@dataclass(frozen=True, slots=True)
class ApprovalMachineState:
    """Immutable live approval projection for one loaded serve session."""

    generation: int = 0
    records: Mapping[ApprovalKey, ApprovalRecord] = MappingProxyType({})
    health: SessionHealth | None = None


@dataclass(frozen=True, slots=True)
class Register:
    request: ApprovalRequest
    store: ConversationStore
    child_instance_id: str


@dataclass(frozen=True, slots=True)
class Start:
    stream_event: StreamEvent
    session_id: str | None
    wire_event: str


@dataclass(frozen=True, slots=True)
class Decide:
    method: str
    request_id: str
    scope: str = "once"


@dataclass(frozen=True, slots=True)
class ExternalEnd:
    key: ApprovalKey
    session_id: str | None
    data: Mapping[str, object]
    suppress_write_errors: bool = False


@dataclass(frozen=True, slots=True)
class Cancel:
    request_id: str
    child_instance_id: str | None = None
    reason: str = "turn_canceled"


@dataclass(frozen=True, slots=True)
class CleanupChild:
    child_instance_id: str


@dataclass(frozen=True, slots=True)
class Terminate:
    foreground_only: bool = False
    suppress_write_errors: bool = False
    reason: str = "controller_disconnected"


@dataclass(frozen=True, slots=True)
class Replace:
    pass


@dataclass(frozen=True, slots=True)
class Close:
    pass


ApprovalEvent = (
    Register
    | Start
    | Decide
    | ExternalEnd
    | Cancel
    | CleanupChild
    | Terminate
    | Replace
    | Close
)


class _Completed:
    def __init__(self, value: object = None) -> None:
        self._value = value

    def __await__(self):
        async def result() -> object:
            return self._value

        return result().__await__()


def approval_display_fields(request: ApprovalRequest) -> dict[str, object]:
    """Return immutable trusted display facts for an approval request."""

    fields: dict[str, object] = {}
    if request.action is not None:
        fields["approval_action"] = request.action
    display = request.audit_display()
    if display:
        fields["approval_display"] = display
    return fields


class ApprovalLifecycle:
    """Own stable approval IDs and make terminal events exactly once."""

    def __init__(
        self,
        *,
        publish: Publish,
        request_turn_failure: Callable[[], None],
        policy: Callable[[], ApprovalPolicy | None],
        session_id: Callable[[], str | None],
        parent_store: Callable[[], ConversationStore | None],
        offload: Offload,
        is_closing: Callable[[], bool],
    ) -> None:
        self._publish = publish
        self._request_turn_failure = request_turn_failure
        self._policy = policy
        self._session_id = session_id
        self._parent_store = parent_store
        self._offload = offload
        self._is_closing = is_closing
        self._state = ApprovalMachineState()

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
        if record.phase != "starting" or record.disposition != "active":
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

    def pending_approvals_now(self) -> list[dict[str, object]]:
        """Return the loop-owned visible projection without touching a store."""

        return [
            {
                **self.observe(record.request),
                **approval_display_fields(record.request),
            }
            for record in self._state.records.values()
            if record.phase == "pending" and record.request is not None
        ]

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

        def strict() -> tuple[
            dict[str, tuple[ToolCall, str | None]],
            dict[ConversationStore, dict[str, tuple[ToolCall, str | None]]],
        ]:
            child_stores = {target.store for target in delegated}
            return foreground.approval_states_snapshot(), {
                store: store.approval_states_snapshot() for store in child_stores
            }

        foreground_states, child_states = await self._offload(strict)
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
            {**self.observe(item), **approval_display_fields(item)} for item in requests
        ]

    def clear_approvals(self) -> None:
        self.transition(Replace())

    def session_replaced(self) -> None:
        self.transition(Replace())

    async def wait_for_approval_settlement(self) -> None:
        while tasks := [
            record.settlement
            for record in self._state.records.values()
            if record.settlement is not None and not record.settlement.done()
        ]:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _new_record(self, target: _ApprovalTarget) -> ApprovalRecord:
        token = (
            max((item.token for item in self._state.records.values()), default=0) + 1
        )
        return self._put(
            ApprovalRecord(
                target=target,
                wire_id=self.wire_id(target.key),
                generation=self._state.generation,
                token=token,
                request=target.request,
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
            task = asyncio.create_task(self._finish_external(record))
            self._change(record, settlement=task)
        elif record.disposition != "active" or self._is_closing():
            reason = (
                "turn_canceled" if record.disposition == "abort" else record.disposition
            )
            record = self._reserve_record(record, reason)
            task = asyncio.create_task(self._settle_record(record, "abort", reason))
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
                task = asyncio.create_task(
                    self._settle_record(reserved, "abort", "child_cleanup")
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
        token = (
            max((item.token for item in self._state.records.values()), default=0) + 1
        )
        record = self._change(record, token=token)
        return self._reserve_record(record, decision)

    async def resolve_approval(
        self, method: str, request_id: str, scope: str = "once"
    ) -> tuple[dict[str, object], ApprovalKey]:
        policy = self._policy()
        if policy is None:
            raise ProtocolError(-32003, "no active session")
        self.reject_if_unhealthy()
        core_key = self.core_key(request_id)
        record = await self._record_for_decision(core_key, policy)
        if record is None or record.phase != "pending" or record.request is None:
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
        record = self._reserve(record, decision)
        if record is None:
            raise ProtocolError(
                -32006, f"approval request not found or already resolved: {request_id}"
            )
        settlement = asyncio.create_task(
            self._settle_record(
                record,
                decision,
                grant=pending if scope == "always_tool" else None,
            )
        )
        self._change(record, settlement=settlement)
        try:
            resolution = await settlement
        except OSError as exc:
            raise ProtocolError(-32000, f"approval persistence failed: {exc}") from exc
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
                -32000,
                "approval persistence failed",
                {"code": health.code},
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
        state = await self._offload(
            lambda: record.target.store.approval_states_snapshot().get(core_key)
        )
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
        request = policy.request_from_store(core_key, state[0])
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

        resolution = await self._offload(persist)
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
                await self._finish_record(record, synthetic)
                return synthetic
            if owns_record:
                self._fail(
                    SessionHealth(
                        "approval_persistence_failed",
                        "approval decision could not be persisted",
                    )
                )
                with contextlib.suppress(ConnectionError, OSError):
                    await self._publish(
                        event_contract.STREAM_EVENT_WIRE_EVENTS[StreamEventType.ERROR],
                        self._session_id(),
                        error={
                            "code": "server_error",
                            "message": "approval decision could not be persisted",
                        },
                        data={"request_id": record.wire_id},
                    )
                self._request_turn_failure()
                record = self._health_abort(record, reason or "turn_canceled")
                self._change(record, settlement=None)
            raise OSError("approval decision could not be persisted")
        if not owns_record:
            return resolution
        if grant is not None and resolution["decision"] == "allow":
            policy = self._policy()
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
        settlement = self.transition(Cancel(request_id, child_instance_id, reason))
        assert not isinstance(settlement, _Completed)
        return asyncio.create_task(settlement)

    async def _settle_canceled_approval(
        self, key: ApprovalKey, reason: str
    ) -> dict[str, str] | None:
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
                settlement = record.settlement
                if settlement is not None and settlement is not asyncio.current_task():
                    with contextlib.suppress(OSError):
                        return await settlement
                return await self._offload(
                    lambda: record.target.store.approval_resolution_data(
                        record.target.store_request_id
                    )
                )
            record = reserved
        if record.phase == "starting":
            settlement = record.settlement
            if settlement is not None and settlement is not asyncio.current_task():
                with contextlib.suppress(OSError):
                    await settlement
            return await self._offload(
                lambda: record.target.store.approval_resolution_data(
                    record.target.store_request_id
                )
            )
        settlement = asyncio.create_task(self._settle_record(record, "abort", reason))
        self._change(record, settlement=settlement)
        return await settlement

    async def _finish_external(self, record: ApprovalRecord) -> None:
        resolution = await self._offload(
            lambda: record.target.store.approval_resolution_data(
                record.target.store_request_id
            )
        )
        if resolution is not None:
            await self._finish_record(record, resolution)
        else:
            self._remove(record.target.key)

    async def _finish_record(
        self, record: ApprovalRecord, resolution: dict[str, str]
    ) -> None:
        current = self._state.records.get(record.target.key)
        if current is None or current.token != record.token:
            return
        record = current
        data: dict[str, object] = {
            key: value for key, value in resolution.items() if key != "request_id"
        }
        payload = self.end(record.target.key, data=data)
        try:
            if payload is not None and record.history_published:
                assert record.target.history_store is not None
                await record.target.history_store.append_approval_history_async(
                    {"event": "approval_end", **payload}
                )
            if payload is not None and record.published:
                await self._publish(
                    event_contract.STREAM_EVENT_WIRE_EVENTS[
                        StreamEventType.TOOL_APPROVAL_END
                    ],
                    self._session_id(),
                    **payload,
                )
        finally:
            if self._owns(record):
                self._remove(record.target.key)

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

        def lookup() -> tuple[ToolCall, str | None] | None:
            return target.store.approval_states_snapshot().get(
                target.store_request_id
            )

        try:
            durable_state = await self._offload(lookup)
        except OSError:
            await self._settle_failed_start(record)
            return
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
        policy: ApprovalPolicy,
        token: int,
        generation: int,
        session_id: str | None,
        wire_event: str,
    ) -> None:
        target = record.target
        if durable_state is None:
            self._remove(target.key)
            await self._publish(
                event_contract.STREAM_EVENT_WIRE_EVENTS[StreamEventType.ERROR],
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
            target.store_request_id, durable_state[0]
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
            resolution = await self._offload(
                lambda: target.store.approval_resolution_data(target.store_request_id)
            )
            if resolution is not None:
                await self._finish_record(record, resolution)
            else:
                self._remove(target.key)
            return
        if record.disposition != "active":
            record = self._reserve_record(record, record.disposition)
            settlement = asyncio.create_task(
                self._settle_record(record, "abort", record.disposition)
            )
            self._change(record, settlement=settlement)
            await settlement
            return
        record = self._mark_pending(record)
        self._change(record, settlement=None, published=True)
        await self._publish(wire_event, session_id, **payload)

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
        settlement = asyncio.create_task(self._settle_record(record, "abort", reason))
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
        if record.phase != "pending":
            return
        resolution = await self._offload(
            lambda: record.target.store.approval_resolution_data(
                record.target.store_request_id
            )
        )
        if resolution is not None:
            await self._finish_record(record, resolution)

    async def _end_stale_approvals(self, session_id: str | None) -> None:
        del session_id
        for record in tuple(self._state.records.values()):
            if record.phase != "pending":
                continue
            resolution = await self._offload(
                lambda record=record: record.target.store.approval_resolution_data(
                    record.target.store_request_id
                )
            )
            if resolution is not None:
                await self._finish_record(record, resolution)

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
        settlement = asyncio.create_task(self._settle_record(record, "abort", reason))
        self._change(record, settlement=settlement)
        return settlement
