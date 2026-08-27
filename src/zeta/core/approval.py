"""Durable tool approval decisions."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Protocol

from .abort import AbortSignal
from .store import ConversationStore
from ..types import Message, MessageRole, ToolCall, ToolResult, ToolUseContent


class ApprovalDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    tool_call: ToolCall
    label: str | None = None


class _AbortSignal(Protocol):
    def is_set(self) -> bool: ...

    async def wait(self) -> None: ...


class ApprovalPolicy:
    """Choose, persist, and resolve decisions for tool calls."""

    def __init__(
        self,
        *,
        always_allow: Iterable[str] = (),
        always_deny: Iterable[str] = (),
        default: ApprovalDecision | str = ApprovalDecision.ASK,
        store: ConversationStore | None = None,
    ) -> None:
        self.always_allow = frozenset(always_allow)
        self.always_deny = frozenset(always_deny)
        self.default = _decision(default)
        self._store = store
        self._delegated: dict[str, tuple[ApprovalRequest, ConversationStore]] = {}

    def bind_store(self, store: ConversationStore) -> None:
        self._store = store

    def decide(
        self,
        tool_name: str,
        arguments: dict[str, object],
    ) -> ApprovalDecision:
        del arguments
        if tool_name in self.always_deny:
            return ApprovalDecision.DENY
        if tool_name in self.always_allow:
            return ApprovalDecision.ALLOW
        return self.default

    def pending_requests(self) -> list[ApprovalRequest]:
        store = self._require_store()
        requests = [
            ApprovalRequest(request_id, tool_call)
            for request_id, tool_call in store.pending_approvals()
        ]
        for request_id, (request, delegated_store) in list(self._delegated.items()):
            state = delegated_store.approval_states().get(request_id)
            if state is None or state[1] is not None:
                self._delegated.pop(request_id, None)
            else:
                requests.append(request)
        return requests

    def register_delegated(
        self,
        request: ApprovalRequest,
        store: ConversationStore,
    ) -> None:
        """Expose a child request without writing it into this store."""

        if request.request_id in self._delegated:
            return
        self._delegated[request.request_id] = (request, store)

    def approve(self, request_id: str) -> bool:
        return self.resolve(request_id, ApprovalDecision.ALLOW)

    def deny(self, request_id: str) -> bool:
        return self.resolve(request_id, ApprovalDecision.DENY)

    def abort(self, request_id: str) -> bool:
        delegated = self._delegated.get(request_id)
        if delegated is not None:
            return delegated[1].resolve_approval(request_id, "abort")
        store = self._require_store()
        return store.resolve_approval(request_id, "abort")

    def abort_or_winner(self, request_id: str) -> ApprovalDecision | None:
        delegated = self._delegated.get(request_id)
        if delegated is not None:
            delegated[1].resolve_approval(request_id, "abort")
            state = delegated[1].approval_states().get(request_id)
            return _resolved_decision(state[1] if state is not None else None)
        store = self._require_store()
        store.resolve_approval(request_id, "abort")
        state = store.approval_states().get(request_id)
        return _resolved_decision(state[1] if state is not None else None)

    def durable_decision(self, request_id: str) -> str | None:
        delegated = self._delegated.get(request_id)
        if delegated is not None:
            state = delegated[1].approval_states().get(request_id)
            return None if state is None else state[1]
        state = self._require_store().approval_states().get(request_id)
        return None if state is None else state[1]

    def resolve(self, request_id: str, decision: ApprovalDecision | str) -> bool:
        resolved = _decision(decision)
        if resolved is ApprovalDecision.ASK:
            raise ValueError("approval resolution must allow or deny")
        delegated = self._delegated.get(request_id)
        if delegated is not None:
            return delegated[1].resolve_approval(request_id, resolved.value)
        store = self._require_store()
        return store.resolve_approval(request_id, resolved.value)

    def prepare(self, tool_call: ToolCall) -> ApprovalRequest | None:
        """Build an ask request for atomic persistence with its assistant anchor."""

        store = self._require_store()
        state = store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            return None
        if self.decide(tool_call.name, tool_call.arguments) is ApprovalDecision.ASK:
            return ApprovalRequest(tool_call.id, tool_call)
        return None

    async def authorize(
        self,
        tool_call: ToolCall,
        abort_signal: _AbortSignal,
    ) -> ApprovalDecision | None:
        store = self._require_store()
        state = store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            if state[1] == ApprovalDecision.ALLOW.value:
                return ApprovalDecision.ALLOW
            if state[1] == ApprovalDecision.DENY.value:
                return ApprovalDecision.DENY
        else:
            decision = self.decide(tool_call.name, tool_call.arguments)
            if decision is not ApprovalDecision.ASK:
                return decision
            store.append_message_with_approval_requests(
                Message(MessageRole.ASSISTANT, [ToolUseContent(tool_call)]),
                [(tool_call.id, tool_call)],
            )

        while True:
            state = store.approval_states().get(tool_call.id)
            if state is not None and state[1] is not None:
                if state[1] == ApprovalDecision.ALLOW.value:
                    return ApprovalDecision.ALLOW
                if state[1] == ApprovalDecision.DENY.value:
                    return ApprovalDecision.DENY
                return None
            if abort_signal.is_set():
                return self._resolve_abort_or_winner(tool_call.id)
            abort_task = asyncio.create_task(abort_signal.wait())
            poll_task = asyncio.create_task(asyncio.sleep(0.05))
            try:
                done, pending = await asyncio.wait(
                    {abort_task, poll_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
            except asyncio.CancelledError:
                abort_task.cancel()
                poll_task.cancel()
                await asyncio.gather(abort_task, poll_task, return_exceptions=True)
                raise
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if abort_task in done:
                return self._resolve_abort_or_winner(tool_call.id)

    def _resolve_abort_or_winner(
        self,
        request_id: str,
    ) -> ApprovalDecision | None:
        return self.abort_or_winner(request_id)

    def _require_store(self) -> ConversationStore:
        if self._store is None:
            raise RuntimeError("approval policy requires a conversation store")
        return self._store


def _decision(value: ApprovalDecision | str) -> ApprovalDecision:
    try:
        return ApprovalDecision(value)
    except ValueError as exc:
        raise ValueError(f"invalid approval decision: {value}") from exc


def _resolved_decision(value: str | None) -> ApprovalDecision | None:
    if value == ApprovalDecision.ALLOW.value:
        return ApprovalDecision.ALLOW
    if value == ApprovalDecision.DENY.value:
        return ApprovalDecision.DENY
    return None


ApprovalHook = Callable[
    [str, dict[str, object]], bool | str | Awaitable[bool | str] | None
]
AdvanceGeneration = Callable[[AbortSignal], AbortSignal]


def canceled_result(tool_call_id: str) -> ToolResult:
    return ToolResult(tool_call_id, "tool execution canceled", True)


@dataclass(slots=True)
class ApprovalGate:
    """Run durable approval and the optional pre-execution hook."""

    policy: ApprovalPolicy | None
    hook: ApprovalHook | None

    async def run(
        self,
        tool_call: ToolCall,
        arguments: dict[str, object],
        signal: AbortSignal,
        advance_generation: AdvanceGeneration,
        lifecycle: Callable[[str], None] | None = None,
        *,
        skip_approval: bool = False,
    ) -> tuple[ToolResult | None, AbortSignal]:
        execution_signal = signal
        if self.policy is not None and not skip_approval:
            approval_started = (
                self.policy.durable_decision(tool_call.id) is None
                and self.policy.decide(tool_call.name, arguments)
                is ApprovalDecision.ASK
            )
            if approval_started and lifecycle is not None:
                lifecycle("approval_start")
            try:
                decision = await self.policy.authorize(tool_call, signal)
            except Exception as exc:
                return ToolResult(tool_call.id, f"approval failed: {exc}", True), execution_signal
            finally:
                if approval_started and lifecycle is not None:
                    lifecycle("approval_end")
            if decision is None:
                return canceled_result(tool_call.id), execution_signal
            if signal.is_set():
                durable_decision = self.policy.durable_decision(tool_call.id)
                if durable_decision == ApprovalDecision.DENY.value:
                    return ToolResult(tool_call.id, "tool execution denied", True), execution_signal
                if durable_decision != ApprovalDecision.ALLOW.value:
                    return canceled_result(tool_call.id), execution_signal
                execution_signal = advance_generation(signal)
                if execution_signal.is_set():
                    return canceled_result(tool_call.id), execution_signal
            if decision is ApprovalDecision.DENY:
                return ToolResult(tool_call.id, "tool execution denied", True), execution_signal
        if self.hook is None:
            if signal.is_set() and execution_signal is signal:
                return canceled_result(tool_call.id), execution_signal
            return None, execution_signal
        try:
            allowed = self.hook(tool_call.name, arguments)
            if inspect.isawaitable(allowed):
                allowed = await allowed
        except Exception as exc:
            return ToolResult(tool_call.id, f"pre-execution hook failed: {exc}", True), execution_signal
        if isinstance(allowed, str):
            return ToolResult(tool_call.id, allowed, True), execution_signal
        if allowed is False:
            return ToolResult(tool_call.id, "tool execution denied", True), execution_signal
        if signal.is_set() and execution_signal is signal:
            return canceled_result(tool_call.id), execution_signal
        return None, execution_signal
