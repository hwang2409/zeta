"""Durable tool approval decisions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Protocol

from .store import ConversationStore
from .types import Message, MessageRole, ToolCall, ToolUseContent


class ApprovalDecision(StrEnum):
    ALLOW = "allow"
    DENY = "deny"
    ASK = "ask"


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    request_id: str
    tool_call: ToolCall


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
        return [
            ApprovalRequest(request_id, tool_call)
            for request_id, tool_call in store.pending_approvals()
        ]

    def approve(self, request_id: str) -> bool:
        return self.resolve(request_id, ApprovalDecision.ALLOW)

    def deny(self, request_id: str) -> bool:
        return self.resolve(request_id, ApprovalDecision.DENY)

    def abort(self, request_id: str) -> bool:
        store = self._require_store()
        return store.resolve_approval(request_id, "abort")

    def abort_or_winner(self, request_id: str) -> ApprovalDecision | None:
        store = self._require_store()
        store.resolve_approval(request_id, "abort")
        state = store.approval_states().get(request_id)
        if state is None:
            return None
        if state[1] == ApprovalDecision.ALLOW.value:
            return ApprovalDecision.ALLOW
        if state[1] == ApprovalDecision.DENY.value:
            return ApprovalDecision.DENY
        return None

    def durable_decision(self, request_id: str) -> str | None:
        state = self._require_store().approval_states().get(request_id)
        return None if state is None else state[1]

    def resolve(self, request_id: str, decision: ApprovalDecision | str) -> bool:
        resolved = _decision(decision)
        if resolved is ApprovalDecision.ASK:
            raise ValueError("approval resolution must allow or deny")
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
