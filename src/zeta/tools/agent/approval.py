"""Approval policy adapter for delegated child-agent tool calls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...core.approval import (
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovedExecution,
)
from ...core.store import ConversationStore
from ...protocol.types import Message, MessageRole, ToolCall, ToolUseContent
from ..registry import AbortSignal


class ChildApprovalPolicy:
    """Keep child approval state in both the child and parent stores."""

    def __init__(
        self,
        parent: ApprovalPolicy | ChildApprovalPolicy,
        child_store: ConversationStore,
        description: str,
        child_instance_id: str,
        *,
        parent_cwd: str | Path | None = None,
        child_cwd: str | Path | None = None,
    ) -> None:
        self.parent = parent
        self.child_store = child_store
        self.description = description
        self.child_instance_id = child_instance_id
        self.child_cwd = Path(
            os.path.abspath(os.fspath(child_cwd or child_store.cwd))
        )
        self.parent_cwd = Path(
            os.path.abspath(os.fspath(parent_cwd or self.child_cwd))
        )
        self._execution_bindings: dict[str, ApprovedExecution] = {}

    def bind_store(self, store: ConversationStore) -> None:
        del store

    def register_delegated(
        self,
        request: ApprovalRequest,
        store: ConversationStore,
        *,
        child_instance_id: str | None = None,
    ) -> None:
        self.parent.register_delegated(
            request,
            store,
            child_instance_id=child_instance_id,
        )

    def cleanup_delegated(self, child_instance_id: str) -> None:
        self.parent.cleanup_delegated(child_instance_id)

    def declare_subjects(self, subjects: Mapping[str, str | None]) -> tuple[str, ...]:
        # Child tools are clones of the parent's, so the parent already holds
        # every subject; declarations merge, so pushing the subset is safe.
        return self.parent.declare_subjects(subjects)

    @property
    def notices(self) -> tuple[str, ...]:
        return self.parent.notices

    def approval_subject(self, tool_name: str) -> str | None:
        return self.parent.approval_subject(tool_name)

    def decide_for_child(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        parent_cwd: str | os.PathLike[str],
        child_cwd: str | os.PathLike[str],
    ) -> ApprovalDecision:
        del parent_cwd
        return self.parent.decide_for_child(
            tool_name,
            arguments,
            parent_cwd=self.parent_cwd,
            child_cwd=child_cwd,
        )

    def decide_for_child_with_binding(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        *,
        parent_cwd: str | os.PathLike[str],
        child_cwd: str | os.PathLike[str],
    ) -> tuple[ApprovalDecision, ApprovedExecution | None]:
        del parent_cwd
        return self.parent.decide_for_child_with_binding(
            tool_name,
            arguments,
            parent_cwd=self.parent_cwd,
            child_cwd=child_cwd,
        )

    def decide(self, tool_name: str, arguments: dict[str, Any]) -> ApprovalDecision:
        decision, _binding = self._decide_with_binding(tool_name, arguments)
        return decision

    def _decide_with_binding(
        self, tool_name: str, arguments: dict[str, Any]
    ) -> tuple[ApprovalDecision, ApprovedExecution | None]:
        return self.parent.decide_for_child_with_binding(
            tool_name,
            arguments,
            parent_cwd=self.parent_cwd,
            child_cwd=self._effective_cwd(tool_name, arguments),
        )

    def consume_execution_binding(self, request_id: str) -> ApprovedExecution | None:
        return self._execution_bindings.pop(request_id, None)

    def _effective_cwd(
        self, tool_name: str, arguments: Mapping[str, object]
    ) -> Path:
        effective_cwd = self.child_cwd
        cwd_argument = arguments.get("cwd")
        if tool_name == "bash" and cwd_argument is None:
            effective_cwd = Path(self.child_store.bash_cwd)
        elif tool_name in {"bash", "run_background"} and isinstance(
            cwd_argument, str
        ):
            candidate = Path(cwd_argument).expanduser()
            effective_cwd = (
                candidate if candidate.is_absolute() else self.child_cwd / candidate
            )
        return Path(os.path.abspath(effective_cwd))

    def _request(self, tool_call: ToolCall) -> ApprovalRequest:
        effective_cwd = self._effective_cwd(tool_call.name, tool_call.arguments)
        resolved_path = None
        if self.parent.approval_subject(tool_call.name) == "path":
            raw_path = tool_call.arguments.get("path")
            if isinstance(raw_path, str):
                candidate = Path(raw_path).expanduser()
                if not candidate.is_absolute():
                    candidate = self.child_cwd / candidate
                resolved_path = os.path.abspath(candidate)
        return ApprovalRequest(
            tool_call.id,
            tool_call,
            label=f"{self.description}: {tool_call.name}",
            effective_cwd=str(effective_cwd),
            resolved_path=resolved_path,
        )

    def prepare(self, tool_call: ToolCall) -> ApprovalRequest | None:
        state = self.child_store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            return None
        if self.decide(tool_call.name, tool_call.arguments) is not ApprovalDecision.ASK:
            return None
        return self._request(tool_call)

    def durable_decision(self, request_id: str) -> str | None:
        state = self.child_store.approval_states().get(request_id)
        return None if state is None else state[1]

    async def authorize(
        self,
        tool_call: ToolCall,
        abort_signal: AbortSignal,
    ) -> ApprovalDecision | None:
        state = self.child_store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            if state[1] == ApprovalDecision.ALLOW.value:
                return ApprovalDecision.ALLOW
            if state[1] == ApprovalDecision.DENY.value:
                return ApprovalDecision.DENY
        else:
            self._execution_bindings.pop(tool_call.id, None)
            decision, binding = self._decide_with_binding(
                tool_call.name, tool_call.arguments
            )
            if decision is not ApprovalDecision.ASK:
                if decision is ApprovalDecision.ALLOW and binding is not None:
                    self._execution_bindings[tool_call.id] = binding
                return decision
            self.child_store.append_message_with_approval_requests(
                Message(MessageRole.ASSISTANT, [ToolUseContent(tool_call)]),
                [(tool_call.id, tool_call)],
            )

        request = self._request(tool_call)
        self.parent.register_delegated(
            request,
            self.child_store,
            child_instance_id=self.child_instance_id,
        )
        while True:
            state = self.child_store.approval_states().get(tool_call.id)
            if state is not None and state[1] is not None:
                return _approval_decision(state[1])
            if abort_signal.is_set():
                return self.abort_or_winner(tool_call.id)
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
                return self.abort_or_winner(tool_call.id)

    def abort_or_winner(self, request_id: str) -> ApprovalDecision | None:
        self.child_store.resolve_approval(request_id, "abort")
        state = self.child_store.approval_states().get(request_id)
        return _approval_decision(state[1] if state is not None else None)

    def cleanup(self) -> None:
        self._execution_bindings.clear()
        self.parent.cleanup_delegated(self.child_instance_id)


def _approval_decision(value: str | None) -> ApprovalDecision | None:
    if value == ApprovalDecision.ALLOW.value:
        return ApprovalDecision.ALLOW
    if value == ApprovalDecision.DENY.value:
        return ApprovalDecision.DENY
    return None
