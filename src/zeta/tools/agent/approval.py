"""Approval policy adapter for delegated child-agent tool calls."""

from __future__ import annotations

import asyncio
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...core.approval import (
    ApprovalCapability,
    ApprovalDecision,
    ApprovalPolicy,
    ApprovalRequest,
    ApprovedCwdExecution,
    ApprovedExecution,
    ApprovedPathExecution,
)
from ...core.store import ConversationStore
from ...protocol.types import (
    ASSISTANT_RESPONSE_SYNTHETIC,
    Message,
    MessageRole,
    ToolCall,
    ToolUseContent,
)
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
        # Object facts captured before a manual request is persisted/displayed.
        # Membership is significant: ``None`` records a failed capture and must
        # never be replaced by a later view of the filesystem.
        self._pending_bindings: dict[str, ApprovedExecution | None] = {}
        self._denial_reasons: dict[str, str] = {}

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

    def bind_rule_resolver(self, resolver: Any) -> None:
        self.parent.bind_rule_resolver(resolver)

    def bind_capability_resolver(self, resolver: Any) -> None:
        self.parent.bind_capability_resolver(resolver)

    @property
    def notices(self) -> tuple[str, ...]:
        return self.parent.notices

    def decide_for_child(
        self,
        capability: ApprovalCapability,
        *,
        parent_cwd: str | os.PathLike[str],
        child_cwd: str | os.PathLike[str],
    ) -> ApprovalDecision:
        del parent_cwd
        return self.parent.decide_for_child(
            capability, parent_cwd=self.parent_cwd, child_cwd=child_cwd
        )

    def decide_for_child_with_binding(
        self,
        capability: ApprovalCapability,
        *,
        parent_cwd: str | os.PathLike[str],
        child_cwd: str | os.PathLike[str],
    ) -> tuple[ApprovalDecision, ApprovedExecution | None]:
        del parent_cwd
        return self.parent.decide_for_child_with_binding(
            capability, parent_cwd=self.parent_cwd, child_cwd=child_cwd
        )

    def capture_child_binding(
        self,
        capability: ApprovalCapability,
        *,
        child_cwd: str | os.PathLike[str],
    ) -> tuple[ApprovedExecution | None, bool]:
        return self.parent.capture_child_binding(capability, child_cwd=child_cwd)

    def decide(self, capability: ApprovalCapability) -> ApprovalDecision:
        effective_cwd = self._effective_cwd(capability.tool, capability.arguments)
        decision, _binding = self._decide_with_binding(capability, effective_cwd)
        return decision

    def _decide_with_binding(
        self,
        capability: ApprovalCapability,
        effective_cwd: str | os.PathLike[str],
    ) -> tuple[ApprovalDecision, ApprovedExecution | None]:
        return self.parent.decide_for_child_with_binding(
            capability,
            parent_cwd=self.parent_cwd,
            child_cwd=effective_cwd,
        )

    def consume_execution_binding(
        self, execution_token: str
    ) -> ApprovedExecution | None:
        return self._execution_bindings.pop(execution_token, None)

    def cleanup_execution_binding(self, execution_token: str) -> None:
        self._execution_bindings.pop(execution_token, None)

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

    def _capture_pending_binding(
        self,
        tool_call: ToolCall,
        capability: ApprovalCapability,
    ) -> None:
        """Capture an approval's object identity once, before persistence."""
        if tool_call.id in self._pending_bindings:
            return
        capture = getattr(self.parent, "capture_child_binding", None)
        if not callable(capture):
            return
        binding, required = capture(
            capability,
            child_cwd=self._effective_cwd(tool_call.name, tool_call.arguments),
        )
        if required:
            # Store failed captures too.  A later filesystem state must not turn
            # a request that was unsafe to bind when shown into an executable one.
            self._pending_bindings[tool_call.id] = binding

    def _request(
        self,
        tool_call: ToolCall,
        capability: ApprovalCapability,
    ) -> ApprovalRequest | None:
        """Format a request exclusively from its captured execution binding."""
        binding = self._pending_bindings.get(tool_call.id)
        binding_required = self._binding_required(tool_call, capability)
        if binding_required and binding is None:
            return None
        effective_cwd = binding.cwd if isinstance(binding, ApprovedCwdExecution) else None
        resolved_path = (
            binding.target if isinstance(binding, ApprovedPathExecution) else None
        )
        return ApprovalRequest(
            tool_call.id,
            tool_call,
            label=f"{self.description}: {tool_call.name}",
            effective_cwd=effective_cwd,
            resolved_path=resolved_path,
            action=capability.action,
        )

    def _binding_required(
        self,
        tool_call: ToolCall,
        capability: ApprovalCapability | None,
    ) -> bool:
        return str(capability.binding) != "none"

    def _deny_unavailable_binding(self, tool_call: ToolCall) -> ApprovalDecision:
        self._pending_bindings.pop(tool_call.id, None)
        self._denial_reasons[tool_call.id] = (
            "approval binding unavailable; submit a new tool request"
        )
        self.child_store.resolve_approval(tool_call.id, ApprovalDecision.DENY.value)
        return ApprovalDecision.DENY

    def prepare(
        self,
        tool_call: ToolCall,
        *,
        capability: ApprovalCapability,
    ) -> ApprovalRequest | None:
        state = self.child_store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            return None
        if self.decide(capability) is not ApprovalDecision.ASK:
            return None
        self._capture_pending_binding(tool_call, capability)
        request = self._request(tool_call, capability)
        if request is None:
            self._deny_unavailable_binding(tool_call)
        return request

    def durable_decision(self, request_id: str) -> str | None:
        state = self.child_store.approval_states().get(request_id)
        return None if state is None else state[1]

    def denial_reason(self, request_id: str) -> str | None:
        return self._denial_reasons.pop(request_id, None)

    async def authorize(
        self,
        tool_call: ToolCall,
        abort_signal: AbortSignal,
        *,
        execution_token: str | None = None,
        capability: ApprovalCapability,
    ) -> ApprovalDecision | None:
        state = self.child_store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            if state[1] == ApprovalDecision.ALLOW.value:
                return self._consume_allow_binding(
                    tool_call, execution_token, capability
                )
            if state[1] == ApprovalDecision.DENY.value:
                self._pending_bindings.pop(tool_call.id, None)
                return ApprovalDecision.DENY
            if state[1] == "abort":
                self._pending_bindings.pop(tool_call.id, None)
                return None
        else:
            decision, binding = self._decide_with_binding(
                capability, self._effective_cwd(tool_call.name, tool_call.arguments)
            )
            if decision is not ApprovalDecision.ASK:
                if (
                    decision is ApprovalDecision.ALLOW
                    and binding is not None
                    and execution_token is not None
                ):
                    self._execution_bindings[execution_token] = binding
                return decision
            # Direct callers that did not run prepare() still capture before the
            # request becomes durable or visible.  Existing durable requests are
            # deliberately never reconstructed here (not even after restart).
            self._capture_pending_binding(tool_call, capability)
            request = self._request(tool_call, capability)
            if request is None:
                return self._deny_unavailable_binding(tool_call)
            display = request.audit_display()
            self.child_store.append_message_with_approval_requests(
                Message(
                    MessageRole.ASSISTANT,
                    [ToolUseContent(tool_call)],
                    metadata={"response_state": ASSISTANT_RESPONSE_SYNTHETIC},
                ),
                [
                    (
                        tool_call.id,
                        tool_call,
                        request.audit_facts(),
                        display,
                    )
                ],
            )

        request = self._request(tool_call, capability)
        if request is None:
            return self._deny_unavailable_binding(tool_call)
        self.parent.register_delegated(
            request,
            self.child_store,
            child_instance_id=self.child_instance_id,
        )
        while True:
            state = self.child_store.approval_states().get(tool_call.id)
            if state is not None and state[1] is not None:
                if state[1] == ApprovalDecision.ALLOW.value:
                    return self._consume_allow_binding(
                        tool_call, execution_token, capability
                    )
                self._pending_bindings.pop(tool_call.id, None)
                if state[1] == "abort":
                    return None
                return ApprovalDecision.DENY
            if abort_signal.is_set():
                return self.abort_or_winner(
                    tool_call.id,
                    execution_token=execution_token,
                    capability=capability,
                )
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
                return self.abort_or_winner(
                    tool_call.id,
                    execution_token=execution_token,
                    capability=capability,
                )

    def _consume_allow_binding(
        self,
        tool_call: ToolCall,
        execution_token: str | None,
        capability: ApprovalCapability,
    ) -> ApprovalDecision:
        """Consume the pre-approval object fact and bind it to this execution."""
        binding_required = self._binding_required(tool_call, capability)
        captured = tool_call.id in self._pending_bindings
        binding = self._pending_bindings.pop(tool_call.id, None)
        if binding_required and (not captured or binding is None):
            # A restart, lost in-memory request, or failed initial capture cannot
            # safely bind this execution to the object shown to the reviewer.
            self._denial_reasons[tool_call.id] = (
                "approval binding unavailable; submit a new tool request"
            )
            return ApprovalDecision.DENY
        if binding is not None and execution_token is not None:
            self._execution_bindings[execution_token] = binding
        return ApprovalDecision.ALLOW

    def abort_or_winner(
        self,
        request_id: str,
        *,
        execution_token: str | None = None,
        capability: ApprovalCapability,
    ) -> ApprovalDecision | None:
        # Resolve first.  If ALLOW won concurrently, the pending binding still
        # belongs to that approval and must be transferred to this execution.
        self.child_store.resolve_approval(request_id, "abort")
        state = self.child_store.approval_states().get(request_id)
        decision = _approval_decision(state[1] if state is not None else None)
        if decision is ApprovalDecision.ALLOW and state is not None:
            return self._consume_allow_binding(
                state[0], execution_token, capability
            )
        self._pending_bindings.pop(request_id, None)
        return decision

    def cleanup(self) -> None:
        self._execution_bindings.clear()
        self._pending_bindings.clear()
        self._denial_reasons.clear()
        self.parent.cleanup_delegated(self.child_instance_id)


def _approval_decision(value: str | None) -> ApprovalDecision | None:
    if value == ApprovalDecision.ALLOW.value:
        return ApprovalDecision.ALLOW
    if value == ApprovalDecision.DENY.value:
        return ApprovalDecision.DENY
    return None
