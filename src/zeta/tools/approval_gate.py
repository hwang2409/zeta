"""Run durable tool approval and the optional pre-execution hook."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..core.abort import AbortSignal
from ..core.approval import (
    ApprovalCapability,
    ApprovalDecision,
    ApprovalPolicy,
    canceled_result,
)
from ..protocol.types import ToolCall, ToolResult

ApprovalHook = Callable[
    [str, dict[str, object]], bool | str | Awaitable[bool | str] | None
]
AdvanceGeneration = Callable[[AbortSignal], AbortSignal]


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
        persist_request: bool = True,
        execution_token: str | None = None,
        capability: ApprovalCapability,
    ) -> tuple[ToolResult | None, AbortSignal]:
        execution_signal = signal
        if self.policy is not None and not skip_approval:
            approval_started = (
                await self.policy.durable_decision(tool_call.id) is None
                and self.policy.decide(capability) is ApprovalDecision.ASK
            )
            slot_acquired = False
            if approval_started:
                slot_acquired = await self.policy.acquire_approval_slot()
                if lifecycle is not None:
                    lifecycle("approval_start")

            try:
                authorize = self.policy.authorize
                parameters = inspect.signature(authorize).parameters
                kwargs: dict[str, object] = {}
                if not persist_request and "persist_request" in parameters:
                    kwargs["persist_request"] = False
                if execution_token is not None and "execution_token" in parameters:
                    kwargs["execution_token"] = execution_token
                if capability is not None and "capability" in parameters:
                    kwargs["capability"] = capability
                decision = await authorize(tool_call, signal, **kwargs)
            except Exception as exc:  # noqa: BLE001 - report approval failures
                return ToolResult(
                    tool_call.id, f"approval failed: {exc}", True
                ), execution_signal
            finally:
                if approval_started:
                    if lifecycle is not None:
                        lifecycle("approval_end")
                    if slot_acquired:
                        self.policy.release_approval_slot()
            if decision is None:
                return canceled_result(tool_call.id), execution_signal
            if signal.is_set():
                durable_decision = await self.policy.durable_decision(tool_call.id)
                if durable_decision == ApprovalDecision.DENY.value:
                    return ToolResult(
                        tool_call.id, "tool execution denied", True
                    ), execution_signal
                if durable_decision != ApprovalDecision.ALLOW.value:
                    return canceled_result(tool_call.id), execution_signal
                execution_signal = advance_generation(signal)
                if execution_signal.is_set():
                    return canceled_result(tool_call.id), execution_signal
            if decision is ApprovalDecision.DENY:
                message = "tool execution denied"
                denial_reason = getattr(self.policy, "denial_reason", None)
                if callable(denial_reason):
                    reason = denial_reason(tool_call.id)
                    if reason:
                        message = f"{message}: {reason}"
                return ToolResult(tool_call.id, message, True), execution_signal
        if self.hook is None:
            if signal.is_set() and execution_signal is signal:
                return canceled_result(tool_call.id), execution_signal
            return None, execution_signal
        try:
            allowed = self.hook(tool_call.name, arguments)
            if inspect.isawaitable(allowed):
                allowed = await allowed
        except Exception as exc:  # noqa: BLE001 - report hook failures
            return ToolResult(
                tool_call.id, f"pre-execution hook failed: {exc}", True
            ), execution_signal
        if isinstance(allowed, str):
            return ToolResult(tool_call.id, allowed, True), execution_signal
        if allowed is False:
            return ToolResult(
                tool_call.id, "tool execution denied by hook", True
            ), execution_signal
        if signal.is_set() and execution_signal is signal:
            return canceled_result(tool_call.id), execution_signal
        return None, execution_signal
