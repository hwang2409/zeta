"""Approval and pre-execution gating."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .abort import AbortSignal
from .approval import ApprovalDecision, ApprovalPolicy
from .types import ToolCall, ToolResult


ApprovalHook = Callable[[str, dict[str, Any]], bool | Awaitable[bool] | None]
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
        arguments: dict[str, Any],
        signal: AbortSignal,
        advance_generation: AdvanceGeneration,
    ) -> tuple[ToolResult | None, AbortSignal]:
        execution_signal = signal
        if self.policy is not None:
            try:
                decision = await self.policy.authorize(tool_call, signal)
            except Exception as exc:
                return ToolResult(tool_call.id, f"approval failed: {exc}", True), execution_signal
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
        if allowed is False:
            return ToolResult(tool_call.id, "tool execution denied", True), execution_signal
        if signal.is_set() and execution_signal is signal:
            return canceled_result(tool_call.id), execution_signal
        return None, execution_signal
