"""Run durable tool approval and the optional pre-execution hook."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass

from ..core.abort import AbortSignal
from ..core.approval import (
    ApprovalCapability,
    ApprovalDecision,
    ApprovalOrigin,
    ApprovalPolicy,
    canceled_result,
)
from ..protocol.types import ToolCall, ToolResult

ApprovalHook = Callable[
    [str, dict[str, object]], bool | str | Awaitable[bool | str] | None
]
AdvanceGeneration = Callable[[AbortSignal], AbortSignal]


@dataclass(frozen=True, slots=True)
class ApprovalGateResult:
    result: ToolResult | None
    signal: AbortSignal
    origin: ApprovalOrigin

    def __iter__(self) -> Iterator[object]:
        yield self.result
        yield self.signal


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
    ) -> ApprovalGateResult:
        execution_signal = signal
        approval_origin: ApprovalOrigin = "automatic"

        def done(result: ToolResult | None, signal: AbortSignal | None = None) -> ApprovalGateResult:
            return ApprovalGateResult(
                result, execution_signal if signal is None else signal, approval_origin
            )
        if self.policy is not None and not skip_approval:
            # Ephemeral (inline-shell) approvals are never durable rows, so they
            # need no store read here, which keeps the submission-to-registration
            # path free of an off-loop hop and lets concurrent inline approvals
            # register in submit order. Durable approvals keep the log-synced
            # async pre-check, which a delegated child needs to see a resolution
            # a parent (a different store instance) wrote.
            already_decided = (
                persist_request
                and await self.policy.durable_decision(tool_call.id) is not None
            )
            approval_started = (
                not already_decided
                and self.policy.decide(capability) is ApprovalDecision.ASK
            )
            slot_acquired = False
            if approval_started:
                slot_acquired = await self.policy.acquire_approval_slot()
                if lifecycle is not None:
                    lifecycle("approval_start")

            try:
                authorize = getattr(
                    self.policy, "authorize_with_origin", self.policy.authorize
                )
                parameters = inspect.signature(authorize).parameters
                kwargs: dict[str, object] = {}
                if not persist_request and "persist_request" in parameters:
                    kwargs["persist_request"] = False
                if execution_token is not None and "execution_token" in parameters:
                    kwargs["execution_token"] = execution_token
                if capability is not None and "capability" in parameters:
                    kwargs["capability"] = capability
                outcome = await authorize(tool_call, signal, **kwargs)
                if hasattr(outcome, "decision"):
                    decision = outcome.decision
                    approval_origin = outcome.origin
                else:
                    decision = outcome
            except Exception as exc:  # noqa: BLE001 - report approval failures
                return done(ToolResult(
                    tool_call.id, f"approval failed: {exc}", True
                ))
            finally:
                if approval_started:
                    if lifecycle is not None:
                        lifecycle("approval_end")
                    if slot_acquired:
                        self.policy.release_approval_slot()
            if decision is None:
                return done(canceled_result(tool_call.id))
            if signal.is_set():
                durable_decision = await self.policy.durable_decision(tool_call.id)
                if durable_decision == ApprovalDecision.DENY.value:
                    return done(ToolResult(
                        tool_call.id, "tool execution denied", True
                    ))
                if durable_decision != ApprovalDecision.ALLOW.value:
                    return done(canceled_result(tool_call.id))
                execution_signal = advance_generation(signal)
                if execution_signal.is_set():
                    return done(canceled_result(tool_call.id))
            if decision is ApprovalDecision.DENY:
                message = "tool execution denied"
                denial_reason = getattr(self.policy, "denial_reason", None)
                if callable(denial_reason):
                    reason = denial_reason(tool_call.id)
                    if reason:
                        message = f"{message}: {reason}"
                return done(ToolResult(tool_call.id, message, True))
        if self.hook is None:
            if signal.is_set() and execution_signal is signal:
                return done(canceled_result(tool_call.id))
            return done(None)
        try:
            allowed = self.hook(tool_call.name, arguments)
            if inspect.isawaitable(allowed):
                allowed = await allowed
        except Exception as exc:  # noqa: BLE001 - report hook failures
            return done(ToolResult(
                tool_call.id, f"pre-execution hook failed: {exc}", True
            ))
        if isinstance(allowed, str):
            return done(ToolResult(tool_call.id, allowed, True))
        if allowed is False:
            return done(ToolResult(
                tool_call.id, "tool execution denied by hook", True
            ))
        if signal.is_set() and execution_signal is signal:
            return done(canceled_result(tool_call.id))
        return done(None)
