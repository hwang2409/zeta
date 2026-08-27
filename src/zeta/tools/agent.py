"""The built-in bounded sub-agent tool."""

from __future__ import annotations

import asyncio
from typing import Any

from ..core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from ..core.store import ConversationStore
from ..types import Message, MessageRole, ToolCall, ToolUseContent
from .registry import AbortSignal, ToolRegistry, ToolStreamPublisher, text_block


CHILD_TURN_CAP = 25


class ChildApprovalPolicy:
    """Keep child approval state in both the child and parent stores."""

    def __init__(
        self,
        parent: ApprovalPolicy,
        child_store: ConversationStore,
        description: str,
    ) -> None:
        self.parent = parent
        self.child_store = child_store
        self.description = description

    def bind_store(self, store: ConversationStore) -> None:
        del store

    def decide(self, tool_name: str, arguments: dict[str, Any]) -> ApprovalDecision:
        return self.parent.decide(tool_name, arguments)

    def prepare(self, tool_call: ToolCall) -> Any:
        state = self.child_store.approval_states().get(tool_call.id)
        if state is not None:
            if state[0] != tool_call:
                raise ValueError(f"approval request tool call mismatch: {tool_call.id}")
            return None
        if self.decide(tool_call.name, tool_call.arguments) is not ApprovalDecision.ASK:
            return None
        return ApprovalRequest(
            tool_call.id,
            tool_call,
            label=f"{self.description}: {tool_call.name}",
        )

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
            decision = self.decide(tool_call.name, tool_call.arguments)
            if decision is not ApprovalDecision.ASK:
                return decision
            self.child_store.append_message_with_approval_requests(
                Message(MessageRole.ASSISTANT, [ToolUseContent(tool_call)]),
                [(tool_call.id, tool_call)],
            )

        request = ApprovalRequest(
            tool_call.id,
            tool_call,
            label=f"{self.description}: {tool_call.name}",
        )
        self.parent.register_delegated(request, self.child_store)
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


def _approval_decision(value: str | None) -> ApprovalDecision | None:
    if value == ApprovalDecision.ALLOW.value:
        return ApprovalDecision.ALLOW
    if value == ApprovalDecision.DENY.value:
        return ApprovalDecision.DENY
    return None


def agent_result(
    text: str,
    *,
    error: bool,
    turns_used: int,
    child_session_path: str,
) -> dict[str, object]:
    return {
        "content": [text_block(text)],
        "isError": error,
        "structuredContent": {
            "turns_used": turns_used,
            "child_session_path": child_session_path,
        },
    }


def register(registry: ToolRegistry) -> None:
    async def handler(
        arguments: dict[str, Any],
        abort_signal: AbortSignal,
        stream_publisher: ToolStreamPublisher | None = None,
    ) -> dict[str, object]:
        runner = registry.agent_runner
        call = registry.active_tool_call
        if runner is None or call is None:
            return agent_result(
                "agent error: agent tool is unavailable outside an agent loop",
                error=True,
                turns_used=0,
                child_session_path="",
            )
        return await runner(call, arguments, abort_signal, stream_publisher)

    registry.register(
        "agent",
        handler,
        description=(
            "Delegate multi-step exploration or research that would pollute the "
            "main context. The child has its own bounded context and cannot "
            "spawn further agents."
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {"type": "string"},
            },
            "required": ["prompt", "description"],
            "additionalProperties": False,
        },
        requires_approval=False,
    )
