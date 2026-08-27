"""The built-in bounded sub-agent tool."""

from __future__ import annotations

from typing import Any

from ..core.approval import ApprovalDecision, ApprovalPolicy
from ..core.store import ConversationStore
from ..types import ToolCall
from .registry import AbortSignal, ToolRegistry, ToolStreamPublisher, text_block


CHILD_TURN_CAP = 25


class ChildApprovalPolicy:
    """Keep child approval state in both the child and parent stores."""

    def __init__(self, parent: ApprovalPolicy, child_store: ConversationStore) -> None:
        self.parent = parent
        self.child_store = child_store

    def bind_store(self, store: ConversationStore) -> None:
        del store

    def decide(self, tool_name: str, arguments: dict[str, Any]) -> ApprovalDecision:
        return self.parent.decide(tool_name, arguments)

    def prepare(self, tool_call: ToolCall) -> Any:
        return self.parent.prepare(tool_call)

    def durable_decision(self, request_id: str) -> str | None:
        return self.parent.durable_decision(request_id)

    async def authorize(
        self,
        tool_call: ToolCall,
        abort_signal: AbortSignal,
    ) -> ApprovalDecision | None:
        decision = await self.parent.authorize(tool_call, abort_signal)
        self._persist_resolution(tool_call.id, decision)
        return decision

    def abort_or_winner(self, request_id: str) -> ApprovalDecision | None:
        decision = self.parent.abort_or_winner(request_id)
        self._persist_resolution(request_id, decision)
        return decision

    def _persist_resolution(
        self,
        request_id: str,
        decision: ApprovalDecision | None,
    ) -> None:
        self.child_store.resolve_approval(
            request_id,
            "abort" if decision is None else decision.value,
        )


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
