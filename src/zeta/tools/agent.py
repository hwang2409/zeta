"""The built-in bounded sub-agent tool."""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from ..core.store import ConversationStore
from ..model_catalog import known_model_names
from ..types import (
    Message,
    MessageRole,
    StructuredContentValue,
    ToolCall,
    ToolUseContent,
)
from .agent_presets import (
    GENERAL_PRESET,
    AgentType,
    agent_type_description,
    agent_type_names,
)
from .registry import (
    AbortSignal,
    ToolExecutionContext,
    ToolRegistry,
    ToolStreamPublisher,
    text_block,
)

MAX_AGENT_STATUS_STEP = 160
MAX_AGENT_STATUS_RESULT = 4_000
_TRUNCATION_NOTE = "\n[truncated]"


class ChildApprovalPolicy:
    """Keep child approval state in both the child and parent stores."""

    def __init__(
        self,
        parent: ApprovalPolicy,
        child_store: ConversationStore,
        description: str,
        child_instance_id: str,
    ) -> None:
        self.parent = parent
        self.child_store = child_store
        self.description = description
        self.child_instance_id = child_instance_id

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
        self.parent.cleanup_delegated(self.child_instance_id)


def _approval_decision(value: str | None) -> ApprovalDecision | None:
    if value == ApprovalDecision.ALLOW.value:
        return ApprovalDecision.ALLOW
    if value == ApprovalDecision.DENY.value:
        return ApprovalDecision.DENY
    return None


def _agent_status_elapsed(
    started_at: str,
    finished_at: str | None,
    *,
    started_monotonic: float | None = None,
    monotonic_pid: int | None = None,
    stored_elapsed: float | None = None,
) -> float:
    if finished_at is not None and type(stored_elapsed) in {int, float}:
        return max(0.0, float(stored_elapsed))
    if (
        type(started_monotonic) in {int, float}
        and monotonic_pid == os.getpid()
    ):
        return max(0.0, time.monotonic() - started_monotonic)
    try:
        started = datetime.fromisoformat(started_at)
        ended = (
            datetime.fromisoformat(finished_at)
            if finished_at is not None
            else datetime.now(UTC)
        )
        return max(0.0, (ended - started).total_seconds())
    except (TypeError, ValueError):
        return 0.0


def _bounded_status_text(value: object, limit: int) -> str:
    if type(value) is not str:
        return ""
    if len(value) <= limit:
        return value
    return value[: limit - len(_TRUNCATION_NOTE)] + _TRUNCATION_NOTE


def _active_agent_receipts(store: object) -> dict[str, Path]:
    replay = getattr(store, "replay", None)
    if not callable(replay):
        raise TypeError("agent status is unavailable outside an agent session")
    receipts: dict[str, Path] = {}
    for entry in replay():
        if entry.type != "message":
            continue
        message = Message.from_dict(entry.data["message"])
        result = message.tool_result
        structured = result.structured_content if result is not None else None
        if structured is None:
            continue
        handle = structured.get("child_instance_id")
        child_path = structured.get("child_session_path")
        if type(handle) is str and handle and type(child_path) is str and child_path:
            receipts[handle] = Path(child_path)
    return receipts


def _read_agent_status(
    store: object,
    *,
    requested_handle: str | None = None,
) -> list[dict[str, StructuredContentValue]]:
    session_dir = getattr(store, "session_dir", None)
    if not isinstance(session_dir, Path):
        raise TypeError("agent status is unavailable outside an agent session")
    active_receipts = _active_agent_receipts(store)
    if requested_handle is not None and requested_handle not in active_receipts:
        raise ValueError(f"unknown child handle: {requested_handle}")
    children: list[dict[str, StructuredContentValue]] = []
    for handle, child_path in active_receipts.items():
        if requested_handle is not None and handle != requested_handle:
            continue
        lifecycle_path = child_path / "agent_lifecycle.json"
        try:
            lifecycle = json.loads(lifecycle_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError(f"could not read child state: {lifecycle_path}") from exc
        if type(lifecycle) is not dict:
            continue
        lifecycle_handle = lifecycle.get("handle")
        if lifecycle_handle != handle:
            raise ValueError(f"child handle mismatch: {lifecycle_path}")
        started_at = lifecycle.get("started_at")
        finished_at = lifecycle.get("finished_at")
        if type(started_at) is not str or (
            finished_at is not None and type(finished_at) is not str
        ):
            raise ValueError(f"invalid child timestamps: {lifecycle_path}")
        item: dict[str, StructuredContentValue] = {
            "handle": handle,
            "state": lifecycle.get("state", "failed"),
            "started_at": started_at,
            "finished_at": finished_at,
            "elapsed": _agent_status_elapsed(
                started_at,
                finished_at,
                started_monotonic=lifecycle.get("started_monotonic"),
                monotonic_pid=lifecycle.get("monotonic_pid"),
                stored_elapsed=lifecycle.get("elapsed"),
            ),
            "turns_used": lifecycle.get("turns_used", 0),
            "tree_budget": lifecycle.get("tree_budget", 0),
            "current_step": _bounded_status_text(
                lifecycle.get("current_step", "unknown"), MAX_AGENT_STATUS_STEP
            ),
            "depth": lifecycle.get("depth", 0),
            "agent_type": lifecycle.get("agent_type", "general"),
            "description": lifecycle.get("description", ""),
        }
        if finished_at is not None and requested_handle is not None:
            item["final_result"] = _bounded_status_text(
                lifecycle.get("final_result", ""), MAX_AGENT_STATUS_RESULT
            )
        children.append(item)
    return sorted(children, key=lambda child: str(child["started_at"]))


async def _agent_status(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: object = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    del abort_signal, stream_publisher, execution_context
    requested = arguments.get("handle")
    if requested is not None and (type(requested) is not str or not requested):
        return {
            "content": [text_block("agent error: handle must be a nonempty string")],
            "isError": True,
            "structuredContent": None,
        }
    try:
        children = _read_agent_status(
            registry.session_store,
            requested_handle=requested,
        )
    except (TypeError, ValueError) as exc:
        return {
            "content": [text_block(f"agent error: {exc}")],
            "isError": True,
            "structuredContent": None,
        }
    if requested is not None:
        matching = [child for child in children if child["handle"] == requested]
        if not matching:
            return {
                "content": [text_block(f"agent error: unknown child handle: {requested}")],
                "isError": True,
                "structuredContent": None,
            }
        children = matching
    count = len(children)
    label = "child" if count == 1 else "children"
    return {
        "content": [text_block(f"agent status: {count} {label}")],
        "isError": False,
        "structuredContent": {"children": children},
    }


def agent_result(
    text: str,
    *,
    error: bool,
    turns_used: int,
    child_session_path: str,
    agent_type: AgentType | None = None,
    status: str | None = None,
    child_instance_id: str | None = None,
    description: str | None = None,
    depth: int | None = None,
    budget_exhausted: bool = False,
) -> dict[str, object]:
    structured_content: dict[str, object] = {
        "turns_used": turns_used,
        "child_session_path": child_session_path,
    }
    if agent_type is not None and agent_type != GENERAL_PRESET.name:
        structured_content["agent_type"] = agent_type
    if status is not None:
        structured_content["status"] = status
    if child_instance_id is not None:
        structured_content["child_instance_id"] = child_instance_id
    if description is not None:
        structured_content["description"] = description
    # Keep depth-one result payloads byte-compatible with the pre-nesting shape.
    if depth is not None and depth != 1:
        structured_content["depth"] = depth
    if budget_exhausted:
        structured_content["error_code"] = "agent_turn_budget"
    return {
        "content": [text_block(text)],
        "isError": error,
        "structuredContent": structured_content,
    }


async def _agent(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: ToolStreamPublisher | None = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    del registry
    runner = execution_context.agent_runner if execution_context is not None else None
    call = execution_context.tool_call if execution_context is not None else None
    if runner is None or call is None:
        return agent_result(
            "agent error: agent tool is unavailable outside an agent loop",
            error=True,
            turns_used=0,
            child_session_path="",
        )
    return await runner(
        call,
        arguments,
        abort_signal,
        stream_publisher,
        execution_context,
    )


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "agent",
        _agent,
        description=(
            "Delegate multi-step exploration or research that would pollute the "
            "main context. The child has its own bounded context and may spawn "
            "one level of grandchildren, but grandchildren cannot spawn agents. "
            "Pass model to run the child on another provider's model and "
            "orchestrate it from here. The returned child_instance_id is the "
            "stable handle for agent_status. Built-in types: "
            f"{agent_type_description()}"
        ),
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "description": {"type": "string"},
                "agent_type": {
                    "type": "string",
                    "enum": agent_type_names(),
                    "description": agent_type_description(),
                },
                "model": {
                    "type": "string",
                    "enum": known_model_names(),
                    "description": (
                        "Run the child on this model instead of inheriting the "
                        "parent's. The provider follows from the model, so this "
                        "is how one provider delegates to another. Implies "
                        "background unless background is passed explicitly."
                    ),
                },
                "background": {
                    "type": "boolean",
                    "description": "Keep the child running across parent turns and return a handle.",
                },
            },
            "required": ["prompt", "description"],
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=True,
    )
    registry.register_session_tool(
        "agent_status",
        _agent_status,
        description=(
            "Inspect child agents from this session. Pass a child handle for "
            "one child, or omit it to list every child. This is read-only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "description": "Stable child handle returned by the agent tool.",
                }
            },
            "additionalProperties": False,
        },
        parallel_safe=True,
        requires_approval=False,
    )
