"""The built-in bounded sub-agent tool."""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..agent_receipt import (
    MAX_AGENT_RESULT_BYTES,
    agent_stats,
    build_agent_progress,
    build_agent_receipt,
    encode_json,
    format_agent_stats,
    terminal_state,
)
from ..core.approval import ApprovalDecision, ApprovalPolicy, ApprovalRequest
from ..core.checkpoints import ConversationEntry, ConversationIntegrityError
from ..core.store import (
    ConversationStore,
    PendingPromptCommitTimeoutError,
    PendingPromptsClosedError,
)
from ..model_catalog import known_model_names
from ..types import (
    Message,
    MessageRole,
    StructuredContentValue,
    TextContent,
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
MAX_AGENT_STATUS_DESCRIPTION = 160
AGENT_SEND_COMMIT_TIMEOUT_SECONDS = 1.0
_TRUNCATION_NOTE = "\n[truncated]"


def _read_agent_lifecycle(path: str) -> dict[str, object]:
    if not path:
        return {}
    try:
        value = json.loads(
            (Path(path) / "agent_lifecycle.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError, RecursionError):
        return {}
    return value if type(value) is dict else {}


def _agent_error(message: str, max_bytes: int) -> dict[str, object]:
    result = {
        "content": [text_block(f"agent error: {message}")],
        "isError": True,
        "structuredContent": None,
    }
    if len(encode_json(result)) <= max_bytes:
        return result
    return {
        "content": [text_block("agent error: response exceeds response limit")],
        "isError": True,
        "structuredContent": None,
    }


def _bounded_agent_result(
    result: dict[str, object], max_bytes: int
) -> dict[str, object]:
    if len(encode_json(result)) <= max_bytes:
        return result
    return _agent_error("response exceeds response limit", max_bytes)


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

    def declare_subjects(
        self, subjects: Mapping[str, str | None]
    ) -> tuple[str, ...]:
        # Child tools are clones of the parent's, so the parent already holds
        # every subject; declarations merge, so pushing the subset is safe.
        return self.parent.declare_subjects(subjects)

    @property
    def notices(self) -> tuple[str, ...]:
        return self.parent.notices

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
    if type(started_monotonic) in {int, float} and monotonic_pid == os.getpid():
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


def _canonical_child_path(session_dir: Path, value: object) -> Path | None:
    if type(value) is not str or not value:
        return None
    try:
        session_root = session_dir.resolve()
        child_path = Path(value).resolve()
        relative = child_path.relative_to(session_root)
    except (OSError, RuntimeError, ValueError):
        return None
    parts = relative.parts
    if (
        len(parts) < 2
        or len(parts) % 2
        or any(parts[index] != "agents" for index in range(0, len(parts), 2))
        or any(not parts[index].isdigit() for index in range(1, len(parts), 2))
    ):
        return None
    return child_path


def _active_agent_receipts(store: object) -> dict[str, Path]:
    replay = getattr(store, "replay", None)
    if not callable(replay):
        raise TypeError("agent status is unavailable outside an agent session")
    session_dir = getattr(store, "session_dir", None)
    if not isinstance(session_dir, Path):
        raise TypeError("agent status is unavailable outside an agent session")
    entries = replay()
    agent_call_ids: set[str] = set()
    for entry in entries:
        if entry.type != "message":
            continue
        message = Message.from_dict(entry.data["message"])
        agent_call_ids.update(
            block.tool_call.id
            for block in message.content
            if isinstance(block, ToolUseContent)
            and block.tool_call.name.casefold() == "agent"
        )

    receipts: dict[str, Path] = {}

    def add_receipt(handle: object, child_path: object) -> None:
        if type(handle) is not str or not handle:
            return
        canonical_path = _canonical_child_path(session_dir, child_path)
        if canonical_path is not None:
            receipts.setdefault(handle, canonical_path)

    for entry in entries:
        if entry.type == "message":
            message = Message.from_dict(entry.data["message"])
            result = message.tool_result
            if result is None or result.tool_call_id not in agent_call_ids:
                continue
            structured = result.structured_content
            if structured is not None:
                add_receipt(
                    structured.get("child_instance_id"),
                    structured.get("child_session_path"),
                )
        elif entry.type == "notification":
            data = entry.data
            add_receipt(
                data.get("child_instance_id"),
                data.get("child_session_path"),
            )

    agent_children = getattr(store, "agent_children", None)
    if callable(agent_children):
        for marker in agent_children().values():
            if type(marker) is not dict:
                continue
            raw_tool_call = marker.get("tool_call")
            if type(raw_tool_call) is not dict:
                continue
            try:
                tool_call = ToolCall.from_dict(raw_tool_call)
            except ValueError:
                continue
            if tool_call.name.casefold() != "agent":
                continue
            add_receipt(
                marker.get("child_instance_id"),
                marker.get("child_session_path"),
            )
    return receipts


def _agent_output_text(message: Message) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
        elif isinstance(block, ToolUseContent):
            arguments = json.dumps(
                block.tool_call.arguments, ensure_ascii=False, sort_keys=True
            )
            parts.append(f"[tool call: {block.tool_call.name} {arguments}]")
    if message.tool_result is not None and not parts:
        parts.append(message.tool_result.content)
    return "\n".join(parts)


def _serialize_agent_transcript(messages: list[Message]) -> str:
    lines: list[str] = []
    for message in messages:
        text = _agent_output_text(message)
        if text:
            lines.append(f"{message.role.value}: {text}")
    return "\n".join(lines) + ("\n" if lines else "")


def _agent_output_page(
    output: str,
    *,
    handle: str,
    offset: int,
    limit: int | None,
    max_bytes: int,
) -> dict[str, object] | None:
    total = len(output)
    requested_end = total if limit is None else min(total, offset + limit)

    def build(candidate: str, candidate_end: int) -> dict[str, object]:
        block = text_block(candidate, full_size=len(output.encode("utf-8")))
        block["full_size_chars"] = total
        block["truncated"] = candidate_end < total
        if candidate_end < total:
            block["next_offset"] = candidate_end
        return {
            "content": [block],
            "isError": False,
            "structuredContent": {
                "handle": handle,
                "offset": offset,
                "truncated": candidate_end < total,
                "total": total,
                **({"next_offset": candidate_end} if candidate_end < total else {}),
            },
        }

    low = offset
    high = requested_end
    while low < high:
        middle = (low + high + 1) // 2
        result = build(output[offset:middle], middle)
        if len(encode_json(result)) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    result = build(output[offset:low], low)
    return result if len(encode_json(result)) <= max_bytes else None


def _read_agent_output(
    store: object,
    *,
    requested_handle: str,
    offset: int,
    limit: int | None,
    max_bytes: int,
) -> dict[str, object]:
    receipts = _active_agent_receipts(store)
    child_path = receipts.get(requested_handle)
    if child_path is None:
        raise ValueError("unknown child handle")
    try:
        if not child_path.is_dir() or not (child_path / "conversation.jsonl").is_file():
            raise OSError("child transcript is missing")
        raw = (child_path / "conversation.jsonl").read_bytes()
        rows = raw.splitlines(keepends=True)
        entries: list[ConversationEntry] = []
        for index, line in enumerate(rows):
            try:
                row = json.loads(line)
            except (ValueError, RecursionError) as exc:
                if index == len(rows) - 1 and not line.endswith(b"\n"):
                    break
                raise ConversationIntegrityError(
                    f"invalid conversation row {index + 1}: {child_path}"
                ) from exc
            if index == 0:
                if not isinstance(row, dict) or row.get("type") != "header":
                    raise ConversationIntegrityError(
                        f"unsupported conversation schema: {child_path}"
                    )
                continue
            if not isinstance(row, dict):
                raise ConversationIntegrityError(
                    f"conversation row {index + 1} is not an object: {child_path}"
                )
            entries.append(ConversationEntry.from_dict(row))
        by_id = {entry.id: entry for entry in entries}
        current = entries[-1] if entries else None
        branch: list[ConversationEntry] = []
        seen: set[str] = set()
        while current is not None:
            if current.id in seen:
                raise ConversationIntegrityError(
                    f"conversation parent cycle at {current.id}"
                )
            seen.add(current.id)
            branch.append(current)
            current = by_id.get(current.parent_id) if current.parent_id else None
        messages = [
            Message.from_dict(entry.data["message"])
            for entry in reversed(branch)
            if entry.type == "message"
        ]
        output = _serialize_agent_transcript(messages)
    except (OSError, ValueError, ConversationIntegrityError) as exc:
        raise ValueError("could not read child transcript") from exc
    result = _agent_output_page(
        output,
        handle=requested_handle,
        offset=offset,
        limit=limit,
        max_bytes=max_bytes,
    )
    if result is None:
        raise ValueError("transcript page exceeds response limit")
    return result


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
        raise ValueError("unknown child handle")
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
            "tool_calls": lifecycle.get("tool_calls", 0),
            "tree_budget": lifecycle.get("tree_budget", 0),
            "current_step": _bounded_status_text(
                lifecycle.get("current_step", "unknown"), MAX_AGENT_STATUS_STEP
            ),
            "depth": lifecycle.get("depth", 0),
            "agent_type": lifecycle.get("agent_type", "general"),
            "description": _bounded_status_text(
                lifecycle.get("description", ""), MAX_AGENT_STATUS_DESCRIPTION
            ),
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
    max_bytes = registry.max_output_chars
    requested = arguments.get("handle")
    if requested is not None and (type(requested) is not str or not requested):
        return _agent_error("handle must be a nonempty string", max_bytes)
    try:
        children = _read_agent_status(
            registry.session_store,
            requested_handle=requested,
        )
    except (TypeError, ValueError) as exc:
        return _agent_error(str(exc), max_bytes)
    if requested is not None:
        matching = [child for child in children if child["handle"] == requested]
        if not matching:
            return _agent_error("unknown child handle", max_bytes)
        children = matching
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    if type(offset) is not int or offset < 0:
        return _agent_error("offset must be a nonnegative integer", max_bytes)
    if limit is not None and (type(limit) is not int or limit < 1):
        return _agent_error("limit must be a positive integer", max_bytes)
    page = children[offset : offset + limit if limit is not None else None]
    while page:
        next_offset = offset + len(page)
        truncated = next_offset < len(children)
        notice = (
            f"\nmore children available: call agent_status with offset={next_offset}"
            if truncated
            else ""
        )
        details = "\n".join(
            (
                "child {handle}: state: {state}; started_at: {started_at}; "
                "finished_at: {finished_at}; elapsed: {elapsed:.2f}s; "
                "turns_used: {turns_used}/{tree_budget}; step: {current_step}; "
                "result: {final_result}"
            ).format(**{**child, "final_result": child.get("final_result", "")})
            + format_agent_stats(child)
            for child in page
        )
        content_text = f"agent status: {len(children)} children\n{details}{notice}"
        structured = {
            "children": page,
            "offset": offset,
            "truncated": truncated,
            "total": len(children),
        }
        if truncated:
            structured["next_offset"] = next_offset
        result = {
            "content": [text_block(content_text)],
            "isError": False,
            "structuredContent": structured,
        }
        if len(encode_json(result)) <= max_bytes:
            return result
        page.pop()
    if offset < len(children):
        return _agent_error("status item exceeds response limit", max_bytes)
    count = len(children)
    label = "child" if count == 1 else "children"
    if count == 0 and offset == 0 and limit is None:
        return _bounded_agent_result(
            {
                "content": [text_block(f"agent status: {count} {label}")],
                "isError": False,
                "structuredContent": {"children": []},
            },
            max_bytes,
        )
    return _bounded_agent_result(
        {
            "content": [text_block(f"agent status: {count} {label}")],
            "isError": False,
            "structuredContent": {
                "children": [],
                "offset": offset,
                "truncated": False,
                "total": count,
            },
        },
        max_bytes,
    )


async def _agent_output(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    stream_publisher: object = None,
    execution_context: ToolExecutionContext | None = None,
) -> dict[str, object]:
    del abort_signal, stream_publisher, execution_context
    max_bytes = registry.max_output_chars
    handle = arguments.get("handle")
    if type(handle) is not str or not handle:
        return _agent_error("handle must be a nonempty string", max_bytes)
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    if type(offset) is not int or offset < 0:
        return _agent_error("offset must be a nonnegative integer", max_bytes)
    if limit is not None and (type(limit) is not int or limit < 1):
        return _agent_error("limit must be a positive integer", max_bytes)
    try:
        return _bounded_agent_result(
            _read_agent_output(
                registry.session_store,
                requested_handle=handle,
                offset=offset,
                limit=limit,
                max_bytes=max_bytes,
            ),
            max_bytes,
        )
    except (TypeError, ValueError) as exc:
        return _agent_error(str(exc), max_bytes)


def agent_result(
    text: str,
    *,
    tool_call_id: str = "",
    error: bool,
    turns_used: int,
    child_session_path: str,
    agent_type: AgentType | None = None,
    status: str | None = None,
    child_instance_id: str | None = None,
    description: str | None = None,
    depth: int | None = None,
    budget_exhausted: bool = False,
    stats: dict[str, object] | None = None,
    include_stats: bool = True,
    canceled: bool = False,
    max_bytes: int = MAX_AGENT_RESULT_BYTES,
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
    if stats is None:
        lifecycle = _read_agent_lifecycle(child_session_path)
        stats = agent_stats(
            lifecycle,
            status=status
            or ("canceled" if canceled else "failed" if error else "completed"),
            turns_used=turns_used,
        )
    if status == "running":
        return build_agent_progress(
            text,
            stats,
            structured_content=structured_content,
            tool_call_id=tool_call_id,
            max_bytes=max_bytes,
            include_stats=include_stats,
        )
    state = terminal_state(error=error, canceled=canceled, status=status)
    return build_agent_receipt(
        state,
        text,
        stats,
        structured_content=structured_content,
        tool_call_id=tool_call_id,
        max_bytes=max_bytes,
    )


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
                "max_turns": {
                    "type": "integer",
                    "minimum": 1,
                    "description": (
                        "Raise the shared turn budget for this agent tree "
                        "(root + descendants). Only accepted on the top-level "
                        "agent call; children inherit the tree budget."
                    ),
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
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Child index for a status page continuation.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of children in this status page.",
                },
            },
            "additionalProperties": False,
        },
        parallel_safe=True,
        requires_approval=False,
    )
    registry.register_session_tool(
        "agent_output",
        _agent_output,
        description=(
            "Read a child agent transcript by handle. Output is bounded and "
            "paginated by character offset. This is read-only."
        ),
        parameters={
            "type": "object",
            "properties": {
                "handle": {
                    "type": "string",
                    "minLength": 1,
                    "description": "Stable child handle returned by the agent tool.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Readable transcript character offset.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum readable transcript characters.",
                },
            },
            "required": ["handle"],
            "additionalProperties": False,
        },
        parallel_safe=True,
        requires_approval=False,
    )
    register_send(registry)


def send_to_run(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
) -> str | None:
    """Queue a follow-up for a live run, returning an error message if not possible.

    Shared by the agent_send tool and the /send command so both reach a run the
    same way.
    """

    if type(child_instance_id) is not str or not child_instance_id.strip():
        return "child_instance_id must be a nonempty string"
    if type(message) is not str or not message.strip():
        return "message must be a nonempty string"
    # The marker is removed once a run's result is durable, so a missing one
    # means the run already finished rather than that it never existed.
    no_live_run = (
        f"no live run {child_instance_id!r}; it already finished or was "
        "never started"
    )
    marker = parent_store.agent_children().get(child_instance_id)
    if marker is None:
        return no_live_run
    # Only runs drain queued follow-ups. Other agent types would leave the
    # prompt in the child's store with no one to consume it.
    agent_type = marker.get("agent_type") or "general"
    if agent_type != "run":
        return (
            f"{child_instance_id!r} is a {agent_type} agent, not a run; "
            "agent_send only works with agent_type=run"
        )
    deadline = time.monotonic() + AGENT_SEND_COMMIT_TIMEOUT_SECONDS
    try:
        child_path = Path(str(marker["child_session_path"]))
        child_store = ConversationStore(
            child_path.parent,
            session_id=child_path.name,
            cwd=parent_store.cwd,
            _lock_deadline=deadline,
        )
        child_store.pending_prompt_queue.append(message, deadline=deadline)
    except PendingPromptCommitTimeoutError:
        return "pending prompt commit timed out before the queue could be changed"
    except PendingPromptsClosedError:
        # consume_run closed the queue while we were checking the marker.
        return no_live_run
    return None


async def _agent_send(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> dict[str, object]:
    # send_to_run reloads the run's own conversation.jsonl and appends under
    # flock+fsync; runs with large logs would stall the event loop, so hop
    # to a worker thread while the marker check and durable append happen.
    commit = asyncio.create_task(
        asyncio.to_thread(
            send_to_run,
            registry.session_store,
            arguments.get("child_instance_id"),
            arguments.get("message"),
        )
    )
    while True:
        try:
            error = await asyncio.shield(commit)
        except asyncio.CancelledError:
            # The worker cannot be canceled. Absorb every caller cancellation
            # until its one commit decision is known.
            if commit.cancelled():
                raise
            continue
        break
    if error is not None:
        return {
            "content": [text_block(f"agent_send error: {error}")],
            "isError": True,
            "structuredContent": None,
        }
    child_instance_id = arguments["child_instance_id"]
    return {
        "content": [
            text_block(
                f"queued a follow-up for {child_instance_id}; it is delivered "
                "when the run finishes its current turn"
            )
        ],
        "isError": False,
        "structuredContent": {"child_instance_id": child_instance_id},
    }


def register_send(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "agent_send",
        _agent_send,
        description=(
            "Send a follow-up instruction to a run you started that is still "
            "working. The run picks it up at its next turn boundary, so it "
            "never interrupts a tool call. Use the child_instance_id the agent "
            "tool returned."
        ),
        parameters={
            "type": "object",
            "properties": {
                "child_instance_id": {"type": "string", "minLength": 1},
                "message": {"type": "string", "minLength": 1},
            },
            "required": ["child_instance_id", "message"],
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=True,
    )
