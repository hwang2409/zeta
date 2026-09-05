"""Build bounded terminal receipts for child agents."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from .core.checkpoints import ConversationEntry
from .types import (
    Message,
    MessageRole,
    StructuredToolResult,
    TextContent,
    ToolCall,
    ToolResult,
    flatten_tool_content,
)

ReceiptState = Literal["running", "completed", "failed", "canceled"]
TerminalState = Literal["completed", "failed", "canceled"]
MAX_AGENT_RESULT_BYTES = 10_000
_TRUNCATION_NOTE = "\n[truncated]"
_PERSISTED_ENTRY_ID = "0" * 32
_PERSISTED_ENTRY_SEQ = 10**100
_SUFFIX_RE = re.compile(
    r" · (?P<turns>\d+) turns · (?P<elapsed>\d+\.\d+)s"
    r" · (?P<tool_calls>\d+) tool calls"
    r" · error=(?P<error>true|false) · canceled=(?P<canceled>true|false)$"
)


def encode_json(value: object) -> bytes:
    """Encode JSON exactly as receipt rows and receipt payloads are encoded."""

    return json.dumps(
        value,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def json_size(value: object) -> int:
    return len(encode_json(value))


def format_agent_stats(
    stats: object, *, state: TerminalState | None = None
) -> str:
    if type(stats) is not dict:
        return ""
    turns = stats.get("turns_used")
    elapsed = stats.get("elapsed")
    tool_calls = stats.get("tool_calls")
    stats_state = stats.get("state")
    error = stats.get("error")
    canceled = stats.get("canceled")
    if (
        type(turns) is not int
        or turns < 0
        or type(elapsed) not in {int, float}
        or elapsed < 0
        or type(tool_calls) is not int
        or tool_calls < 0
    ):
        return ""
    if state is not None:
        error = state == "failed"
        canceled = state == "canceled"
    elif type(stats_state) is str:
        if stats_state not in {
            "running",
            "completed",
            "failed",
            "canceled",
            "error",
        }:
            return ""
        error = stats_state in {"failed", "error"}
        canceled = stats_state == "canceled"
    elif type(error) is not bool or type(canceled) is not bool:
        return ""
    return (
        f" · {turns} turns · {elapsed:.1f}s · {tool_calls} tool calls"
        f" · error={str(error).lower()} · canceled={str(canceled).lower()}"
    )


def agent_stats(
    lifecycle: dict[str, object] | None,
    *,
    status: str | None = None,
    turns_used: int = 0,
) -> dict[str, object]:
    lifecycle = lifecycle or {}
    elapsed = lifecycle.get("elapsed", 0.0)
    if type(elapsed) not in {int, float} or elapsed < 0:
        elapsed = 0.0
    turns = lifecycle.get("turns_used", turns_used)
    if type(turns) is not int or turns < 0:
        turns = turns_used
    tool_calls = lifecycle.get("tool_calls", 0)
    if type(tool_calls) is not int or tool_calls < 0:
        tool_calls = 0
    state = status or lifecycle.get("state")
    if state == "error":
        state = "failed"
    return {
        "turns_used": turns,
        "elapsed": elapsed,
        "tool_calls": tool_calls,
        "error": state in {"failed", "error"},
        "canceled": state == "canceled",
    }


def terminal_state(
    *, error: bool = False, canceled: bool = False, status: str | None = None
) -> TerminalState:
    if status == "canceled" or canceled:
        return "canceled"
    if status in {"failed", "error"} or error:
        return "failed"
    if status == "completed" or status is None:
        return "completed"
    raise ValueError(f"unsupported terminal agent state: {status}")


def _text_block(text: str) -> dict[str, object]:
    return {
        "type": "text",
        "text": text,
        "truncated": False,
        "full_size": len(text.encode("utf-8")),
    }


def _candidate(
    state: ReceiptState,
    answer: str,
    suffix: str,
    structured_content: dict[str, Any] | None,
    tool_call_id: str,
) -> StructuredToolResult:
    text = answer + suffix
    result: StructuredToolResult = {
        "content": [_text_block(text)],
        "isError": state == "failed",
        "structuredContent": structured_content,
    }
    if state == "canceled":
        result["isCanceled"] = True
    return result


def _serialized_sizes(
    result: StructuredToolResult,
    tool_call_id: str,
    envelope: Callable[[StructuredToolResult], StructuredToolResult] | None = None,
) -> tuple[int, int]:
    measured = envelope(result) if envelope is not None else result
    payload_size = json_size(measured)
    text = measured["content"][0]["text"]
    tool_result = ToolResult(
        tool_call_id,
        text,
        measured["isError"],
        content_blocks=measured["content"],
        structured_content=measured["structuredContent"],
        is_canceled=measured.get("isCanceled", False),
    )
    message = Message(
        MessageRole.TOOL_RESULT,
        [TextContent(text)],
        tool_result=tool_result,
    )
    message_data = message.to_dict()
    persisted_entry = ConversationEntry(
        seq=_PERSISTED_ENTRY_SEQ,
        id=_PERSISTED_ENTRY_ID,
        parent_id=_PERSISTED_ENTRY_ID,
        lane="main",
        type="message",
        data={"message": message_data},
    )
    persisted_row_size = json_size(persisted_entry.to_dict()) + 1
    return payload_size, persisted_row_size


def _with_answer_limit(
    state: ReceiptState,
    answer: str,
    suffix: str,
    structured_content: dict[str, Any] | None,
    tool_call_id: str,
    max_bytes: int,
    envelope: Callable[[StructuredToolResult], StructuredToolResult] | None = None,
) -> StructuredToolResult:
    full = _candidate(state, answer, suffix, structured_content, tool_call_id)
    if max(_serialized_sizes(full, tool_call_id, envelope)) <= max_bytes:
        return full

    def candidate(length: int) -> StructuredToolResult:
        shown = answer[:length]
        if shown != answer:
            shown += _TRUNCATION_NOTE
        return _candidate(state, shown, suffix, structured_content, tool_call_id)

    low = 0
    high = len(answer)
    while low < high:
        middle = (low + high + 1) // 2
        if max(_serialized_sizes(candidate(middle), tool_call_id, envelope)) <= max_bytes:
            low = middle
        else:
            high = middle - 1
    bounded = candidate(low)
    if max(_serialized_sizes(bounded, tool_call_id, envelope)) <= max_bytes:
        return bounded

    minimal = _candidate(state, "", suffix, structured_content, tool_call_id)
    if max(_serialized_sizes(minimal, tool_call_id, envelope)) <= max_bytes:
        return minimal

    reduced = _candidate(state, "", suffix, None, tool_call_id)
    if max(_serialized_sizes(reduced, tool_call_id, envelope)) <= max_bytes:
        return reduced
    return _candidate(state, "", "", None, tool_call_id)


def _governance_envelope(
    tool_name: str,
) -> Callable[[StructuredToolResult], StructuredToolResult]:
    from .tools.registry import _apply_error_governance

    return lambda result: _apply_error_governance(result, tool_name)


def build_agent_receipt(
    state: TerminalState,
    answer: str,
    stats: Mapping[str, object] | None,
    *,
    structured_content: dict[str, Any] | None = None,
    tool_call_id: str = "",
    max_bytes: int = MAX_AGENT_RESULT_BYTES,
) -> StructuredToolResult:
    """Build one bounded terminal result with canonical flags and stats."""

    if state not in {"completed", "failed", "canceled"}:
        raise ValueError(f"unsupported terminal agent state: {state}")
    if type(answer) is not str:
        raise TypeError("agent receipt answer must be a string")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("agent receipt byte limit must be positive")
    answer = _without_agent_receipt_suffix(answer)
    suffix = format_agent_stats(
        dict(stats) if stats is not None else {}, state=state
    )
    envelope = _governance_envelope("agent") if state == "failed" else None
    return _with_answer_limit(
        state,
        answer,
        suffix,
        dict(structured_content) if structured_content is not None else None,
        tool_call_id,
        max_bytes,
        envelope=envelope,
    )


def build_agent_progress(
    answer: str,
    stats: Mapping[str, object] | None,
    *,
    structured_content: dict[str, Any] | None = None,
    tool_call_id: str = "",
    max_bytes: int = MAX_AGENT_RESULT_BYTES,
    include_stats: bool = True,
) -> StructuredToolResult:
    """Build a bounded non-terminal result for a running child."""

    if type(answer) is not str:
        raise TypeError("agent progress answer must be a string")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("agent progress byte limit must be positive")
    answer = _without_agent_receipt_suffix(answer)
    suffix = (
        format_agent_stats(dict(stats) if stats is not None else {})
        if include_stats
        else ""
    )
    return _with_answer_limit(
        "running",
        answer,
        suffix,
        dict(structured_content) if structured_content is not None else None,
        tool_call_id,
        max_bytes,
    )


def has_agent_receipt_suffix(value: str) -> bool:
    """Return whether a string ends with the canonical receipt suffix."""

    return _SUFFIX_RE.search(value) is not None


def _without_agent_receipt_suffix(value: str) -> str:
    return _SUFFIX_RE.sub("", value)


def ensure_agent_receipt_text(
    answer: str,
    state: TerminalState,
    stats: Mapping[str, object] | None,
) -> str:
    match = _SUFFIX_RE.search(answer)
    if match is not None:
        expected_error = state == "failed"
        expected_canceled = state == "canceled"
        if (
            match.group("error") == str(expected_error).lower()
            and match.group("canceled") == str(expected_canceled).lower()
        ):
            return answer
        stats = {
            "turns_used": int(match.group("turns")),
            "elapsed": float(match.group("elapsed")),
            "tool_calls": int(match.group("tool_calls")),
            "error": expected_error,
            "canceled": expected_canceled,
        }
    result = build_agent_receipt(state, answer, stats)
    content = result["content"][0]
    return content["text"] if content["type"] == "text" else answer


def receipt_message_size(result: StructuredToolResult, tool_call_id: str = "") -> int:
    """Measure the exact message JSON used when a receipt is persisted."""

    return _serialized_sizes(result, tool_call_id)[1]


def receipt_tool_result(
    tool_call_id: str, result: StructuredToolResult
) -> ToolResult:
    return ToolResult(
        tool_call_id,
        flatten_tool_content(result["content"]),
        result["isError"],
        content_blocks=result["content"],
        structured_content=result["structuredContent"],
        is_canceled=result.get("isCanceled", False),
    )


class _ReceiptLoop(Protocol):
    store: Any
    hooks: Any
    _agent_child_stores: Any
    _agent_child_turns: Any
    _agent_child_types: Any
    _background_owner: Any
    agent_instance_id: str | None

    def _existing_tool_result(self, tool_call_id: str) -> ToolResult | None: ...

    def _child_result_payload(
        self, tool_call_id: str, content: str, **kwargs: Any
    ) -> dict[str, Any]: ...

    def _canceled_agent_result(
        self, tool_call_id: str, **kwargs: Any
    ) -> ToolResult: ...


def finalize_agent_results(
    owner: _ReceiptLoop,
    calls: Sequence[ToolCall],
    slots: Sequence[ToolResult | None],
) -> list[ToolResult]:
    results: list[ToolResult] = []
    new_results: list[tuple[ToolCall, ToolResult]] = []
    for call, slot in zip(calls, slots, strict=True):
        result = owner._existing_tool_result(call.id)
        stored_result = result
        child_store = owner._agent_child_stores.get(call.id)
        candidate = result if result is not None else slot
        if (
            call.name.casefold() == "agent"
            and candidate is not None
            and not (
                candidate.structured_content is not None
                and candidate.structured_content.get("status") == "running"
            )
        ):
            metadata = candidate.structured_content or {}
            payload = owner._child_result_payload(
                call.id,
                candidate.content,
                state=terminal_state(
                    error=candidate.is_error,
                    canceled=candidate.is_canceled,
                ),
                child_session_path=(
                    str(child_store.session_dir) if child_store is not None else None
                ),
                child_instance_id=(
                    metadata.get("child_instance_id")
                    if type(metadata.get("child_instance_id")) is str
                    else child_store.agent_handle()
                    if child_store is not None
                    else None
                ),
                agent_type=(
                    metadata.get("agent_type")
                    if type(metadata.get("agent_type")) is str
                    else owner._agent_child_types.get(call.id)
                ),
                description=(
                    metadata.get("description")
                    if type(metadata.get("description")) is str
                    else None
                ),
                depth=(
                    metadata.get("depth")
                    if type(metadata.get("depth")) is int
                    else None
                ),
                budget_exhausted=metadata.get("error_code") == "agent_turn_budget",
            )
            result = receipt_tool_result(call.id, payload)
            candidate = result
        if child_store is not None and (
            candidate is None or candidate.is_canceled
        ):
            result = owner._canceled_agent_result(
                call.id,
                child_session_path=str(child_store.session_dir),
                agent_type=owner._agent_child_types.get(call.id),
            )
        if result is None:
            result = slot
            if result is None and call.name.casefold() == "agent":
                result = owner._canceled_agent_result(call.id)
            if result is None:
                result = ToolResult(
                    call.id,
                    "tool execution canceled",
                    is_error=True,
                    is_canceled=True,
                )
        if stored_result is None:
            new_results.append((call, result))
        results.append(result)
    for call, result in new_results:
        owner.store.append_message(
            Message(
                MessageRole.TOOL_RESULT,
                [TextContent(result.content)],
                tool_result=result,
            )
        )
        if owner.hooks is not None:
            owner.hooks.post_tool(call.name, result.content)
        if call.name == "agent":
            is_background = (
                result.structured_content is not None
                and result.structured_content.get("status") == "running"
            )
            if is_background:
                continue
            child_store = owner._agent_child_stores.pop(call.id, None)
            if child_store is not None:
                if result.is_canceled:
                    child_store.mark_agent_canceled(call.id)
                else:
                    from .agent_background import adopt_agent_children

                    adopt_agent_children(
                        child_store,
                        owner.store,
                        background_owner=owner._background_owner,
                    )
                    child_store.finish_agent_parent()
            owner.store.finish_agent_child(
                f"{owner.agent_instance_id}:{child_store.session_id}"
                if child_store is not None and owner.agent_instance_id is not None
                else call.id
            )
            owner._agent_child_turns.pop(call.id, None)
            owner._agent_child_types.pop(call.id, None)
    return results
