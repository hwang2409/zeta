"""Build bounded terminal receipts for child agents."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal, Protocol

from ..core.checkpoints import ConversationEntry
from ..protocol.types import (
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
# A terminal agent receipt has a fixed persisted-message envelope even when its
# text and metadata are empty. Receipt call sites clamp configured tool-output
# limits to this floor; direct callers are clamped here as a final safeguard.
MIN_AGENT_RECEIPT_BYTES = 1_000
_TRUNCATION_NOTE = "\n[truncated]"
# A run that delivers a follow-up after the child's final response joins the
# report (produced before the follow-up) and the reply (after it) with this
# separator. build_agent_receipt is the single place that trims the combined
# text to the byte budget: it preserves the reply whole and trims the report's
# head so the newest content survives even when the head-first byte bound would
# otherwise drop it.
RUN_REPORT_SEPARATOR = "\n\n--- follow-up ---\n\n"
_REPORT_TRUNCATION_NOTE = "[earlier report truncated]\n"
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


def format_agent_stats(stats: object, *, state: TerminalState | None = None) -> str:
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


def _fits(
    result: StructuredToolResult,
    tool_call_id: str,
    envelope: Callable[[StructuredToolResult], StructuredToolResult] | None,
    max_bytes: int,
) -> bool:
    return max(_serialized_sizes(result, tool_call_id, envelope)) <= max_bytes


def _with_answer_limit(
    state: ReceiptState,
    answer: str,
    suffix: str,
    structured_content: dict[str, Any] | None,
    tool_call_id: str,
    max_bytes: int,
    envelope: Callable[[StructuredToolResult], StructuredToolResult] | None = None,
) -> StructuredToolResult:
    def candidate(length: int) -> StructuredToolResult:
        shown = answer[:length]
        if shown != answer:
            shown += _TRUNCATION_NOTE
        return _candidate(state, shown, suffix, structured_content, tool_call_id)

    full = candidate(len(answer))
    if _fits(full, tool_call_id, envelope, max_bytes):
        return full
    low = 0
    high = len(answer)
    while low < high:
        middle = (low + high + 1) // 2
        if _fits(candidate(middle), tool_call_id, envelope, max_bytes):
            low = middle
        else:
            high = middle - 1
    bounded = candidate(low)
    if _fits(bounded, tool_call_id, envelope, max_bytes):
        return bounded
    minimal = _candidate(state, "", suffix, structured_content, tool_call_id)
    if _fits(minimal, tool_call_id, envelope, max_bytes):
        return minimal
    reduced = _candidate(state, "", suffix, None, tool_call_id)
    if _fits(reduced, tool_call_id, envelope, max_bytes):
        return reduced
    return _candidate(state, "", "", None, tool_call_id)


def _notice_summary(items: Sequence[str], kept: int) -> str:
    total = len(items)
    if kept == 0:
        return f"\nkilled {total} tasks"
    shown = ", ".join(items[:kept])
    remainder = total - kept
    more = f", … (+{remainder} more)" if remainder else ""
    return f"\nkilled {total} tasks: {shown}{more}"


def _compact_structured_content(
    state: TerminalState,
    suffix: str,
    structured_content: dict[str, Any] | None,
    tool_call_id: str,
    max_bytes: int,
    envelope: Callable[[StructuredToolResult], StructuredToolResult] | None,
) -> dict[str, Any] | None:
    """Fit metadata without ever making receipt construction fail."""

    if structured_content is None:
        return None

    def fits(value: dict[str, Any] | None) -> bool:
        return _fits(
            _candidate(state, "", suffix, value, tool_call_id),
            tool_call_id,
            envelope,
            max_bytes,
        )

    candidate = dict(structured_content)
    if fits(candidate):
        return candidate
    task_ids = candidate.get("killed_task_ids")
    if isinstance(task_ids, list):
        original_count = candidate.get("killed_task_count")
        if type(original_count) is not int or original_count < len(task_ids):
            original_count = len(task_ids)
        original_truncated = candidate.get("killed_task_ids_truncated") is True
        candidate["killed_task_count"] = original_count
        candidate["killed_task_ids_truncated"] = original_truncated
        low, high = 0, len(task_ids)
        while low < high:
            middle = (low + high + 1) // 2
            trial = dict(candidate)
            trial["killed_task_ids"] = task_ids[:middle]
            trial["killed_task_ids_truncated"] = (
                original_truncated or middle < len(task_ids)
            )
            if fits(trial):
                low = middle
            else:
                high = middle - 1
        candidate["killed_task_ids"] = task_ids[:low]
        candidate["killed_task_ids_truncated"] = (
            original_truncated or low < len(task_ids)
        )
        if fits(candidate):
            return candidate
        candidate.pop("killed_task_ids", None)
        if fits(candidate):
            return candidate

    # Preserve the fields used to locate and interpret a child before optional
    # descriptive fields. Values that are themselves oversized are skipped.
    compact: dict[str, Any] = {}
    for key in (
        "child_instance_id",
        "status",
        "turns_used",
        "agent_type",
        "depth",
        "killed_task_count",
        "killed_task_ids_truncated",
    ):
        if key not in candidate:
            continue
        trial = {**compact, key: candidate[key]}
        if fits(trial):
            compact = trial
    return compact if compact and fits(compact) else None


def _build_component_receipt(
    state: TerminalState,
    *,
    report: str | None,
    reply: str | None,
    notice: str | None,
    notice_items: Sequence[str] | None,
    suffix: str,
    structured_content: dict[str, Any] | None,
    tool_call_id: str,
    max_bytes: int,
    envelope: Callable[[StructuredToolResult], StructuredToolResult] | None,
) -> StructuredToolResult:
    """Bound stats (always) > notice (full/summary) > reply (whole/tail) > report tail."""

    def candidate(
        shown_report: str = "", shown_reply: str = "", shown_notice: str = ""
    ) -> StructuredToolResult:
        text = shown_report
        if shown_reply:
            text += (RUN_REPORT_SEPARATOR if shown_report else "") + shown_reply
        return _candidate(
            state,
            text + shown_notice,
            suffix,
            structured_content,
            tool_call_id,
        )

    def fits(parts: tuple[str, str, str]) -> bool:
        return _fits(candidate(*parts), tool_call_id, envelope, max_bytes)

    def tail(value: str, count: int) -> str:
        return value[len(value) - count :]

    def largest(high: int, parts: Callable[[int], tuple[str, str, str]]) -> int:
        low = 0
        while low < high:
            middle = (low + high + 1) // 2
            if fits(parts(middle)):
                low = middle
            else:
                high = middle - 1
        return low

    empty = ("", "", "")
    if not fits(empty):
        # max_bytes is clamped by build_agent_receipt, but retain a total
        # fallback if a caller supplies an unusually large external envelope.
        structured_content = None
        suffix = ""

    shown_notice = notice or ""
    if shown_notice and not fits(("", "", shown_notice)):
        if notice_items is None:
            shown_notice = "\n[notice truncated]"
        else:
            items = tuple(notice_items)
            kept = largest(
                len(items), lambda count: ("", "", _notice_summary(items, count))
            )
            shown_notice = _notice_summary(items, kept)
        if not fits(("", "", shown_notice)):
            shown_notice = ""

    shown_reply = reply or ""
    if shown_reply and not fits(("", shown_reply, shown_notice)):
        marker = "[earlier reply truncated]\n"
        if fits(("", marker, shown_notice)):
            kept = largest(
                len(shown_reply),
                lambda count: ("", marker + tail(shown_reply, count), shown_notice),
            )
            shown_reply = marker + tail(shown_reply, kept)
        else:
            shown_reply = ""

    shown_report = report or ""
    full = (shown_report, shown_reply, shown_notice)
    if fits(full):
        return candidate(*full)
    marker = _REPORT_TRUNCATION_NOTE
    if not fits((marker, shown_reply, shown_notice)):
        return candidate("", shown_reply, shown_notice)
    kept = largest(
        len(shown_report),
        lambda count: (marker + tail(shown_report, count), shown_reply, shown_notice),
    )
    return candidate(marker + tail(shown_report, kept), shown_reply, shown_notice)


def _governance_envelope(
    tool_name: str,
) -> Callable[[StructuredToolResult], StructuredToolResult]:
    from ..tools.registry import _apply_error_governance

    return lambda result: _apply_error_governance(result, tool_name)


def build_agent_receipt(
    state: TerminalState,
    answer: str,
    stats: Mapping[str, object] | None,
    *,
    structured_content: dict[str, Any] | None = None,
    tool_call_id: str = "",
    max_bytes: int = MAX_AGENT_RESULT_BYTES,
    report: str | None = None,
    reply: str | None = None,
    notice: str | None = None,
    notice_items: Sequence[str] | None = None,
) -> StructuredToolResult:
    """Build the final terminal receipt from explicit prioritized components."""

    if state not in {"completed", "failed", "canceled"}:
        raise ValueError(f"unsupported terminal agent state: {state}")
    if type(answer) is not str:
        raise TypeError("agent receipt answer must be a string")
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError("agent receipt byte limit must be positive")
    # Registry limits are allowed to be small, but terminal receipts have a
    # fixed envelope. Use the same effective floor for direct and registry
    # callers instead of raising during child finalization.
    max_bytes = max(max_bytes, MIN_AGENT_RECEIPT_BYTES)
    if notice_items is not None and not all(
        type(item) is str and item for item in notice_items
    ):
        raise ValueError("agent receipt notice items must be nonempty strings")
    answer = _without_agent_receipt_suffix(answer)
    if report is None:
        report = answer
    else:
        report = _without_agent_receipt_suffix(report)
    if reply is not None:
        reply = _without_agent_receipt_suffix(reply)
    suffix = format_agent_stats(dict(stats) if stats is not None else {}, state=state)
    envelope = _governance_envelope("agent") if state == "failed" else None
    structured_content = _compact_structured_content(
        state,
        suffix,
        dict(structured_content) if structured_content is not None else None,
        tool_call_id,
        max_bytes,
        envelope,
    )
    return _build_component_receipt(
        state,
        report=report,
        reply=reply,
        notice=notice,
        notice_items=notice_items,
        suffix=suffix,
        structured_content=structured_content,
        tool_call_id=tool_call_id,
        max_bytes=max_bytes,
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


def receipt_tool_result(tool_call_id: str, result: StructuredToolResult) -> ToolResult:
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
            )
            result = receipt_tool_result(call.id, payload)
            candidate = result
        if child_store is not None and (candidate is None or candidate.is_canceled):
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
                    from .background import adopt_agent_children

                    adopt_agent_children(
                        child_store,
                        owner.store,
                        background_owner=owner._background_owner,
                        parent_instance_id=owner.agent_instance_id,
                    )
                    child_store.finish_agent_parent()
            if child_store is not None:
                prefix = owner.agent_instance_id or owner.store.session_id
                owner.store.finish_agent_child(f"{prefix}:{child_store.session_id}")
                owner._background_owner.mark_store_finished(child_store)
            else:
                owner.store.finish_agent_child(call.id)
            owner._agent_child_turns.pop(call.id, None)
            owner._agent_child_types.pop(call.id, None)
    owner._background_owner.release_unused_stores()
    return results
