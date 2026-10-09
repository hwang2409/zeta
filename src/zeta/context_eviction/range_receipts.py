"""Deterministic range receipts for old per-item eviction receipts."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..protocol.types import Message, MessageRole, TextContent, ToolUseContent

_RANGE_LIMIT = 440
_STRUCTURED_RECEIPT_LIMIT = 8_192
_MAX_TOOL_KINDS = 12
_SHA256 = re.compile(r"[0-9a-f]{64}").fullmatch
RECEIPT_KIND_METADATA = "eviction_receipt"


@dataclass(frozen=True, slots=True)
class RangeCandidate:
    """A complete replacement view and the source sequences it coalesces."""

    records: list[tuple[int, Message]]
    coalesced_source_seqs: frozenset[int]


@dataclass(frozen=True, slots=True)
class _ReceiptUnit:
    records: tuple[tuple[int, Message], ...]
    kind: str
    tool_names: tuple[str, ...] = ()


def range_receipt_candidate(
    records: Sequence[tuple[int, Message]],
    *,
    allows: Callable[[int], bool],
    is_smaller: Callable[[Sequence[Message], Sequence[Message]], bool],
) -> RangeCandidate:
    """Coalesce smaller maximal runs; existing ranges permanently break runs.

    A provider tool call and all its receipted results are one unit. This removes
    the complete exchange or leaves it intact, so no provider sees an orphan.
    Each run is accepted separately through the caller's shared size policy.
    """

    output: list[tuple[int, Message]] = []
    coalesced: set[int] = set()
    run: list[_ReceiptUnit] = []
    index = 0

    def flush() -> None:
        flattened = [record for unit in run for record in unit.records]
        if len(run) >= 2:
            start, end = flattened[0][0], flattened[-1][0]
            receipt = _range_receipt(start, end, run)
            if is_smaller([message for _, message in flattened], [receipt]):
                output.append((start, receipt))
                coalesced.update(seq for seq, _ in flattened)
            else:
                output.extend(flattened)
        else:
            output.extend(flattened)
        run.clear()

    while index < len(records):
        unit, next_index = _receipt_unit(records, index, allows)
        if unit is None:
            flush()
            output.append(records[index])
            index += 1
            continue
        if run and unit.records[0][0] != run[-1].records[-1][0] + 1:
            flush()
        run.append(unit)
        index = next_index
    flush()
    return RangeCandidate(output, frozenset(coalesced))


def _receipt_unit(
    records: Sequence[tuple[int, Message]],
    index: int,
    allows: Callable[[int], bool],
) -> tuple[_ReceiptUnit | None, int]:
    seq, message = records[index]
    if not allows(seq) or message.metadata.get("eviction_range"):
        return None, index + 1
    if message.role is MessageRole.USER:
        return None, index + 1

    calls = [
        block.tool_call
        for block in message.content
        if isinstance(block, ToolUseContent)
    ]
    if calls:
        if any(
            not isinstance(block, ToolUseContent)
            and not (
                isinstance(block, TextContent)
                and _is_generated_assistant_receipt(block.text, message, seq)
            )
            for block in message.content
        ):
            return None, index + 1
        expected = {call.id: call.name for call in calls}
        grouped = [records[index]]
        names: list[str] = []
        cursor = index + 1
        while cursor < len(records):
            result_seq, result_message = records[cursor]
            result = result_message.tool_result
            if result is None or result.tool_call_id not in expected:
                break
            if not allows(result_seq) or not _is_receipt_kind(
                result_message,
                result_seq,
                "tool_result",
                tool_name=expected[result.tool_call_id],
            ):
                return None, index + 1
            grouped.append(records[cursor])
            names.append(expected.pop(result.tool_call_id))
            cursor += 1
        if expected:
            return None, index + 1
        return _ReceiptUnit(tuple(grouped), "tool", tuple(names)), cursor

    for kind in ("assistant", "notification"):
        if _is_receipt_kind(message, seq, kind):
            return _ReceiptUnit((records[index],), kind), index + 1
    return None, index + 1


def _is_receipt_kind(
    message: Message, seq: int, kind: str, *, tool_name: str | None = None
) -> bool:
    marked = message.metadata.get(RECEIPT_KIND_METADATA)
    if marked is not None:
        return (
            marked == kind
            and message.metadata.get("source_seq") == seq
            and (
                (
                    kind == "tool_result"
                    and message.tool_result is not None
                    and not message.content
                    and message.tool_result.content_blocks is None
                    and message.tool_result.structured_content is None
                )
                or (
                    kind != "tool_result"
                    and message.tool_result is None
                    and len(message.content) == 1
                    and isinstance(message.content[0], TextContent)
                )
            )
        )
    if (
        not message.metadata.get("context_evicted")
        or message.metadata.get("source_seq") != seq
    ):
        return False
    if kind == "tool_result":
        result = message.tool_result
        digest = message.metadata.get("eviction_content_digest")
        if (
            result is None
            or message.content
            or result.content_blocks is not None
            or result.structured_content is not None
            or type(digest) is not str
            or _SHA256(digest) is None
            or tool_name is None
        ):
            return False
        return _is_legacy_tool_receipt(result.content, seq, tool_name, result.tool_call_id)
    if (
        message.tool_result is not None
        or len(message.content) != 1
        or not isinstance(message.content[0], TextContent)
    ):
        return False
    text = message.content[0].text
    if kind == "notification":
        digest = message.metadata.get("eviction_content_digest")
        return (
            type(digest) is str
            and _SHA256(digest) is not None
            and _is_legacy_structured_receipt(
                text, "notification receipt", seq, "notification", "notification"
            )
        )
    if _is_generated_assistant_receipt(text, message, seq):
        return True
    collapsed_into = message.metadata.get("collapsed_into_seq")
    return type(collapsed_into) is int and text in {
        f"[older duplicate read collapsed into seq {collapsed_into}]",
        f"[duplicate result collapsed into seq {collapsed_into}]",
    }


def _is_legacy_tool_receipt(
    text: str, seq: int, tool_name: str, tool_call_id: str
) -> bool:
    semantic_prefix = f"[semantic {tool_name} digest · seq {seq}] "
    semantic_suffix = (
        f". recall_history seq_start={seq}, seq_end={seq} for exact output; "
        "re-read only if the source may have changed."
    )
    if text.startswith(semantic_prefix):
        body = text[len(semantic_prefix) : -len(semantic_suffix)]
        return (
            len(text) <= _RANGE_LIMIT
            and "\n" not in text
            and text.endswith(semantic_suffix)
            and bool(body)
        )
    if tool_name in {"agent", "agent_output", "task_output"}:
        return _is_legacy_structured_receipt(
            text,
            "orchestration result receipt",
            seq,
            "result",
            "orchestration",
            tool_name=tool_name,
            tool_call_id=tool_call_id,
        )
    if tool_name in {"inbox", "project", "recall_history", "run_background"}:
        return _is_legacy_structured_receipt(
            text,
            "workflow result receipt",
            seq,
            "result",
            "workflow",
            tool_name=tool_name,
        )
    return False


def _is_legacy_structured_receipt(
    text: str,
    prefix: str,
    seq: int,
    exact_kind: str,
    receipt_kind: str,
    *,
    tool_name: str | None = None,
    tool_call_id: str | None = None,
) -> bool:
    start = f"[{prefix}] "
    end = (
        f"; recall_history seq_start={seq}, seq_end={seq} for exact {exact_kind}"
    )
    if (
        len(text) > _STRUCTURED_RECEIPT_LIMIT
        or "\n" in text
        or not text.startswith(start)
        or not text.endswith(end)
    ):
        return False
    encoded = text[len(start) : -len(end)]
    try:
        payload = json.loads(encoded)
    except json.JSONDecodeError:
        return False
    if (
        not isinstance(payload, dict)
        or json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        )
        != encoded
    ):
        return False
    if receipt_kind == "notification":
        return _is_notification_payload(payload)
    if receipt_kind == "orchestration":
        return _is_orchestration_payload(payload, tool_name, tool_call_id)
    return _is_workflow_payload(payload, tool_name)


def _is_notification_payload(payload: dict[str, object]) -> bool:
    if set(payload) != {"notifications", "original_chars", "sha256"}:
        return False
    notifications = payload["notifications"]
    allowed = {
        "kind",
        "child_instance_id",
        "task_id",
        "status",
        "exit_code",
        "description",
    }
    return (
        _nonnegative_int(payload["original_chars"])
        and _short_sha(payload["sha256"])
        and isinstance(notifications, list)
        and len(notifications) <= 32
        and all(
            isinstance(item, dict)
            and "kind" in item
            and set(item) <= allowed
            and _generated_string(item["kind"])
            and all(
                _generated_string(item[key])
                for key in (
                    "child_instance_id",
                    "task_id",
                    "status",
                    "description",
                )
                if key in item
            )
            and (
                "exit_code" not in item or type(item["exit_code"]) is int
            )
            for item in notifications
        )
    )


def _is_orchestration_payload(
    payload: dict[str, object], tool_name: str | None, tool_call_id: str | None
) -> bool:
    required = {"tool", "call", "status", "original_chars", "sha256"}
    optional = {"child_instance_id", "task_id", "handle", "description"}
    return (
        required <= set(payload) <= required | optional
        and payload["tool"] == tool_name
        and payload["call"] == tool_call_id
        and _generated_string(payload["status"])
        and _nonnegative_int(payload["original_chars"])
        and _short_sha(payload["sha256"])
        and all(_generated_string(payload[key]) for key in optional if key in payload)
    )


def _is_workflow_payload(payload: dict[str, object], tool_name: str | None) -> bool:
    if tool_name == "inbox":
        return _is_inbox_payload(payload)
    if tool_name == "project":
        return _is_project_payload(payload)
    if tool_name == "recall_history":
        return _is_recall_payload(payload)
    if tool_name == "run_background":
        return _is_background_payload(payload)
    return False


def _is_inbox_payload(payload: dict[str, object]) -> bool:
    allowed = {
        "action",
        "target_project",
        "id",
        "projects",
        "omitted_projects",
        "messages",
        "omitted_messages",
    }
    if "action" not in payload or not set(payload) <= allowed:
        return False
    if not _generated_string(payload["action"]):
        return False
    if any(
        not _generated_string(payload[key])
        for key in ("target_project", "id")
        if key in payload
    ):
        return False
    if any(
        key in payload and not _nonnegative_int(payload[key])
        for key in ("omitted_projects", "omitted_messages")
    ):
        return False
    projects = payload.get("projects", [])
    if not isinstance(projects, list) or len(projects) > 4:
        return False
    if any(
        not isinstance(project, dict)
        or not set(project) <= {"id", "name", "scope"}
        or any(not _generated_string(value) for value in project.values())
        for project in projects
    ):
        return False
    messages = payload.get("messages", [])
    required = {"id", "kind", "title", "status", "claimed", "done"}
    optional = {"from_project", "from_session", "outcome"}
    return (
        isinstance(messages, list)
        and len(messages) <= 4
        and all(
            isinstance(message, dict)
            and required <= set(message) <= required | optional
            and all(
                _generated_string(message[key], limit=120 if key == "outcome" else 160)
                for key in {"id", "kind", "title", "status"} | (set(message) & optional)
            )
            and type(message["claimed"]) is bool
            and type(message["done"]) is bool
            for message in messages
        )
    )


def _is_project_payload(payload: dict[str, object]) -> bool:
    required = {"action", "sections"}
    optional = {"project_id", "project_name", "files"}
    return (
        required <= set(payload) <= required | optional
        and _generated_string(payload["action"])
        and all(
            _generated_string(payload[key])
            for key in ("project_id", "project_name")
            if key in payload
        )
        and _string_list(payload["sections"], limit=64)
        and ("files" not in payload or _string_list(payload["files"], limit=16))
    )


def _is_recall_payload(payload: dict[str, object]) -> bool:
    required = {"action", "result_chars"}
    optional = {"seq_start", "seq_end", "offset", "max_chars", "query"}
    return (
        required <= set(payload) <= required | optional
        and payload["action"] in {"query", "range"}
        and _nonnegative_int(payload["result_chars"])
        and all(
            type(payload[key]) is int
            for key in ("seq_start", "seq_end", "offset", "max_chars")
            if key in payload
        )
        and ("query" not in payload or _generated_string(payload["query"]))
    )


def _is_background_payload(payload: dict[str, object]) -> bool:
    return (
        {"command", "status"} <= set(payload) <= {"command", "status", "task_id"}
        and _generated_string(payload["command"], limit=120)
        and payload["status"] in {"canceled", "error", "running", "started"}
        and ("task_id" not in payload or _generated_string(payload["task_id"]))
    )


def _generated_string(value: object, *, limit: int = 160) -> bool:
    return isinstance(value, str) and value == " ".join(value.split()) and len(value) <= limit


def _string_list(value: object, *, limit: int) -> bool:
    return (
        isinstance(value, list)
        and len(value) <= limit
        and all(_generated_string(item) for item in value)
    )


def _nonnegative_int(value: object) -> bool:
    return type(value) is int and value >= 0


def _short_sha(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 16
        and all(character in "0123456789abcdef" for character in value)
    )


def _is_generated_assistant_receipt(text: str, message: Message, seq: int) -> bool:
    return message.metadata.get("source_seq") == seq and text in {
        f"[assistant text evicted · seq {seq}]",
        f"[assistant reasoning evicted · seq {seq}]",
    }


def _range_receipt(start: int, end: int, units: Sequence[_ReceiptUnit]) -> Message:
    kinds = Counter(unit.kind for unit in units)
    tools = Counter(name for unit in units for name in unit.tool_names)
    parts: list[str] = []
    if tools:
        sorted_tools = sorted(tools.items())
        shown = sorted_tools[:_MAX_TOOL_KINDS]
        breakdown = ", ".join(f"{name} {count}" for name, count in shown)
        if len(shown) < len(sorted_tools):
            breakdown += f", other {sum(count for _, count in sorted_tools[len(shown):])}"
        count = sum(tools.values())
        parts.append(f"{count} tool {'result' if count == 1 else 'results'} ({breakdown})")
    if kinds["notification"]:
        count = kinds["notification"]
        parts.append(f"{count} {'notification' if count == 1 else 'notifications'}")
    if kinds["assistant"]:
        count = kinds["assistant"]
        parts.append(f"{count} assistant {'note' if count == 1 else 'notes'}")
    summary = ", ".join(parts)
    suffix = (
        f"; recall_history seq_start={start}, seq_end={end} for exact content"
    )
    text = f"[evicted range] seq {start}-{end}: {summary}{suffix}"
    if len(text) > _RANGE_LIMIT:
        text = f"[evicted range] seq {start}-{end}: {len(units)} receipt items{suffix}"
    return Message(
        MessageRole.ASSISTANT,
        [TextContent(text)],
        metadata={
            "context_evicted": True,
            RECEIPT_KIND_METADATA: "range",
            "eviction_range": True,
            "source_seq": start,
            "range_seq_end": end,
        },
    )
