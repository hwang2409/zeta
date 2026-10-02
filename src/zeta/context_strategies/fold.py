"""Deterministic folding of re-derivable tool results."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import ceil

from ..protocol.types import Message, ToolCall, ToolResult, ToolUseContent

# These are the re-derivable read/search/listing tools present in the built-in
# registry, plus conventional names accepted from MCP/custom registries.
REDERIVABLE_TOOLS = frozenset(
    {"bash", "read", "websearch", "glob", "grep", "ls", "find", "search", "list"}
)


@dataclass(frozen=True, slots=True)
class FoldResult:
    messages: list[Message]
    items_folded: int
    tokens_before: int
    tokens_after: int


def estimated_tokens(message: Message) -> int:
    encoded = json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":"))
    return max(1, ceil(len(encoded) / 4))


def tool_calls(messages: Sequence[tuple[int, Message]]) -> dict[str, ToolCall]:
    return {
        block.tool_call.id: block.tool_call
        for _, message in messages
        for block in message.content
        if isinstance(block, ToolUseContent)
    }


def is_failed_result(message: Message) -> bool:
    result = message.tool_result
    if result is None:
        return False
    if result.is_error or result.is_canceled:
        return True
    structured = result.structured_content
    return bool(
        isinstance(structured, dict)
        and type(structured.get("exit_code")) is int
        and structured["exit_code"] != 0
    )


def result_stub(tool_call: ToolCall, message: Message, seq: int) -> str:
    result = message.tool_result
    if result is None:  # pragma: no cover - callers enforce this invariant
        raise ValueError("folded message must have a tool result")
    token_count = estimated_tokens(message)
    subject = tool_subject(tool_call)
    label = f" {subject}" if subject else ""
    return (
        f"[folded {tool_call.name}{label} · ~{token_count} tok · seq {seq} · "
        f"re-run the tool or recall_history seq_start={seq}]"
    )


def folded_result(message: Message, tool_call: ToolCall, seq: int) -> Message:
    result = message.tool_result
    if result is None:
        return message
    return Message(
        message.role,
        list(message.content),
        tool_result=ToolResult(
            result.tool_call_id,
            result_stub(tool_call, message, seq),
            is_error=False,
        ),
        # Provider replay metadata may contain the original bulky output. The
        # neutral call/result IDs are sufficient to rebuild a valid payload.
        metadata={"context_folded": True, "source_seq": seq},
    )


def fold_messages(
    records: Sequence[tuple[int, Message]],
    *,
    token_counter: Callable[[Message], int] = estimated_tokens,
) -> FoldResult:
    """Return folded copies without mutating the transcript or input messages."""

    calls = tool_calls(records)
    output: list[Message] = []
    folded = 0
    for seq, message in records:
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        if (
            result is not None
            and call is not None
            and call.name in REDERIVABLE_TOOLS
            and not is_failed_result(message)
            and not message.metadata.get("context_folded")
        ):
            message = folded_result(message, call, seq)
            folded += 1
        output.append(message)
    return FoldResult(
        messages=output,
        items_folded=folded,
        tokens_before=sum(token_counter(message) for _, message in records),
        tokens_after=sum(token_counter(message) for message in output),
    )


def tool_subject(call: ToolCall) -> str:
    keys = {
        "read": ("path",),
        "bash": ("command", "cmd"),
        "websearch": ("query",),
        "glob": ("pattern",),
        "grep": ("pattern", "query"),
        "ls": ("path",),
        "find": ("path", "pattern"),
        "search": ("query", "pattern"),
        "list": ("path",),
    }.get(call.name, ())
    for key in keys:
        value = call.arguments.get(key)
        if isinstance(value, str) and value:
            compact = " ".join(value.split())
            return compact if len(compact) <= 80 else f"{compact[:77]}..."
    return ""
