"""Deterministic semantic context eviction."""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from collections.abc import Callable, Sequence

from ..protocol.types import (
    Message,
    MessageRole,
    RedactedThinkingContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from .evict import EVICTION_KINDS, EvictionResult
from .fold import estimated_tokens, is_failed_result, tool_calls

EVICTION2_KINDS = EVICTION_KINDS - {"eviction"}
_DIGEST_LIMIT = 440
_LOAD_BEARING = re.compile(
    r"(^\s*#{1,6}\s)|\b(must|never|required|normative|deprecated|todo)\b|do not",
    re.IGNORECASE,
)


def evict2_messages(
    records: Sequence[tuple[int, Message]],
    *,
    fixed_tokens: int,
    target_tokens: int,
    recall_enabled: bool,
    token_counter: Callable[[Message], int] = estimated_tokens,
) -> EvictionResult:
    """Replace re-derivable outputs with bounded semantic digests."""

    messages = [message for _, message in records]
    before = fixed_tokens + sum(token_counter(message) for message in messages)
    calls = tool_calls(records)
    call_indexes = _call_indexes(messages)
    changed: set[int] = set()
    read_counts = _collapse_repeated_reads(records, messages, calls, call_indexes, changed)

    def total() -> int:
        return fixed_tokens + sum(token_counter(message) for message in messages)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        if result is None or call is None or is_failed_result(message):
            continue
        if call.name not in {"read", "bash", "search", "grep", "websearch"}:
            continue
        path = _read_path(call)
        count = read_counts.get((path, _content_digest(result.content)), 1)
        messages[index] = _digest_result(
            message,
            call,
            seq,
            recall_enabled=recall_enabled,
            read_count=count,
        )
        changed.add(index)
        if total() <= target_tokens:
            return _result(messages, changed, before, total(), True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if message.role is not MessageRole.ASSISTANT or not any(
            isinstance(block, (ThinkingContent, RedactedThinkingContent))
            for block in message.content
        ):
            continue
        content = [
            block
            for block in message.content
            if not isinstance(block, (ThinkingContent, RedactedThinkingContent))
        ]
        content.append(TextContent(f"[assistant reasoning evicted · seq {seq}]"))
        messages[index] = Message(
            message.role,
            content,
            tool_result=message.tool_result,
            metadata={"context_evicted": True, "source_seq": seq},
        )
        changed.add(index)
        if total() <= target_tokens:
            return _result(messages, changed, before, total(), True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if (
            message.role is not MessageRole.ASSISTANT
            or message.tool_result is not None
            or any(isinstance(block, ToolUseContent) for block in message.content)
            or message.metadata.get("context_evicted")
        ):
            continue
        messages[index] = Message(
            MessageRole.ASSISTANT,
            [TextContent(f"[assistant text evicted · seq {seq}]")],
            metadata={"context_evicted": True, "source_seq": seq},
        )
        changed.add(index)
        if total() <= target_tokens:
            return _result(messages, changed, before, total(), True)

    return _result(messages, changed, before, total(), total() <= target_tokens)


def assistant_summary_messages(
    records: Sequence[tuple[int, Message]],
) -> list[Message]:
    """Return only visible assistant prose from an eviction source range."""

    output: list[Message] = []
    for _, message in records:
        if message.role is not MessageRole.ASSISTANT:
            continue
        content = [block for block in message.content if isinstance(block, TextContent)]
        if content:
            output.append(Message(MessageRole.ASSISTANT, content))
    return output


def _collapse_repeated_reads(
    records: Sequence[tuple[int, Message]],
    messages: list[Message],
    calls: dict[str, ToolCall],
    call_indexes: dict[str, int],
    changed: set[int],
) -> dict[tuple[str, str], int]:
    groups: dict[str, list[tuple[int, str, int, str]]] = defaultdict(list)
    for result_index, (seq, message) in enumerate(records):
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        path = _read_path(call) if call is not None else None
        if call is None or call.name != "read" or result is None or path is None:
            continue
        digest = str(message.metadata.get("evict2_content_digest") or _content_digest(result.content))
        groups[path].append((result_index, result.tool_call_id, seq, digest))

    counts: dict[tuple[str, str], int] = {}
    for path, occurrences in groups.items():
        newest = occurrences[-1]
        same_digest = sum(item[3] == newest[3] for item in occurrences)
        counts[(path, newest[3])] = same_digest
        if len(occurrences) < 2:
            continue
        for result_index, call_id, seq, _ in occurrences[:-1]:
            call_index = call_indexes.get(call_id)
            if call_index is None or len(_tool_uses(messages[call_index])) != 1:
                continue
            messages[call_index] = Message(
                MessageRole.ASSISTANT,
                [TextContent(f"[older duplicate read collapsed into seq {newest[2]}]")],
                metadata={"context_evicted": True, "source_seq": records[call_index][0]},
            )
            messages[result_index] = Message(
                MessageRole.ASSISTANT,
                [TextContent(f"[duplicate result collapsed into seq {newest[2]}]")],
                metadata={"context_evicted": True, "source_seq": seq},
            )
            changed.update((call_index, result_index))
    return counts


def _digest_result(
    message: Message,
    call: ToolCall,
    seq: int,
    *,
    recall_enabled: bool,
    read_count: int,
) -> Message:
    result = message.tool_result
    if result is None:
        return message
    content_digest = _content_digest(result.content)
    if call.name == "read":
        digest = _read_digest(call, result.content, read_count)
        path = _read_path(call)
    elif call.name == "bash":
        digest = _bash_digest(call, result)
        path = None
    else:
        digest = _search_digest(call, result.content)
        path = None
    hint = (
        f" recall_history seq_start={seq}, seq_end={seq} for exact output; "
        "re-read only if the file may have changed."
        if recall_enabled
        else " Exact output remains in session history."
    )
    bounded = _bounded(f"[semantic {call.name} digest · seq {seq}] {digest}.{hint}")
    metadata = {
        "context_evicted": True,
        "source_seq": seq,
        "evict2_content_digest": content_digest,
    }
    if path is not None:
        metadata["evict2_path"] = path
    return Message(
        message.role,
        list(message.content),
        tool_result=ToolResult(
            result.tool_call_id,
            bounded,
            is_error=result.is_error,
            is_canceled=result.is_canceled,
        ),
        metadata=metadata,
    )


def _read_digest(call: ToolCall, content: str, read_count: int) -> str:
    path = _read_path(call) or "unknown path"
    lines = content.splitlines()
    selected = _selected_lines(lines)
    count = f"; read {read_count} times" if read_count > 1 else ""
    excerpt = " | ".join(selected) or "(empty file)"
    return f"read {path}; {len(lines)} lines{count}; {excerpt}"


def _bash_digest(call: ToolCall, result: ToolResult) -> str:
    command = str(call.arguments.get("command", call.arguments.get("cmd", "")))
    lines = result.content.splitlines()
    errors = [line for line in lines if re.search(r"error|fatal|failed", line, re.IGNORECASE)]
    selected = _unique([*errors[:3], *lines[-4:]])
    excerpt = " | ".join(selected) or "(no output)"
    return f"command={command[:120]!r}; exit={'1' if result.is_error else '0'}; tail={excerpt}"


def _search_digest(call: ToolCall, content: str) -> str:
    subject = ", ".join(
        f"{key}={value}"
        for key, value in sorted(call.arguments.items())
        if key in {"query", "path", "glob", "pattern"}
    )
    lines = content.splitlines()
    excerpt = " | ".join(_selected_lines(lines)) or "(no matches)"
    return f"{subject or 'search'}; {len(lines)} lines; {excerpt}"


def _selected_lines(lines: Sequence[str]) -> list[str]:
    important = [line.strip() for line in lines if _LOAD_BEARING.search(line)]
    edges = [line.strip() for line in [*lines[:2], *lines[-2:]] if line.strip()]
    return _unique([*important, *edges])


def _bounded(value: str) -> str:
    if len(value) <= _DIGEST_LIMIT:
        return value
    return value[: _DIGEST_LIMIT - 3].rstrip() + "..."


def _read_path(call: ToolCall | None) -> str | None:
    if call is None:
        return None
    path = call.arguments.get("path")
    return path if isinstance(path, str) and path else None


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _call_indexes(messages: Sequence[Message]) -> dict[str, int]:
    return {
        block.tool_call.id: index
        for index, message in enumerate(messages)
        for block in message.content
        if isinstance(block, ToolUseContent)
    }


def _tool_uses(message: Message) -> list[ToolUseContent]:
    return [block for block in message.content if isinstance(block, ToolUseContent)]


def _unique(values: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _result(
    messages: list[Message],
    changed: set[int],
    before: int,
    after: int,
    reached: bool,
) -> EvictionResult:
    return EvictionResult(messages, len(changed), 0, before, after, reached)
