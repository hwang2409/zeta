"""Deterministic context eviction and exact hidden-history recall."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil

from ..core.store import ConversationEntry, ConversationStore
from ..protocol.types import (
    ContentBlock,
    Message,
    MessageRole,
    RedactedThinkingContent,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)

EVICTION_KIND = "evict"
TARGET_RATIO = 0.55
HYSTERESIS_RATIO = 0.15
DIGEST_LIMIT = 440
RECALL_DEFAULT_MAX_CHARS = 8_000
RECALL_HARD_MAX_CHARS = 20_000
RECALL_NO_RANGE_MATCH = "No compacted messages in that range on the active branch."
RECALL_NO_QUERY_MATCH = "No matching compacted messages on the active branch."
_REDERIVABLE_TOOLS = frozenset(
    {
        "read",
        "bash",
        "search",
        "grep",
        "glob",
        "list",
        "find",
        "websearch",
        "fetch",
        "mcp_discover",
        "agent_status",
    }
)
_LOAD_BEARING = re.compile(
    r"(^\s*#{1,6}\s)|\b(must|never|required|normative|deprecated|todo|incident|contract|guarantee|authoritative|first-seen|root cause)\b|do not",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class EvictionResult:
    """One deterministic replacement view and its accounting."""

    messages: list[Message]
    items_evicted: int
    tokens_before: int
    tokens_after: int
    reached_target: bool


def estimated_tokens(message: Message) -> int:
    """Use the same stable approximation as normal context accounting."""

    encoded = json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":"))
    return max(1, ceil(len(encoded) / 4))


def evict_messages(
    records: Sequence[tuple[int, Message]],
    *,
    fixed_tokens: int,
    target_tokens: int,
    token_counter: Callable[[Message], int] = estimated_tokens,
) -> EvictionResult:
    """Replace old re-derivable results with bounded semantic digests."""

    messages = [message for _, message in records]
    before = fixed_tokens + sum(token_counter(message) for message in messages)
    calls = _tool_calls(messages)
    call_indexes = _call_indexes(messages)
    changed: set[int] = set()
    read_counts = _collapse_repeated_reads(
        records, messages, calls, call_indexes, changed
    )

    def total() -> int:
        return fixed_tokens + sum(token_counter(message) for message in messages)

    def digest_results(*, failed: bool) -> EvictionResult | None:
        for index, (seq, _) in enumerate(records):
            message = messages[index]
            result = message.tool_result
            call = calls.get(result.tool_call_id) if result is not None else None
            if (
                result is None
                or call is None
                or call.name not in _REDERIVABLE_TOOLS
                or bool(result.is_error or result.is_canceled) is not failed
                or message.metadata.get("context_evicted")
            ):
                continue
            path = _read_path(call)
            count = read_counts.get((path, _content_digest(result.content)), 1)
            messages[index] = _digest_result(message, call, seq, read_count=count)
            changed.add(index)
            if total() <= target_tokens:
                return _result(messages, changed, before, total(), True)
        return None

    reached = digest_results(failed=False)
    if reached is not None:
        return reached

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

    reached = digest_results(failed=True)
    if reached is not None:
        return reached

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if not _is_notification_message(message):
            continue
        messages[index] = _notification_receipt(message, seq)
        changed.add(index)
        if total() <= target_tokens:
            return _result(messages, changed, before, total(), True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        replacement = _digest_agent_prompts(message, seq)
        if replacement is message:
            continue
        messages[index] = replacement
        changed.add(index)
        if total() <= target_tokens:
            return _result(messages, changed, before, total(), True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        if result is None or call is None or call.name not in {
            "agent",
            "agent_output",
            "task_output",
        }:
            continue
        messages[index] = _orchestration_result_receipt(message, call, seq)
        changed.add(index)
        if total() <= target_tokens:
            return _result(messages, changed, before, total(), True)

    return _result(messages, changed, before, total(), total() <= target_tokens)


def eviction_view(
    records: Sequence[tuple[int, Message]], result: EvictionResult
) -> list[dict[str, object]]:
    """Serialize an eviction result for durable replay."""

    return [
        {"seq": seq, "message": message.to_dict()}
        for (seq, _), message in zip(records, result.messages, strict=True)
    ]


def active_compacted_ranges(
    entries: Sequence[ConversationEntry],
) -> list[tuple[int, int]]:
    """Return effective hidden ranges for an already-active branch."""

    markers = [entry for entry in entries if entry.type == "compaction"]
    superseded = {
        marker_id for marker in markers for marker_id in marker.data.get("replaces", [])
    }
    return [
        (marker.data["source_seq_start"], marker.data["source_seq_end"])
        for marker in markers
        if marker.id not in superseded
    ]


def recall_history(
    store: ConversationStore,
    *,
    query: str | None = None,
    seq_start: int | None = None,
    seq_end: int | None = None,
    offset: int = 0,
    max_chars: int = RECALL_DEFAULT_MAX_CHARS,
) -> str:
    """Read exact hidden messages from the active branch without mutation."""

    if type(max_chars) is not int or max_chars < 1:
        raise ValueError("max_chars must be a positive integer")
    if type(offset) is not int or offset < 0:
        raise ValueError("offset must be a nonnegative integer")
    max_chars = min(max_chars, RECALL_HARD_MAX_CHARS)
    branch = store.replay()
    ranges = active_compacted_ranges(branch)
    hidden = [
        entry
        for entry in branch
        if entry.type == "message" and _covered(entry.seq, ranges)
    ]
    if seq_start is not None or seq_end is not None:
        if type(seq_start) is not int or type(seq_end) is not int:
            raise ValueError("seq_start and seq_end must be provided together")
        if seq_start < 1 or seq_end < seq_start:
            raise ValueError("invalid sequence range")
        selected = [entry for entry in hidden if seq_start <= entry.seq <= seq_end]
        if not selected:
            return RECALL_NO_RANGE_MATCH
        return _render_range(
            selected,
            offset=offset,
            max_chars=max_chars,
            requested_start=seq_start,
            requested_end=seq_end,
        )
    if query is None or not query.strip():
        raise ValueError("provide query or both seq_start and seq_end")
    folded = query.casefold().strip()
    tokens = re.findall(r"\w+", folded)
    matches: list[tuple[int, int, str]] = []
    for entry in hidden:
        rendered = _render_entry(entry)
        # Search the unescaped text so non-ASCII queries (CJK, emoji, accents)
        # match; the returned snippet keeps the stable escaped rendering.
        searchable = _searchable_entry(entry).casefold()
        score = (10 if folded in searchable else 0) + sum(
            searchable.count(token) for token in tokens
        )
        if score:
            snippet = rendered if len(rendered) <= 400 else f"{rendered[:397]}..."
            matches.append((score, entry.seq, snippet))
    matches.sort(key=lambda item: (-item[0], item[1]))
    body = "\n".join(item[2] for item in matches[:20])
    if not body:
        body = RECALL_NO_QUERY_MATCH
    return _bounded_with_hint(body, max_chars, "refine the query for more matches")


def _collapse_repeated_reads(
    records: Sequence[tuple[int, Message]],
    messages: list[Message],
    calls: Mapping[str, ToolCall],
    call_indexes: Mapping[str, int],
    changed: set[int],
) -> dict[tuple[str | None, str], int]:
    groups: dict[tuple[str, str], list[tuple[int, str, int]]] = defaultdict(list)
    for result_index, (seq, message) in enumerate(records):
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        path = _read_path(call)
        if call is None or call.name != "read" or result is None or path is None:
            continue
        digest = str(
            message.metadata.get("eviction_content_digest")
            or _content_digest(result.content)
        )
        groups[(path, digest)].append((result_index, result.tool_call_id, seq))

    counts: dict[tuple[str | None, str], int] = {}
    for (path, digest), occurrences in groups.items():
        newest = occurrences[-1]
        counts[(path, digest)] = len(occurrences)
        for result_index, call_id, seq in occurrences[:-1]:
            call_index = call_indexes.get(call_id)
            if call_index is None or len(_tool_uses(messages[call_index])) != 1:
                continue
            messages[call_index] = Message(
                MessageRole.ASSISTANT,
                [TextContent(f"[older duplicate read collapsed into seq {newest[2]}]")],
                metadata={
                    "context_evicted": True,
                    "source_seq": records[call_index][0],
                    "collapsed_into_seq": newest[2],
                },
            )
            messages[result_index] = Message(
                MessageRole.ASSISTANT,
                [TextContent(f"[duplicate result collapsed into seq {newest[2]}]")],
                metadata={
                    "context_evicted": True,
                    "source_seq": seq,
                    "collapsed_into_seq": newest[2],
                },
            )
            changed.update((call_index, result_index))
    return counts


def _digest_result(
    message: Message, call: ToolCall, seq: int, *, read_count: int
) -> Message:
    result = message.tool_result
    if result is None:
        return message
    if call.name == "read":
        digest = _read_digest(call, result.content, read_count)
    elif call.name == "bash":
        digest = _bash_digest(call, result)
    else:
        digest = _search_digest(call, result.content)
    prefix = f"[semantic {call.name} digest · seq {seq}] "
    hint = (
        f". recall_history seq_start={seq}, seq_end={seq} for exact output; "
        "re-read only if the source may have changed."
    )
    body_limit = DIGEST_LIMIT - len(prefix) - len(hint)
    body = digest if len(digest) <= body_limit else digest[: body_limit - 3].rstrip() + "..."
    bounded = f"{prefix}{body}{hint}"
    return Message(
        message.role,
        list(message.content),
        tool_result=ToolResult(
            result.tool_call_id,
            bounded,
            is_error=result.is_error,
            is_canceled=result.is_canceled,
        ),
        metadata={
            **message.metadata,
            "context_evicted": True,
            "source_seq": seq,
            "eviction_content_digest": _content_digest(result.content),
        },
    )


def _digest_agent_prompts(message: Message, seq: int) -> Message:
    changed = False
    content: list[ContentBlock] = []
    for block in message.content:
        if not isinstance(block, ToolUseContent):
            content.append(block)
            continue
        call = block.tool_call
        prompt = call.arguments.get("prompt")
        if call.name != "agent" or not isinstance(prompt, str):
            content.append(block)
            continue
        arguments = dict(call.arguments)
        arguments["prompt"] = (
            f"[agent prompt receipt · seq {seq}] original_chars={len(prompt)} "
            f"sha256={_content_digest(prompt)[:16]}; recall_history "
            f"seq_start={seq}, seq_end={seq} for exact prompt"
        )
        content.append(
            ToolUseContent(ToolCall(call.id, call.name, arguments))
        )
        changed = True
    if not changed:
        return message
    return Message(
        message.role,
        content,
        tool_result=message.tool_result,
        metadata={
            **message.metadata,
            "context_evicted": True,
            "source_seq": seq,
        },
    )


def _orchestration_result_receipt(
    message: Message, call: ToolCall, seq: int
) -> Message:
    result = message.tool_result
    if result is None:
        return message
    structured = result.structured_content or {}
    status = structured.get("status")
    if not isinstance(status, str):
        status = (
            "canceled"
            if result.is_canceled
            else "error"
            if result.is_error
            else "success"
        )
    fields = [
        f"tool={call.name}",
        f"call={result.tool_call_id}",
        f"status={_one_line(status)}",
    ]
    for label, key in (
        ("child", "child_instance_id"),
        ("task", "task_id"),
        ("handle", "handle"),
    ):
        value = structured.get(key, call.arguments.get(key))
        if compact := _one_line(value):
            fields.append(f"{label}={compact}")
    fields.extend(
        (
            f"original_chars={len(result.content)}",
            f"sha256={_content_digest(result.content)[:16]}",
        )
    )
    receipt = (
        f"[orchestration result receipt · seq {seq}] {' '.join(fields)}; "
        f"recall_history seq_start={seq}, seq_end={seq} for exact result"
    )
    return Message(
        message.role,
        list(message.content),
        tool_result=ToolResult(
            result.tool_call_id,
            receipt,
            is_error=result.is_error,
            is_canceled=result.is_canceled,
        ),
        metadata={
            **message.metadata,
            "context_evicted": True,
            "source_seq": seq,
            "eviction_content_digest": _content_digest(result.content),
        },
    )


def _is_notification_message(message: Message) -> bool:
    return (
        message.metadata.get("zeta_event") == "agent_notifications"
        and isinstance(message.metadata.get("notifications"), list)
    )


def _notification_receipt(message: Message, seq: int) -> Message:
    notifications = message.metadata["notifications"]
    summaries: list[str] = []
    for raw in notifications:
        if not isinstance(raw, Mapping):
            continue
        kind = _one_line(raw.get("kind", "agent_completion"))
        fields = [kind]
        if child_id := _one_line(raw.get("child_instance_id")):
            fields.append(f"child={child_id}")
        if task_id := _one_line(raw.get("task_id")):
            fields.append(f"task={task_id}")
        if status := _one_line(raw.get("status")):
            fields.append(f"status={status}")
        exit_code = raw.get("exit_code")
        if type(exit_code) is int:
            fields.append(f"exit_code={exit_code}")
        if description := _one_line(
            raw.get("description", raw.get("headline", raw.get("command")))
        ):
            fields.append(f"description={description}")
        summaries.append(" ".join(fields))
    encoded = json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":"))
    details = "; ".join(summaries) or "unknown notification"
    receipt = (
        f"[notification receipt · seq {seq}] {details}; "
        f"original_chars={len(encoded)} sha256={_content_digest(encoded)[:16]}; "
        f"recall_history seq_start={seq}, seq_end={seq} for exact notification"
    )
    return Message(
        message.role,
        [TextContent(receipt)],
        metadata={
            "context_evicted": True,
            "source_seq": seq,
            "eviction_content_digest": _content_digest(encoded),
        },
    )


def _one_line(value: object, *, limit: int = 160) -> str:
    if value is None:
        return ""
    compact = " ".join(str(value).split())
    return compact if len(compact) <= limit else compact[: limit - 3].rstrip() + "..."


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
    errors = [
        line for line in lines if re.search(r"error|fatal|failed", line, re.IGNORECASE)
    ]
    excerpt = (
        " | ".join(_unique([*errors[:3], *_selected_lines(lines), *lines[-4:]]))
        or "(no output)"
    )
    status = "error" if result.is_error or result.is_canceled else "success"
    return f"command={command[:120]!r}; status={status}; tail={excerpt}"


def _search_digest(call: ToolCall, content: str) -> str:
    subject = ", ".join(
        f"{key}={value}"
        for key, value in sorted(call.arguments.items())
        if key in {"query", "path", "glob", "pattern", "url"}
    )
    lines = content.splitlines()
    excerpt = " | ".join(_selected_lines(lines)) or "(no matches)"
    return f"{subject or 'result'}; {len(lines)} lines; {excerpt}"


def _selected_lines(lines: Sequence[str]) -> list[str]:
    important = [line.strip() for line in lines if _LOAD_BEARING.search(line)]
    edges = [line.strip() for line in [*lines[:2], *lines[-2:]] if line.strip()]
    return _unique([*important, *edges])


def _result(
    messages: list[Message],
    changed: set[int],
    before: int,
    after: int,
    reached: bool,
) -> EvictionResult:
    return EvictionResult(messages, len(changed), before, after, reached)


def _tool_calls(messages: Sequence[Message]) -> dict[str, ToolCall]:
    return {
        block.tool_call.id: block.tool_call
        for message in messages
        for block in message.content
        if isinstance(block, ToolUseContent)
    }


def _call_indexes(messages: Sequence[Message]) -> dict[str, int]:
    return {
        block.tool_call.id: index
        for index, message in enumerate(messages)
        for block in message.content
        if isinstance(block, ToolUseContent)
    }


def _tool_uses(message: Message) -> list[ToolUseContent]:
    return [block for block in message.content if isinstance(block, ToolUseContent)]


def _read_path(call: ToolCall | None) -> str | None:
    if call is None:
        return None
    path = call.arguments.get("path")
    return path if isinstance(path, str) and path else None


def _content_digest(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _unique(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _covered(seq: int, ranges: Iterable[tuple[int, int]]) -> bool:
    return any(start <= seq <= end for start, end in ranges)


def _render_entry(entry: ConversationEntry) -> str:
    message = Message.from_dict(entry.data["message"])
    encoded = json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":"))
    return f"seq {entry.seq}: {encoded}"



def _searchable_entry(entry: ConversationEntry) -> str:
    message = Message.from_dict(entry.data["message"])
    return json.dumps(
        message.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )

def _render_range(
    entries: Sequence[ConversationEntry],
    *,
    offset: int,
    max_chars: int,
    requested_start: int,
    requested_end: int,
) -> str:
    rendered = "\n".join(_render_entry(entry) for entry in entries)
    if offset >= len(rendered):
        return "[end of range]"
    content = rendered[offset : offset + max_chars]
    next_offset = offset + len(content)
    if next_offset == len(rendered):
        return f"{content}\n[end of range]"
    return (
        f"{content}\n[truncated; continue with seq_start={requested_start}, "
        f"seq_end={requested_end}, offset={next_offset}]"
    )


def _bounded_with_hint(value: str, max_chars: int, hint: str) -> str:
    if len(value) <= max_chars:
        return value
    suffix = f"\n[truncated; {hint}]"
    if len(suffix) >= max_chars:
        return suffix[:max_chars]
    return value[: max_chars - len(suffix)] + suffix
