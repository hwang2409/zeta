"""Deterministic context eviction and exact hidden-history recall."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil

from ..context_accounting import message_token_count
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
WORKFLOW_MESSAGE_LIMIT = 4
BASH_CALL_TAIL = 20
NOTIFICATION_TAIL = 3
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


def estimated_text_tokens(text: str) -> int:
    """Use the project's stable token approximation for persisted text."""
    return max(1, ceil(len(text) / 4))


@dataclass(frozen=True, slots=True)
class _EvictionEligibility:
    """The single old-versus-protected classification for one eviction pass."""

    evictable_source_seqs: frozenset[int]

    def allows(self, source_seq: int) -> bool:
        return source_seq in self.evictable_source_seqs


def estimated_tokens(message: Message) -> int:
    """Use the same stable approximation as normal context accounting."""

    return message_token_count(message)


def normalize_evicted_tool_result(message: Message) -> Message:
    """Drop legacy display-only copies from a persisted eviction receipt."""

    result = message.tool_result
    if (
        result is None
        or not message.metadata.get("context_evicted")
        or "eviction_content_digest" not in message.metadata
    ):
        return message
    return Message(
        message.role,
        tool_result=ToolResult(
            result.tool_call_id,
            result.content,
            is_error=result.is_error,
            content_blocks=result.content_blocks,
            is_canceled=result.is_canceled,
        ),
        metadata=message.metadata,
    )


def evict_messages(
    records: Sequence[tuple[int, Message]],
    *,
    fixed_tokens: int,
    target_tokens: int,
    token_counter: Callable[[Message], int] = estimated_tokens,
    unconsumed_source_seqs: Collection[int] = (),
) -> EvictionResult:
    """Replace old re-derivable results with bounded semantic digests.

    ``unconsumed_source_seqs`` identifies tool results and notifications stored
    after the latest persisted assistant response. The module protects their
    required call records and adds its other workflow protections once, before
    any transformation.
    """

    messages = [message for _, message in records]
    before = fixed_tokens + sum(token_counter(message) for message in messages)
    calls = _tool_calls(messages)
    call_indexes = _call_indexes(messages)
    results = _tool_results(messages)
    eligibility = _eviction_eligibility(records, unconsumed_source_seqs)
    changed: set[int] = set()
    read_counts = _collapse_repeated_reads(
        records, messages, calls, call_indexes, changed, eligibility
    )
    message_tokens = [token_counter(message) for message in messages]
    running_total = fixed_tokens + sum(message_tokens)

    def replace(index: int, replacement: Message) -> None:
        nonlocal running_total
        replacement_tokens = token_counter(replacement)
        running_total += replacement_tokens - message_tokens[index]
        message_tokens[index] = replacement_tokens
        messages[index] = replacement
        changed.add(index)

    def digest_results(*, failed: bool) -> EvictionResult | None:
        for index, (seq, _) in enumerate(records):
            message = messages[index]
            result = message.tool_result
            call = calls.get(result.tool_call_id) if result is not None else None
            if (
                not eligibility.allows(seq)
                or result is None
                or call is None
                or call.name not in _REDERIVABLE_TOOLS
                or bool(result.is_error or result.is_canceled) is not failed
                or message.metadata.get("context_evicted")
            ):
                continue
            path = _read_path(call)
            count = read_counts.get((path, _content_digest(result.content)), 1)
            replace(index, _digest_result(message, call, seq, read_count=count))
            if running_total <= target_tokens:
                return _result(messages, changed, before, running_total, True)
        return None

    reached = digest_results(failed=False)
    if reached is not None:
        return reached

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if (
            not eligibility.allows(seq)
            or message.role is not MessageRole.ASSISTANT
            or not any(
                isinstance(block, (ThinkingContent, RedactedThinkingContent))
                for block in message.content
            )
        ):
            continue
        content = [
            block
            for block in message.content
            if not isinstance(block, (ThinkingContent, RedactedThinkingContent))
        ]
        content.append(TextContent(f"[assistant reasoning evicted · seq {seq}]"))
        replace(
            index,
            Message(
                message.role,
                content,
                tool_result=message.tool_result,
                metadata={"context_evicted": True, "source_seq": seq},
            ),
        )
        if running_total <= target_tokens:
            return _result(messages, changed, before, running_total, True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if (
            not eligibility.allows(seq)
            or message.role is not MessageRole.ASSISTANT
            or message.tool_result is not None
            or any(isinstance(block, ToolUseContent) for block in message.content)
            or message.metadata.get("context_evicted")
        ):
            continue
        replace(
            index,
            Message(
                MessageRole.ASSISTANT,
                [TextContent(f"[assistant text evicted · seq {seq}]")],
                metadata={"context_evicted": True, "source_seq": seq},
            ),
        )
        if running_total <= target_tokens:
            return _result(messages, changed, before, running_total, True)

    reached = digest_results(failed=True)
    if reached is not None:
        return reached

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if not eligibility.allows(seq) or not _is_notification_message(message):
            continue
        replace(index, _notification_receipt(message, seq))
        if running_total <= target_tokens:
            return _result(messages, changed, before, running_total, True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if not eligibility.allows(seq):
            continue
        replacement = _digest_agent_prompts(message, seq)
        if replacement is message:
            continue
        replace(index, replacement)
        if running_total <= target_tokens:
            return _result(messages, changed, before, running_total, True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        if (
            not eligibility.allows(seq)
            or result is None
            or call is None
            or message.metadata.get("context_evicted")
            or call.name not in {"agent", "agent_output", "task_output"}
        ):
            continue
        replace(index, _orchestration_result_receipt(message, call, seq))
        if running_total <= target_tokens:
            return _result(messages, changed, before, running_total, True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        if (
            not eligibility.allows(seq)
            or result is None
            or call is None
            or message.metadata.get("context_evicted")
            or call.name not in {"inbox", "project", "recall_history", "run_background"}
        ):
            continue
        replacement = _workflow_result_receipt(message, call, seq)
        if token_counter(replacement) >= message_tokens[index]:
            continue
        replace(index, replacement)
        if running_total <= target_tokens:
            return _result(messages, changed, before, running_total, True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if not eligibility.allows(seq):
            continue
        replacement = _digest_edit_write_payloads(message, seq, results)
        if replacement is message:
            continue
        replace(index, replacement)
        if running_total <= target_tokens:
            return _result(messages, changed, before, running_total, True)

    for index, (seq, _) in enumerate(records):
        message = messages[index]
        if not eligibility.allows(seq):
            continue
        replacement = _digest_bash_commands(message, seq)
        if replacement is message:
            continue
        replace(index, replacement)
        if running_total <= target_tokens:
            return _result(messages, changed, before, running_total, True)

    return _result(
        messages,
        changed,
        before,
        running_total,
        running_total <= target_tokens,
    )


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


def _eviction_eligibility(
    records: Sequence[tuple[int, Message]],
    unconsumed_source_seqs: Collection[int],
) -> _EvictionEligibility:
    """Classify all records once so transformations cannot infer consumption."""

    protected = set(unconsumed_source_seqs)
    unconsumed_result_ids = {
        result.tool_call_id
        for seq, message in records
        if seq in protected and (result := message.tool_result) is not None
    }
    completed_call_ids = {
        result.tool_call_id
        for _, message in records
        if (result := message.tool_result) is not None
    }
    notification_seqs: list[int] = []
    bash_call_seqs: list[int] = []
    for seq, message in records:
        if _is_notification_message(message):
            notification_seqs.append(seq)
        for block in _tool_uses(message):
            call = block.tool_call
            if call.id in unconsumed_result_ids:
                protected.add(seq)
            if call.name == "agent" and call.id not in completed_call_ids:
                protected.add(seq)
            if call.name == "bash":
                bash_call_seqs.append(seq)
    protected.update(notification_seqs[-NOTIFICATION_TAIL:])
    protected.update(bash_call_seqs[-BASH_CALL_TAIL:])
    return _EvictionEligibility(
        frozenset(seq for seq, _ in records if seq not in protected)
    )


def _collapse_repeated_reads(
    records: Sequence[tuple[int, Message]],
    messages: list[Message],
    calls: Mapping[str, ToolCall],
    call_indexes: Mapping[str, int],
    changed: set[int],
    eligibility: _EvictionEligibility,
) -> dict[tuple[str | None, str], int]:
    groups: dict[tuple[str, str], list[tuple[int, str, int]]] = defaultdict(list)
    for result_index, (seq, message) in enumerate(records):
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        path = _read_path(call)
        if (
            not eligibility.allows(seq)
            or call is None
            or call.name != "read"
            or result is None
            or path is None
        ):
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
            if (
                call_index is None
                or not eligibility.allows(records[call_index][0])
                or len(_tool_uses(messages[call_index])) != 1
            ):
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


def _digest_edit_write_payloads(
    message: Message, seq: int, results: Mapping[str, ToolResult]
) -> Message:
    changed = False
    content: list[ContentBlock] = []
    for block in message.content:
        if not isinstance(block, ToolUseContent):
            content.append(block)
            continue
        call = block.tool_call
        result = results.get(call.id)
        if (
            call.name not in {"edit", "write"}
            or result is None
            or result.is_error
            or result.is_canceled
        ):
            content.append(block)
            continue
        payload_keys = (
            ("content",)
            if call.name == "write"
            else ("old_string", "new_string", "edits")
        )
        payload = {
            key: call.arguments[key] for key in payload_keys if key in call.arguments
        }
        if not payload or all(_is_edit_write_receipt(value) for value in payload.values()):
            content.append(block)
            continue
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        receipt = (
            f"[edit/write payload receipt · seq {seq}] original_chars={len(encoded)} "
            f"sha256={_content_digest(encoded)[:16]}; recall_history "
            f"seq_start={seq}, seq_end={seq} for exact payload"
        )
        arguments = dict(call.arguments)
        for key in payload:
            arguments[key] = (
                [{"old_string": receipt, "new_string": receipt}]
                if key == "edits"
                else receipt
            )
        content.append(ToolUseContent(ToolCall(call.id, call.name, arguments)))
        changed = True
    return _replaced_tool_call_message(message, content, seq) if changed else message


def _is_edit_write_receipt(value: object) -> bool:
    if isinstance(value, str):
        return value.startswith("[edit/write payload receipt · seq ")
    if isinstance(value, list):
        return bool(value) and all(
            isinstance(item, Mapping)
            and bool(item)
            and all(_is_edit_write_receipt(field) for field in item.values())
            for item in value
        )
    return False


def _digest_bash_commands(message: Message, seq: int) -> Message:
    changed = False
    content: list[ContentBlock] = []
    for block in message.content:
        if not isinstance(block, ToolUseContent):
            content.append(block)
            continue
        call = block.tool_call
        command_key = "command" if "command" in call.arguments else "cmd"
        command = call.arguments.get(command_key)
        if (
            call.name != "bash"
            or not isinstance(command, str)
            or command.startswith("[bash command receipt · seq ")
        ):
            content.append(block)
            continue
        arguments = dict(call.arguments)
        arguments[command_key] = (
            f"[bash command receipt · seq {seq}] original_chars={len(command)} "
            f"sha256={_content_digest(command)[:16]}; recall_history "
            f"seq_start={seq}, seq_end={seq} for exact command"
        )
        content.append(ToolUseContent(ToolCall(call.id, call.name, arguments)))
        changed = True
    return _replaced_tool_call_message(message, content, seq) if changed else message


def _replaced_tool_call_message(
    message: Message, content: list[ContentBlock], seq: int
) -> Message:
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


def _digest_agent_prompts(message: Message, seq: int) -> Message:
    changed = False
    content: list[ContentBlock] = []
    for block in message.content:
        if not isinstance(block, ToolUseContent):
            content.append(block)
            continue
        call = block.tool_call
        prompt = call.arguments.get("prompt")
        if (
            call.name != "agent"
            or not isinstance(prompt, str)
            or prompt.startswith("[agent prompt receipt · seq ")
        ):
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
    return _replaced_tool_call_message(message, content, seq) if changed else message


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
    payload: dict[str, object] = {
        "tool": call.name,
        "call": result.tool_call_id,
        "status": _one_line(status),
    }
    for key in ("child_instance_id", "task_id", "handle"):
        value = structured.get(key, call.arguments.get(key))
        if compact := _one_line(value):
            payload[key] = compact
    description = structured.get("description", call.arguments.get("description"))
    if compact_description := _one_line(description):
        payload["description"] = compact_description
    payload.update(
        original_chars=len(result.content),
        sha256=_content_digest(result.content)[:16],
    )
    receipt = _structured_receipt(
        "orchestration result receipt", payload, seq, "result"
    )
    return Message(
        message.role,
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


def _workflow_result_receipt(message: Message, call: ToolCall, seq: int) -> Message:
    result = message.tool_result
    if result is None:
        return message
    structured = result.structured_content or {}
    if call.name == "inbox":
        payload = _inbox_receipt_payload(call, structured)
    elif call.name == "project":
        payload = _project_receipt_payload(call, structured)
    elif call.name == "recall_history":
        payload = _recall_receipt_payload(call, result.content)
    else:
        payload = _background_receipt_payload(call, structured, result)
    receipt = _structured_receipt("workflow result receipt", payload, seq, "result")
    return Message(
        message.role,
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


def _inbox_receipt_payload(
    call: ToolCall, structured: Mapping[str, object]
) -> dict[str, object]:
    action = _one_line(call.arguments.get("action")) or "unknown"
    payload: dict[str, object] = {"action": action}
    if action == "send":
        if target := _one_line(call.arguments.get("project")):
            payload["target_project"] = target
        if message_id := _one_line(structured.get("id")):
            payload["id"] = message_id
        return payload
    if action == "projects":
        raw_projects = structured.get("projects")
        projects = (
            [item for item in raw_projects if isinstance(item, Mapping)]
            if isinstance(raw_projects, list)
            else []
        )
        payload["projects"] = [
            {
                key: compact
                for key in ("id", "name", "scope")
                if (compact := _one_line(project.get(key)))
            }
            for project in projects[:WORKFLOW_MESSAGE_LIMIT]
        ]
        if len(projects) > WORKFLOW_MESSAGE_LIMIT:
            payload["omitted_projects"] = len(projects) - WORKFLOW_MESSAGE_LIMIT
        return payload

    raw_message = structured.get("message")
    raw_messages = structured.get("messages")
    if isinstance(raw_message, Mapping):
        messages: list[Mapping[str, object]] = [raw_message]
    elif isinstance(raw_messages, list):
        messages = [item for item in raw_messages if isinstance(item, Mapping)]
    else:
        messages = []
    if (
        not messages
        and action in {"claim", "done"}
        and (message_id := _one_line(call.arguments.get("id")))
    ):
        messages = [{"id": message_id}]
    summaries: list[dict[str, object]] = []
    default_status = _one_line(structured.get("status"))
    for raw in messages[:WORKFLOW_MESSAGE_LIMIT]:
        status = _one_line(raw.get("status")) or (
            "done"
            if raw.get("done_at") is not None or action == "done"
            else "claimed"
            if raw.get("claimer_session") is not None or action == "claim"
            else default_status or "new"
        )
        summary: dict[str, object] = {
            "id": _one_line(raw.get("id")),
            "kind": _one_line(raw.get("kind")),
            "title": _one_line(raw.get("title"), limit=80),
            "status": status,
            "claimed": status in {"claimed", "done"}
            or raw.get("claimer_session") is not None,
            "done": status == "done" or raw.get("done_at") is not None,
        }
        sender = raw.get("from")
        if isinstance(sender, Mapping):
            if project := _one_line(sender.get("project")):
                summary["from_project"] = project
            if session := _one_line(sender.get("session")):
                summary["from_session"] = session
        if outcome := _one_line(raw.get("outcome"), limit=120):
            summary["outcome"] = outcome
        summaries.append(summary)
    payload["messages"] = summaries
    if len(messages) > WORKFLOW_MESSAGE_LIMIT:
        payload["omitted_messages"] = len(messages) - WORKFLOW_MESSAGE_LIMIT
    return payload


def _project_receipt_payload(
    call: ToolCall, structured: Mapping[str, object]
) -> dict[str, object]:
    payload: dict[str, object] = {
        "action": _one_line(call.arguments.get("action")) or "inspect"
    }
    project = structured.get("project")
    if isinstance(project, Mapping):
        if project_id := _one_line(project.get("project_id")):
            payload["project_id"] = project_id
        if name := _one_line(project.get("name")):
            payload["project_name"] = name
    sections = sorted(key for key in structured if key != "project")
    payload["sections"] = sections
    memory = structured.get("memory")
    if isinstance(memory, Mapping):
        payload["files"] = sorted(_one_line(name) for name in memory)[:16]
    return payload


def _recall_receipt_payload(call: ToolCall, content: str) -> dict[str, object]:
    arguments = call.arguments
    payload: dict[str, object] = {
        "action": "query" if arguments.get("query") is not None else "range",
        "result_chars": len(content),
    }
    for key in ("seq_start", "seq_end", "offset", "max_chars"):
        value = arguments.get(key)
        if type(value) is int:
            payload[key] = value
    if query := _one_line(arguments.get("query")):
        payload["query"] = query
    return payload


def _background_receipt_payload(
    call: ToolCall, structured: Mapping[str, object], result: ToolResult
) -> dict[str, object]:
    running = structured.get("running")
    status = (
        "canceled"
        if result.is_canceled
        else "error"
        if result.is_error
        else "running"
        if running is True
        else "started"
    )
    payload: dict[str, object] = {
        "command": _one_line(call.arguments.get("command"), limit=120),
        "status": status,
    }
    task_id = _one_line(structured.get("task_id"))
    if not task_id:
        match = re.search(r"\bstarted background task ([^\s()]+)", result.content)
        task_id = _one_line(match.group(1)) if match is not None else ""
    if task_id:
        payload["task_id"] = task_id
    return payload


def _is_notification_message(message: Message) -> bool:
    return (
        message.metadata.get("zeta_event") == "agent_notifications"
        and isinstance(message.metadata.get("notifications"), list)
    )


def _notification_receipt(message: Message, seq: int) -> Message:
    notifications = message.metadata["notifications"]
    summaries: list[dict[str, object]] = []
    for raw in notifications:
        if not isinstance(raw, Mapping):
            continue
        summary: dict[str, object] = {
            "kind": _one_line(raw.get("kind", "agent_completion"))
        }
        for key in ("child_instance_id", "task_id", "status"):
            if compact := _one_line(raw.get(key)):
                summary[key] = compact
        exit_code = raw.get("exit_code")
        if type(exit_code) is int:
            summary["exit_code"] = exit_code
        description = raw.get("description", raw.get("headline", raw.get("command")))
        if compact_description := _one_line(description):
            summary["description"] = compact_description
        summaries.append(summary)
    encoded = json.dumps(message.to_dict(), sort_keys=True, separators=(",", ":"))
    payload = {
        "notifications": summaries,
        "original_chars": len(encoded),
        "sha256": _content_digest(encoded)[:16],
    }
    receipt = _structured_receipt("notification receipt", payload, seq, "notification")
    return Message(
        message.role,
        [TextContent(receipt)],
        metadata={
            "context_evicted": True,
            "source_seq": seq,
            "eviction_content_digest": _content_digest(encoded),
        },
    )


def _structured_receipt(
    prefix: str, payload: Mapping[str, object], seq: int, exact_kind: str
) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return (
        f"[{prefix}] {encoded}; recall_history "
        f"seq_start={seq}, seq_end={seq} for exact {exact_kind}"
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


def _tool_results(messages: Sequence[Message]) -> dict[str, ToolResult]:
    return {
        result.tool_call_id: result
        for message in messages
        if (result := message.tool_result) is not None
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
