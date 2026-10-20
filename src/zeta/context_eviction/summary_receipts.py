"""Canonical source projection and reversible eviction-receipt decoration."""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Mapping, Sequence

from ..protocol.types import (
    Message,
    MessageRole,
    TextContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)
from .receipt_constructors import RECEIPT_FIELDS_METADATA, RECEIPT_KIND_METADATA
from .summaries import SummaryKey, SummaryRequest, source_hash, valid_summary
from .token_estimates import estimated_text_tokens

MIN_SUMMARY_SOURCE_TOKENS = 256
# Short single items are common and carried the recall gain in the experiment.
MIN_SHORT_ITEM_SOURCE_TOKENS = 48
SHORT_ITEM_SUMMARY_KINDS = frozenset({"assistant", "notification", "orchestration"})


def summary_source_is_large_enough(source: str, *, kind: str) -> bool:
    """Apply the source threshold for this summary kind."""
    minimum = (
        MIN_SHORT_ITEM_SOURCE_TOKENS
        if kind in SHORT_ITEM_SUMMARY_KINDS
        else MIN_SUMMARY_SOURCE_TOKENS
    )
    return estimated_text_tokens(source) >= minimum


def summary_decoration_fits_source(
    kind: str, *, decorated_tokens: int, source_tokens: int
) -> bool:
    """Keep range savings strict while allowing non-growing single items."""
    if kind in SHORT_ITEM_SUMMARY_KINDS:
        return decorated_tokens <= source_tokens
    return decorated_tokens < source_tokens


def normalize_receipt(message: Message) -> Message:
    """Restore the exact deterministic receipt kept in decoration metadata."""
    value = message.metadata.get("eviction_deterministic_receipt")
    if not isinstance(value, Mapping):
        return message
    try:
        return Message.from_dict(value)
    except (KeyError, TypeError, ValueError):
        return message


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


def _decorate_text(text: str, summary: str, *, kind: str) -> str:
    encoded = json.dumps(summary, ensure_ascii=False)
    if kind in {"notification", "tool_result"} and "}; recall_history" in text:
        return text.replace(
            "}; recall_history", f',"summary_data":{encoded}}}; recall_history', 1
        )
    if kind == "range" and "; recall_history" in text:
        return text.replace(
            "; recall_history", f"; summary_data={encoded}; recall_history", 1
        )
    if kind == "assistant":
        match = re.search(r"seq (\d+)", text)
        seq = int(match.group(1)) if match else 0
        return (
            f'{text} {{"summary_data":{encoded}}}; recall_history '
            f"seq_start={seq}, seq_end={seq} for exact assistant text"
        )
    return text


def decorate_receipt(message: Message, summary: str | None) -> Message:
    """Decorate one receipt reversibly without changing canonical metadata."""
    if (
        not summary
        or not valid_summary(summary)
        or not message.metadata.get("context_evicted")
    ):
        return message
    kind = str(message.metadata.get(RECEIPT_KIND_METADATA, ""))
    metadata = dict(message.metadata)
    metadata["eviction_deterministic_receipt"] = message.to_dict()
    metadata["summary_data"] = summary
    if message.tool_result is not None:
        result = message.tool_result
        decorated_result = ToolResult(
            result.tool_call_id,
            _decorate_text(result.content, summary, kind=kind),
            is_error=result.is_error,
            content_blocks=result.content_blocks,
            is_canceled=result.is_canceled,
        )
        return Message(
            message.role,
            message.content,
            tool_result=decorated_result,
            metadata=metadata,
        )
    content = [
        TextContent(_decorate_text(block.text, summary, kind=kind))
        if isinstance(block, TextContent)
        else block
        for block in message.content
    ]
    return Message(message.role, content, metadata=metadata)


def _tool_calls(messages: Sequence[Message]) -> dict[str, ToolCall]:
    return {
        block.tool_call.id: block.tool_call
        for message in messages
        for block in message.content
        if isinstance(block, ToolUseContent)
    }


def _notification(message: Message) -> bool:
    return message.metadata.get("zeta_event") == "agent_notifications" and isinstance(
        message.metadata.get("notifications"), list
    )


def _summary_source(
    source_messages: Mapping[int, Message], seq_start: int, seq_end: int
) -> str:
    rows: list[str] = []
    calls = _tool_calls(list(source_messages.values()))
    for seq in sorted(source_messages):
        if not seq_start <= seq <= seq_end:
            continue
        message = source_messages[seq]
        if _summary_kind(message, calls) is None:
            continue
        parts = [
            block.text for block in message.content if isinstance(block, TextContent)
        ]
        if message.tool_result is not None:
            parts.append(message.tool_result.content)
        if _notification(message):
            parts.append(
                "notifications="
                + json.dumps(
                    message.metadata["notifications"],
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )
        rows.append(f"seq {seq} {message.role.value}: {' '.join(parts)}")
    return "\n".join(rows)


def _summary_kind(message: Message, calls: Mapping[str, ToolCall]) -> str | None:
    if message.metadata.get("context_evicted"):
        return None
    if _notification(message):
        return "notification"
    result = message.tool_result
    call = calls.get(result.tool_call_id) if result is not None else None
    if call is not None and call.name in {"agent", "task_output"}:
        return "orchestration"
    if (
        message.role is MessageRole.ASSISTANT
        and result is None
        and not any(isinstance(block, ToolUseContent) for block in message.content)
        and any(
            isinstance(block, TextContent) and block.text.strip()
            for block in message.content
        )
    ):
        return "assistant"
    return None


def eviction_summary_candidates(
    records: Sequence[tuple[int, Message]],
    *,
    evictable_source_seqs: Collection[int],
) -> tuple[SummaryRequest, ...]:
    """Project covered single-item work from shared eligibility output."""
    source_messages = dict(records)
    calls = _tool_calls(list(source_messages.values()))
    requests: list[SummaryRequest] = []
    for seq, message in records:
        kind = _summary_kind(message, calls)
        if kind is None or seq not in evictable_source_seqs:
            continue
        source = _summary_source(source_messages, seq, seq)
        if not summary_source_is_large_enough(source, kind=kind):
            continue
        requests.append(
            SummaryRequest(SummaryKey(kind, seq, seq, source_hash(source)), source)
        )
    return tuple(requests)


def receipt_summary_request(
    message: Message, source_messages: Mapping[int, Message]
) -> SummaryRequest | None:
    """Build the exact cache key for a finished eligible receipt."""
    if not message.metadata.get("context_evicted"):
        return None
    start = message.metadata.get("source_seq")
    end = message.metadata.get("range_seq_end", start)
    if type(start) is not int or type(end) is not int:
        return None
    receipt_kind = message.metadata.get(RECEIPT_KIND_METADATA)
    kind: str | None = None
    if receipt_kind == "range":
        calls = _tool_calls(list(source_messages.values()))
        if any(
            _summary_kind(source, calls) is not None
            for seq, source in source_messages.items()
            if start <= seq <= end
        ):
            kind = "range"
    elif receipt_kind == "assistant":
        fields = message.metadata.get(RECEIPT_FIELDS_METADATA)
        if isinstance(fields, Mapping) and fields.get("kind") == "assistant text":
            kind = "assistant"
    elif receipt_kind == "notification":
        kind = "notification"
    elif receipt_kind == "tool_result":
        fields = message.metadata.get(RECEIPT_FIELDS_METADATA)
        if (
            isinstance(fields, Mapping)
            and fields.get("receipt_kind") == "orchestration"
        ):
            kind = "orchestration"
    if kind is None:
        return None
    source = _summary_source(source_messages, start, end)
    if not summary_source_is_large_enough(source, kind=kind):
        return None
    return SummaryRequest(
        SummaryKey(kind, start, end, source_hash(source)),
        source,
        exact_range=kind == "range",
    )
