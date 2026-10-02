"""Deterministic durable context eviction."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from ..core.store import ConversationEntry
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
from .fold import (
    REDERIVABLE_TOOLS,
    estimated_tokens,
    fold_messages,
    is_failed_result,
    tool_calls,
    tool_subject,
)

EVICTION_KIND = "eviction"
MUTATING_TOOLS = frozenset({"edit", "write"})


@dataclass(frozen=True, slots=True)
class EvictionResult:
    messages: list[Message]
    items_evicted: int
    items_folded: int
    tokens_before: int
    tokens_after: int
    reached_target: bool


def evict_messages(
    records: Sequence[tuple[int, Message]],
    *,
    fixed_tokens: int,
    target_tokens: int,
    token_counter: Callable[[Message], int] = estimated_tokens,
) -> EvictionResult:
    """Apply ordered eviction stages and return one deterministic replacement view."""

    original = [message for _, message in records]
    messages = list(original)
    before = fixed_tokens + sum(token_counter(message) for message in messages)
    calls = tool_calls(records)
    seqs = [seq for seq, _ in records]
    changed: set[int] = set()

    def total() -> int:
        return fixed_tokens + sum(token_counter(message) for message in messages)

    def done() -> bool:
        return total() <= target_tokens

    folded = fold_messages(records, token_counter=token_counter)
    applied_folds = 0
    for index, candidate in enumerate(folded.messages):
        if candidate != messages[index]:
            messages[index] = candidate
            changed.add(index)
            applied_folds += 1
            if done():
                return _result(messages, changed, applied_folds, before, total(), True)

    # Reasoning is disposable before conversational text.
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
        content.append(TextContent(_stub("assistant reasoning", seq)))
        messages[index] = Message(
            message.role,
            content,
            tool_result=message.tool_result,
            metadata={"context_reasoning_evicted": True, "source_seq": seq},
        )
        changed.add(index)
        if done():
            return _result(
                messages, changed, folded.items_folded, before, total(), True
            )

    # Remove successful re-derivable call/result pairs together, so no provider
    # can observe an orphan function call or output.
    result_indexes = {
        message.tool_result.tool_call_id: index
        for index, message in enumerate(messages)
        if message.tool_result is not None and not is_failed_result(message)
    }
    for call_index, call_message in enumerate(list(messages)):
        call_blocks = [
            block.tool_call
            for block in call_message.content
            if isinstance(block, ToolUseContent)
        ]
        # Parallel calls share one assistant message. Remove the whole eligible
        # batch atomically; partial removal could put assistant prose between a
        # surviving call and its result in provider payloads.
        if not call_blocks or not all(
            call.name in REDERIVABLE_TOOLS and call.id in result_indexes
            for call in call_blocks
        ):
            continue
        remaining = [
            block
            for block in call_message.content
            if not isinstance(block, ToolUseContent)
        ]
        remaining.extend(
            TextContent(
                _stub(
                    " ".join(
                        part
                        for part in (call.name, tool_subject(call), "tool call")
                        if part
                    ),
                    seqs[call_index],
                )
            )
            for call in call_blocks
        )
        messages[call_index] = Message(
            call_message.role,
            remaining,
            metadata={"context_evicted": True, "source_seq": seqs[call_index]},
        )
        changed.add(call_index)
        for call in call_blocks:
            result_index = result_indexes[call.id]
            messages[result_index] = Message(
                MessageRole.ASSISTANT,
                [
                    TextContent(
                        _stub(
                            " ".join(
                                part
                                for part in (
                                    call.name,
                                    tool_subject(call),
                                    "tool result",
                                )
                                if part
                            ),
                            seqs[result_index],
                        )
                    )
                ],
                metadata={"context_evicted": True, "source_seq": seqs[result_index]},
            )
            changed.add(result_index)
        if done():
            return _result(
                messages, changed, folded.items_folded, before, total(), True
            )

    # Old standalone assistant prose is lower priority than user input.
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
            [TextContent(_stub("assistant text", seq))],
            metadata={"context_evicted": True, "source_seq": seq},
        )
        changed.add(index)
        if done():
            return _result(
                messages, changed, folded.items_folded, before, total(), True
            )

    # Mutation calls stay paired and retain compact argument evidence; successful
    # results may be reduced to a result stub.
    for call_id, call in calls.items():
        if call.name not in MUTATING_TOOLS or call_id not in result_indexes:
            continue
        call_index = _call_index(messages, call_id)
        result_index = result_indexes[call_id]
        if call_index is None:
            continue
        messages[call_index] = _summarize_mutating_call(
            messages[call_index], call_id, seqs[call_index]
        )
        messages[result_index] = _tool_result_stub(
            messages[result_index], call, seqs[result_index], failed=False
        )
        changed.update((call_index, result_index))
        if done():
            return _result(
                messages, changed, folded.items_folded, before, total(), True
            )

    # Other successful results remain paired but can lose bulky output.
    for index, (seq, _) in enumerate(records):
        message = messages[index]
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        if result is None or call is None or is_failed_result(message):
            continue
        messages[index] = _tool_result_stub(message, call, seq, failed=False)
        changed.add(index)
        if done():
            return _result(
                messages, changed, folded.items_folded, before, total(), True
            )

    # Failures are last-resort evidence: retain the failed status and pairing.
    for index, (seq, _) in enumerate(records):
        message = messages[index]
        result = message.tool_result
        call = calls.get(result.tool_call_id) if result is not None else None
        if result is None or call is None or not is_failed_result(message):
            continue
        messages[index] = _tool_result_stub(message, call, seq, failed=True)
        changed.add(index)
        if done():
            return _result(
                messages, changed, folded.items_folded, before, total(), True
            )

    return _result(messages, changed, folded.items_folded, before, total(), done())


def eviction_view(
    records: Sequence[tuple[int, Message]], result: EvictionResult
) -> list[dict]:
    return [
        {"seq": seq, "message": message.to_dict()}
        for (seq, _), message in zip(records, result.messages, strict=True)
    ]


def materialize_evictions(
    entries: Sequence[ConversationEntry],
) -> list[ConversationEntry]:
    """Replay active eviction markers as their persisted replacement messages."""

    markers = [entry for entry in entries if entry.type == "compaction"]
    superseded = {
        marker_id for marker in markers for marker_id in marker.data.get("replaces", [])
    }
    evictions = [
        entry
        for entry in markers
        if entry.id not in superseded and entry.data.get("kind") == EVICTION_KIND
    ]
    by_start = {entry.data["source_seq_start"]: entry for entry in evictions}
    ranges = [
        (entry.data["source_seq_start"], entry.data["source_seq_end"])
        for entry in evictions
    ]
    output: list[ConversationEntry] = []
    emitted: set[str] = set()
    for entry in entries:
        marker = by_start.get(entry.seq)
        if marker is not None:
            output.extend(_view_entries(marker))
            emitted.add(marker.id)
        if entry.type == "compaction" and (
            entry.data.get("kind") == EVICTION_KIND or entry.id in superseded
        ):
            continue
        if any(start <= entry.seq <= end for start, end in ranges):
            continue
        output.append(entry)
    for marker in evictions:
        if marker.id not in emitted:
            output.extend(_view_entries(marker))
    return output


def _view_entries(marker: ConversationEntry) -> list[ConversationEntry]:
    return [
        ConversationEntry(
            seq=row["seq"],
            id=f"{marker.id}:evicted:{index}",
            parent_id=marker.parent_id,
            lane=marker.lane,
            type="message",
            data={
                "message": row["message"],
                "eviction_marker_id": marker.id,
                "eviction_source_seq_start": marker.data["source_seq_start"],
                "eviction_source_seq_end": marker.data["source_seq_end"],
            },
        )
        for index, row in enumerate(marker.data["view"])
    ]


def _result(
    messages: list[Message],
    changed: set[int],
    folded: int,
    before: int,
    after: int,
    reached: bool,
) -> EvictionResult:
    return EvictionResult(messages, len(changed), folded, before, after, reached)


def _stub(label: str, seq: int) -> str:
    return f"[evicted {label} · seq {seq} · recall_history seq_start={seq}]"


def _call_index(messages: Sequence[Message], call_id: str) -> int | None:
    return next(
        (
            index
            for index, message in enumerate(messages)
            if any(
                isinstance(block, ToolUseContent) and block.tool_call.id == call_id
                for block in message.content
            )
        ),
        None,
    )


def _tool_result_stub(
    message: Message, call: ToolCall, seq: int, *, failed: bool
) -> Message:
    result = message.tool_result
    if result is None:
        return message
    label = f"failed {call.name} tool result" if failed else f"{call.name} tool result"
    return Message(
        message.role,
        list(message.content),
        tool_result=ToolResult(
            result.tool_call_id,
            _stub(label, seq),
            is_error=result.is_error,
            is_canceled=result.is_canceled,
        ),
        metadata={"context_evicted": True, "source_seq": seq},
    )


def _summarize_mutating_call(message: Message, call_id: str, seq: int) -> Message:
    content = []
    for block in message.content:
        if not isinstance(block, ToolUseContent) or block.tool_call.id != call_id:
            content.append(block)
            continue
        call = block.tool_call
        arguments = {
            key: value
            if not isinstance(value, str) or len(value) <= 160
            else f"{value[:157]}... [argument evicted; seq {seq}]"
            for key, value in call.arguments.items()
        }
        content.append(ToolUseContent(ToolCall(call.id, call.name, arguments)))
    return Message(
        message.role,
        content,
        metadata={"context_evicted": True, "source_seq": seq},
    )
