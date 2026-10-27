"""Context-budget truncation for durable tool-result messages."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from ...protocol.types import Message, ToolResult, flatten_tool_content


def truncate_tool_results(
    messages: Sequence[Message],
    target: int,
    result_seqs: Mapping[int, int],
    token_counter: Callable[[Message], int],
) -> list[Message] | None:
    """Fit tool results by truncating the largest durable outputs first."""

    result = list(messages)
    message_tokens = [token_counter(message) for message in result]
    running_total = sum(message_tokens)
    if running_total <= target:
        return result
    candidates: list[tuple[int, int, int, str]] = []
    for index, message in enumerate(messages):
        tool_result = message.tool_result
        seq = result_seqs.get(id(message))
        if tool_result is None or seq is None:
            continue
        content = (
            flatten_tool_content(tool_result.content_blocks, detailed_images=True)
            if tool_result.content_blocks is not None
            else tool_result.content
        )
        candidates.append((-len(content), seq, index, content))
    for _, seq, index, content in sorted(candidates):
        if running_total <= target:
            break
        original = result[index].tool_result
        if original is None:
            continue
        low = 0
        high = len(content)
        best: tuple[Message, int] | None = None
        while low <= high:
            shown = (low + high) // 2
            replacement = _truncated_tool_message(result[index], content, shown, seq)
            replacement_tokens = token_counter(replacement)
            proposed_total = running_total - message_tokens[index] + replacement_tokens
            if proposed_total <= target:
                best = (replacement, replacement_tokens)
                low = shown + 1
            else:
                high = shown - 1
        if best is None:
            replacement = _truncated_tool_message(result[index], content, 0, seq)
            best = (replacement, token_counter(replacement))
        result[index], replacement_tokens = best
        running_total += replacement_tokens - message_tokens[index]
        message_tokens[index] = replacement_tokens
    return result if running_total <= target else None


def _truncated_tool_message(
    message: Message,
    content: str,
    shown: int,
    seq: int,
) -> Message:
    head_size = (shown + 1) // 2
    tail_size = shown - head_size
    excerpt = content[:head_size]
    if tail_size:
        excerpt += "\n…\n" + content[len(content) - tail_size :]
    marker = (
        f"[output truncated for context: showed {shown} of {len(content)} chars; "
        f"full output is in the session log at seq {seq}]"
    )
    excerpt = f"{excerpt}\n{marker}" if excerpt else marker
    original = message.tool_result
    if original is None:
        return message
    return Message(
        message.role,
        [],
        tool_result=ToolResult(
            original.tool_call_id,
            excerpt,
            is_error=original.is_error,
            is_canceled=original.is_canceled,
        ),
        metadata=dict(message.metadata),
    )
