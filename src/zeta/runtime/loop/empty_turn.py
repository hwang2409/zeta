"""Empty (thinking-only) turn detection, stop-reason metadata, and nudge recovery.

Models sometimes end a turn with an assistant message that carries only a
thinking block -- no text and no tool call. For an ordinary user turn that reads
as a dropped reply, so the loop nudges once to recover. For a notification turn
it is a legitimate "nothing to report" and stays silent. When the model hit its
output-token limit mid-thinking we surface that instead of nudging.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ...protocol.types import (
    MESSAGE_ORIGIN_METADATA,
    Message,
    MessageOrigin,
    MessageRole,
    TextContent,
    ToolUseContent,
)

# Marks the hidden nudge message so UIs can hide it from the transcript.
EMPTY_TURN_NUDGE_EVENT = "empty_turn_nudge"

# Short hidden prompt appended once when a turn ends with no visible reply.
EMPTY_TURN_NUDGE_TEXT = (
    "You ended your turn without a visible response. "
    "Reply to the user or continue the task."
)

# Surfaced (TUI and subagent receipt) when a thinking-only turn hit the limit.
MAX_TOKENS_THINKING_NOTICE = (
    "The model reached its output-token limit while thinking "
    "and did not produce a reply."
)

# Provider stop reasons are short; keep the persisted value bounded regardless.
MAX_STOP_REASON_CHARS = 64


def has_visible_output(message: Message) -> bool:
    """True when the assistant produced text or a tool call the user can act on."""

    for block in message.content:
        if isinstance(block, ToolUseContent):
            return True
        if isinstance(block, TextContent) and block.text.strip():
            return True
    return False


def read_turn_metadata(data: Mapping[str, Any]) -> tuple[str | None, int | None]:
    """Pull the normalized stop reason and output-token usage from event data."""

    reason = data.get("stop_reason")
    stop_reason = (
        reason[:MAX_STOP_REASON_CHARS]
        if type(reason) is str and reason
        else None
    )
    usage = data.get("usage")
    tokens = usage.get("output_tokens") if isinstance(usage, Mapping) else None
    output_tokens = tokens if type(tokens) is int else None
    return stop_reason, output_tokens


def annotate_turn_metadata(
    message: Message,
    *,
    stop_reason: str | None,
    output_tokens: int | None,
) -> Message:
    """Record the provider stop reason and output-token usage on the message."""

    if stop_reason is None and output_tokens is None:
        return message
    metadata: dict[str, Any] = dict(message.metadata)
    if stop_reason is not None:
        metadata["stop_reason"] = stop_reason
    if output_tokens is not None:
        metadata["output_tokens"] = output_tokens
    return Message(
        message.role,
        message.content,
        tool_result=message.tool_result,
        metadata=metadata,
    )


def build_nudge_message() -> Message:
    """A hidden user message that asks the model to actually respond."""

    return Message(
        MessageRole.USER,
        [TextContent(EMPTY_TURN_NUDGE_TEXT)],
        metadata={
            "zeta_event": EMPTY_TURN_NUDGE_EVENT,
            MESSAGE_ORIGIN_METADATA: MessageOrigin.HARNESS_NUDGE.value,
        },
    )


def should_nudge_empty_turn(
    message: Message,
    *,
    stop_reason: str | None,
    notification_turn: bool,
    already_nudged: bool,
) -> bool:
    """Whether a turn with no tool calls should be nudged once to recover.

    Skip notification turns (silence is fine), turns that already produced
    visible output, turns truncated at the output-token limit (surfaced
    separately), and any turn already nudged this user turn.
    """

    return (
        not notification_turn
        and not already_nudged
        and stop_reason != "max_tokens"
        and not has_visible_output(message)
    )
