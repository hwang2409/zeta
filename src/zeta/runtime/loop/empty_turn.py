"""Empty (thinking-only) turn detection, stop-reason metadata, and nudge recovery.

Models sometimes end a turn with an assistant message that carries only a
thinking block -- no text and no tool call. For an ordinary user turn that reads
as a dropped reply, so the loop nudges once to recover. For a notification turn
it is a legitimate "nothing to report" and stays silent. When the model hit its
output-token limit mid-thinking we surface that instead of nudging.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...protocol.types import (
    ErrorInfo,
    Message,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolUseContent,
    harness_notice,
)

# Marks the hidden nudge message so UIs can hide it from the transcript.
EMPTY_TURN_NUDGE_EVENT = "empty_turn_nudge"

# Short hidden prompt appended once when a turn ends with no visible reply.
EMPTY_TURN_NUDGE_TEXT = (
    "You ended your turn without a visible response. "
    "Reply to the user or continue the task."
)
TOOL_LIMIT_RECOVERY_EVENT = "tool_call_output_limit_recovery"

# Provider stop reasons are short; keep the persisted value bounded regardless.
MAX_STOP_REASON_CHARS = 64
OUTPUT_LIMIT_ERROR_MESSAGE = (
    "the model reached its output-token limit while thinking "
    "and did not produce a reply"
)
MAX_EMPTY_TURN_NUDGES = 3
MAX_EMPTY_TURN_DETAIL_CHARS = 128


@dataclass(slots=True)
class EmptyTurnRecoveryState:
    nudge_count: int = 0
    nudge_episode: int = -1
    tool_result_episode: int = 0
    last_tool_call: Any = None
    last_tool_result: Any = None

    def record_tool_result(self, tool_call: Any, tool_result: Any) -> None:
        self.last_tool_call = tool_call
        self.last_tool_result = tool_result
        self.tool_result_episode += 1

    def record_event(self, event: Any) -> None:
        if event.type is StreamEventType.TOOL_EXECUTION_END:
            self.record_tool_result(event.tool_call, event.tool_result)


async def complete_empty_turn(
    transition: Any,
    assistant_message: Message,
    turn_number: int,
    *,
    foreground: bool,
    admit_terminal: Any,
    schedule_index: Any,
) -> tuple[tuple[Any, ...], bool]:
    if transition.error is not None:
        if foreground:
            await admit_terminal("foreground_end")
        return (
            (
                StreamEvent(StreamEventType.ERROR, error=transition.error),
                StreamEvent(StreamEventType.AGENT_END),
            ),
            True,
        )
    turn_end = StreamEvent(
        StreamEventType.TURN_END,
        message=assistant_message,
        data={"turn": turn_number, "tool_calls": 0},
    )
    if transition.continue_turn:
        return (turn_end,), False
    schedule_index()
    return (turn_end, StreamEvent(StreamEventType.AGENT_END)), True


def _bounded_detail(value: str) -> str:
    return value[:MAX_EMPTY_TURN_DETAIL_CHARS]


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


# One identical re-dispatch is legitimate: a gate deadline can cancel a turn
# before the provider acknowledges its delivery, and the next call repeats it.
# A second identical call means the same cancel route fired again; stop there.
MAX_REPEATED_DISPATCHES = 1
NO_PROGRESS_LOOP_CODE = "no_progress_loop"


class DispatchProgress:
    """Bound provider calls that repeat a request with nothing new committed."""

    def __init__(self) -> None:
        self._mark: tuple[object, ...] | None = None
        self._repeats = 0

    def reset(self) -> None:
        self._mark = None
        self._repeats = 0

    def admit(
        self, store: Any, prepared: Any, flow: Any, retrying_context: bool
    ) -> str | ErrorInfo:
        """Return "dispatch", or stop with a typed error and keep the input pending."""

        mark = (
            store.active_branch_head_sequence(),
            prepared.prompt_id,
            prepared.notification_ids,
        )
        if retrying_context or prepared.steering is not None or mark != self._mark:
            self._mark = mark
            self._repeats = 0
            return "dispatch"
        self._repeats += 1
        if self._repeats <= MAX_REPEATED_DISPATCHES:
            return "dispatch"
        self.reset()
        flow.release_active()
        # keep the input durable but stop it from waking the model again; the
        # next new input carries it into the following request.
        stopped = prepared.notification_ids + (
            () if prepared.prompt_id is None else (prepared.prompt_id,)
        )
        flow.suppress_pending(stopped)
        return ErrorInfo(
            NO_PROGRESS_LOOP_CODE,
            f"the model was called {MAX_REPEATED_DISPATCHES + 1} times with the "
            "same request and nothing new was saved; stopped to avoid repeat "
            "spend. Send a new message to continue.",
        )


def build_nudge_message() -> Message:
    """Build hidden zeta guidance that asks the model to actually respond."""

    return harness_notice(
        "empty-turn recovery",
        EMPTY_TURN_NUDGE_TEXT,
        event=EMPTY_TURN_NUDGE_EVENT,
    )


def build_tool_limit_recovery_message(tool_name: str) -> Message:
    text = (
        f"Your output hit the output token limit while writing a <{tool_name[:128]}> "
        "call. You should split the work into smaller calls. For files, create a "
        "smaller file, then edit/append it."
    )
    return harness_notice(
        "output-limit recovery",
        text,
        event=TOOL_LIMIT_RECOVERY_EVENT,
    )


def output_limit_error(
    message: Message, *, stop_reason: str | None
) -> ErrorInfo | None:
    if stop_reason != "max_tokens" or has_visible_output(message):
        return None
    return ErrorInfo("max_tokens", OUTPUT_LIMIT_ERROR_MESSAGE)


def should_nudge_empty_turn(
    message: Message,
    *,
    stop_reason: str | None,
    notification_turn: bool,
    already_nudged: bool,
    nudge_count: int = 0,
    max_nudges: int = MAX_EMPTY_TURN_NUDGES,
) -> bool:
    """Whether a turn with no tool calls should be nudged once to recover.

    Skip notification turns (silence is fine), turns that already produced
    visible output, turns truncated at the output-token limit (surfaced
    separately), and any turn already nudged this user turn.
    """

    return (
        not notification_turn
        and not already_nudged
        and nudge_count < max_nudges
        and stop_reason != "max_tokens"
        and not has_visible_output(message)
    )


def empty_turn_error(
    tool_name: str | None,
    tool_call_id: str | None,
    tool_state: str | None,
) -> ErrorInfo:
    if tool_name is None or tool_call_id is None or tool_state is None:
        return ErrorInfo("empty_turn", "child returned an empty final assistant message")
    return ErrorInfo(
        "empty_turn",
        "child returned an empty final assistant message after tool "
        f"`{_bounded_detail(tool_name)}` ({_bounded_detail(tool_call_id)}) "
        f"{tool_state}",
    )


def empty_turn_tool_error(tool_call: Any, result: Any) -> ErrorInfo:
    state = "cancelled" if result.is_canceled else "error" if result.is_error else "ok"
    return empty_turn_error(tool_call.name, tool_call.id, state)


def empty_turn_recovery_error(
    message: Message,
    *,
    notification_turn: bool,
    consuming_notifications: bool,
    should_nudge: bool,
    recovery_state: EmptyTurnRecoveryState,
) -> ErrorInfo | None:
    if (
        notification_turn
        or consuming_notifications
        or should_nudge
        or has_visible_output(message)
        or recovery_state.tool_result_episode == 0
        or recovery_state.last_tool_call is None
        or recovery_state.last_tool_result is None
    ):
        return None
    return empty_turn_tool_error(
        recovery_state.last_tool_call, recovery_state.last_tool_result
    )
