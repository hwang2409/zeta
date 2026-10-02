"""Non-persisted CLM context-usage reminders."""

from __future__ import annotations

from collections.abc import Collection

from ..protocol.types import Message, MessageRole, TextContent
from . import ContextTelemetry

_THRESHOLDS = ((50, "half-full"), (75, "three-quarters full"), (90, "near limit"))


def enabled_context_tools(strategies: Collection[str]) -> tuple[str, ...]:
    tools: list[str] = []
    if "archive" in strategies:
        tools.extend(("context_archive", "context_restore"))
    if "edit" in strategies:
        tools.append("context_replace")
    if "recall" in strategies:
        tools.append("recall_history")
    return tuple(tools)


def build_nudges(
    *,
    estimated_tokens: int,
    budget: int,
    strategies: Collection[str],
    emitted: set[int],
) -> list[Message]:
    """Return each newly crossed threshold once in the current compaction cycle."""

    percent = 100.0 * estimated_tokens / budget
    tools = enabled_context_tools(strategies)
    tool_text = ", ".join(tools) if tools else "no optional context tools"
    messages: list[Message] = []
    for threshold, label in _THRESHOLDS:
        if percent < threshold or threshold in emitted:
            continue
        emitted.add(threshold)
        text = (
            f"[context nudge] Usage is {percent:.0f}% ({label}); available context "
            f"tools: {tool_text}. Normal compaction remains the safety net."
        )
        messages.append(
            Message(
                MessageRole.USER,
                [TextContent(text)],
                metadata={"context_nudge": threshold},
            )
        )
        ContextTelemetry().emit(
            "nudge",
            kind="nudge",
            range=None,
            tokens=estimated_tokens,
            threshold=threshold,
        )
    return messages
