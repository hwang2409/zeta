"""Run child-agent turns and preserve their bounded result shape."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import TYPE_CHECKING

from .types import Message, StreamEvent, StreamEventType, ToolCall, assistant_text

if TYPE_CHECKING:
    from .loop import AgentLoop


async def consume_child(
    child_loop: AgentLoop,
    prompt: str,
    *,
    turn_cap: int,
    child_path: str,
    publish: Callable[[str], None],
    child_turns: Callable[[], int],
    update_turns: Callable[[int], None],
    publish_lifecycle: Callable[..., None],
    child_result: Callable[..., dict[str, object]],
    error_message: Callable[[BaseException], str],
) -> dict[str, object]:
    """Consume one child loop, including nested lifecycle events."""

    final_message: Message | None = None
    last_assistant_text = ""
    cap_hit = False
    failure_message: str | None = None
    budget_exhausted = False

    def lifecycle_depth(call: ToolCall | None) -> int:
        if call is not None and call.name.casefold() == "agent":
            return child_loop.agent_depth + 1
        return child_loop.agent_depth

    try:
        async for event in child_loop.run_turn(prompt):
            if event.type is StreamEventType.TURN_START:
                publish(f"turn {child_turns() + 1}: thinking")
            elif event.type is StreamEventType.TOOL_APPROVAL_START:
                name = event.tool_call.name if event.tool_call is not None else "tool"
                publish(f"turn {child_turns() + 1}: approval pending: {name}")
                publish_lifecycle(
                    "approval_start",
                    event.tool_call,
                    depth=lifecycle_depth(event.tool_call),
                )
            elif event.type is StreamEventType.TOOL_APPROVAL_END:
                publish_lifecycle(
                    "approval_end",
                    event.tool_call,
                    depth=lifecycle_depth(event.tool_call),
                )
            elif event.type is StreamEventType.TOOL_EXECUTION_START:
                name = event.tool_call.name if event.tool_call is not None else "tool"
                arguments = (
                    event.tool_call.arguments if event.tool_call is not None else {}
                )
                summary = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
                publish(f"turn {child_turns() + 1}: tool: {name} {summary}")
                publish_lifecycle(
                    "execution_start",
                    event.tool_call,
                    depth=lifecycle_depth(event.tool_call),
                )
            elif event.type is StreamEventType.TOOL_EXECUTION_END:
                publish_lifecycle(
                    "execution_end",
                    event.tool_call,
                    tool_result=event.tool_result,
                    depth=lifecycle_depth(event.tool_call),
                )
                if (
                    event.tool_result is not None
                    and event.tool_result.structured_content is not None
                    and event.tool_result.structured_content.get("error_code")
                    == "agent_turn_budget"
                ):
                    budget_exhausted = True
                    failure_message = event.tool_result.content
            elif event.type is StreamEventType.TURN_END:
                turns = child_turns() + 1
                update_turns(turns)
                if event.message is not None:
                    last_assistant_text = _assistant_text_snippet(event.message)
                if event.data.get("tool_calls") == 0 and event.message is not None:
                    final_message = event.message
            elif event.type is StreamEventType.ERROR and event.error is not None:
                if event.error.code == "agent_turn_budget":
                    budget_exhausted = True
                    failure_message = event.error.message
                elif event.error.code == "max_turns":
                    cap_hit = True
                else:
                    failure_message = event.error.message
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        failure_message = error_message(exc)
    if budget_exhausted:
        return child_result(
            f"agent error: {failure_message or 'shared agent turn budget exhausted'}",
            error=True,
            budget_exhausted=True,
        )
    if cap_hit:
        return child_result(
            f"agent error: child reached the {turn_cap}-turn cap; "
            f"partial state is saved at {child_path}; "
            f"last assistant text: {last_assistant_text or '[none]'}; "
            f"turns used: {child_turns()}",
            error=True,
        )
    if failure_message is not None:
        return child_result(f"agent error: {failure_message}", error=True)
    if final_message is None:
        return child_result(
            "agent error: child ended without a final response", error=True
        )
    final_text = assistant_text(final_message)
    if not final_text.strip():
        return child_result(
            "agent error: child returned an empty final assistant message",
            error=True,
        )
    return child_result(final_text, error=False)


def _assistant_text_snippet(message: Message) -> str:
    text = assistant_text(message).replace("\r", " ").replace("\n", " ")
    return text if len(text) <= 160 else f"{text[:157]}..."
