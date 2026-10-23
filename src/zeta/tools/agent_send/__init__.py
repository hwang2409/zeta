"""Send follow-up prompts to eligible live child agents."""

from __future__ import annotations

from typing import Any

from ...agent.pending_delivery import (  # noqa: F401
    AGENT_SEND_COMMIT_TIMEOUT_SECONDS,
    send_to_run,
)
from ..registry import ToolRegistry, text_block


async def _agent_send(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> dict[str, object]:
    child_instance_id = arguments.get("child_instance_id")
    error = await send_to_run(
        registry.session_store,
        child_instance_id,
        arguments.get("message"),
        channel=registry._agent_owner.conversation_channel,
    )
    if error is not None:
        return {
            "content": [text_block(f"agent_send error: {error}")],
            "isError": True,
            "structuredContent": None,
        }
    return {
        "content": [
            text_block(
                f"queued a follow-up for {child_instance_id}; it is delivered "
                "when the child finishes its current turn"
            )
        ],
        "isError": False,
        "structuredContent": {"child_instance_id": child_instance_id},
    }


def register_send(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "agent_send",
        _agent_send,
        description=(
            "Send a follow-up to an eligible child you started. It arrives at "
            "the next turn boundary and never interrupts a tool call. When "
            "answering ask_parent, you can reference its question id in the message."
        ),
        parameters={
            "type": "object",
            "properties": {
                "child_instance_id": {"type": "string", "minLength": 1},
                "message": {"type": "string", "minLength": 1},
            },
            "required": ["child_instance_id", "message"],
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=True,
    )
