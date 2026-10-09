"""Send follow-up prompts to eligible live child agents."""

from __future__ import annotations

import asyncio
from typing import Any

from ...agent.conversation_channel import ConversationChannel
from ...core.store import ConversationStore
from ..registry import ToolRegistry, text_block

AGENT_SEND_COMMIT_TIMEOUT_SECONDS = 1.0


def send_to_run(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
    *,
    channel: ConversationChannel,
) -> str | None:
    """Queue and wake an eligible live child through its conversation channel."""

    return channel.publish_follow_up(
        parent_store,
        child_instance_id,
        message,
        commit_timeout=AGENT_SEND_COMMIT_TIMEOUT_SECONDS,
    )


async def _agent_send(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> dict[str, object]:
    # The channel reloads the run's own conversation.jsonl and appends under
    # flock+fsync; runs with large logs would stall the event loop, so hop
    # to a worker thread while the marker check and durable append happen.
    commit = asyncio.create_task(
        asyncio.to_thread(
            send_to_run,
            registry.session_store,
            arguments.get("child_instance_id"),
            arguments.get("message"),
            channel=registry._agent_owner.conversation_channel,
        )
    )
    while True:
        try:
            error = await asyncio.shield(commit)
        except asyncio.CancelledError:
            # The worker cannot be canceled. Absorb every caller cancellation
            # until its one commit decision is known.
            if commit.cancelled():
                raise
            continue
        break
    if error is not None:
        return {
            "content": [text_block(f"agent_send error: {error}")],
            "isError": True,
            "structuredContent": None,
        }
    child_instance_id = arguments["child_instance_id"]
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
