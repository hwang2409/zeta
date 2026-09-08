"""Send follow-up prompts to live run agents."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

from ...core.store import (
    ConversationStore,
    PendingPromptCommitTimeoutError,
    PendingPromptsClosedError,
)
from ..registry import ToolRegistry, text_block

AGENT_SEND_COMMIT_TIMEOUT_SECONDS = 1.0


def send_to_run(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
) -> str | None:
    """Queue a follow-up for a live run, returning an error if not possible."""

    if type(child_instance_id) is not str or not child_instance_id.strip():
        return "child_instance_id must be a nonempty string"
    if type(message) is not str or not message.strip():
        return "message must be a nonempty string"
    # The marker is removed once a run's result is durable, so a missing one
    # means the run already finished rather than that it never existed.
    no_live_run = (
        f"no live run {child_instance_id!r}; it already finished or was "
        "never started"
    )
    marker = parent_store.agent_children().get(child_instance_id)
    if marker is None:
        return no_live_run
    # Only runs drain queued follow-ups. Other agent types would leave the
    # prompt in the child's store with no one to consume it.
    agent_type = marker.get("agent_type") or "general"
    if agent_type != "run":
        return (
            f"{child_instance_id!r} is a {agent_type} agent, not a run; "
            "agent_send only works with agent_type=run"
        )
    deadline = time.monotonic() + AGENT_SEND_COMMIT_TIMEOUT_SECONDS
    try:
        child_path = Path(str(marker["child_session_path"]))
        child_store = ConversationStore(
            child_path.parent,
            session_id=child_path.name,
            cwd=parent_store.cwd,
            _lock_deadline=deadline,
        )
        child_store.pending_prompt_queue.append(message, deadline=deadline)
    except PendingPromptCommitTimeoutError:
        return "pending prompt commit timed out before the queue could be changed"
    except PendingPromptsClosedError:
        # consume_run closed the queue while we were checking the marker.
        return no_live_run
    return None


async def _agent_send(
    registry: ToolRegistry,
    arguments: dict[str, Any],
) -> dict[str, object]:
    # send_to_run reloads the run's own conversation.jsonl and appends under
    # flock+fsync; runs with large logs would stall the event loop, so hop
    # to a worker thread while the marker check and durable append happen.
    commit = asyncio.create_task(
        asyncio.to_thread(
            send_to_run,
            registry.session_store,
            arguments.get("child_instance_id"),
            arguments.get("message"),
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
                "when the run finishes its current turn"
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
            "Send a follow-up instruction to a run you started that is still "
            "working. The run picks it up at its next turn boundary, so it "
            "never interrupts a tool call. Use the child_instance_id the agent "
            "tool returned."
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
