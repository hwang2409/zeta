"""Send follow-up prompts to eligible live child agents."""

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
from ...protocol.types import MessageOrigin
from ..registry import ToolRegistry, text_block

AGENT_SEND_COMMIT_TIMEOUT_SECONDS = 1.0


def send_to_run(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
    question_id: object = None,
) -> str | None:
    """Queue a follow-up for an eligible live child."""

    if type(child_instance_id) is not str or not child_instance_id.strip():
        return "child_instance_id must be a nonempty string"
    if type(message) is not str or not message.strip():
        return "message must be a nonempty string"
    if question_id is not None and (
        type(question_id) is not str or not question_id.strip()
    ):
        return "question_id must be a nonempty string when provided"
    no_live_run = (
        f"no live run/child {child_instance_id!r}; it was canceled, finished, "
        "or never started"
    )
    marker = parent_store.agent_children().get(child_instance_id)
    if marker is None:
        return no_live_run
    if marker.get("accepts_follow_ups") is not True:
        agent_type = marker.get("agent_type") or "general"
        return (
            f"agent_send rejected for {agent_type} child {child_instance_id!r}: "
            "it does not accept follow-ups; reviewers are one-shot; start a "
            "fresh reviewer"
        )
    deadline = time.monotonic() + AGENT_SEND_COMMIT_TIMEOUT_SECONDS
    try:
        child_path = Path(str(marker["child_session_path"]))
        with ConversationStore(
            child_path.parent,
            session_id=child_path.name,
            cwd=parent_store.cwd,
            _lock_deadline=deadline,
        ) as child_store:
            child_store.pending_prompt_queue.append(
                message,
                origin=MessageOrigin.AGENT_SEND,
                deadline=deadline,
                question_id=question_id if isinstance(question_id, str) else None,
            )
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
            arguments.get("question_id"),
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
            "the next turn boundary and never interrupts a tool call. Include "
            "question_id when answering ask_parent."
        ),
        parameters={
            "type": "object",
            "properties": {
                "child_instance_id": {"type": "string", "minLength": 1},
                "message": {"type": "string", "minLength": 1},
                "question_id": {"type": "string", "minLength": 1},
            },
            "required": ["child_instance_id", "message"],
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=True,
    )
