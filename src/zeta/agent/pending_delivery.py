"""Own pending-prompt sends, resume recovery, delivery, and acknowledgement."""

from __future__ import annotations

import asyncio
import re
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path

from ..core.checkpoints import ConversationEntry, ConversationIntegrityError
from ..core.session_files import SessionNotFoundError
from ..core.store import (
    ConversationStore,
    PendingPromptCommitTimeoutError,
    PendingPromptsClosedError,
)
from ..core.store._pending_prompts import PENDING_PROMPT_ID_METADATA
from ..protocol.types import Message, MessageOrigin, MessageRole, TextContent
from .conversation_channel import ConversationChannel, has_follow_up_loop

AGENT_SEND_COMMIT_TIMEOUT_SECONDS = 1.0
QUESTION_ANSWER_PATTERN = re.compile(r"^Re ([^:\s]+):(?:\s|$)")


@dataclass(frozen=True)
class PendingDelivery:
    """One queued prompt and the persisted user row used for its next turn."""

    prompt_id: str
    message: Message
    persist_message: bool

    @classmethod
    def resume(
        cls,
        store: ConversationStore,
        prompt_id: str,
        text: str,
        origin: MessageOrigin,
    ) -> PendingDelivery:
        messages = store.messages()
        current = next(
            (
                message
                for message in reversed(messages)
                if message.metadata.get(PENDING_PROMPT_ID_METADATA) == prompt_id
            ),
            None,
        )
        if current is None and messages:
            trailing = messages[-1]
            if _is_legacy_delivery_row(trailing, text, origin):
                current = trailing
        if current is not None:
            return cls(prompt_id, current, False)
        return cls(
            prompt_id,
            Message(
                MessageRole.USER,
                [TextContent(text)],
                metadata={
                    "zeta.origin": origin.value,
                    PENDING_PROMPT_ID_METADATA: prompt_id,
                },
            ),
            True,
        )

    async def acknowledge(
        self, store: ConversationStore, context_messages: list[Message]
    ) -> None:
        await store.acknowledge_pending_prompt_messages(
            context_messages, prompt_ids=(self.prompt_id,)
        )


def pending_prompt_acknowledgement(
    store: ConversationStore,
    delivery: PendingDelivery | None,
    context_messages: list[Message],
) -> Callable[[], Awaitable[None]] | None:
    """Select the first-provider-event acknowledgement for this context."""

    if delivery is not None:
        return partial(delivery.acknowledge, store, context_messages)
    if any(
        PENDING_PROMPT_ID_METADATA in message.metadata for message in context_messages
    ):
        return partial(store.acknowledge_pending_prompt_messages, context_messages)
    return None


def _is_legacy_delivery_row(
    message: Message, text: str, origin: MessageOrigin
) -> bool:
    return (
        message.role is MessageRole.USER
        and message.metadata.get("zeta.origin") == origin.value
        and message.content == [TextContent(text)]
    )


def _answered_question_notification_id(
    store: ConversationStore, child_instance_id: str, message: str
) -> str | None:
    match = QUESTION_ANSWER_PATTERN.match(message)
    if match is None:
        return None
    question_id = match.group(1)
    return next(
        (
            entry.id
            for entry in store.agent_notifications()
            if entry.data.get("kind") == "child_question"
            and entry.data.get("child_instance_id") == child_instance_id
            and entry.data.get("question_id") == question_id
        ),
        None,
    )


async def pending_deliveries(
    store: ConversationStore,
) -> list[ConversationEntry]:
    """Read pending deliveries without file I/O on the event loop."""

    return await store._to_thread_durable(store.pending_prompts)


async def close_pending_delivery_queue_if_empty(
    store: ConversationStore,
) -> list[ConversationEntry]:
    """Atomically return pending deliveries or close empty admission off-loop."""

    return await store._to_thread_durable(store.close_pending_queue_if_empty)


async def close_pending_delivery_queue(store: ConversationStore) -> None:
    """Close pending-delivery admission off-loop."""

    await store._to_thread_durable(store.close_pending_queue)


def _commit_follow_up(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
    *,
    channel: ConversationChannel,
) -> str | None:
    if type(child_instance_id) is not str or not child_instance_id.strip():
        return "child_instance_id must be a nonempty string"
    if type(message) is not str or not message.strip():
        return "message must be a nonempty string"
    no_live_run = (
        f"no live run/child {child_instance_id!r}; it was canceled, finished, "
        "or never started"
    )
    marker = parent_store.agent_children().get(child_instance_id)
    if marker is None:
        return no_live_run
    if not has_follow_up_loop(
        accepts_follow_ups=marker.get("accepts_follow_ups") is True,
        background=marker.get("background") is True,
    ):
        agent_type = marker.get("agent_type") or "general"
        reason = (
            "it is a foreground child"
            if marker.get("background") is not True
            else "its profile does not accept follow-ups"
        )
        return (
            f"agent_send rejected for {agent_type} child {child_instance_id!r}: "
            f"{reason}"
        )
    deadline = time.monotonic() + AGENT_SEND_COMMIT_TIMEOUT_SECONDS
    try:
        child_path = Path(str(marker["child_session_path"]))
        with ConversationStore(
            child_path.parent,
            session_id=child_path.name,
            cwd=parent_store.cwd,
            _lock_deadline=deadline,
            _must_exist=True,
        ) as child_store:
            child_store.pending_prompt_queue.append(
                message,
                origin=MessageOrigin.AGENT_SEND,
                deadline=deadline,
            )
    except PendingPromptCommitTimeoutError:
        return "pending prompt commit timed out before the queue could be changed"
    except (PendingPromptsClosedError, SessionNotFoundError):
        return no_live_run
    except ConversationIntegrityError:
        if (child_path / "conversation.jsonl").exists():
            raise
        return no_live_run
    channel.wake(child_instance_id)
    return None


async def send_to_run(
    parent_store: ConversationStore,
    child_instance_id: object,
    message: object,
    *,
    channel: ConversationChannel,
) -> str | None:
    """Queue one follow-up and route an exact question answer off-loop."""

    commit = asyncio.create_task(
        asyncio.to_thread(
            _commit_follow_up,
            parent_store,
            child_instance_id,
            message,
            channel=channel,
        )
    )
    while True:
        try:
            error = await asyncio.shield(commit)
        except asyncio.CancelledError:
            if commit.cancelled():
                raise
            continue
        break
    if error is not None:
        return error
    assert isinstance(child_instance_id, str)
    assert isinstance(message, str)
    notification_id = _answered_question_notification_id(
        parent_store, child_instance_id, message
    )
    if notification_id is not None:
        await parent_store.acknowledge_agent_notifications_async([notification_id])
    return None
