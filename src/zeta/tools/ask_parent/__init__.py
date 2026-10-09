"""Let an eligible child ask its direct parent a bounded question."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ...core.store import ConversationStore
from ..registry import ToolRegistry, text_block

MAX_WAIT_SECONDS = 300.0
_POLL_SECONDS = 0.05


@dataclass(slots=True)
class AskParentContext:
    parent_store: ConversationStore
    child_store: ConversationStore
    child_instance_id: str
    notify_parent: Callable[[], None]
    pending: asyncio.Lock = field(default_factory=asyncio.Lock)

    def persist_question(
        self, question_id: str, question: str, options: list[str] | None
    ) -> None:
        self.parent_store.append_child_question(
            child_instance_id=self.child_instance_id,
            question_id=question_id,
            question=question,
            options=options,
        )

    def take_answer(self, question_id: str) -> str | None:
        answer = next(
            (
                entry
                for entry in self.child_store.pending_prompts()
                if entry.data.get("question_id") in {None, question_id}
            ),
            None,
        )
        if answer is None:
            return None
        self.child_store.acknowledge_pending_prompt(answer.id)
        return str(answer.data["text"])


async def _ask_parent(
    context: AskParentContext,
    _registry: ToolRegistry,
    arguments: dict[str, Any],
) -> dict[str, object]:
    question = arguments.get("question")
    options = arguments.get("options")
    wait_seconds = arguments.get("wait_seconds", MAX_WAIT_SECONDS)
    if type(question) is not str or not question.strip():
        return _error("question must be a nonempty string")
    if options is not None and (
        type(options) is not list
        or not options
        or len(options) > 10
        or any(
            type(option) is not str or not option.strip() or len(option) > 1_000
            for option in options
        )
    ):
        return _error("options must contain 1 to 10 nonempty strings")
    if type(wait_seconds) not in {int, float} or not 0 <= wait_seconds <= MAX_WAIT_SECONDS:
        return _error(f"wait_seconds must be between 0 and {MAX_WAIT_SECONDS:g}")
    if context.pending.locked():
        return _error("this child already has a pending parent question")

    async with context.pending:
        question_id = uuid.uuid4().hex
        await asyncio.to_thread(
            context.persist_question,
            question_id,
            question,
            options,
        )
        context.notify_parent()
        deadline = time.monotonic() + float(wait_seconds)
        while True:
            answer = await asyncio.to_thread(context.take_answer, question_id)
            if answer is not None:
                return {
                    "content": [text_block(answer)],
                    "isError": False,
                    "structuredContent": {
                        "question_id": question_id,
                        "answered": True,
                        "answer": answer,
                    },
                }
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                text = "no answer yet — proceed with a stated assumption"
                return {
                    "content": [text_block(text)],
                    "isError": False,
                    "structuredContent": {
                        "question_id": question_id,
                        "answered": False,
                    },
                }
            await asyncio.sleep(min(_POLL_SECONDS, remaining))


def _error(message: str) -> dict[str, object]:
    return {
        "content": [text_block(f"ask_parent error: {message}")],
        "isError": True,
        "structuredContent": None,
    }


def register_ask_parent(
    registry: ToolRegistry,
    *,
    parent_store: ConversationStore,
    child_store: ConversationStore,
    child_instance_id: str,
    notify_parent: Callable[[], None],
) -> None:
    """Register ask_parent only in one eligible child registry."""
    context = AskParentContext(
        parent_store=parent_store,
        child_store=child_store,
        child_instance_id=child_instance_id,
        notify_parent=notify_parent,
    )

    async def ask_parent(
        bound_registry: ToolRegistry, arguments: dict[str, Any]
    ) -> dict[str, object]:
        return await _ask_parent(context, bound_registry, arguments)

    registry.register_session_tool(
        "ask_parent",
        ask_parent,
        description=(
            "Ask your direct parent one question. Wait briefly for an answer; "
            "if none arrives, continue with a stated assumption."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {"type": "string", "minLength": 1},
                "options": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 1_000,
                    },
                    "minItems": 1,
                    "maxItems": 10,
                },
                "wait_seconds": {
                    "type": "number",
                    "minimum": 0,
                    "maximum": MAX_WAIT_SECONDS,
                },
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=True,
    )
