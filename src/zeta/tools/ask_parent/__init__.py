"""Let an eligible background child ask its parent without blocking."""

from __future__ import annotations

import uuid
from typing import Any

from ...agent.conversation_channel import ConversationChannel
from ..registry import ToolRegistry, text_block

MAX_QUESTION_LENGTH = 4_000


def _error(message: str) -> dict[str, object]:
    return {
        "content": [text_block(f"ask_parent error: {message}")],
        "isError": True,
        "structuredContent": None,
    }


def register_ask_parent(
    registry: ToolRegistry,
    *,
    channel: ConversationChannel,
    child_instance_id: str,
) -> None:
    """Register non-blocking parent questions for one eligible child."""

    async def ask_parent(
        _registry: ToolRegistry, arguments: dict[str, Any]
    ) -> dict[str, object]:
        question = arguments.get("question")
        options = arguments.get("options")
        if (
            type(question) is not str
            or not question.strip()
            or len(question) > MAX_QUESTION_LENGTH
        ):
            return _error(
                f"question must be a nonempty string of at most {MAX_QUESTION_LENGTH} characters"
            )
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
        question_id = uuid.uuid4().hex
        channel.publish_question(
            child_instance_id=child_instance_id,
            question_id=question_id,
            question=question,
            options=options,
        )
        text = (
            f"question sent (id {question_id}); continue your work; "
            "the answer arrives as a follow-up message"
        )
        return {
            "content": [text_block(text)],
            "isError": False,
            "structuredContent": {"question_id": question_id},
        }

    registry.register_session_tool(
        "ask_parent",
        ask_parent,
        description=(
            "Send your parent one non-blocking question. Continue working; "
            "the answer arrives later as an ordinary follow-up message."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUESTION_LENGTH,
                },
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
            },
            "required": ["question"],
            "additionalProperties": False,
        },
        requires_approval=False,
        parallel_safe=True,
    )
