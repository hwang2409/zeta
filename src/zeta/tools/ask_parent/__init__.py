"""Let an eligible background child ask or notify its parent without blocking."""

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
        kind = arguments.get("kind", "question")
        if kind not in ("question", "note"):
            return _error('kind must be "question" or "note"')
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
        if kind == "note":
            if options is not None:
                return _error("options apply only to questions; a note needs no reply")
            await channel.publish_note(
                child_instance_id=child_instance_id,
                child_session_path=str(_registry.session_store.session_dir),
                note=question,
            )
            return {
                "content": [
                    text_block("note sent; no reply will come; continue your work")
                ],
                "isError": False,
                "structuredContent": None,
            }
        question_id = uuid.uuid4().hex
        await channel.publish_question(
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
        binding="loop",
        description=(
            "Send your parent one non-blocking question. Continue working; "
            "the answer arrives later as an ordinary follow-up message. "
            'Use kind "note" for a milestone or status that needs no reply: '
            "the parent gets it as a background update, and it is never an "
            "open question."
        ),
        parameters={
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": MAX_QUESTION_LENGTH,
                },
                "kind": {"type": "string", "enum": ["question", "note"]},
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
