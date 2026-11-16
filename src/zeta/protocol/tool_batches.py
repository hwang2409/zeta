from __future__ import annotations

import logging
from collections.abc import Sequence

from .types import Message, MessageRole, TextContent, ToolResult, ToolUseContent

logger = logging.getLogger(__name__)

NOT_RUN_TEXT = "not run: the turn was interrupted"
MAYBE_RUN_TEXT = (
    "interrupted: may or may not have taken effect; check before retrying"
)


def interrupted_tool_result(call_id: str, *, not_run: bool) -> Message:
    """Stand in for the result a dead process never wrote."""

    text = NOT_RUN_TEXT if not_run else MAYBE_RUN_TEXT
    return Message(
        MessageRole.TOOL_RESULT,
        [TextContent(text)],
        tool_result=ToolResult(call_id, text, is_error=True),
    )


def complete_tool_batches(messages: Sequence[Message]) -> list[Message]:
    """Pair every tool call of an interrupted batch with exactly one result.

    A batch is interrupted when a later message follows before all its results
    exist (the process died mid-batch). Its real results stay, in call order,
    and each missing one gets an interrupted result, so the model sees what
    already ran. A trailing incomplete batch is still in flight and stays as is.
    Context assembly fills these first with approval facts; here none exist, so
    a missing result may or may not have taken effect.
    """

    result: list[Message] = []
    orphan_results = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role is MessageRole.TOOL_RESULT:
            previous = messages[index - 1] if index else None
            fields = (
                previous.metadata.get("eviction_receipt_fields")
                if previous is not None
                else None
            )
            is_eviction_orphan = (
                isinstance(fields, dict)
                and type(fields.get("collapsed_into_seq")) is int
                and type(fields.get("older_read")) is bool
            ) or (
                previous is not None
                and type(previous.metadata.get("collapsed_into_seq")) is int
                and type(previous.metadata.get("older_read")) is bool
            )
            if is_eviction_orphan:
                orphan_results += 1
                index += 1
                continue
            result.append(message)
            index += 1
            continue
        call_ids = [
            block.tool_call.id
            for block in message.content
            if isinstance(block, ToolUseContent)
        ]
        if message.role is not MessageRole.ASSISTANT or not call_ids:
            result.append(message)
            index += 1
            continue

        following = index + 1
        batch_results: list[Message] = []
        result_ids: list[str] = []
        while following < len(messages):
            candidate = messages[following]
            if candidate.role is not MessageRole.TOOL_RESULT:
                break
            batch_results.append(candidate)
            if candidate.tool_result is not None:
                result_ids.append(candidate.tool_result.tool_call_id)
            following += 1

        complete = (
            len(set(call_ids)) == len(call_ids)
            and len(result_ids) == len(call_ids)
            and set(result_ids) == set(call_ids)
        )
        if complete or following == len(messages):
            result.append(message)
            result.extend(batch_results)
        elif len(set(call_ids)) == len(call_ids):
            by_id: dict[str, Message] = {}
            for candidate in batch_results:
                if candidate.tool_result is not None:
                    by_id.setdefault(candidate.tool_result.tool_call_id, candidate)
            result.append(message)
            result.extend(
                by_id.get(call_id) or interrupted_tool_result(call_id, not_run=False)
                for call_id in call_ids
            )
        index = following
    if orphan_results:
        logger.warning(
            "dropped %d orphan tool result(s) from provider payload", orphan_results
        )
    return result
