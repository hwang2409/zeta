from __future__ import annotations

import logging
from collections.abc import Sequence

from .types import Message, MessageRole, ToolUseContent

logger = logging.getLogger(__name__)


def without_incomplete_tool_batches(messages: Sequence[Message]) -> list[Message]:
    """Remove incomplete tool batches before later conversation messages."""

    result: list[Message] = []
    orphan_results = 0
    index = 0
    while index < len(messages):
        message = messages[index]
        if message.role is MessageRole.TOOL_RESULT:
            orphan_results += 1
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
        index = following
    if orphan_results:
        logger.warning(
            "dropped %d orphan tool result(s) from provider payload", orphan_results
        )
    return result
