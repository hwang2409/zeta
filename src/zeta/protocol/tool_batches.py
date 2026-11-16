from __future__ import annotations

import logging
from collections.abc import Callable, Sequence

from .types import (
    ASSISTANT_RESPONSE_STATE,
    ASSISTANT_RESPONSE_SYNTHETIC,
    FAILED_TURN_MARKER,
    Message,
    MessageRole,
    TextContent,
    ToolResult,
    ToolUseContent,
    is_passive_harness_message,
)

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


def repair_tool_batches[T](
    items: Sequence[T],
    message_of: Callable[[T], Message],
    stand_in: Callable[[str], T],
) -> list[T]:
    """Pair every tool call of an interrupted batch with exactly one result.

    A batch is interrupted when a later item follows before all its results
    exist (the process died mid-batch). Its real results stay, in call order,
    and ``stand_in`` supplies each missing one, so the model sees what already
    ran. A trailing incomplete batch is still in flight and stays as is. A
    failed provider reply's calls never ran, so that batch is left out.

    A call that asks for approval only at execute time stores a synthetic
    assistant anchor holding that call between the batch and its result; the
    anchor repeats a call of the open batch, so it is skipped, not sent.
    A harness notice written between a batch's results moves after them.
    """

    result: list[T] = []
    index = 0
    while index < len(items):
        item = items[index]
        message = message_of(item)
        index += 1
        call_ids = _call_ids(message) if message.role is MessageRole.ASSISTANT else []
        if not call_ids:
            result.append(item)
            continue
        batch_results: list[T] = []
        notices: list[T] = []
        while index < len(items):
            candidate = message_of(items[index])
            if candidate.role is MessageRole.TOOL_RESULT:
                batch_results.append(items[index])
            elif is_passive_harness_message(candidate):
                # written mid-batch (shell cwd recovery); it follows the results
                notices.append(items[index])
            elif not _is_anchor(candidate, call_ids):
                break
            index += 1
        by_id: dict[str, T] = {}
        for candidate in batch_results:
            tool_result = message_of(candidate).tool_result
            if tool_result is not None:
                by_id.setdefault(tool_result.tool_call_id, candidate)
        unique = len(set(call_ids)) == len(call_ids)
        complete = (
            unique and len(batch_results) == len(call_ids) and set(by_id) == set(call_ids)
        )
        if complete or index == len(items):
            result.append(item)
            result.extend(batch_results)
        elif unique and not message.metadata.get(FAILED_TURN_MARKER):
            result.append(item)
            result.extend(by_id.get(call_id) or stand_in(call_id) for call_id in call_ids)
        result.extend(notices)
    return result


def _call_ids(message: Message) -> list[str]:
    return [
        block.tool_call.id
        for block in message.content
        if isinstance(block, ToolUseContent)
    ]


def _is_anchor(message: Message, call_ids: list[str]) -> bool:
    return (
        message.role is MessageRole.ASSISTANT
        and message.metadata.get(ASSISTANT_RESPONSE_STATE) == ASSISTANT_RESPONSE_SYNTHETIC
        and bool(message.content)
        and all(
            isinstance(block, ToolUseContent) and block.tool_call.id in call_ids
            for block in message.content
        )
    )


def complete_tool_batches(messages: Sequence[Message]) -> list[Message]:
    """Shape tool batches for a provider payload.

    Context assembly repairs interrupted batches first, with approval facts.
    Here none exist, so a missing result may or may not have taken effect.
    """

    kept: list[Message] = []
    for index, message in enumerate(messages):
        previous = messages[index - 1] if index else None
        if (
            message.role is MessageRole.TOOL_RESULT
            and previous is not None
            and not _call_ids(previous)
            and _is_eviction_receipt(previous)
        ):
            continue
        kept.append(message)
    if len(kept) != len(messages):
        logger.warning(
            "dropped %d orphan tool result(s) from provider payload",
            len(messages) - len(kept),
        )
    return repair_tool_batches(
        kept,
        lambda message: message,
        lambda call_id: interrupted_tool_result(call_id, not_run=False),
    )


def _is_eviction_receipt(message: Message) -> bool:
    fields = message.metadata.get("eviction_receipt_fields")
    return (
        isinstance(fields, dict)
        and type(fields.get("collapsed_into_seq")) is int
        and type(fields.get("older_read")) is bool
    ) or (
        type(message.metadata.get("collapsed_into_seq")) is int
        and type(message.metadata.get("older_read")) is bool
    )
