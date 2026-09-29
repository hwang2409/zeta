"""Turn-completion lifecycle helpers shared by the agent loop.

Extracted from ``agent.py`` to keep that module within its size budget; these
helpers own how a streaming completion is closed and when a context-limit
retry is safe.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from ...protocol.types import ContentBlock, ErrorInfo, Message, StreamEvent


async def close_completion(
    completion: AsyncIterator[StreamEvent] | None,
) -> BaseException | None:
    if completion is None:
        return None
    close = getattr(completion, "aclose", None)
    if close is None:
        return None
    try:
        await close()
    except BaseException as exc:  # noqa: BLE001 - preserve close errors
        return exc
    return None


def task_is_cancelling() -> bool:
    task = asyncio.current_task()
    return task is not None and task.cancelling() > 0


def can_retry_context(
    error: ErrorInfo, retrying: bool, partial: list[ContentBlock], message: Message | None
) -> bool:
    return error.code == "context_length_exceeded" and not retrying and not partial and message is None
