"""Completion-backend adapter for one memory reconciliation request."""

from __future__ import annotations

import asyncio
from collections.abc import Sequence

from ..protocol.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEventType,
    TextContent,
    ToolSchema,
)
from ..providers.retry_policy import (
    ProviderRetryBudget,
    current_retry_budget,
    use_retry_budget,
)
from .reconciler import ReconciliationResponse


async def complete_reconciliation(
    backend: CompletionBackend, prompt: str
) -> ReconciliationResponse:
    """Return one accepted response using the shared provider retry budget.

    No streamed reconciliation content is externally committed. A retry is
    therefore safe until a complete final message has been accepted, including
    after message-start or text-delta events.
    """
    request = [Message(MessageRole.USER, [TextContent(prompt)])]
    budget = current_retry_budget() or ProviderRetryBudget()
    while budget.start_attempt("memory"):
        final: Message | None = None
        usage: dict[str, int] = {}
        failure: BaseException | object | None = None
        event_data: dict[str, object] = {}
        try:
            with use_retry_budget(budget):
                async for event in backend.complete(request, _NO_TOOLS):
                    if event.type is StreamEventType.ERROR and event.error is not None:
                        failure = event.error
                        event_data = dict(event.data)
                        break
                    if (
                        event.type is StreamEventType.MESSAGE_END
                        and event.message is not None
                    ):
                        final = event.message
                        raw_usage = event.data.get("usage")
                        if isinstance(raw_usage, dict):
                            usage = {
                                key: value
                                for key, value in raw_usage.items()
                                if isinstance(key, str)
                                and type(value) is int
                                and value >= 0
                            }
        except BaseException as exc:  # noqa: BLE001 - classified by shared policy
            failure = exc
        if final is not None:
            return ReconciliationResponse(
                "".join(
                    block.text
                    for block in final.content
                    if isinstance(block, TextContent)
                ),
                usage,
            )
        failure = failure or RuntimeError("memory reconciler returned no final message")
        plan = budget.plan(failure, owner="memory", event_data=event_data)
        if plan is None:
            if isinstance(failure, BaseException):
                raise failure
            raise RuntimeError("memory reconciliation provider failed")
        budget.record_retry(plan)
        await asyncio.sleep(plan.delay)
    raise RuntimeError("memory reconciliation provider retry budget exhausted")


_NO_TOOLS: Sequence[ToolSchema] = ()
