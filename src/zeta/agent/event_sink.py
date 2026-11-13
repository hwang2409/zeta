"""Await event sinks without changing synchronous adapters."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable

from ..protocol.types import StreamEvent

EventSink = Callable[[StreamEvent], Awaitable[None] | None]


async def await_event_sink(sink: EventSink, event: StreamEvent) -> None:
    result = sink(event)
    if inspect.isawaitable(result):
        await result
