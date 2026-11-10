"""Bounded publication of locally running foreground turn events."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator

from ..protocol.types import StreamEvent


class ForegroundEventSubscription:
    """A bounded, async view of events admitted by a publisher."""

    def __init__(self, max_pending: int) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._events: asyncio.Queue[StreamEvent] = asyncio.Queue(max_pending)
        self._closed = False
        self._waiter: asyncio.Future[None] | None = None

    def _admit(self, event: StreamEvent) -> bool:
        if self._closed:
            return False
        try:
            self._events.put_nowait(event)
        except asyncio.QueueFull:
            return False
        self._wake()
        return True

    def _wake(self) -> None:
        waiter = self._waiter
        if waiter is not None and not waiter.done():
            waiter.set_result(None)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._wake()

    def __aiter__(self) -> AsyncIterator[StreamEvent]:
        return self

    async def __anext__(self) -> StreamEvent:
        while True:
            try:
                return self._events.get_nowait()
            except asyncio.QueueEmpty:
                if self._closed:
                    raise StopAsyncIteration
                loop = asyncio.get_running_loop()
                waiter = loop.create_future()
                self._waiter = waiter
                try:
                    await waiter
                finally:
                    if self._waiter is waiter:
                        self._waiter = None


class ForegroundEventPublisher:
    """Fan out foreground events without making their producer wait."""

    def __init__(self, *, max_pending: int = 256) -> None:
        if max_pending < 1:
            raise ValueError("max_pending must be positive")
        self._max_pending = max_pending
        self._subscriptions: set[ForegroundEventSubscription] = set()
        self._closed = False

    def subscribe(self) -> ForegroundEventSubscription:
        if self._closed:
            raise RuntimeError("foreground event publisher is closed")
        subscription = ForegroundEventSubscription(self._max_pending)
        self._subscriptions.add(subscription)
        return subscription

    def unsubscribe(self, subscription: ForegroundEventSubscription) -> None:
        if subscription in self._subscriptions:
            self._subscriptions.remove(subscription)
            subscription.close()

    def close(self) -> None:
        self._closed = True
        subscriptions = tuple(self._subscriptions)
        self._subscriptions.clear()
        for subscription in subscriptions:
            subscription.close()

    async def events(self, source: AsyncIterable[StreamEvent]) -> AsyncIterator[StreamEvent]:
        async for event in source:
            for subscription in tuple(self._subscriptions):
                if not subscription._admit(event):
                    self.unsubscribe(subscription)
            yield event
