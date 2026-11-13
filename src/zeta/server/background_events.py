"""Bounded, ordered publication of background serve events."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..protocol.types import StreamEvent

PublishEvent = Callable[[str, StreamEvent], Awaitable[None]]
_MAX_PENDING_BYTES = 4 * 1024 * 1024
_MAX_PENDING_EVENTS = 255


class BackgroundEventTooLarge(RuntimeError):
    """One event cannot fit in the semantic publication budget."""


@dataclass(slots=True, eq=False)
class _QueuedEvent:
    session_id: str
    event: StreamEvent
    size: int
    result: asyncio.Future[None]
    admitted: asyncio.Future[None]


class BackgroundEventPublisher:
    """Record one bounded FIFO and report each event's terminal result."""

    def __init__(
        self,
        publish_event: PublishEvent,
        *,
        max_events: int = _MAX_PENDING_EVENTS,
        max_bytes: int = _MAX_PENDING_BYTES,
    ) -> None:
        self._publish_event = publish_event
        self._max_events = max_events
        self._max_bytes = max_bytes
        self._queue: asyncio.Queue[_QueuedEvent] = asyncio.Queue()
        self._waiting: deque[_QueuedEvent] = deque()
        self._task: asyncio.Task[None] | None = None
        self._events = 0
        self._bytes = 0
        self._failures: list[BaseException] = []
        self._changed = asyncio.Event()
        self._changed.set()
        self._closed = False
        self._sealed = False
        self._active: set[_QueuedEvent] = set()

    @staticmethod
    def _close_error() -> RuntimeError:
        return RuntimeError("background event publisher is closed")

    @staticmethod
    def _size(session_id: str, event: StreamEvent) -> int:
        return len(
            json.dumps(
                {"session_id": session_id, "event": event.to_dict()},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )

    async def publish(self, session_id: str, event: StreamEvent) -> None:
        if self._closed:
            raise RuntimeError("background event publisher is closed")
        size = self._size(session_id, event)
        if size > self._max_bytes:
            raise BackgroundEventTooLarge(
                f"background event is {size} bytes; limit is {self._max_bytes}"
            )
        loop = asyncio.get_running_loop()
        item = _QueuedEvent(
            session_id,
            event,
            size,
            loop.create_future(),
            loop.create_future(),
        )
        self._waiting.append(item)
        self._admit_waiters()
        try:
            await asyncio.shield(item.admitted)
        except asyncio.CancelledError:
            if not item.admitted.done():
                self._waiting.remove(item)
                item.admitted.cancel()
                self._admit_waiters()
                self._mark_changed()
                raise

        cancelled = False
        while not item.result.done():
            try:
                await asyncio.shield(item.result)
            except asyncio.CancelledError:
                cancelled = True
                current = asyncio.current_task()
                if current is not None:
                    current.uncancel()
        error = item.result.exception()
        if cancelled:
            raise asyncio.CancelledError
        if error is not None:
            raise error

    def _admit_waiters(self) -> None:
        while self._waiting:
            if self._sealed:
                return
            item = self._waiting[0]
            if self._events >= self._max_events or self._bytes + item.size > self._max_bytes:
                return
            self._waiting.popleft()
            self._events += 1
            self._bytes += item.size
            self._active.add(item)
            self._queue.put_nowait(item)
            if not item.admitted.done():
                item.admitted.set_result(None)
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(self._drain())
        self._mark_changed()

    def _mark_changed(self) -> None:
        self._changed.set()

    async def _drain(self) -> None:
        while True:
            item = await self._queue.get()
            error: BaseException | None = None
            try:
                await self._publish_event(item.session_id, item.event)
            except asyncio.CancelledError:
                raise
            except BaseException as exc:  # noqa: BLE001 - publish failures include cancellation
                error = exc
                self._failures.append(exc)
            finally:
                self._events -= 1
                self._bytes -= item.size
                self._active.discard(item)
                self._queue.task_done()
                if error is None and self._sealed:
                    error = self._close_error()
                if error is None:
                    if not item.result.done():
                        item.result.set_result(None)
                elif not item.result.done():
                    item.result.set_exception(error)
                if not self._sealed:
                    self._admit_waiters()
                self._mark_changed()

    async def flush(self) -> None:
        while self._waiting or self._events:
            self._changed.clear()
            if self._waiting or self._events:
                await self._changed.wait()
        if self._failures:
            raise self._failures[0]

    def _settle_close(self) -> None:
        error = self._close_error()
        while self._waiting:
            item = self._waiting.popleft()
            if not item.admitted.done():
                item.admitted.set_exception(error)
            if not item.result.done():
                item.result.cancel()
        for item in self._active:
            if not item.result.done():
                item.result.set_exception(error)
        self._active.clear()
        while not self._queue.empty():
            self._queue.get_nowait()
            self._queue.task_done()
        self._events = 0
        self._bytes = 0
        self._mark_changed()

    async def close(self, timeout: float) -> None:
        self._closed = True
        error: BaseException | None = None
        try:
            await asyncio.wait_for(self.flush(), timeout)
        except BaseException as exc:  # noqa: BLE001 - close preserves the first failure
            error = exc
        self._sealed = True
        task = self._task
        if task is not None:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        self._settle_close()
        self._task = None
        if error is not None:
            raise error
