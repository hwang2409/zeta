"""Bounded, ordered publication of background serve events."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from ..protocol.types import StreamEvent
from .connections import OUTPUT_MAX_BYTES, OUTPUT_MAX_FRAMES
from .protocol import ErrorDescriptor, FrameCodec

logger = logging.getLogger(__name__)

PublishEvent = Callable[[str, StreamEvent], Awaitable[None]]
Overflow = Callable[[], None]
_OVERFLOW_FRAME_BYTES = len(
    FrameCodec().error_response(
        None,
        ErrorDescriptor.CONTROLLER_OUTPUT_OVERFLOW,
        ErrorDescriptor.CONTROLLER_OUTPUT_OVERFLOW.meaning,
    )
)
_MAX_PENDING_BYTES = OUTPUT_MAX_BYTES - _OVERFLOW_FRAME_BYTES
_MAX_PENDING_FRAMES = OUTPUT_MAX_FRAMES - 1


@dataclass(frozen=True, slots=True)
class _QueuedEvent:
    session_id: str
    event: StreamEvent
    size: int


class BackgroundEventPublisher:
    """Publish one bounded FIFO without making event producers wait."""

    def __init__(self, publish_event: PublishEvent, overflow: Overflow) -> None:
        self._publish_event = publish_event
        self._overflow = overflow
        self._queue: asyncio.Queue[_QueuedEvent] = asyncio.Queue()
        self._task: asyncio.Task[None] | None = None
        self._frames = 0
        self._bytes = 0
        self._record_only = False
        self._closed = False

    def publish(self, session_id: str, event: StreamEvent) -> None:
        if self._closed:
            return
        size = len(
            json.dumps(
                {"session_id": session_id, "event": event.to_dict()},
                ensure_ascii=False,
                separators=(",", ":"),
            ).encode()
        )
        if not self._record_only and (
            self._frames + 1 > _MAX_PENDING_FRAMES
            or self._bytes + size > _MAX_PENDING_BYTES
        ):
            self._record_only = True
            self._overflow()
        retained_size = 0 if self._record_only else size
        if retained_size:
            self._frames += 1
            self._bytes += retained_size
        self._queue.put_nowait(_QueuedEvent(session_id, event, retained_size))
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._drain())

    async def _drain(self) -> None:
        while True:
            item = await self._queue.get()
            try:
                await self._publish_event(item.session_id, item.event)
            except asyncio.CancelledError:
                raise
            except Exception as error:  # noqa: BLE001 - isolate each event failure
                logger.error(
                    "background event publication failed",
                    extra={
                        "event_type": item.event.type.value,
                        "error_type": type(error).__name__,
                    },
                )
            finally:
                if item.size:
                    self._frames -= 1
                    self._bytes -= item.size
                self._queue.task_done()
                if self._queue.empty() and self._frames == 0:
                    self._record_only = False

    async def flush(self) -> None:
        await self._queue.join()

    async def close(self, timeout: float) -> None:
        self._closed = True
        task = self._task
        if task is None:
            return
        try:
            await asyncio.wait_for(self.flush(), timeout)
        except TimeoutError:
            pass
        finally:
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
            self._task = None
