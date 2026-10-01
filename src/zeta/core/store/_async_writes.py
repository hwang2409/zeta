"""Cancellation-safe coordination for off-loop durable store writes."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._store import ConversationStore


class AsyncDurableWritesMixin:
    """Track, drain, and cancellation-shield writes running in worker threads."""

    def _initialize_async_writes(self: ConversationStore) -> None:
        self._async_write_lock = asyncio.Lock()
        self._durable_write_condition = threading.Condition()
        self._durable_writes_in_flight = 0

    def _drain_durable_writes_for_close(self: ConversationStore) -> bool:
        with self._durable_write_condition:
            self._closing = True
            while self._durable_writes_in_flight:
                self._durable_write_condition.wait()
            if self._closed:
                return False
            self._closed = True
            return True

    def _run_durable_write(
        self: ConversationStore,
        function: Any,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        try:
            return function(*args, **kwargs)
        finally:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()

    async def _to_thread_durable(
        self: ConversationStore, function: Any, /, *args: Any, **kwargs: Any
    ) -> Any:
        """Serialize a blocking write and defer cancellation until it finishes."""
        async with self._async_write_lock:
            with self._durable_write_condition:
                if self._closing or self._closed:
                    raise ValueError("session store is closed")
                self._durable_writes_in_flight += 1
            write = asyncio.create_task(
                asyncio.to_thread(self._run_durable_write, function, args, kwargs)
            )
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(write)
                    break
                except asyncio.CancelledError:
                    # Repeated cancellation must not release the caller while a
                    # started append can still be before its write or fsync.
                    cancelled = True
            if cancelled:
                raise asyncio.CancelledError
            return result
