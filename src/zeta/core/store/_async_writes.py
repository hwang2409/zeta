"""Cancellation-safe coordination for off-loop durable store writes."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
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
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run a write against isolated state; the resident store stays loop-owned."""
        try:
            writer = type(self)(
                self.root_dir,
                session_id=self.session_id,
                _must_exist=True,
                _collect_persisted_appends=self._collect_persisted_appends,
            )
            try:
                result = getattr(writer, method_name)(*args, **kwargs)
                return result, writer.take_persisted_appends()
            finally:
                writer.close()
        finally:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()

    async def _to_thread_durable(
        self: ConversationStore,
        function: Any,
        /,
        *args: Any,
        _on_persisted: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Serialize a blocking write and defer cancellation until it finishes."""
        async with self._async_write_lock:
            method_name = function.__name__
            if not hasattr(type(self), method_name):
                # Preserve instance-level test/instrumentation overrides without
                # ever letting them mutate resident state from a worker thread.
                return function(*args, **kwargs)
            with self._durable_write_condition:
                if self._closing or self._closed:
                    raise ValueError("session store is closed")
                self._durable_writes_in_flight += 1
            loop = asyncio.get_running_loop()
            try:
                # Keep the executor future separate from a Task: asyncio.run()
                # cancels pending Tasks during shutdown, but the submitted thread
                # must remain awaitable until its durable write has completed.
                write = loop.run_in_executor(
                    None, self._run_durable_write, method_name, args, kwargs
                )
            except BaseException:
                with self._durable_write_condition:
                    self._durable_writes_in_flight -= 1
                    self._durable_write_condition.notify_all()
                raise
            cancelled = False
            while True:
                try:
                    result, receipts = await asyncio.shield(write)
                    break
                except asyncio.CancelledError:
                    # Repeated cancellation must not release the caller while a
                    # started append can still be before its write or fsync.
                    cancelled = True
            # The worker changes only the log. Publish its committed tail
            # on-loop unless close has already started draining this store.
            with self._durable_write_condition:
                if not self._closing and not self._closed:
                    self._accept_persisted_appends(receipts)
                    self.refresh()
            if _on_persisted is not None:
                _on_persisted()
            if cancelled:
                raise asyncio.CancelledError
            return result
