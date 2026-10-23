"""Cancellation-safe coordination for off-loop durable store writes."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..todo import TodoItem
    from ._log import _DetachedLogState
    from ._store import ConversationStore


@dataclass(frozen=True, slots=True)
class _DetachedConversationState:
    log: _DetachedLogState
    bash_cwd: str
    todo_items: list[TodoItem]
    todo_dismissed: bool
    agent_counter: int
    agent_children: dict[str, dict[str, Any]]
    agent_parent: dict[str, Any] | None
    agent_canceled: dict[str, Any] | None
    agent_lifecycle: dict[str, Any] | None


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
                _collect_persisted_appends=True,
            )
            try:
                result = getattr(writer, method_name)(*args, **kwargs)
                return result, writer._detach_log_state()
            finally:
                writer.close()
        finally:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()

    def _load_detached_log_state(self: ConversationStore) -> Any:
        """Load and validate current durable state without touching resident state."""
        reader = type(self)(
            self.root_dir,
            session_id=self.session_id,
            _must_exist=True,
        )
        try:
            return reader._detach_log_state()
        finally:
            reader.close()

    def _load_detached_store_state(
        self: ConversationStore,
    ) -> _DetachedConversationState:
        """Load all durable store state for publication by the owner loop."""
        reader = type(self)(
            self.root_dir,
            session_id=self.session_id,
            _must_exist=True,
        )
        try:
            return _DetachedConversationState(
                log=reader._detach_log_state(),
                bash_cwd=reader.bash_cwd,
                todo_items=reader._todo_items,
                todo_dismissed=reader._todo_dismissed,
                agent_counter=reader._agent_counter,
                agent_children=reader._agent_children,
                agent_parent=reader._agent_parent,
                agent_canceled=reader._agent_canceled,
                agent_lifecycle=reader._agent_lifecycle,
            )
        finally:
            reader.close()

    def _install_store_state(
        self: ConversationStore, state: _DetachedConversationState
    ) -> _DetachedLogState:
        """Install one detached full-store snapshot on the owner loop."""
        for marker_key, loaded_marker in state.agent_children.items():
            live_marker = self._agent_children.get(marker_key)
            if (
                live_marker is not None
                and live_marker.get("child_session_path")
                == loaded_marker.get("child_session_path")
            ):
                loaded_marker["turns_used"] = max(
                    loaded_marker.get("turns_used", 0),
                    live_marker.get("turns_used", 0),
                )
        previous = self._install_log_state(state.log)
        self.bash_cwd = state.bash_cwd
        self._todo_items = state.todo_items
        self._todo_dismissed = state.todo_dismissed
        self._agent_counter = state.agent_counter
        self._agent_children = state.agent_children
        self._agent_parent = state.agent_parent
        self._agent_canceled = state.agent_canceled
        self._agent_lifecycle = state.agent_lifecycle
        return previous

    @staticmethod
    def _release_detached_log_state(state: _DetachedLogState) -> None:
        """Drop superseded transcript objects on a worker thread."""
        del state

    async def _refresh_off_loop(self: ConversationStore) -> None:
        """Load durable state on a worker and install it on the owner loop."""
        loop = asyncio.get_running_loop()
        cancelled = False
        while True:
            generation = self._transcript_generation
            refresh = loop.run_in_executor(None, self._load_detached_store_state)
            while True:
                try:
                    state = await asyncio.shield(refresh)
                    break
                except asyncio.CancelledError:
                    cancelled = True
            if generation == self._transcript_generation:
                break
        previous = None
        with self._durable_write_condition:
            if not self._closing and not self._closed:
                previous = self._install_store_state(state)
        if previous is not None:
            release = loop.run_in_executor(
                None, self._release_detached_log_state, previous
            )
            previous = None
            while True:
                try:
                    await asyncio.shield(release)
                    break
                except asyncio.CancelledError:
                    cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _to_thread_store(self: ConversationStore, function: Any, /) -> Any:
        """Run one resident-store lease operation off-loop despite cancellation."""

        write = asyncio.get_running_loop().run_in_executor(None, function)
        cancelled = False
        while True:
            try:
                result = await asyncio.shield(write)
                break
            except asyncio.CancelledError:
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError
        return result

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
            generation = self._transcript_generation
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
                    result, state = await asyncio.shield(write)
                    break
                except asyncio.CancelledError:
                    # Repeated cancellation must not release the caller while a
                    # started append can still be before its write or fsync.
                    cancelled = True
            # A synchronous owner write can run while this task awaits the worker.
            # Refresh again off-loop until the detached state matches that generation.
            while generation != self._transcript_generation:
                generation = self._transcript_generation
                refresh = loop.run_in_executor(None, self._load_detached_log_state)
                while True:
                    try:
                        state = await asyncio.shield(refresh)
                        break
                    except asyncio.CancelledError:
                        cancelled = True
            # Publish only validated worker state, and only while the store is open.
            previous = None
            with self._durable_write_condition:
                if not self._closing and not self._closed:
                    previous = self._install_log_state(state)
            if previous is not None:
                release = loop.run_in_executor(
                    None, self._release_detached_log_state, previous
                )
                previous = None
                while True:
                    try:
                        await asyncio.shield(release)
                        break
                    except asyncio.CancelledError:
                        cancelled = True
            if _on_persisted is not None:
                _on_persisted()
            if cancelled:
                raise asyncio.CancelledError
            return result
