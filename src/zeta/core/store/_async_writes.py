"""Cancellation-safe coordination for off-loop durable store writes."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._log import _DetachedLogState
    from ._store import ConversationStore


@dataclass(frozen=True, slots=True)
class _DetachedConversationState:
    """Durable fields that child recovery can change."""

    log: _DetachedLogState
    agent_counter: int
    agent_children: dict[str, dict[str, Any]]
    agent_parent: dict[str, Any] | None
    agent_canceled: dict[str, Any] | None
    agent_lifecycle: dict[str, Any] | None


@dataclass(frozen=True, slots=True)
class _PublicationFence:
    """One owner snapshot covering every field recovery publication installs."""

    transcript_generation: int
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
                return (
                    result,
                    writer._detach_log_state(),
                    writer.take_persisted_appends(),
                )
            finally:
                writer.close()
        finally:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()

    def _load_detached_log_state(self: ConversationStore) -> _DetachedLogState:
        """Load one validated transcript snapshot without touching resident state."""
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
        """Load the durable fields that recovery can change without resident writes."""
        reader = type(self)(
            self.root_dir,
            session_id=self.session_id,
            _must_exist=True,
        )
        try:
            return _DetachedConversationState(
                log=reader._detach_log_state(),
                agent_counter=reader._agent_counter,
                agent_children=reader._agent_children,
                agent_parent=reader._agent_parent,
                agent_canceled=reader._agent_canceled,
                agent_lifecycle=reader._agent_lifecycle,
            )
        finally:
            reader.close()

    def _publication_fence(self: ConversationStore) -> _PublicationFence:
        """Capture every owner field that a recovery refresh may replace."""
        return _PublicationFence(
            transcript_generation=self._transcript_generation,
            agent_counter=self._agent_counter,
            agent_children=copy.deepcopy(self._agent_children),
            agent_parent=copy.deepcopy(self._agent_parent),
            agent_canceled=copy.deepcopy(self._agent_canceled),
            agent_lifecycle=copy.deepcopy(self._agent_lifecycle),
        )

    @staticmethod
    def _markers_without_turns(
        markers: dict[str, dict[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        normalized = copy.deepcopy(markers)
        for marker in normalized.values():
            marker.pop("turns_used", None)
        return normalized

    def _install_agent_state(
        self: ConversationStore,
        state: _DetachedConversationState,
        fence: _PublicationFence,
    ) -> None:
        """Install recovery fields that their owner has not changed."""
        if self._agent_counter == fence.agent_counter:
            self._agent_counter = state.agent_counter
        if self._agent_parent == fence.agent_parent:
            self._agent_parent = state.agent_parent
        if self._agent_canceled == fence.agent_canceled:
            self._agent_canceled = state.agent_canceled
        if self._agent_lifecycle == fence.agent_lifecycle:
            self._agent_lifecycle = state.agent_lifecycle

        live_markers = self._agent_children
        if self._markers_without_turns(live_markers) != self._markers_without_turns(
            fence.agent_children
        ):
            return
        for marker_key, loaded_marker in state.agent_children.items():
            live_marker = live_markers.get(marker_key)
            if (
                live_marker is not None
                and live_marker.get("child_session_path")
                == loaded_marker.get("child_session_path")
            ):
                loaded_marker["turns_used"] = max(
                    loaded_marker.get("turns_used", 0),
                    live_marker.get("turns_used", 0),
                )
        self._agent_children = state.agent_children

    def _install_store_state(
        self: ConversationStore,
        state: _DetachedConversationState,
        fence: _PublicationFence,
    ) -> _DetachedLogState:
        """Publish unchanged recovery fields and preserve newer owner mutations."""
        if self._transcript_generation == fence.transcript_generation:
            released_log = self._install_log_state(state.log)
        else:
            released_log = state.log

        self._install_agent_state(state, fence)
        return released_log

    def _publish_log_state(
        self: ConversationStore,
        state: _DetachedLogState,
        generation: int,
        receipts: Any,
    ) -> _DetachedLogState:
        """Publish one durable write unless a newer owner snapshot is authoritative."""
        released_log = state
        with self._durable_write_condition:
            if self._closing or self._closed:
                return released_log
            if self._transcript_generation == generation:
                released_log = self._install_log_state(state)
            self._accept_persisted_appends(receipts)
        return released_log

    @staticmethod
    def _release_detached_log_state(state: _DetachedLogState) -> None:
        """Drop superseded transcript objects on a worker thread."""
        del state

    @staticmethod
    async def _await_started_worker(worker: asyncio.Future[Any]) -> tuple[Any, bool]:
        """Wait for a started durable operation, then report deferred cancellation."""
        cancelled = False
        while True:
            try:
                return await asyncio.shield(worker), cancelled
            except asyncio.CancelledError:
                cancelled = True

    async def _release_log_state_off_loop(
        self: ConversationStore, state: _DetachedLogState
    ) -> bool:
        loop = asyncio.get_running_loop()
        release = loop.run_in_executor(None, self._release_detached_log_state, state)
        _, cancelled = await self._await_started_worker(release)
        return cancelled

    async def _refresh_off_loop(self: ConversationStore) -> None:
        """Load recovery state once off-loop and publish it through one fence."""
        async with self._async_write_lock:
            loop = asyncio.get_running_loop()
            fence = self._publication_fence()
            state = await loop.run_in_executor(None, self._load_detached_store_state)
            with self._durable_write_condition:
                if self._closing or self._closed:
                    released_log = state.log
                else:
                    released_log = self._install_store_state(state, fence)
            cancelled = await self._release_log_state_off_loop(released_log)
            if cancelled:
                raise asyncio.CancelledError

    async def _to_thread_store(self: ConversationStore, function: Any, /) -> Any:
        """Run one resident-store lease operation off-loop despite cancellation."""
        worker = asyncio.get_running_loop().run_in_executor(None, function)
        result, cancelled = await self._await_started_worker(worker)
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _start_durable_write(
        self: ConversationStore,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> asyncio.Future[Any]:
        with self._durable_write_condition:
            if self._closing or self._closed:
                raise ValueError("session store is closed")
            self._durable_writes_in_flight += 1
        try:
            return asyncio.get_running_loop().run_in_executor(
                None, self._run_durable_write, method_name, args, kwargs
            )
        except BaseException:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()
            raise

    async def _to_thread_durable(
        self: ConversationStore,
        function: Any,
        /,
        *args: Any,
        _on_persisted: Callable[[], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """Serialize one blocking commit and one bounded owner publication."""
        async with self._async_write_lock:
            method_name = function.__name__
            if not hasattr(type(self), method_name):
                # Preserve instance-level instrumentation without mutating resident
                # state from a worker thread.
                return function(*args, **kwargs)
            generation = self._transcript_generation
            worker = self._start_durable_write(method_name, args, kwargs)
            outcome, cancelled = await self._await_started_worker(worker)
            result, state, receipts = outcome
            if self._transcript_generation == generation:
                released_log = self._publish_log_state(state, generation, receipts)
                cancelled |= await self._release_log_state_off_loop(released_log)
            else:
                reload_generation = self._transcript_generation
                reload_worker = asyncio.get_running_loop().run_in_executor(
                    None, self._load_detached_log_state
                )
                reloaded, reload_cancelled = await self._await_started_worker(
                    reload_worker
                )
                cancelled |= reload_cancelled
                with self._durable_write_condition:
                    if self._closing or self._closed:
                        released_reload = reloaded
                    else:
                        if self._transcript_generation == reload_generation:
                            released_reload = self._install_log_state(reloaded)
                        else:
                            # A later owner write synchronized this completed commit.
                            released_reload = reloaded
                        self._accept_persisted_appends(receipts)
                cancelled |= await self._release_log_state_off_loop(state)
                cancelled |= await self._release_log_state_off_loop(released_reload)
            if _on_persisted is not None:
                _on_persisted()
            if cancelled:
                raise asyncio.CancelledError
            return result
