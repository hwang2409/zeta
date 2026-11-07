"""Cancellation-safe coordination for typed off-loop store mutations."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from ..checkpoints import ConversationIntegrityError
from ..checkpoints.mutation import (
    CommittedDelta,
    ConversationGenerationConflict,
    ConversationMutation,
    _MethodMutation,
    execute_mutation,
)
from ._pending_prompts import PendingPromptWriterReloadedError

_REFRESH_PATCH_FIELDS = (
    "cwd",
    "_active_approval_requests",
    "_active_approval_resolutions",
    "_active_tool_results",
    "_active_agent_call_ids",
    "_active_agent_receipts",
    "_notification_ids",
    "_active_notifications",
    "_active_completion_notifications",
    "_active_notification_presentations",
    "_pending_prompt_ids",
    "_active_pending_prompt_entries",
    "_active_pending_prompt_messages",
    "_active_notification_delivery_message",
    "_active_last_message",
    "_active_last_turn_end",
    "_task_notification_ids",
    "_sent_message_receipts",
    "_client_deliveries",
    "_client_delivery_outcomes",
)

if TYPE_CHECKING:
    from ..checkpoints import ConversationEntry
    from ._log import _DetachedLogState
    from ._store import ConversationStore


class _MutationCancellation:
    def __init__(self) -> None:
        self.cancelled = threading.Event()
        self.write_started = threading.Event()

    def set(self) -> None:
        self.cancelled.set()

    def is_set(self) -> bool:
        return self.cancelled.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self.cancelled.wait(timeout)


class AsyncDurableWritesMixin:
    """Commit typed mutations off-loop and install only their deltas."""

    def _initialize_async_writes(self: ConversationStore) -> None:
        self._async_write_lock = asyncio.Lock()
        self._mutation_lock = threading.RLock()
        self._durable_write_condition = threading.Condition()
        self._durable_writes_in_flight = 0
        self._durable_writer: ConversationStore | None = None
        self._durable_writer_snapshot: tuple[int, _DetachedLogState] | None = None

    @property
    def conversation_generation(self: ConversationStore) -> int:
        return self._log_offset

    def _install_committed_delta(
        self: ConversationStore, delta: CommittedDelta[Any]
    ) -> tuple[ConversationEntry, ...] | str:
        self._assert_owner_thread()
        if delta.duplicate:
            return ()
        entries_to_install = delta.entries_to_install
        if (
            self.conversation_generation != delta.generation_before
            and not delta._resident_patch.replace_entries
        ):
            existing_ids = {entry.id for entry in self._entries}
            missing = next(
                (
                    index
                    for index, entry in enumerate(entries_to_install)
                    if entry.id not in existing_ids
                ),
                len(entries_to_install),
            )
            if missing == len(entries_to_install):
                return ()
            if any(entry.id in existing_ids for entry in entries_to_install[missing:]):
                raise ConversationIntegrityError(
                    "resident generation changed before committed delta installation"
                )
            entries_to_install = entries_to_install[missing:]
        entries = (
            list(entries_to_install)
            if delta._resident_patch.replace_entries
            else list(copy.deepcopy(entries_to_install))
        )
        if delta._resident_patch.replace_entries:
            self._before_incremental_state_install()
            self._entries = entries
            self._entry_ids = {entry.id for entry in entries}
        elif entries:
            self._before_incremental_state_install()
            if delta._resident_patch.raw_entries:
                self._entries.extend(entries)
                self._entry_ids.update(entry.id for entry in entries)
            else:
                self._install_validated_linear_entries(entries)
                self._record_client_delivery_entries(entries)
        for field, value in delta._resident_patch.values.items():
            if getattr(self, field) == delta._resident_patch.expected[field]:
                setattr(self, field, copy.deepcopy(value))
        cursor = delta.log_cursor
        self._entry_prefix_digest = cursor.entry_prefix_digest
        self._log_identity = cursor.identity
        self._log_offset = cursor.offset
        self._log_mtime_ns = cursor.mtime_ns
        self._log_prefix_fingerprint = cursor.prefix_fingerprint
        if entries:
            self._transcript_generation += 1
        return (
            "reset"
            if delta._resident_patch.replace_entries
            else tuple(entries_to_install)
        )

    def commit(
        self: ConversationStore,
        command: ConversationMutation[Any],
        expected_generation: int,
    ) -> CommittedDelta[Any]:
        self._assert_owner_thread()
        with self._mutation_lock:
            delta = execute_mutation(self, command, expected_generation)
        if delta._resident_patch.replace_entries:
            self._publish_entries_installed("reset")
        elif delta.entries_to_install:
            self._publish_entries_installed(delta.entries_to_install)
        if (
            isinstance(command, _MethodMutation)
            and command.persisted_callback is not None
        ):
            command.persisted_callback()
        return delta

    def _before_committed_mutation(self: ConversationStore) -> None:
        """Test seam immediately before a dispatched mutation can write."""

    def _run_committed_mutation(
        self: ConversationStore,
        command: ConversationMutation[Any],
        expected_generation: int,
        cancelled_before_write: _MutationCancellation,
    ) -> tuple[CommittedDelta[Any], Any]:
        writer: ConversationStore | None = None
        try:
            writer = self._durable_writer
            if writer is None:
                writer = self._new_persistent_durable_writer()
                self._durable_writer = writer
            self._before_committed_mutation()
            if cancelled_before_write.is_set():
                raise asyncio.CancelledError
            cancelled_before_write.write_started.set()
            delta = execute_mutation(writer, command, expected_generation)
            return delta, writer.take_persisted_appends()
        except (ConversationGenerationConflict, PendingPromptWriterReloadedError):
            raise
        except BaseException:
            if writer is not None:
                writer.close()
                with self._durable_write_condition:
                    if self._durable_writer is writer:
                        self._durable_writer = None
            raise
        finally:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()

    def _start_committed_mutation(
        self: ConversationStore,
        command: ConversationMutation[Any],
        expected_generation: int,
        cancelled_before_write: _MutationCancellation,
    ) -> asyncio.Future[Any]:
        with self._durable_write_condition:
            if self._closing or self._closed:
                raise ValueError("session store is closed")
            self._durable_writes_in_flight += 1
        try:
            return asyncio.get_running_loop().run_in_executor(
                None,
                self._run_committed_mutation,
                command,
                expected_generation,
                cancelled_before_write,
            )
        except BaseException:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()
            raise

    @staticmethod
    async def _await_started_worker(
        worker: asyncio.Future[Any],
        on_cancel: Callable[[], None] | None = None,
        can_abandon: Callable[[], bool] | None = None,
    ) -> tuple[Any, bool]:
        cancelled = False
        while True:
            try:
                return await asyncio.shield(worker), cancelled
            except asyncio.CancelledError:
                cancelled = True
                if on_cancel is not None:
                    on_cancel()
                if can_abandon is not None and can_abandon():
                    raise
                if worker.done():
                    return worker.result(), cancelled

    async def commit_async(
        self: ConversationStore,
        command: ConversationMutation[Any],
        expected_generation: int,
    ) -> CommittedDelta[Any]:
        async with self._async_write_lock:
            return await self._commit_async_unlocked(command, expected_generation)

    async def _commit_async_unlocked(
        self: ConversationStore,
        command: ConversationMutation[Any],
        expected_generation: int,
    ) -> CommittedDelta[Any]:
        cancelled_before_write = _MutationCancellation()
        worker = self._start_committed_mutation(
            command, expected_generation, cancelled_before_write
        )
        (delta, receipts), cancelled = await self._await_started_worker(
            worker,
            cancelled_before_write.set,
            lambda: not cancelled_before_write.write_started.is_set(),
        )
        task = asyncio.current_task()
        cancelled = cancelled or (task is not None and task.cancelling() > 0)
        installed: tuple[ConversationEntry, ...] | str = ()
        with self._durable_write_condition:
            if not self._closing and not self._closed:
                installed = self._install_committed_delta(delta)
            self._accept_persisted_appends(receipts)
        if installed == "reset":
            self._publish_entries_installed("reset")
        elif installed:
            self._publish_prepared_entries_installed(installed)
        if (
            isinstance(command, _MethodMutation)
            and command.persisted_callback is not None
        ):
            command.persisted_callback()
        if cancelled:
            raise asyncio.CancelledError
        return delta

    async def _commit_address_async(
        self: ConversationStore, command: ConversationMutation[Any]
    ) -> CommittedDelta[Any]:
        async with self._async_write_lock:
            return await self._commit_async_unlocked(
                command, self.conversation_generation
            )

    async def _commit_method_async(
        self: ConversationStore,
        method: Callable[..., Any],
        /,
        *args: Any,
        _patch_fields: tuple[str, ...] = _REFRESH_PATCH_FIELDS
        + (
            "_agent_counter",
            "_agent_children",
            "_agent_parent",
            "_agent_canceled",
            "_agent_lifecycle",
        ),
        _on_persisted: Callable[[], None] | None = None,
        _accept_external_generation: bool = False,
        _method_accepts_store: bool | None = None,
        **kwargs: Any,
    ) -> Any:
        async with self._async_write_lock:
            is_bound_method = getattr(method, "__self__", None) is not None
            unbound = getattr(method, "__func__", method)
            command = _MethodMutation(
                method=unbound,
                method_accepts_store=(
                    is_bound_method
                    if _method_accepts_store is None
                    else _method_accepts_store
                ),
                args=args,
                kwargs=kwargs,
                patch_fields=_patch_fields,
                persisted_callback=_on_persisted,
                accept_external_generation=_accept_external_generation,
                baseline_count=len(self._entries),
                baseline_digest=self._entry_prefix_digest,
                baseline_patch={
                    field: copy.deepcopy(getattr(self, field))
                    for field in _patch_fields
                },
            )
            resident_generation = self.conversation_generation
            try:
                delta = await self._commit_async_unlocked(command, resident_generation)
            except ConversationGenerationConflict:
                if _accept_external_generation:
                    raise
                if self.conversation_generation != resident_generation:
                    command = replace(
                        command,
                        baseline_count=len(self._entries),
                        baseline_digest=self._entry_prefix_digest,
                        baseline_patch={
                            field: copy.deepcopy(getattr(self, field))
                            for field in command.patch_fields
                        },
                    )
                command = replace(command, accept_external_generation=True)
                delta = await self._commit_async_unlocked(
                    command, self.conversation_generation
                )
            return delta.result

    async def _refresh_off_loop(self: ConversationStore) -> None:
        await self._commit_method_async(
            type(self).refresh,
            _patch_fields=_REFRESH_PATCH_FIELDS
            + (
                "_agent_counter",
                "_agent_children",
                "_agent_parent",
                "_agent_canceled",
                "_agent_lifecycle",
            ),
            _accept_external_generation=True,
            _method_accepts_store=True,
        )

    async def _to_thread_store(self: ConversationStore, function: Any, /) -> Any:
        """Run one resident-store lease operation off-loop despite cancellation."""
        worker = asyncio.get_running_loop().run_in_executor(None, function)
        result, cancelled = await self._await_started_worker(worker)
        if cancelled:
            raise asyncio.CancelledError
        return result

    def _prepare_durable_writer_snapshot(self: ConversationStore) -> None:
        """Detach one validated generation before the store becomes loop-owned."""
        snapshot = self._snapshot_log_state()
        with self._durable_write_condition:
            if self._closing or self._closed:
                return
            self._durable_writer_snapshot = (self._transcript_generation, snapshot)

    def _new_persistent_durable_writer(self: ConversationStore) -> ConversationStore:
        with self._durable_write_condition:
            prepared = self._durable_writer_snapshot
            if prepared is not None and prepared[0] != self._transcript_generation:
                prepared = None
        return type(self)(
            self.root_dir,
            session_id=self.session_id,
            _must_exist=True,
            _collect_persisted_appends=True,
            _validated_log_state=None if prepared is None else prepared[1],
            _storage_lease=self._storage_lease,
            _execution_lease=self.execution_lease,
            _mutation_target=True,
        )

    def _drain_durable_writes_for_close(self: ConversationStore) -> bool:
        with self._durable_write_condition:
            self._closing = True
            while self._durable_writes_in_flight:
                self._durable_write_condition.wait()
            if self._closed:
                return False
            self._closed = True
            writer = self._durable_writer
            self._durable_writer = None
            self._durable_writer_snapshot = None
        if writer is not None:
            writer.close()
        return True
