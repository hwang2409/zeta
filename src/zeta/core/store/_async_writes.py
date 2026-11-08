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
    execute_mutation,
)
from ._mutation_commands import (
    _AcceptExplicitFinishCommand,
    _AcknowledgeStagedDeliveryCommand,
    _AdmitStepDeliveryTerminalCommand,
    _AppendPendingPromptCommand,
    _CancelExplicitChildCommand,
    _CommitDeliveryInputCommand,
    _ExplicitFinishReportCommand,
    _IdleNudgeEligibleCommand,
    _RefreshCommand,
    _ResolveApprovalCommand,
    _StageDeliveryRequestCommand,
    _StoreMutationCommand,
    _WarmStepDeliveryCommand,
)
from ._pending_prompts import PendingPromptWriterReloadedError

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
        self._address_commit_waiters = 0

    @property
    def conversation_generation(self: ConversationStore) -> int:
        return self._log_offset

    def _rebase_committed_entries(
        self: ConversationStore, delta: CommittedDelta[Any]
    ) -> tuple[ConversationEntry, ...] | None:
        entries = delta.entries_to_install
        if (
            self.conversation_generation == delta.generation_before
            or delta._resident_patch.replace_entries
        ):
            return entries
        existing_ids = {entry.id for entry in self._entries}
        missing = next(
            (
                index
                for index, entry in enumerate(entries)
                if entry.id not in existing_ids
            ),
            len(entries),
        )
        if missing == len(entries):
            return None
        if any(entry.id in existing_ids for entry in entries[missing:]):
            raise ConversationIntegrityError(
                "resident generation changed before committed delta installation"
            )
        return entries[missing:]

    def _install_delta_entries(
        self: ConversationStore,
        entries_to_install: tuple[ConversationEntry, ...],
        delta: CommittedDelta[Any],
    ) -> list[ConversationEntry]:
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
        return entries

    def _install_committed_delta(
        self: ConversationStore, delta: CommittedDelta[Any]
    ) -> tuple[ConversationEntry, ...] | str:
        self._assert_owner_thread()
        if delta.duplicate:
            return ()
        entries_to_install = self._rebase_committed_entries(delta)
        if entries_to_install is None:
            return ()
        entries = self._install_delta_entries(entries_to_install, delta)
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
        if isinstance(command, _StoreMutationCommand):
            command.after_persisted()
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
            if cancelled_before_write.is_set() and not isinstance(
                command, _StoreMutationCommand
            ):
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
        if isinstance(command, _StoreMutationCommand):
            command.after_persisted()
        if cancelled:
            raise asyncio.CancelledError
        return delta

    def _address_commit_queued(self: ConversationStore) -> None:
        """Test seam after an address commit joins the resident lock queue."""

    async def _commit_address_async(
        self: ConversationStore, command: ConversationMutation[Any]
    ) -> CommittedDelta[Any]:
        self._address_commit_waiters += 1
        self._address_commit_queued()
        try:
            async with self._async_write_lock:
                return await self._commit_async_unlocked(
                    command, self.conversation_generation
                )
        finally:
            self._address_commit_waiters -= 1

    async def _commit_storage_command(
        self: ConversationStore, command: _StoreMutationCommand
    ) -> Any:
        async with self._async_write_lock:
            command = replace(
                command,
                baseline_count=len(self._entries),
                baseline_digest=self._entry_prefix_digest,
                baseline_patch={
                    field: copy.deepcopy(getattr(self, field))
                    for field in command.patch_fields
                },
            )
            resident_generation = self.conversation_generation
            try:
                delta = await self._commit_async_unlocked(command, resident_generation)
            except ConversationGenerationConflict:
                if command.accept_external_generation:
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
                delta = await self._commit_async_unlocked(
                    replace(command, accept_external_generation=True),
                    self.conversation_generation,
                )
            return delta.result

    async def _refresh_off_loop(self: ConversationStore) -> None:
        await self._commit_storage_command(
            _RefreshCommand(accept_external_generation=True)
        )

    async def warm_step_delivery_writer(self: ConversationStore) -> None:
        await self._commit_storage_command(_WarmStepDeliveryCommand())

    async def commit_delivery_input(self: ConversationStore, item: Any) -> Any:
        return await self._commit_storage_command(_CommitDeliveryInputCommand(item=item))

    async def stage_delivery_request(
        self: ConversationStore,
        notification_budget: int | None,
        token_counter: Any,
        notification_selector: Any,
        notification_context: Any,
    ) -> Any:
        return await self._commit_storage_command(
            _StageDeliveryRequestCommand(
                notification_budget=notification_budget,
                token_counter=token_counter,
                notification_selector=notification_selector,
                notification_context=notification_context,
            )
        )

    async def acknowledge_staged_delivery(
        self: ConversationStore,
        prompt_id: str | None,
        notification_ids: tuple[str, ...],
    ) -> Any:
        return await self._commit_storage_command(
            _AcknowledgeStagedDeliveryCommand(
                prompt_id=prompt_id, notification_ids=notification_ids
            )
        )

    async def admit_step_delivery_terminal(
        self: ConversationStore, intent: Any
    ) -> Any:
        return await self._commit_storage_command(
            _AdmitStepDeliveryTerminalCommand(intent=intent)
        )

    async def accept_explicit_finish(
        self: ConversationStore,
        report: dict[str, object],
        notification_ids: frozenset[str],
        prompt_id: str | None,
    ) -> Any:
        return await self._commit_storage_command(
            _AcceptExplicitFinishCommand(
                report=report,
                notification_ids=notification_ids,
                prompt_id=prompt_id,
            )
        )

    async def cancel_explicit_child(self: ConversationStore) -> bool:
        return await self._commit_storage_command(_CancelExplicitChildCommand())

    async def explicit_finish_report(self: ConversationStore) -> Any:
        return await self._commit_storage_command(_ExplicitFinishReportCommand())

    async def idle_nudge_eligible(
        self: ConversationStore, notification_ids: frozenset[str]
    ) -> bool:
        return await self._commit_storage_command(
            _IdleNudgeEligibleCommand(notification_ids=notification_ids)
        )

    async def resolve_approval_async(
        self: ConversationStore,
        request_id: str,
        decision: str,
        *,
        reason: str | None = None,
    ) -> None:
        await self._commit_storage_command(
            _ResolveApprovalCommand(
                request_id=request_id, decision=decision, reason=reason
            )
        )

    async def append_pending_prompt_async(
        self: ConversationStore, text: str
    ) -> Any:
        return await self._commit_storage_command(_AppendPendingPromptCommand(text=text))

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
