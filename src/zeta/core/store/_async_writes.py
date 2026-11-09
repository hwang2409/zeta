"""Cancellation-safe coordination for typed off-loop store mutations."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, ClassVar

from ..checkpoints import ConversationIntegrityError
from ..checkpoints.mutation import (
    CommittedDelta,
    ConversationGenerationConflict,
    ConversationMutation,
    MutationCancellationPolicy,
    _cursor,
    _ResidentPatch,
    execute_mutation,
)
from ._pending_prompts import PendingPromptWriterReloadedError

if TYPE_CHECKING:
    from ..checkpoints import ConversationEntry
    from ._log import _DetachedLogState
    from ._store import ConversationStore


_PATCH_FIELDS = (
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
    "_agent_counter",
    "_agent_children",
    "_agent_parent",
    "_agent_canceled",
    "_agent_lifecycle",
)
_DEEP_PATCH_FIELDS = frozenset({"_agent_children"})


def _snapshot_patch_fields(
    store: ConversationStore, fields: tuple[str, ...]
) -> dict[str, object]:
    return {
        field: (
            copy.deepcopy(getattr(store, field))
            if field in _DEEP_PATCH_FIELDS
            else copy.copy(getattr(store, field))
        )
        for field in fields
    }


@dataclass(frozen=True, slots=True, kw_only=True)
class _StoreMutationCommand:
    """Shared reconciliation facts; each subclass fixes one storage operation."""

    baseline_count: int | None = None
    baseline_digest: bytes | None = None
    baseline_patch: Mapping[str, object] | None = None
    cancellation_policy: ClassVar[MutationCancellationPolicy] = (
        MutationCancellationPolicy.FINALIZE_ONCE_SUBMITTED
    )
    patch_fields: ClassVar[tuple[str, ...]] = _PATCH_FIELDS
    accept_external_generation: bool = False

    def apply(self, store: Any) -> Any:
        raise NotImplementedError

    def after_persisted(self) -> None:
        pass


@dataclass(frozen=True, slots=True, kw_only=True)
class _RefreshCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> None:
        store.refresh()


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendMessageCommand(_StoreMutationCommand):
    message: Any
    parent_id: str | None
    callback: Callable[[], None] | None = None

    def apply(self, store: Any) -> Any:
        return store.append_message(self.message, parent_id=self.parent_id)

    def after_persisted(self) -> None:
        if self.callback is not None:
            self.callback()


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendChildRunStartedCommand(_StoreMutationCommand):
    run_id: str
    parent_run_id: str | None
    name: str | None
    task: str
    started_at: str
    callback: Callable[[], None] | None = None

    def apply(self, store: Any) -> Any:
        return store.append_child_run_started(
            self.run_id, self.parent_run_id, self.name, self.task, self.started_at
        )

    def after_persisted(self) -> None:
        if self.callback is not None:
            self.callback()


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendApprovalHistoryCommand(_StoreMutationCommand):
    payload: Mapping[str, Any]

    def apply(self, store: Any) -> Any:
        return store.append_approval_history(self.payload)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendClientDeliveryCommand(_StoreMutationCommand):
    delivery_id: str
    method: Any
    status: Any
    outcome: dict[str, object]
    message: Any = None
    accepted_message: Any = None
    callback: Callable[[], None] | None = None

    def apply(self, store: Any) -> Any:
        return store.append_client_delivery(
            self.delivery_id,
            self.method,
            self.status,
            self.outcome,
            message=self.message,
            accepted_message=self.accepted_message,
        )

    def after_persisted(self) -> None:
        if self.callback is not None:
            self.callback()


@dataclass(frozen=True, slots=True, kw_only=True)
class _DeliverClientSteeringCommand(_StoreMutationCommand):
    batch: Any

    def apply(self, store: Any) -> None:
        store._deliver_client_steering(self.batch)


@dataclass(frozen=True, slots=True, kw_only=True)
class _DropClientSteeringCommand(_StoreMutationCommand):
    queued: tuple[Any, ...]
    reason: Any

    def apply(self, store: Any) -> Any:
        return store._drop_client_steering(self.queued, self.reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendTurnEndCommand(_StoreMutationCommand):
    data: dict[str, Any]

    def apply(self, store: Any) -> None:
        store._append_row("turn_end", self.data)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RecordNotificationDeliveryCommand(_StoreMutationCommand):
    notification_id: str
    presented: bool
    acknowledged: bool

    def apply(self, store: Any) -> Any:
        return store.record_agent_notification_delivery(
            self.notification_id,
            presented=self.presented,
            acknowledged=self.acknowledged,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _PresentNotificationsCommand(_StoreMutationCommand):
    notification_ids: list[str]

    def apply(self, store: Any) -> None:
        store.mark_agent_notifications_presented_to_tui(self.notification_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class _WarmStepDeliveryCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> None:
        store._warm_step_delivery_writer()


@dataclass(frozen=True, slots=True, kw_only=True)
class _CommitDeliveryInputCommand(_StoreMutationCommand):
    item: Any

    def apply(self, store: Any) -> Any:
        return store._commit_delivery_input(self.item)


@dataclass(frozen=True, slots=True, kw_only=True)
class _StageDeliveryRequestCommand(_StoreMutationCommand):
    notification_budget: int | None
    token_counter: Any
    notification_selector: Any
    notification_context: Any

    def apply(self, store: Any) -> Any:
        return store._stage_delivery_request(
            self.notification_budget,
            self.token_counter,
            self.notification_selector,
            self.notification_context,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _AcknowledgeStagedDeliveryCommand(_StoreMutationCommand):
    prompt_id: str | None
    notification_ids: tuple[str, ...]

    def apply(self, store: Any) -> Any:
        return store._acknowledge_staged_delivery(self.prompt_id, self.notification_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AdmitStepDeliveryTerminalCommand(_StoreMutationCommand):
    intent: Any

    def apply(self, store: Any) -> Any:
        return store._admit_step_delivery_terminal(self.intent)


@dataclass(frozen=True, slots=True, kw_only=True)
class _AcceptExplicitFinishCommand(_StoreMutationCommand):
    report: dict[str, object]
    notification_ids: frozenset[str]
    prompt_id: str | None

    def apply(self, store: Any) -> Any:
        return store._accept_explicit_finish(
            self.report, self.notification_ids, self.prompt_id
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _CancelExplicitChildCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> bool:
        return store._cancel_explicit_child()


@dataclass(frozen=True, slots=True, kw_only=True)
class _ExplicitFinishReportCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> Any:
        return store._explicit_finish_report()


@dataclass(frozen=True, slots=True, kw_only=True)
class _IdleNudgeEligibleCommand(_StoreMutationCommand):
    notification_ids: frozenset[str]

    def apply(self, store: Any) -> bool:
        return store._idle_nudge_eligible(self.notification_ids)


@dataclass(frozen=True, slots=True, kw_only=True)
class _ResolveApprovalCommand(_StoreMutationCommand):
    request_id: str
    decision: str
    reason: str | None

    def apply(self, store: Any) -> None:
        store.resolve_approval(self.request_id, self.decision, reason=self.reason)


@dataclass(frozen=True, slots=True, kw_only=True)
class _MarkAgentCanceledCommand(_StoreMutationCommand):
    tool_call_id: str

    def apply(self, store: Any) -> None:
        store.mark_agent_canceled(self.tool_call_id)


@dataclass(frozen=True, slots=True, kw_only=True)
class _RegisterAgentChildCommand(_StoreMutationCommand):
    tool_call: Any
    child_session_path: str
    description: str
    agent_type: str | None
    background: bool
    child_instance_id: str | None
    accepts_follow_ups: bool
    explicit_finish: bool
    turns_used: int

    def apply(self, store: Any) -> Any:
        return store.register_agent_child(
            self.tool_call,
            child_session_path=self.child_session_path,
            description=self.description,
            agent_type=self.agent_type,
            background=self.background,
            child_instance_id=self.child_instance_id,
            accepts_follow_ups=self.accepts_follow_ups,
            explicit_finish=self.explicit_finish,
            turns_used=self.turns_used,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _FinishAgentChildCommand(_StoreMutationCommand):
    marker_key: str

    def apply(self, store: Any) -> None:
        store.finish_agent_child(self.marker_key)


@dataclass(frozen=True, slots=True, kw_only=True)
class _FinishAgentParentCommand(_StoreMutationCommand):
    def apply(self, store: Any) -> None:
        store.finish_agent_parent()


@dataclass(frozen=True, slots=True, kw_only=True)
class _FinishAgentLifecycleCommand(_StoreMutationCommand):
    state: str
    final_result: str
    turns_used: int | None
    killed_task_ids: list[str] | None
    killed_task_count: int | None
    killed_task_ids_truncated: bool | None
    canonical_receipt: bool

    def apply(self, store: Any) -> None:
        store.finish_agent_lifecycle(
            self.state,
            final_result=self.final_result,
            turns_used=self.turns_used,
            killed_task_ids=self.killed_task_ids,
            killed_task_count=self.killed_task_count,
            killed_task_ids_truncated=self.killed_task_ids_truncated,
            canonical_receipt=self.canonical_receipt,
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class _AppendPendingPromptCommand(_StoreMutationCommand):
    text: str

    def apply(self, store: Any) -> Any:
        return store.append_pending_prompt(self.text)


class _MutationCancellation:
    def __init__(self, context: object | None = None) -> None:
        self.context = context
        self.cancelled = threading.Event()
        self.write_started = threading.Event()
        self._lock = threading.Lock()

    def abort_before_write(self) -> bool:
        with self._lock:
            if self.write_started.is_set():
                return False
            self.cancelled.set()
            return True

    def begin_write(self) -> None:
        with self._lock:
            if self.cancelled.is_set():
                raise asyncio.CancelledError
            self.write_started.set()

    def is_set(self) -> bool:
        return self.cancelled.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self.cancelled.wait(timeout)


def _mutation_cancellation_observed(
    policy: MutationCancellationPolicy,
) -> None:
    """Test seam after caller cancellation reaches the shared policy boundary."""


def _before_mutation_write_gate() -> None:
    """Test seam on the executor before the one mutation write gate."""


def _run_gated_mutation[DurableT](
    cancellation: _MutationCancellation,
    worker: Callable[[_MutationCancellation], DurableT],
    before_write: Callable[[], None] | None,
    submission_failed: Callable[[], None] | None,
) -> DurableT:
    try:
        _before_mutation_write_gate()
        if before_write is not None:
            before_write()
        cancellation.begin_write()
    except BaseException:
        if submission_failed is not None:
            submission_failed()
        raise
    return worker(cancellation)


async def _adopt_cancelled_mutation[ResultT](
    command: ConversationMutation[Any],
    cancellation: _MutationCancellation,
    operation: asyncio.Task[ResultT],
    cancelled: Callable[[MutationCancellationPolicy], None] | None,
) -> tuple[bool, ResultT | None]:
    policy = command.cancellation_policy
    _mutation_cancellation_observed(policy)
    if cancelled is not None:
        cancelled(policy)
    if (
        policy is MutationCancellationPolicy.ABORT_BEFORE_WRITE
        and cancellation.abort_before_write()
    ):
        # test-contract: durability submitted work must finish after cancellation
        return True, await asyncio.shield(operation)
    if operation.done():
        return True, operation.result()
    return False, None


async def _submit_durable_mutation[DurableT, ResultT](
    command: ConversationMutation[Any],
    worker: Callable[[_MutationCancellation], DurableT],
    finish: Callable[[DurableT], Awaitable[ResultT]],
    *,
    before_submit: Callable[[], None] | None = None,
    before_write: Callable[[], None] | None = None,
    worker_context: object | None = None,
    submission_failed: Callable[[], None] | None = None,
    cancelled: Callable[[MutationCancellationPolicy], None] | None = None,
) -> ResultT:
    """Submit one durable mutation and enforce its cancellation policy."""
    cancellation = _MutationCancellation(worker_context)
    if before_submit is not None:
        before_submit()
    try:
        worker_future = asyncio.get_running_loop().run_in_executor(
            None,
            _run_gated_mutation,
            cancellation,
            worker,
            before_write,
            submission_failed,
        )
    except BaseException:
        if submission_failed is not None:
            submission_failed()
        raise

    async def finish_operation() -> ResultT:
        # test-contract: durability the submitted mutation owns this completion
        return await finish(await worker_future)

    operation = asyncio.create_task(finish_operation())
    was_cancelled = False
    while True:
        try:
            # test-contract: durability submitted work must finish after cancellation
            result = await asyncio.shield(operation)
            break
        except asyncio.CancelledError:
            was_cancelled = True
            finished, adopted = await _adopt_cancelled_mutation(
                command, cancellation, operation, cancelled
            )
            if finished:
                result = adopted
                break
    task = asyncio.current_task()
    if was_cancelled or (task is not None and task.cancelling() > 0):
        raise asyncio.CancelledError
    return result


@dataclass(frozen=True, slots=True)
class _ResidentMutationBaseline:
    generation: int
    cursor: tuple[tuple[int, int, int] | None, int, int | None, bytes | None, bytes]
    patch: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _SubmittedMutationOutcome:
    prelude: CommittedDelta[Any] | None
    delta: CommittedDelta[Any] | None
    error: BaseException | None
    receipts: Any


class _AsyncWriteLock(asyncio.Lock):
    def __init__(self, queued: Callable[[], None]) -> None:
        super().__init__()
        self._queued = queued

    async def acquire(self) -> bool:
        if self.locked():
            self._queued()
        return await super().acquire()


class AsyncDurableWritesMixin:
    """Commit typed mutations off-loop and install only their deltas."""

    def _initialize_async_writes(self: ConversationStore) -> None:
        self._async_write_lock = _AsyncWriteLock(lambda: self._async_write_queued())
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
            self._reset_generation += 1
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
            if delta.committed_entries and self._history_index is not None:
                self._history_index.apply_committed(
                    {**self._history_index_facts(), "rows": delta.committed_entries}
                )
        if delta._resident_patch.replace_entries:
            self._publish_entries_installed("reset")
        elif delta.entries_to_install:
            self._publish_entries_installed(delta.entries_to_install)
        if isinstance(command, _StoreMutationCommand):
            command.after_persisted()
        return delta

    def _before_committed_mutation(self: ConversationStore) -> None:
        """Test seam immediately before a dispatched mutation can write."""

    @staticmethod
    def _reload_durable_writer_if_changed(writer: ConversationStore) -> None:
        if not writer._log_metadata_matches():
            with writer._append_lock():
                pass

    def _materialization_prelude(
        self: ConversationStore,
        writer: ConversationStore,
        baseline: _ResidentMutationBaseline,
    ) -> CommittedDelta[Any] | None:
        cursor = _cursor(writer)
        current = (
            cursor.identity,
            cursor.offset,
            cursor.mtime_ns,
            cursor.prefix_fingerprint,
            cursor.entry_prefix_digest,
        )
        if current == baseline.cursor:
            return None
        return CommittedDelta(
            generation_before=baseline.generation,
            generation_after=writer.conversation_generation,
            committed_entries=(),
            entries_to_install=writer._owned_installed_delta(tuple(writer._entries)),
            _resident_patch=_ResidentPatch(
                {field: copy.deepcopy(getattr(writer, field)) for field in _PATCH_FIELDS},
                baseline.patch,
                replace_entries=True,
            ),
            result=None,
            log_cursor=cursor,
            dedupe_key=None,
            duplicate=False,
        )

    def _run_committed_mutation(
        self: ConversationStore,
        command: ConversationMutation[Any],
        expected_generation: int,
        cancellation: _MutationCancellation,
    ) -> _SubmittedMutationOutcome:
        baseline = cancellation.context
        assert isinstance(baseline, _ResidentMutationBaseline)
        writer: ConversationStore | None = None
        prelude: CommittedDelta[Any] | None = None
        try:
            writer = self._durable_writer
            if writer is None:
                writer = self._new_persistent_durable_writer()
                self._durable_writer = writer
            else:
                self._reload_durable_writer_if_changed(writer)
            prelude = self._materialization_prelude(writer, baseline)
            self._after_committed_mutation_started()
            if (
                prelude is not None
                and isinstance(command, _CommitDeliveryInputCommand)
                and getattr(command.item, "deadline", None) is not None
            ):
                return _SubmittedMutationOutcome(
                    prelude,
                    None,
                    PendingPromptWriterReloadedError(),
                    writer.take_persisted_appends(),
                )
            generation = (
                writer.conversation_generation
                if prelude is not None
                else expected_generation
            )
            durable_command = command
            if prelude is not None and isinstance(command, _StoreMutationCommand):
                durable_command = replace(
                    command,
                    baseline_count=len(writer._entries),
                    baseline_digest=writer._entry_prefix_digest,
                    baseline_patch=prelude._resident_patch.values,
                )
            try:
                delta = execute_mutation(writer, durable_command, generation)
                if delta.committed_entries and writer._history_index is not None:
                    writer._history_index.apply_committed(
                        {
                            **writer._history_index_facts(),
                            "rows": delta.committed_entries,
                        }
                    )
            except BaseException as error:
                if prelude is None:
                    raise
                return _SubmittedMutationOutcome(
                    prelude, None, error, writer.take_persisted_appends()
                )
            return _SubmittedMutationOutcome(
                prelude, delta, None, writer.take_persisted_appends()
            )
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

    def _before_mutation_submit(self: ConversationStore) -> None:
        with self._durable_write_condition:
            if self._closing or self._closed:
                raise ValueError("session store is closed")
            self._durable_writes_in_flight += 1

    def _mutation_submission_failed(self: ConversationStore) -> None:
        with self._durable_write_condition:
            self._durable_writes_in_flight -= 1
            self._durable_write_condition.notify_all()

    def _submitted_mutation_cancelled(
        self: ConversationStore, policy: MutationCancellationPolicy
    ) -> None:
        """Test seam after owner cancellation reaches the policy boundary."""

    def _async_write_queued(self: ConversationStore) -> None:
        """Test seam when a mutation joins the resident lock queue."""

    def _after_committed_mutation_started(self: ConversationStore) -> None:
        """Test seam after cancellation closes but before the durable operation."""

    async def commit_async(
        self: ConversationStore,
        command: ConversationMutation[Any],
        expected_generation: int,
    ) -> CommittedDelta[Any]:
        async with self._async_write_lock:
            return await self._commit_async_unlocked(command, expected_generation)

    async def _before_committed_install(self: ConversationStore) -> None:
        """Test seam after durability and before resident installation."""

    async def _finish_submitted_mutation(
        self: ConversationStore,
        command: ConversationMutation[Any],
        outcome: _SubmittedMutationOutcome,
    ) -> CommittedDelta[Any]:
        await self._before_committed_install()
        installed: list[tuple[ConversationEntry, ...] | str] = []
        with self._durable_write_condition:
            if not self._closing and not self._closed:
                if outcome.prelude is not None:
                    installed.append(self._install_committed_delta(outcome.prelude))
                if outcome.delta is not None:
                    installed.append(self._install_committed_delta(outcome.delta))
            self._accept_persisted_appends(outcome.receipts)
        if "reset" in installed:
            self._publish_entries_installed("reset")
        else:
            appended = tuple(
                entry
                for delta in installed
                if delta
                for entry in delta
            )
            if appended:
                self._publish_prepared_entries_installed(appended)
        if outcome.error is not None:
            raise outcome.error
        assert outcome.delta is not None
        if isinstance(command, _StoreMutationCommand):
            command.after_persisted()
        return outcome.delta

    async def _commit_async_unlocked(
        self: ConversationStore,
        command: ConversationMutation[Any],
        expected_generation: int,
    ) -> CommittedDelta[Any]:
        baseline_patch = (
            dict(command.baseline_patch)
            if isinstance(command, _StoreMutationCommand)
            and command.baseline_patch is not None
            else _snapshot_patch_fields(self, _PATCH_FIELDS)
        )
        baseline = _ResidentMutationBaseline(
            expected_generation,
            (
                self._log_identity,
                self._log_offset,
                self._log_mtime_ns,
                self._log_prefix_fingerprint,
                self._entry_prefix_digest,
            ),
            baseline_patch,
        )
        return await _submit_durable_mutation(
            command,
            lambda cancellation: self._run_committed_mutation(
                command, expected_generation, cancellation
            ),
            lambda outcome: self._finish_submitted_mutation(command, outcome),
            before_submit=self._before_mutation_submit,
            before_write=self._before_committed_mutation,
            worker_context=baseline,
            submission_failed=self._mutation_submission_failed,
            cancelled=self._submitted_mutation_cancelled,
        )

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
                baseline_patch=_snapshot_patch_fields(self, command.patch_fields),
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
                        baseline_patch=_snapshot_patch_fields(
                            self, command.patch_fields
                        ),
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
        return await self._commit_storage_command(
            _CommitDeliveryInputCommand(item=item)
        )

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

    async def admit_step_delivery_terminal(self: ConversationStore, intent: Any) -> Any:
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
        # test-contract: seam this is the public async storage boundary
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

    async def append_pending_prompt_async(self: ConversationStore, text: str) -> Any:
        return await self._commit_storage_command(
            _AppendPendingPromptCommand(text=text)
        )

    @staticmethod
    async def _await_store_worker(worker: asyncio.Future[Any]) -> tuple[Any, bool]:
        cancelled = False
        while True:
            try:
                # test-contract: durability submitted store work must finish
                return await asyncio.shield(worker), cancelled
            except asyncio.CancelledError:
                cancelled = True

    async def _to_thread_store(self: ConversationStore, function: Any, /) -> Any:
        """Run one resident-store lease operation off-loop despite cancellation."""
        worker = asyncio.get_running_loop().run_in_executor(None, function)
        result, cancelled = await self._await_store_worker(worker)
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
            _history_index=self._history_index,
        )

    def _drain_durable_writes_for_close(self: ConversationStore) -> bool:
        with self._durable_write_condition:
            self._closing = True
            while self._durable_writes_in_flight:
                # test-contract: durability close must drain submitted writes
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
