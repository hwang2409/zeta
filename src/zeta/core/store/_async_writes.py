"""Cancellation-safe coordination for off-loop durable store writes."""

from __future__ import annotations

import asyncio
import copy
import threading
from collections.abc import Callable
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import TYPE_CHECKING, Any

from ._pending_prompts import PendingPromptWriterReloadedError

if TYPE_CHECKING:
    from ..checkpoints import ConversationEntry
    from ._log import InstalledEntriesDelta, PersistedAppend, _DetachedLogState
    from ._store import ConversationStore


@dataclass(frozen=True, slots=True)
class _OwnerSyncDelta:
    expected_baseline: tuple[int, bytes]
    expected_log: tuple[tuple[int, int, int], int]
    entries: tuple[ConversationEntry, ...]
    entry_prefix_digest: bytes
    log_identity: tuple[int, int, int]
    log_offset: int
    log_mtime_ns: int
    log_prefix_fingerprint: bytes


@dataclass(frozen=True, slots=True)
class _DurableLogDelta:
    """Small worker-validated suffix plus the durable log facts it produced."""

    entries: tuple[ConversationEntry, ...]
    installed_delta: InstalledEntriesDelta
    entry_prefix_digest: bytes
    log_identity: tuple[int, int, int]
    log_offset: int
    log_mtime_ns: int
    log_prefix_fingerprint: bytes


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
        self._durable_writer: ConversationStore | None = None
        self._durable_writer_snapshot: tuple[int, _DetachedLogState] | None = None
        self._durable_writer_owner_deltas: list[_OwnerSyncDelta] = []
        self._durable_writer_alignment_invalid = False

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
            self._durable_writer_owner_deltas.clear()
            self._durable_writer_alignment_invalid = False
        if writer is not None:
            writer.close()
        return True

    def _prepare_durable_writer_snapshot(self: ConversationStore) -> None:
        """Detach one validated generation before the store becomes loop-owned."""
        snapshot = self._snapshot_log_state()
        with self._durable_write_condition:
            if self._closing or self._closed:
                return
            self._durable_writer_snapshot = (
                self._transcript_generation,
                snapshot,
            )

    def _queue_owner_sync_delta(
        self: ConversationStore,
        entries: tuple[ConversationEntry, ...],
        expected_baseline: tuple[int, bytes],
        expected_log: tuple[tuple[int, int, int], int],
    ) -> None:
        """Let the worker apply one owner-committed row before its next operation."""
        if not entries:
            return
        prior_id = self._entries[-len(entries) - 1].id if expected_baseline[0] else None
        linear = entries[0].parent_id == prior_id and all(
            entry.parent_id == previous.id
            for previous, entry in pairwise(entries)
        )
        with self._durable_write_condition:
            if self._closing or self._closed:
                return
            if self._durable_writes_in_flight or not linear:
                self._durable_writer_owner_deltas.clear()
                self._durable_writer_snapshot = None
                self._durable_writer_alignment_invalid = True
                return
            if self._durable_writer is None and self._durable_writer_snapshot is None:
                return
            if len(self._durable_writer_owner_deltas) >= 64:
                self._durable_writer_owner_deltas.clear()
                self._durable_writer_snapshot = None
                self._durable_writer_alignment_invalid = True
                return
            self._durable_writer_owner_deltas.append(
                _OwnerSyncDelta(
                    expected_baseline=expected_baseline,
                    expected_log=expected_log,
                    entries=copy.deepcopy(entries),
                    entry_prefix_digest=self._entry_prefix_digest,
                    log_identity=self._log_identity,
                    log_offset=self._log_offset,
                    log_mtime_ns=self._log_mtime_ns,
                    log_prefix_fingerprint=self._log_prefix_fingerprint,
                )
            )
            if self._durable_writer_snapshot is not None:
                _, snapshot = self._durable_writer_snapshot
                self._durable_writer_snapshot = self._transcript_generation, snapshot

    def _align_persistent_durable_writer(
        self: ConversationStore, writer: ConversationStore
    ) -> ConversationStore:
        with self._durable_write_condition:
            invalid = self._durable_writer_alignment_invalid
            deltas = tuple(self._durable_writer_owner_deltas)
            self._durable_writer_alignment_invalid = False
            self._durable_writer_owner_deltas.clear()
        aligned = not invalid
        for delta in deltas:
            aligned &= (
                (len(writer._entries), writer._entry_prefix_digest)
                == delta.expected_baseline
                and (writer._log_identity, writer._log_offset) == delta.expected_log
                and self._accept_worker_alignment_delta(writer, delta.entries)
            )
            if not aligned:
                break
            writer._entry_prefix_digest = delta.entry_prefix_digest
            writer._log_identity = delta.log_identity
            writer._log_offset = delta.log_offset
            writer._log_mtime_ns = delta.log_mtime_ns
            writer._log_prefix_fingerprint = delta.log_prefix_fingerprint
        if aligned:
            return writer
        writer.close()
        replacement = self._new_persistent_durable_writer()
        self._durable_writer = replacement
        return replacement

    @staticmethod
    def _accept_worker_alignment_delta(
        writer: ConversationStore, entries: tuple[ConversationEntry, ...]
    ) -> bool:
        """Destructively advance a worker that will be discarded on rejection."""
        for entry in entries:
            if not writer._accept_incremental_entry(entry):
                return False
        writer._record_client_delivery_entries(entries)
        return True

    def _new_persistent_durable_writer(self: ConversationStore) -> ConversationStore:
        with self._durable_write_condition:
            prepared = self._durable_writer_snapshot
            self._durable_writer_snapshot = None
            if prepared is not None and prepared[0] != self._transcript_generation:
                prepared = None
        writer = type(self)(
            self.root_dir,
            session_id=self.session_id,
            _must_exist=True,
            _collect_persisted_appends=True,
            _validated_log_state=None if prepared is None else prepared[1],
            _storage_lease=self._storage_lease,
            _execution_lease=self.execution_lease,
        )
        return writer

    def _run_durable_write(
        self: ConversationStore,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> Any:
        """Run a write against isolated state; the resident store stays loop-owned."""
        writer: ConversationStore | None = None
        try:
            writer = self._durable_writer
            if writer is None:
                writer = self._new_persistent_durable_writer()
                self._durable_writer = writer
            writer = self._align_persistent_durable_writer(writer)
            writer_baseline = (
                len(writer._entries),
                writer._entry_prefix_digest,
            )
            full_reload_count = writer._full_reload_count
            result = getattr(writer, method_name)(*args, **kwargs)
            return (
                result,
                writer,
                writer.take_persisted_appends(),
                writer_baseline,
                writer._full_reload_count != full_reload_count,
            )
        except PendingPromptWriterReloadedError:
            raise
        except BaseException:
            if writer is not None:
                writer.close()
                with self._durable_write_condition:
                    if self._durable_writer is writer:
                        self._durable_writer = None
                        self._durable_writer_snapshot = None
            raise
        finally:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()

    def _load_detached_log_state(self: ConversationStore) -> _DetachedLogState:
        """Load one validated transcript snapshot without touching resident state."""
        reader = type(self)(
            self.root_dir,
            session_id=self.session_id,
            _read_only=True,
            _must_exist=True,
            _storage_lease=self._storage_lease,
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
            _read_only=True,
            _must_exist=True,
            _storage_lease=self._storage_lease,
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

    def _load_detached_log_state_with_delta(
        self: ConversationStore, baseline: tuple[int, bytes]
    ) -> _DetachedLogState:
        state = self._load_detached_log_state()
        return replace(
            state,
            installed_delta=self._installed_delta_from_baseline(
                state.entries, baseline
            ),
        )

    def _load_detached_store_state_with_delta(
        self: ConversationStore, baseline: tuple[int, bytes]
    ) -> _DetachedConversationState:
        state = self._load_detached_store_state()
        delta = self._installed_delta_from_baseline(state.log.entries, baseline)
        return replace(state, log=replace(state.log, installed_delta=delta))

    @classmethod
    def _worker_prefix_matches_baseline(
        cls,
        entries: list[ConversationEntry],
        writer_baseline: tuple[int, bytes],
        owner_baseline: tuple[int, bytes],
    ) -> bool:
        writer_count, digest = writer_baseline
        owner_count, expected_digest = owner_baseline
        if writer_count > owner_count or owner_count - writer_count > 64:
            return False
        for entry in entries[writer_count:owner_count]:
            digest = cls._next_entry_prefix_digest(digest, cls._encoded_entry_line(entry))
        return digest == expected_digest

    @staticmethod
    def _linear_suffix(entries: list[ConversationEntry], baseline_count: int) -> bool:
        previous = entries[baseline_count - 1] if baseline_count else None
        for entry in entries[baseline_count:]:
            expected_seq = previous.seq + 1 if previous is not None else 1
            expected_parent = previous.id if previous is not None else None
            if entry.seq != expected_seq or entry.parent_id != expected_parent:
                return False
            previous = entry
        return True

    def _proven_bounded_durable_delta(
        self: ConversationStore,
        method_name: str,
        writer: ConversationStore,
        receipts: tuple[PersistedAppend, ...] | None,
        writer_baseline: tuple[int, bytes],
        baseline: tuple[int, bytes],
        owner_log: tuple[tuple[int, int, int], int],
        reloaded: bool,
    ) -> _DurableLogDelta | None:
        owner_identity, owner_offset = owner_log
        unchanged_read_proven = (
            method_name in {"_stage_delivery_request", "_warm_step_delivery_writer"}
            and receipts == ()
            and not reloaded
            and writer_baseline == baseline
            and (len(writer._entries), writer._entry_prefix_digest) == writer_baseline
            and (writer._log_identity, writer._log_offset) == owner_log
            and writer._log_metadata_matches()
        )
        receipt_suffix_proven = bool(
            not reloaded
            and receipts
            and receipts[0].start_offset == owner_offset
            and (receipts[0].source_device, receipts[0].source_inode)
            == owner_identity[:2]
            and self._worker_prefix_matches_baseline(
                writer._entries, writer_baseline, baseline
            )
        )
        if not (unchanged_read_proven or receipt_suffix_proven):
            return None

        suffix = tuple(writer._entries[baseline[0] :])
        if len(suffix) > 64 or not self._linear_suffix(writer._entries, baseline[0]):
            return None
        return _DurableLogDelta(
            entries=self._owned_installed_delta(suffix),
            installed_delta=self._owned_installed_delta(suffix),
            entry_prefix_digest=writer._entry_prefix_digest,
            log_identity=writer._log_identity,
            log_offset=writer._log_offset,
            log_mtime_ns=writer._log_mtime_ns,
            log_prefix_fingerprint=writer._log_prefix_fingerprint,
        )

    def _run_durable_write_with_delta(
        self: ConversationStore,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        baseline: tuple[int, bytes],
        owner_log: tuple[tuple[int, int, int], int],
    ) -> tuple[Any, Any, Any]:
        result, writer, receipts, writer_baseline, reloaded = self._run_durable_write(
            method_name, args, kwargs
        )
        delta = self._proven_bounded_durable_delta(
            method_name,
            writer,
            receipts,
            writer_baseline,
            baseline,
            owner_log,
            reloaded,
        )
        if delta is not None:
            return result, delta, receipts

        state = writer._snapshot_log_state()
        delta = self._installed_delta_from_baseline(state.entries, baseline)
        if delta != "reset" and (
            len(delta) > 64 or not self._linear_suffix(state.entries, baseline[0])
        ):
            # Keep owner publication bounded; large or nonlinear suffixes use
            # one worker-owned state copy and an owner-thread pointer swap.
            delta = "reset"
        publication_delta = self._owned_installed_delta(delta)
        state = replace(
            state,
            installed_delta=delta,
            publication_delta=publication_delta,
        )
        if delta == "reset":
            state = copy.deepcopy(state)
        return result, state, receipts

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
        self._assert_owner_thread()
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
            if live_marker is not None and live_marker.get(
                "child_session_path"
            ) == loaded_marker.get("child_session_path"):
                loaded_marker["turns_used"] = max(
                    loaded_marker.get("turns_used", 0),
                    live_marker.get("turns_used", 0),
                )
        self._agent_children = state.agent_children

    def _install_store_state(
        self: ConversationStore,
        state: _DetachedConversationState,
        fence: _PublicationFence,
    ) -> tuple[_DetachedLogState, InstalledEntriesDelta]:
        """Publish unchanged recovery fields and preserve newer owner mutations."""
        installed_delta = ()
        if self._transcript_generation == fence.transcript_generation:
            released_log = self._install_log_state(state.log)
            installed_delta = state.log.installed_delta
        else:
            released_log = state.log

        self._install_agent_state(state, fence)
        return released_log, installed_delta

    def _publish_log_state(
        self: ConversationStore,
        state: _DetachedLogState | _DurableLogDelta,
        generation: int,
        receipts: Any,
    ) -> tuple[_DetachedLogState | _DurableLogDelta, InstalledEntriesDelta]:
        """Publish one durable write unless a newer owner snapshot is authoritative."""
        released_log: _DetachedLogState | _DurableLogDelta = state
        installed_delta: InstalledEntriesDelta = ()
        with self._durable_write_condition:
            if self._closing or self._closed:
                return released_log, installed_delta
            if self._transcript_generation == generation:
                installed_delta = state.installed_delta
                if isinstance(state, _DurableLogDelta):
                    entries = list(state.entries)
                    if entries:
                        self._before_incremental_state_install()
                        self._install_validated_linear_entries(entries)
                        self._record_client_delivery_entries(entries)
                        self._transcript_generation += 1
                elif state.publication_delta == "reset":
                    released_log = self._install_log_state(state)
                else:
                    entries = list(state.publication_delta)
                    if entries:
                        self._before_incremental_state_install()
                        self._install_validated_linear_entries(entries)
                        self._record_client_delivery_entries(entries)
                        self._transcript_generation += 1
                self._entry_prefix_digest = state.entry_prefix_digest
                self._log_identity = state.log_identity
                self._log_offset = state.log_offset
                self._log_mtime_ns = state.log_mtime_ns
                self._log_prefix_fingerprint = state.log_prefix_fingerprint
            self._accept_persisted_appends(receipts)
        return released_log, installed_delta

    @staticmethod
    def _release_detached_log_state(
        state: _DetachedLogState | _DurableLogDelta,
    ) -> None:
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
        self: ConversationStore, state: _DetachedLogState | _DurableLogDelta
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
            baseline = (len(self._entries), self._entry_prefix_digest)
            state = await loop.run_in_executor(
                None, self._load_detached_store_state_with_delta, baseline
            )
            installed_delta = ()
            with self._durable_write_condition:
                if self._closing or self._closed:
                    released_log = state.log
                else:
                    released_log, installed_delta = self._install_store_state(
                        state, fence
                    )
            cancelled = await self._release_log_state_off_loop(released_log)
            self._publish_prepared_entries_installed(installed_delta)
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
        baseline: tuple[int, bytes],
        owner_log: tuple[tuple[int, int, int], int],
    ) -> asyncio.Future[Any]:
        with self._durable_write_condition:
            if self._closing or self._closed:
                raise ValueError("session store is closed")
            self._durable_writes_in_flight += 1
        try:
            return asyncio.get_running_loop().run_in_executor(
                None,
                self._run_durable_write_with_delta,
                method_name,
                args,
                kwargs,
                baseline,
                owner_log,
            )
        except BaseException:
            with self._durable_write_condition:
                self._durable_writes_in_flight -= 1
                self._durable_write_condition.notify_all()
            raise

    def _run_agent_state_write(
        self: ConversationStore,
        method_name: str,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[Any, _DetachedConversationState]:
        writer = type(self)(
            self.root_dir,
            session_id=self.session_id,
            _must_exist=True,
            _execution_lease=self.execution_lease,
        )
        try:
            result = getattr(writer, method_name)(*args, **kwargs)
            state = _DetachedConversationState(
                log=writer._detach_log_state(),
                agent_counter=writer._agent_counter,
                agent_children=writer._agent_children,
                agent_parent=writer._agent_parent,
                agent_canceled=writer._agent_canceled,
                agent_lifecycle=writer._agent_lifecycle,
            )
            return result, state
        finally:
            writer.close()

    async def _to_thread_agent_state(
        self: ConversationStore, function: Any, /, *args: Any, **kwargs: Any
    ) -> Any:
        """Commit one agent-state mutation off-loop and publish its full state."""

        async with self._async_write_lock:
            fence = self._publication_fence()
            worker = asyncio.get_running_loop().run_in_executor(
                None,
                self._run_agent_state_write,
                function.__name__,
                args,
                kwargs,
            )
            (result, state), cancelled = await self._await_started_worker(worker)
            with self._durable_write_condition:
                released_log, installed_delta = self._install_store_state(state, fence)
            cancelled |= await self._release_log_state_off_loop(released_log)
            self._publish_prepared_entries_installed(installed_delta)
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
        """Serialize one blocking commit and one bounded owner publication."""
        async with self._async_write_lock:
            method_name = function.__name__
            if not hasattr(type(self), method_name):
                # Preserve instance-level instrumentation without mutating resident
                # state from a worker thread.
                return function(*args, **kwargs)
            generation = self._transcript_generation
            baseline = (len(self._entries), self._entry_prefix_digest)
            owner_log = (self._log_identity, self._log_offset)
            worker = self._start_durable_write(
                method_name, args, kwargs, baseline, owner_log
            )
            outcome, cancelled = await self._await_started_worker(worker)
            result, state, receipts = outcome
            if self._transcript_generation == generation:
                released_log, installed_delta = self._publish_log_state(
                    state, generation, receipts
                )
                cancelled |= await self._release_log_state_off_loop(released_log)
            else:
                reload_generation = self._transcript_generation
                reload_baseline = (len(self._entries), self._entry_prefix_digest)
                reload_worker = asyncio.get_running_loop().run_in_executor(
                    None,
                    self._load_detached_log_state_with_delta,
                    reload_baseline,
                )
                reloaded, reload_cancelled = await self._await_started_worker(
                    reload_worker
                )
                cancelled |= reload_cancelled
                with self._durable_write_condition:
                    installed_delta = ()
                    if self._closing or self._closed:
                        released_reload = reloaded
                    else:
                        if self._transcript_generation == reload_generation:
                            released_reload = self._install_log_state(reloaded)
                            installed_delta = reloaded.installed_delta
                        else:
                            # A later owner write synchronized this completed commit.
                            released_reload = reloaded
                        self._accept_persisted_appends(receipts)
                cancelled |= await self._release_log_state_off_loop(state)
                cancelled |= await self._release_log_state_off_loop(released_reload)
            try:
                if _on_persisted is not None:
                    _on_persisted()
            finally:
                self._publish_prepared_entries_installed(installed_delta)
            if cancelled:
                raise asyncio.CancelledError
            return result
