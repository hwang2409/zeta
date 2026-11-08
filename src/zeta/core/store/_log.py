"""Incremental append-only conversation log synchronization.

The log contract permits only appends by cooperating Zeta writers under the
session flock. Mutation detection is intentionally bounded: it detects file
replacement, truncation, and rewrites in the most recent 64 KiB, but an in-place
rewrite earlier in the prefix followed by growth is not detected by a resident
store. Reopen the session to force a full load after unsupported external edits.
"""

from __future__ import annotations

import copy
import fcntl
import hashlib
import json
import logging
import os
import threading
import time
import warnings
from collections import OrderedDict, deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any, Literal

from ...agent.receipt import encode_json
from ...storage_format import WRITER_EPOCH
from ..checkpoints import (
    CONVERSATION_ENTRY_TYPES,
    ConversationEntry,
    ConversationIntegrityError,
    _now,
    load_session_json,
)
from ..checkpoints.entry_types import sent_message_receipts
from ..session_files import open_session_file
from ..transcript_recovery import (
    TRANSCRIPT_REPAIR_EPOCH,
    TranscriptSourceFacts,
    ValidatedPrefix,
    reconcile_pending,
    repair_transcript,
)
from ._validation import TASK_EXITED_NOTIFICATION_KIND

if TYPE_CHECKING:
    from ...protocol.types import Message, ToolResult
    from ._client_delivery import ClientDelivery
    from ._store import ConversationStore


@dataclass(frozen=True, slots=True)
class PersistedAppend:
    """Proof of one byte range durably appended by this store process."""

    start_offset: int
    end_offset: int
    digest: str
    source_device: int
    source_inode: int
    before_mtime_ns: int
    before_ctime_ns: int
    after_mtime_ns: int
    after_ctime_ns: int


SCHEMA = "zeta.conversation.v1"
PREFIX_FINGERPRINT_BYTES = 64 * 1024
MAX_PENDING_APPEND_RECEIPTS = 256
type InstalledEntriesDelta = tuple[ConversationEntry, ...] | Literal["reset"]

_LOGGER = logging.getLogger(__name__)


TranscriptFailureKind = Literal[
    "torn_tail",
    "invalid_row",
    "broken_links",
    "unsupported_format",
    "unrecoverable_header",
]


class ObserverSnapshotChanged(ConversationIntegrityError):
    """The observer could not validate one authoritative source view."""


class TranscriptLoadError(ConversationIntegrityError):
    """A typed transcript refusal that never exposes persisted content or paths."""

    def __init__(
        self,
        *,
        kind: TranscriptFailureKind,
        session_id: str,
        row_number: int | None,
        byte_offset: int | None,
        valid_prefix_end: int | None,
        reason_code: str,
        repairable: bool,
        repair_available: bool,
        source_device: int,
        source_inode: int,
        source_size: int,
        prefix_digest: str,
        child_run_id: str | None = None,
        source_facts: TranscriptSourceFacts | None = None,
    ) -> None:
        self.kind = kind
        self.session_id = session_id
        self.row_number = row_number
        self.byte_offset = byte_offset
        self.valid_prefix_end = valid_prefix_end
        self.reason_code = reason_code
        self.repairable = repairable
        self.repair_available = repair_available
        self.source_device = source_device
        self.source_inode = source_inode
        self.source_size = source_size
        self.prefix_digest = prefix_digest
        self.child_run_id = child_run_id
        self._source_facts = source_facts
        location = f" at row {row_number}" if row_number is not None else ""
        detail = {
            "invalid_payload": "invalid payload for conversation entry",
            "missing_parent": "missing prior parent",
            "duplicate_id": "duplicate conversation id",
            "orphaned_root": "orphaned root",
            "invalid_sequence": "non-monotonic conversation sequence",
            "unlinked_approval_resolution": "approval resolution is not linked",
            "header_session_mismatch": "conversation header session id mismatch",
            "invalid_json": (
                "terminated invalid conversation row"
                if kind == "invalid_row"
                else kind.replace("_", " ")
            ),
        }.get(reason_code)
        label = detail or kind.replace("_", " ")
        super().__init__(f"transcript {label}{location} (session {session_id})")

    def protocol_data(self) -> dict[str, object]:
        """Return the bounded public diagnosis used by every serve error path."""
        return {
            "code": "corrupt_transcript",
            "kind": self.kind,
            "row": self.row_number,
            "repairable": self.repairable,
            "session_id": self.session_id,
            "actions": (
                [
                    {
                        "method": "repair_transcript",
                        "session_id": self.session_id,
                        **(
                            {"child_run_id": self.child_run_id}
                            if self.child_run_id is not None
                            else {}
                        ),
                        "requires_confirmation": True,
                    }
                ]
                if self.repair_available
                else []
            ),
        }


@dataclass(frozen=True, slots=True)
class _DetachedLogState:
    """Validated transcript state that is safe to install without file access."""

    cwd: str
    entries: list[ConversationEntry]
    entry_ids: set[str]
    entry_prefix_digest: bytes
    active_approval_requests: dict[str, ConversationEntry]
    active_approval_resolutions: dict[str, str]
    active_tool_results: dict[str, ToolResult]
    active_agent_call_ids: set[str]
    active_agent_receipts: list[tuple[str, str]]
    notification_ids: set[str]
    active_notifications: dict[str, ConversationEntry]
    active_completion_notifications: dict[str, ConversationEntry]
    active_notification_presentations: set[str]
    pending_prompt_ids: set[str]
    active_pending_prompt_entries: dict[str, ConversationEntry]
    active_pending_prompt_messages: dict[str, Message]
    active_notification_delivery_message: Message | None
    active_last_message: Message | None
    active_last_turn_end: ConversationEntry | None
    task_notification_ids: set[str]
    sent_message_receipts: list[tuple[str, str]]
    client_deliveries: OrderedDict[str, ClientDelivery]
    client_delivery_outcomes: dict[bytes, bytes]
    log_identity: tuple[int, int, int]
    log_offset: int
    log_mtime_ns: int
    log_prefix_fingerprint: bytes
    installed_delta: InstalledEntriesDelta = ()
    publication_delta: InstalledEntriesDelta = ()


@dataclass(frozen=True, slots=True)
class _PhysicalRow:
    row_number: int
    byte_offset: int
    value: dict[str, Any] | None
    torn: bool = False


class _FullLoadValidator:
    """Validate rows in order and retain every index needed by the loaded store."""

    _STATE_ATTRIBUTES = (
        "cwd",
        "_entries",
        "_entry_ids",
        "_entry_prefix_digest",
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

    def __init__(
        self, store: ConversationStore, raw: bytes, stat: os.stat_result
    ) -> None:
        self.store = store
        self.raw = raw
        self.stat = stat
        self.state = copy.copy(store)
        self.state._entries = []
        self.state._entry_prefix_digest = b""
        self.state._reset_incremental_validation_state()
        self.state._task_notification_ids = set()
        self.state._sent_message_receipts = []
        self.by_id: dict[str, ConversationEntry] = {}
        self.rows_by_id: dict[str, _PhysicalRow] = {}
        self.header_seen = False

    def _error(
        self,
        row: _PhysicalRow,
        *,
        kind: TranscriptFailureKind,
        reason_code: str,
        repairable: bool = True,
    ) -> TranscriptLoadError:
        return self.store._transcript_error(
            kind=kind,
            row_number=row.row_number,
            byte_offset=row.byte_offset,
            valid_prefix_end=None if row.row_number == 1 else row.byte_offset,
            reason_code=reason_code,
            repairable=repairable,
            raw=self.raw,
            stat=self.stat,
            validated_prefix=(
                self.validated_prefix(row.byte_offset) if row.row_number > 1 else None
            ),
        )

    def accept(self, row: _PhysicalRow) -> ConversationEntry | None:
        value = row.value
        assert value is not None
        if row.row_number == 1:
            self.header_seen = True
            try:
                self.state._validate_header(value)
            except ConversationIntegrityError as exc:
                data = value.get("data")
                unsupported = (
                    value.get("type") == "header"
                    and type(data) is dict
                    and type(data.get("schema")) is str
                    and data.get("schema") != SCHEMA
                )
                mismatch = (
                    type(data) is dict
                    and data.get("schema") == SCHEMA
                    and data.get("session_id") != self.store.session_id
                )
                raise self._error(
                    row,
                    kind="unsupported_format"
                    if unsupported
                    else "unrecoverable_header",
                    reason_code=(
                        "unsupported_schema"
                        if unsupported
                        else "header_session_mismatch"
                        if mismatch
                        else "invalid_header"
                    ),
                    repairable=False,
                ) from exc
            return None

        entry_type = value.get("type")
        if type(entry_type) is str and entry_type not in CONVERSATION_ENTRY_TYPES:
            try:
                ConversationEntry.from_dict({**value, "type": "warning"})
            except ConversationIntegrityError:
                pass
            else:
                raise self._error(
                    row,
                    kind="unsupported_format",
                    reason_code="unknown_entry_type",
                    repairable=False,
                )
        try:
            entry = ConversationEntry.from_dict(value)
        except (ConversationIntegrityError, KeyError, TypeError, ValueError) as exc:
            raise self._error(
                row, kind="invalid_row", reason_code="invalid_common_fields"
            ) from exc
        try:
            self.state._validate_entry_payload(entry)
            if entry.type == "fork":
                self.state._validate_fork_entry(entry)
        except ConversationIntegrityError as exc:
            raise self._error(
                row, kind="invalid_row", reason_code="invalid_payload"
            ) from (exc.__cause__ or exc)

        reason_code = None
        if entry.seq != row.row_number - 1:
            reason_code = "invalid_sequence"
        elif entry.id in self.by_id:
            reason_code = "duplicate_id"
        elif self.state._entries and entry.parent_id is None:
            reason_code = "orphaned_root"
        elif entry.parent_id is not None and entry.parent_id not in self.by_id:
            reason_code = "missing_parent"
        if reason_code is not None:
            raise self._error(row, kind="broken_links", reason_code=reason_code)

        self.state._entries.append(entry)
        self.by_id[entry.id] = entry
        self.rows_by_id[entry.id] = row
        self.state._entry_ids.add(entry.id)
        self.state._record_entry_prefix(entry)
        self.state._sent_message_receipts.extend(sent_message_receipts(entry))
        if (
            entry.type == "notification"
            and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND
        ):
            task_id = entry.data.get("task_id")
            if type(task_id) is str and task_id:
                self.state._task_notification_ids.add(task_id)
        return entry

    def validated_prefix(self, end_offset: int) -> ValidatedPrefix:
        """Return the exact prefix already accepted by this one-pass validator."""
        last = next(
            (
                entry
                for entry in reversed(self.state._entries)
                if self.rows_by_id[entry.id].byte_offset < end_offset
            ),
            None,
        )
        return ValidatedPrefix(
            end_offset=end_offset,
            last_sequence=last.seq if last is not None else 0,
            last_entry_id=last.id if last is not None else None,
            last_lane=last.lane if last is not None else "main",
        )

    def _select_branch(self, head: ConversationEntry) -> None:
        entry_ids = set(self.by_id)
        self.state._reset_incremental_validation_state()
        self.state._entry_ids = entry_ids
        branch: list[ConversationEntry] = []
        current: ConversationEntry | None = head
        while current is not None:
            branch.append(current)
            current = self.by_id.get(current.parent_id) if current.parent_id else None
        for entry in reversed(branch):
            try:
                self.state._record_active_entry(entry)
            except ConversationIntegrityError as exc:
                raise self._error(
                    self.rows_by_id[entry.id],
                    kind="broken_links",
                    reason_code=self._semantic_reason(exc),
                ) from exc

    @staticmethod
    def _semantic_reason(error: ConversationIntegrityError) -> str:
        message = str(error)
        prefixes = (
            ("standalone approval requests", "standalone_approval_request"),
            ("approval resolution is not linked", "unlinked_approval_resolution"),
            ("duplicate approval request", "duplicate_approval_request"),
            ("duplicate approval resolution", "duplicate_approval_resolution"),
            ("notification acknowledgement is not linked", "unlinked_notification_ack"),
            ("duplicate notification acknowledgement", "duplicate_notification_ack"),
            (
                "notification TUI presentation is not linked",
                "unlinked_notification_presentation",
            ),
            (
                "duplicate notification TUI presentation",
                "duplicate_notification_presentation",
            ),
            (
                "pending prompt acknowledgement is not linked",
                "unlinked_pending_prompt_ack",
            ),
            (
                "duplicate pending prompt acknowledgement",
                "duplicate_pending_prompt_ack",
            ),
        )
        return next(
            (code for prefix, code in prefixes if message.startswith(prefix)),
            "invalid_active_branch_links",
        )

    def finish(self, *, install: bool = True) -> None:
        if not self.header_seen:
            raise self.store._transcript_error(
                kind="unrecoverable_header",
                row_number=None,
                byte_offset=None,
                valid_prefix_end=None,
                reason_code="missing_header",
                repairable=False,
                raw=self.raw,
                stat=self.stat,
            )
        if self.state._entries:
            self._select_branch(self.state._entries[-1])
        self.state._rebuild_client_deliveries(self.state._entries)
        if install:
            for attribute in self._STATE_ATTRIBUTES:
                setattr(self.store, attribute, getattr(self.state, attribute))


class ConversationLogMixin:
    """Load a log once, then synchronize only bytes appended by other writers."""

    def _initialize_entries_installed(self: ConversationStore) -> None:
        self.on_entries_installed: Callable[[InstalledEntriesDelta], None] | None = None
        self._installed_callback_lock = threading.Lock()
        self._installed_callback_pending: deque[InstalledEntriesDelta] = deque()
        self._installed_callback_draining = False
        self._installed_callback_local = threading.local()
        self._full_reload_count = 0

    def _detach_log_state(self: ConversationStore) -> _DetachedLogState:
        """Return the worker-owned validated state for an atomic loop install."""
        return _DetachedLogState(
            cwd=self.cwd,
            entries=self._entries,
            entry_ids=self._entry_ids,
            entry_prefix_digest=self._entry_prefix_digest,
            active_approval_requests=self._active_approval_requests,
            active_approval_resolutions=self._active_approval_resolutions,
            active_tool_results=self._active_tool_results,
            active_agent_call_ids=self._active_agent_call_ids,
            active_agent_receipts=self._active_agent_receipts,
            notification_ids=self._notification_ids,
            active_notifications=self._active_notifications,
            active_completion_notifications=self._active_completion_notifications,
            active_notification_presentations=self._active_notification_presentations,
            pending_prompt_ids=self._pending_prompt_ids,
            active_pending_prompt_entries=self._active_pending_prompt_entries,
            active_pending_prompt_messages=self._active_pending_prompt_messages,
            active_notification_delivery_message=(
                self._active_notification_delivery_message
            ),
            active_last_message=self._active_last_message,
            active_last_turn_end=self._active_last_turn_end,
            task_notification_ids=self._task_notification_ids,
            sent_message_receipts=self._sent_message_receipts,
            client_deliveries=self._client_deliveries,
            client_delivery_outcomes=self._client_delivery_outcomes,
            log_identity=self._log_identity,
            log_offset=self._log_offset,
            log_mtime_ns=self._log_mtime_ns,
            log_prefix_fingerprint=self._log_prefix_fingerprint,
        )

    def _snapshot_log_state(self: ConversationStore) -> _DetachedLogState:
        """Copy mutable indexes so a retained writer can keep its own state."""
        state = self._detach_log_state()
        return replace(
            state,
            entries=list(state.entries),
            entry_ids=set(state.entry_ids),
            active_approval_requests=dict(state.active_approval_requests),
            active_approval_resolutions=dict(state.active_approval_resolutions),
            active_tool_results=dict(state.active_tool_results),
            active_agent_call_ids=set(state.active_agent_call_ids),
            active_agent_receipts=list(state.active_agent_receipts),
            notification_ids=set(state.notification_ids),
            active_notifications=dict(state.active_notifications),
            active_completion_notifications=dict(state.active_completion_notifications),
            active_notification_presentations=set(
                state.active_notification_presentations
            ),
            pending_prompt_ids=set(state.pending_prompt_ids),
            active_pending_prompt_entries=dict(state.active_pending_prompt_entries),
            active_pending_prompt_messages=dict(state.active_pending_prompt_messages),
            task_notification_ids=set(state.task_notification_ids),
            sent_message_receipts=list(state.sent_message_receipts),
            client_deliveries=state.client_deliveries.copy(),
            client_delivery_outcomes=dict(state.client_delivery_outcomes),
        )

    @staticmethod
    def _encoded_entry_line(entry: ConversationEntry) -> bytes:
        return encode_json(entry.to_dict()) + b"\n"

    @staticmethod
    def _next_entry_prefix_digest(previous: bytes, encoded_line: bytes) -> bytes:
        return hashlib.sha256(previous + encoded_line).digest()

    def _record_entry_prefix(self: ConversationStore, entry: ConversationEntry) -> None:
        self._entry_prefix_digest = self._next_entry_prefix_digest(
            self._entry_prefix_digest, self._encoded_entry_line(entry)
        )

    @staticmethod
    def _owned_installed_delta(
        delta: InstalledEntriesDelta,
    ) -> InstalledEntriesDelta:
        if delta == "reset":
            return delta
        return copy.deepcopy(delta)

    def _deliver_entries_installed(
        self: ConversationStore, delta: InstalledEntriesDelta
    ) -> None:
        assert not getattr(self._installed_callback_local, "flock_held", False)
        callback = self.on_entries_installed
        if not delta or callback is None:
            return
        try:
            callback(delta)
        except Exception:
            _LOGGER.exception("installed-row callback failed")

    def _start_entries_installed(self: ConversationStore) -> None:
        self._installed_callback_local.defer = True
        self._installed_callback_local.flock_held = True

    def _finish_entries_installed(self: ConversationStore) -> None:
        self._installed_callback_local.defer = False
        self._installed_callback_local.flock_held = False
        if getattr(self._installed_callback_local, "drain_after_unlock", False):
            del self._installed_callback_local.drain_after_unlock
            self._drain_entries_installed()

    def _drain_entries_installed(self: ConversationStore) -> None:
        failure: BaseException | None = None
        while True:
            with self._installed_callback_lock:
                if not self._installed_callback_pending:
                    self._installed_callback_draining = False
                    break
                delta = self._installed_callback_pending.popleft()
            try:
                self._deliver_entries_installed(delta)
            except BaseException as exc:  # noqa: BLE001 - propagate after FIFO cleanup
                if failure is None:
                    failure = exc
        if failure is not None:
            raise failure

    def _publish_prepared_entries_installed(
        self: ConversationStore, delta: InstalledEntriesDelta
    ) -> None:
        """Enqueue one owned delta and let one store-wide drainer deliver it."""
        if not delta or self.on_entries_installed is None:
            return
        should_drain = False
        with self._installed_callback_lock:
            self._installed_callback_pending.append(delta)
            if not self._installed_callback_draining:
                self._installed_callback_draining = True
                should_drain = True
        if not should_drain:
            return
        if getattr(self._installed_callback_local, "defer", False):
            self._installed_callback_local.drain_after_unlock = True
        else:
            self._drain_entries_installed()

    def _publish_entries_installed(
        self: ConversationStore, delta: InstalledEntriesDelta
    ) -> None:
        """Copy and publish only newly installed rows."""
        if not delta or self.on_entries_installed is None:
            return
        self._publish_prepared_entries_installed(self._owned_installed_delta(delta))

    @classmethod
    def _installed_delta_from_baseline(
        cls,
        entries: list[ConversationEntry],
        baseline: tuple[int, bytes],
    ) -> InstalledEntriesDelta:
        """Prepare an owned suffix, or reset when the old log is not a prefix."""
        count, expected_digest = baseline
        if count > len(entries):
            return "reset"
        digest = b""
        for index in range(count):
            digest = cls._next_entry_prefix_digest(
                digest, cls._encoded_entry_line(entries[index])
            )
        if digest != expected_digest:
            return "reset"
        return cls._owned_installed_delta(tuple(entries[count:]))

    def _install_log_state(
        self: ConversationStore, state: _DetachedLogState
    ) -> _DetachedLogState:
        """Install validated state on the owner thread using pointer swaps only."""
        self._assert_owner_thread()
        previous = self._detach_log_state() if hasattr(self, "_entry_ids") else state
        self._before_incremental_state_install()
        self.cwd = state.cwd
        self._entry_ids = state.entry_ids
        self._entry_prefix_digest = state.entry_prefix_digest
        self._active_approval_requests = state.active_approval_requests
        self._active_approval_resolutions = state.active_approval_resolutions
        self._active_tool_results = state.active_tool_results
        self._active_agent_call_ids = state.active_agent_call_ids
        self._active_agent_receipts = state.active_agent_receipts
        self._notification_ids = state.notification_ids
        self._active_notifications = state.active_notifications
        self._active_completion_notifications = state.active_completion_notifications
        self._active_notification_presentations = (
            state.active_notification_presentations
        )
        self._pending_prompt_ids = state.pending_prompt_ids
        self._active_pending_prompt_entries = state.active_pending_prompt_entries
        self._active_pending_prompt_messages = state.active_pending_prompt_messages
        self._active_notification_delivery_message = (
            state.active_notification_delivery_message
        )
        self._active_last_message = state.active_last_message
        self._active_last_turn_end = state.active_last_turn_end
        self._task_notification_ids = state.task_notification_ids
        self._sent_message_receipts = state.sent_message_receipts
        self._client_deliveries = state.client_deliveries
        self._client_delivery_outcomes = state.client_delivery_outcomes
        self._log_identity = state.log_identity
        self._log_offset = state.log_offset
        self._log_mtime_ns = state.log_mtime_ns
        self._log_prefix_fingerprint = state.log_prefix_fingerprint
        # Publish entries last so readers never see rows without matching indexes.
        self._entries = state.entries
        self._transcript_generation += 1
        return previous

    def enable_persisted_append_tracking(self: ConversationStore) -> None:
        """Collect bounded append proofs for incremental transcript indexing."""
        self._collect_persisted_appends = True

    def disable_persisted_append_tracking(self: ConversationStore) -> None:
        """Stop collecting append proofs and discard pending tracking state."""
        self._collect_persisted_appends = False
        self._persisted_appends.clear()
        self._persisted_appends_unverified = False

    def _record_persisted_append(
        self: ConversationStore, receipt: PersistedAppend
    ) -> None:
        if not self._collect_persisted_appends or self._persisted_appends_unverified:
            return
        if len(self._persisted_appends) >= MAX_PENDING_APPEND_RECEIPTS:
            self._persisted_appends.clear()
            self._persisted_appends_unverified = True
            return
        self._persisted_appends.append(receipt)

    def _accept_persisted_appends(
        self: ConversationStore, receipts: tuple[PersistedAppend, ...] | None
    ) -> None:
        if not self._collect_persisted_appends:
            return
        if receipts is None:
            self._persisted_appends.clear()
            self._persisted_appends_unverified = True
            return
        for receipt in receipts:
            self._record_persisted_append(receipt)

    def take_persisted_appends(
        self: ConversationStore,
    ) -> tuple[PersistedAppend, ...] | None:
        """Transfer append proofs, or report that the pending range is unverified."""
        if self._persisted_appends_unverified:
            self._persisted_appends_unverified = False
            return None
        receipts = tuple(self._persisted_appends)
        self._persisted_appends.clear()
        return receipts

    def _log_metadata_matches(self: ConversationStore) -> bool:
        """Return whether the durable log still matches the resident indexes."""
        try:
            fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        except FileNotFoundError:
            return False
        try:
            stat = os.fstat(fd)
        finally:
            os.close(fd)

        known_identity = getattr(self, "_log_identity", None)
        return (
            known_identity == (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
            and getattr(self, "_log_offset", None) == stat.st_size
            and getattr(self, "_log_mtime_ns", None) == stat.st_mtime_ns
        )

    def _sync_log_without_waiting(self: ConversationStore) -> None:
        """Synchronize changed log metadata unless another writer owns it."""
        if self._observer_snapshot:
            self._sync_observer_snapshot()
            return
        if self._log_metadata_matches():
            return
        with self._append_lock(wait=False, observer=self._read_only) as acquired:
            if not acquired:
                # A writer can expose a new size before releasing the lock. A
                # later query synchronizes the append after the writer exits.
                return

    def _sync_observer_snapshot(
        self: ConversationStore, *, retry_source_change: bool = True
    ) -> None:
        """Install only a validated prefix from one authoritative source view."""
        fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        try:
            g0 = os.fstat(fd)
            raw = self._read_fd(fd, g0.st_size)
        finally:
            os.close(fd)
        complete_end = raw.rfind(b"\n") + 1
        if (
            complete_end == g0.st_size
            and complete_end == getattr(self, "_log_offset", 0)
            and complete_end
            and self._known_prefix_matches_bytes(raw, complete_end)
        ):
            self._check_observer_source(g0, retry_source_change)
            return
        candidate = raw[:complete_end]
        baseline = (len(self._entries), self._entry_prefix_digest)
        validator = _FullLoadValidator(self, candidate, g0)
        validation_error: ConversationIntegrityError | None = None
        try:
            for physical_row in self._iter_physical_rows(
                candidate, g0, validator.validated_prefix
            ):
                validator.accept(physical_row)
            validator.finish(install=False)
        except ConversationIntegrityError as exc:
            validation_error = exc
        has_suffix = complete_end < g0.st_size
        full_validator: _FullLoadValidator | None = None
        full_validation_error: ConversationIntegrityError | None = None
        if has_suffix:
            full_validator = _FullLoadValidator(self, raw, g0)
            try:
                for physical_row in self._iter_physical_rows(
                    raw, g0, full_validator.validated_prefix
                ):
                    full_validator.accept(physical_row)
                full_validator.finish(install=False)
            except ConversationIntegrityError as exc:
                full_validation_error = exc
        append_active = has_suffix and self._observer_append_active()
        if self._check_observer_source(g0, retry_source_change):
            return
        if validation_error is not None:
            raise validation_error
        if has_suffix and not append_active:
            if full_validation_error is not None:
                raise full_validation_error
            assert full_validator is not None
            validator = full_validator
            candidate = raw
        for attribute in _FullLoadValidator._STATE_ATTRIBUTES:
            setattr(self, attribute, getattr(validator.state, attribute))
        delta = (
            self._installed_delta_from_baseline(self._entries, baseline)
            if self.on_entries_installed is not None
            else ()
        )
        self._set_observer_log_stat(g0, complete_end, raw)
        self._publish_entries_installed(delta)

    def _check_observer_source(
        self: ConversationStore,
        g0: os.stat_result,
        retry_source_change: bool,
    ) -> bool:
        fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        try:
            g1 = os.fstat(fd)
        finally:
            os.close(fd)
        if (g0.st_dev, g0.st_ino, g0.st_ctime_ns, g0.st_size, g0.st_mtime_ns) == (
            g1.st_dev,
            g1.st_ino,
            g1.st_ctime_ns,
            g1.st_size,
            g1.st_mtime_ns,
        ):
            return False
        replaced = (g0.st_dev, g0.st_ino) != (g1.st_dev, g1.st_ino)
        if retry_source_change and (not self._entries or not replaced):
            self._sync_observer_snapshot(retry_source_change=False)
            return True
        raise ObserverSnapshotChanged("observer transcript changed during read")

    def _observer_append_active(self: ConversationStore) -> bool:
        fd = open_session_file(self.directory_fd, ".lock", os.O_RDWR)
        try:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(fd, fcntl.LOCK_UN)
                return False
        finally:
            os.close(fd)

    def _known_prefix_matches_bytes(self: ConversationStore, raw: bytes, offset: int) -> bool:
        known = getattr(self, "_log_prefix_fingerprint", None)
        if known is None:
            return False
        length = min(offset, PREFIX_FINGERPRINT_BYTES)
        return hashlib.blake2b(raw[offset - length : offset], digest_size=16).digest() == known

    def _set_observer_log_stat(
        self: ConversationStore, stat: os.stat_result, offset: int, raw: bytes
    ) -> None:
        length = min(offset, PREFIX_FINGERPRINT_BYTES)
        self._log_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        self._log_offset = offset
        self._log_mtime_ns = stat.st_mtime_ns
        self._log_prefix_fingerprint = hashlib.blake2b(
            raw[offset - length : offset], digest_size=16
        ).digest()
        self._transcript_generation += 1

    def _load(self: ConversationStore) -> None:
        """Synchronize in-memory state with the durable log.

        Callers hold the session append flock unless the store is read-only.
        Replacements, truncations, and non-linear tails fall back to a complete
        parse; ordinary external appends read and validate only the new bytes.
        """
        self._assert_owner_thread()
        had_entries = bool(self._entries)
        try:
            fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        except FileNotFoundError:
            if self._read_only or self._must_exist:
                raise ConversationIntegrityError(
                    f"conversation file is missing: {self.path}"
                )
            self._entries = []
            self._entry_prefix_digest = b""
            self._sent_message_receipts = []
            self._reset_incremental_validation_state()
            self._rebuild_client_deliveries(self._entries)
            header = {
                "schema": SCHEMA,
                "session_id": self.session_id,
                "cwd": self.cwd,
                "created_at": _now(),
            }
            self._write_line({"type": "header", "data": header})
            if had_entries:
                self._publish_entries_installed("reset")
            return

        try:
            stat = os.fstat(fd)
            identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
            known_identity = getattr(self, "_log_identity", None)
            known_offset = getattr(self, "_log_offset", 0)
            known_mtime = getattr(self, "_log_mtime_ns", None)
            replaced = known_identity is None or known_identity[:2] != identity[:2]
            metadata_changed = (
                known_identity != identity or stat.st_mtime_ns != known_mtime
            )
            if replaced or stat.st_size < known_offset:
                os.close(fd)
                fd = -1
                self._load_full()
                return
            if stat.st_size == known_offset:
                if metadata_changed:
                    os.close(fd)
                    fd = -1
                    self._load_full()
                return
            if not self._known_prefix_matches(fd, known_offset):
                os.close(fd)
                fd = -1
                self._load_full()
                return
            os.lseek(fd, known_offset, os.SEEK_SET)
            raw = self._read_fd(fd, stat.st_size - known_offset)
        finally:
            if fd >= 0:
                os.close(fd)

        try:
            entries, torn_at = self._parse_entry_lines(
                raw, first_row=len(self._entries) + 2
            )
        except ConversationIntegrityError:
            self._load_full()
            return
        if torn_at is not None:
            if self._read_only:
                raise ConversationIntegrityError(
                    f"invalid conversation row {len(self._entries) + len(entries) + 2}: {self.path}"
                )
            self._strict_snapshot_sync = False
            self._load_full()
            return

        try:
            accepted = self._accept_incremental_entries(entries)
        except ConversationIntegrityError:
            self._load_full()
            return
        if not accepted:
            self._load_full()
            return
        if not self._read_only and not raw.endswith(b"\n"):
            # Preserve the loader's historical recovery for a complete JSON row
            # whose writer omitted only the line terminator.
            self._write_bytes(b"\n")
        else:
            self._set_log_stat(stat)
        self._publish_entries_installed(tuple(entries))

    def _transcript_error(
        self: ConversationStore,
        *,
        kind: TranscriptFailureKind,
        row_number: int | None,
        byte_offset: int | None,
        valid_prefix_end: int | None,
        reason_code: str,
        repairable: bool,
        raw: bytes,
        stat: os.stat_result,
        validated_prefix: ValidatedPrefix | None = None,
    ) -> TranscriptLoadError:
        prefix_end = valid_prefix_end or 0
        return TranscriptLoadError(
            kind=kind,
            session_id=self._repair_target.root_session_id,
            row_number=row_number,
            byte_offset=byte_offset,
            valid_prefix_end=valid_prefix_end,
            reason_code=reason_code,
            repairable=repairable,
            repair_available=(
                repairable
                and self._storage_lease.observed_epoch >= TRANSCRIPT_REPAIR_EPOCH
            ),
            source_device=stat.st_dev,
            source_inode=stat.st_ino,
            source_size=stat.st_size,
            prefix_digest=hashlib.sha256(raw[:prefix_end]).hexdigest(),
            child_run_id=self._repair_target.child_run_id,
            source_facts=(
                TranscriptSourceFacts(
                    device=stat.st_dev,
                    inode=stat.st_ino,
                    size=stat.st_size,
                    source_digest=hashlib.sha256(raw).hexdigest(),
                    prefix_digest=hashlib.sha256(raw[:prefix_end]).hexdigest(),
                    first_bad_row=row_number,
                    first_bad_offset=byte_offset,
                    prefix=validated_prefix,
                    diagnosed_at=_now(),
                )
                if repairable
                and row_number is not None
                and byte_offset is not None
                and validated_prefix is not None
                else None
            ),
        )

    @staticmethod
    def _recovery_prefix_for_error(
        validated_prefix: Callable[[int], ValidatedPrefix] | None,
        row_offset: int,
        *,
        repairable: bool,
    ) -> ValidatedPrefix | None:
        if not repairable or validated_prefix is None:
            return None
        return validated_prefix(row_offset)

    def _iter_physical_rows(
        self: ConversationStore,
        raw: bytes,
        stat: os.stat_result,
        validated_prefix: Callable[[int], ValidatedPrefix] | None = None,
    ) -> Iterator[_PhysicalRow]:
        """Yield decoded physical rows and one marker for a writable torn tail."""
        offset = 0
        lines = raw.splitlines(keepends=True)
        for index, line in enumerate(lines):
            if index and index % 64 == 0:
                time.sleep(0.0001)
            row_number = index + 1
            row_offset = offset
            try:
                value = load_session_json(line)
            except ConversationIntegrityError as exc:
                torn = (
                    index == len(lines) - 1
                    and not line.endswith(b"\n")
                    and isinstance(exc.__cause__, (json.JSONDecodeError, UnicodeError))
                )
                if torn and not self._read_only:
                    yield _PhysicalRow(row_number, row_offset, None, torn=True)
                    return
                raise self._transcript_error(
                    kind=(
                        "unrecoverable_header"
                        if index == 0
                        else "torn_tail"
                        if torn
                        else "invalid_row"
                    ),
                    row_number=row_number,
                    byte_offset=row_offset,
                    valid_prefix_end=None if index == 0 else row_offset,
                    reason_code="invalid_json",
                    repairable=index > 0,
                    raw=raw,
                    stat=stat,
                    validated_prefix=self._recovery_prefix_for_error(
                        validated_prefix, row_offset, repairable=index > 0
                    ),
                ) from exc
            if not isinstance(value, dict):
                raise self._transcript_error(
                    kind="unrecoverable_header" if index == 0 else "invalid_row",
                    row_number=row_number,
                    byte_offset=row_offset,
                    valid_prefix_end=None if index == 0 else row_offset,
                    reason_code="row_not_object",
                    repairable=index > 0,
                    raw=raw,
                    stat=stat,
                    validated_prefix=self._recovery_prefix_for_error(
                        validated_prefix, row_offset, repairable=index > 0
                    ),
                )
            offset += len(line)
            yield _PhysicalRow(row_number, row_offset, value)

    def _repair_explicit_load_error(
        self: ConversationStore, error: TranscriptLoadError
    ) -> bool:
        if not self._repair_explicit or error._source_facts is None:
            return False
        if self._storage_lease.observed_epoch != WRITER_EPOCH:
            raise error
        self._last_repair_result = repair_transcript(
            self.directory_fd,
            self._repair_target,
            error._source_facts,
            strict_reopen=self._strict_reopen_repaired_transcript,
            memory_review_required=self._repair_target.child_run_id is None,
        )
        self._repair_explicit = False
        self._load_full()
        return True

    def _repair_torn_row(
        self: ConversationStore,
        torn_offset: int | None,
        validator: _FullLoadValidator,
        stat: os.stat_result,
        raw: bytes,
        delta: tuple[ConversationEntry, ...],
    ) -> bool:
        if torn_offset is None:
            return False
        if self._storage_lease.observed_epoch != WRITER_EPOCH:
            self._repair_torn_tail_legacy(torn_offset)
            self._log_offset = torn_offset
            self._publish_entries_installed(delta)
            self._append_torn_warning_legacy()
            return True
        prefix = validator.validated_prefix(torn_offset)
        source = TranscriptSourceFacts(
            device=stat.st_dev,
            inode=stat.st_ino,
            size=stat.st_size,
            source_digest=hashlib.sha256(raw).hexdigest(),
            prefix_digest=hashlib.sha256(raw[:torn_offset]).hexdigest(),
            first_bad_row=len(raw[:torn_offset].splitlines()) + 1,
            first_bad_offset=torn_offset,
            prefix=prefix,
            diagnosed_at=_now(),
        )
        self._last_repair_result = repair_transcript(
            self.directory_fd,
            self._repair_target,
            source,
            strict_reopen=self._strict_reopen_repaired_transcript,
            memory_review_required=False,
        )
        warnings.warn(
            f"dropped torn final conversation line from {self.path}",
            RuntimeWarning,
            stacklevel=2,
        )
        self._load_full()
        return True

    def _load_full(self: ConversationStore) -> None:
        self._full_reload_count += 1
        if not self._read_only and self._storage_lease.observed_epoch == WRITER_EPOCH:
            reconcile_pending(
                self.directory_fd,
                self._repair_target,
                strict_reopen=self._strict_reopen_repaired_transcript,
            )
        baseline = (len(self._entries), self._entry_prefix_digest)
        fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        try:
            stat = os.fstat(fd)
            raw = self._read_fd(fd, stat.st_size)
        finally:
            os.close(fd)

        validator = _FullLoadValidator(self, raw, stat)
        torn_offset: int | None = None
        try:
            for physical_row in self._iter_physical_rows(
                raw, stat, validator.validated_prefix
            ):
                if physical_row.torn:
                    torn_offset = physical_row.byte_offset
                    break
                validator.accept(physical_row)
            validator.finish()
        except TranscriptLoadError as error:
            if self._repair_explicit_load_error(error):
                return
            raise

        delta = (
            self._installed_delta_from_baseline(self._entries, baseline)
            if self.on_entries_installed is not None
            else ()
        )
        if self._repair_torn_row(torn_offset, validator, stat, raw, delta):
            return
        if not self._read_only and not raw.endswith(b"\n"):
            self._write_bytes(b"\n")
            self._publish_entries_installed(delta)
        else:
            self._set_log_stat(stat)
            self._publish_entries_installed(delta)

    def _validate_header(self: ConversationStore, header: dict[str, Any]) -> None:
        header_data = header.get("data")
        if (
            type(header.get("type")) is not str
            or header.get("type") != "header"
            or type(header_data) is not dict
            or type(header_data.get("schema")) is not str
            or header_data.get("schema") != SCHEMA
        ):
            raise ConversationIntegrityError(
                f"unsupported conversation schema: {self.path}"
            )
        header_session_id = header_data.get("session_id")
        cwd = header_data.get("cwd")
        created_at = header_data.get("created_at")
        if (
            type(header_session_id) is not str
            or not header_session_id
            or type(cwd) is not str
            or not cwd
            or type(created_at) is not str
        ):
            raise ConversationIntegrityError(
                f"conversation header is incomplete: {self.path}"
            )
        self.cwd = cwd
        if header_session_id != self.session_id:
            raise ConversationIntegrityError(
                f"conversation header session id mismatch: {self.path}"
            )

    def _parse_entry_lines(
        self: ConversationStore, raw: bytes, *, first_row: int
    ) -> tuple[list[ConversationEntry], int | None]:
        entries: list[ConversationEntry] = []
        offset = 0
        index = 0
        while offset < len(raw):
            newline = raw.find(b"\n", offset)
            line_end = len(raw) if newline < 0 else newline + 1
            line = raw[offset:line_end]
            # Parsing runs in worker threads for live views. A short sleep
            # releases the GIL long enough for the owner event loop to paint.
            if index and index % 64 == 0:
                time.sleep(0.001)
            try:
                row = load_session_json(line)
            except ConversationIntegrityError as exc:
                if (
                    line_end != len(raw)
                    or line.endswith(b"\n")
                    or not isinstance(
                        exc.__cause__, (json.JSONDecodeError, UnicodeError)
                    )
                ):
                    kind = "terminated " if line.endswith(b"\n") else ""
                    raise ConversationIntegrityError(
                        f"invalid {kind}conversation row {first_row + index}: {self.path}"
                    ) from exc
                return entries, offset
            if not isinstance(row, dict):
                raise ConversationIntegrityError(
                    f"conversation row {first_row + index} is not an object: {self.path}"
                )
            try:
                entries.append(ConversationEntry.from_dict(row))
            except ConversationIntegrityError:
                raise
            except (KeyError, TypeError, ValueError) as exc:
                raise ConversationIntegrityError(
                    f"invalid conversation entry: {self.path}"
                ) from exc
            offset = line_end
            index += 1
        return entries, None

    @staticmethod
    def _read_fd(fd: int, length: int) -> bytes:
        chunks: list[bytes] = []
        remaining = length
        while remaining:
            chunk = os.read(fd, min(remaining, 1024 * 1024))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        return b"".join(chunks)

    def _strict_reopen_repaired_transcript(self: ConversationStore) -> None:
        fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        try:
            stat = os.fstat(fd)
            raw = self._read_fd(fd, stat.st_size)
        finally:
            os.close(fd)
        validator = _FullLoadValidator(self, raw, stat)
        for physical_row in self._iter_physical_rows(
            raw, stat, validator.validated_prefix
        ):
            if physical_row.torn:
                raise ConversationIntegrityError(
                    f"invalid conversation row {physical_row.row_number}: {self.path}"
                )
            validator.accept(physical_row)
        validator.finish()

    def _repair_torn_tail_legacy(self: ConversationStore, offset: int) -> None:
        with os.fdopen(
            open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDWR),
            "r+b",
        ) as handle:
            handle.truncate(offset)
            handle.flush()
            os.fsync(handle.fileno())
            self._set_log_stat(os.fstat(handle.fileno()))

    def _append_torn_warning_legacy(self: ConversationStore) -> None:
        self._append_row_unlocked(
            "warning", {"message": "dropped torn final conversation line"}
        )
        warnings.warn(
            f"dropped torn final conversation line from {self.path}",
            RuntimeWarning,
            stacklevel=2,
        )

    def _write_line(self: ConversationStore, row: dict[str, Any]) -> None:
        encoded = encode_json(row) + b"\n"
        if (
            self._write_deadline is not None
            and time.monotonic() >= self._write_deadline
        ):
            from ._store import PendingPromptCommitTimeoutError

            raise PendingPromptCommitTimeoutError(
                "pending prompt commit deadline exceeded"
            )
        self._write_bytes(encoded)

    def _write_bytes(self: ConversationStore, data: bytes) -> None:
        with os.fdopen(
            open_session_file(
                self.directory_fd,
                "conversation.jsonl",
                os.O_WRONLY | os.O_APPEND | os.O_CREAT,
            ),
            "ab",
        ) as handle:
            before = (
                os.fstat(handle.fileno()) if self._collect_persisted_appends else None
            )
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
            after = os.fstat(handle.fileno())
            self._set_log_stat(after)
            if before is not None:
                self._record_persisted_append(
                    PersistedAppend(
                        start_offset=before.st_size,
                        end_offset=after.st_size,
                        digest=hashlib.sha256(data).hexdigest(),
                        source_device=after.st_dev,
                        source_inode=after.st_ino,
                        before_mtime_ns=before.st_mtime_ns,
                        before_ctime_ns=before.st_ctime_ns,
                        after_mtime_ns=after.st_mtime_ns,
                        after_ctime_ns=after.st_ctime_ns,
                    )
                )

    @staticmethod
    def _prefix_fingerprint(fd: int, offset: int) -> bytes:
        length = min(offset, PREFIX_FINGERPRINT_BYTES)
        start = offset - length
        return hashlib.blake2b(os.pread(fd, length, start), digest_size=16).digest()

    def _known_prefix_matches(self: ConversationStore, fd: int, offset: int) -> bool:
        known = getattr(self, "_log_prefix_fingerprint", None)
        return known is not None and self._prefix_fingerprint(fd, offset) == known

    def _set_log_stat(self: ConversationStore, stat: os.stat_result) -> None:
        fd = open_session_file(self.directory_fd, "conversation.jsonl", os.O_RDONLY)
        try:
            fingerprint = self._prefix_fingerprint(fd, stat.st_size)
        finally:
            os.close(fd)
        self._log_identity = (stat.st_dev, stat.st_ino, stat.st_ctime_ns)
        self._log_offset = stat.st_size
        self._log_mtime_ns = stat.st_mtime_ns
        self._log_prefix_fingerprint = fingerprint
        self._transcript_generation += 1
