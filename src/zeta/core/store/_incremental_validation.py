"""Incremental integrity indexes for conversation log tails."""

from __future__ import annotations

import copy
import time

from ...protocol.types import Message
from ...storage_format import (
    # the stored-format registry owns the reasons a resolution may carry
    APPROVAL_RESOLUTION_REASONS,
)
from ..checkpoints import ConversationEntry, ConversationIntegrityError
from ..checkpoints.entry_types import sent_message_receipts
from ._delivery_fold import DeliveryFold
from ._validation import TASK_EXITED_NOTIFICATION_KIND


def validate_approval_reason(decision: str, reason: str | None) -> None:
    if reason is not None and (
        type(reason) is not str
        or decision != "abort"
        or reason not in APPROVAL_RESOLUTION_REASONS
    ):
        raise ValueError("invalid approval resolution reason")


class _ApprovalValidationOverlay:
    """Operation-scoped approval relations for authoritative row validation."""

    def __init__(self, relations: dict[str, str | None] | None = None) -> None:
        relations = relations or {}
        self.requests = set(relations)
        self.resolutions = {
            request_id for request_id, decision in relations.items() if decision is not None
        }

    @staticmethod
    def touched(entries: list[ConversationEntry]) -> set[str]:
        request_ids: set[str] = set()
        for entry in entries:
            if entry.type == "message":
                request_ids.update(
                    request["request_id"]
                    for request in entry.data.get("approval_requests", ())
                    if isinstance(request, dict)
                    and isinstance(request.get("request_id"), str)
                )
            elif entry.type == "approval_resolution":
                request_id = entry.data.get("request_id")
                if isinstance(request_id, str):
                    request_ids.add(request_id)
        return request_ids

    def record(self, entry: ConversationEntry) -> None:
        if entry.type == "message":
            for request in entry.data.get("approval_requests", ()):
                request_id = request.get("request_id")
                if type(request_id) is str and request_id:
                    if request_id in self.requests:
                        raise ConversationIntegrityError(
                            f"duplicate approval request: {request_id}"
                        )
                    self.requests.add(request_id)
        elif entry.type == "approval_request":
            raise ConversationIntegrityError(
                "standalone approval requests are not supported"
            )
        elif entry.type == "approval_resolution":
            request_id = entry.data.get("request_id")
            if request_id not in self.requests:
                raise ConversationIntegrityError(
                    f"approval resolution is not linked to request: {request_id}"
                )
            reason = entry.data.get("reason")
            if reason is not None and (
                type(reason) is not str
                or entry.data.get("decision") != "abort"
                or reason not in APPROVAL_RESOLUTION_REASONS
            ):
                raise ConversationIntegrityError("invalid approval resolution reason")
            if request_id in self.resolutions:
                raise ConversationIntegrityError(
                    f"duplicate approval resolution: {request_id}"
                )
            self.resolutions.add(request_id)


class IncrementalValidationMixin:
    """Validate linear appended entries without rescanning the full log."""

    def _full_branch_approval_overlay(self) -> _ApprovalValidationOverlay:
        """Build the active-branch approval relations by one authoritative scan.

        Approval-row validation is independent of the derived sidecar: it reads
        the resident branch so a stale or absent index can never fail (or pass)
        an authoritative write. The already-committed branch is known valid, so
        this collects its request/resolution relations in one tight pass rather
        than re-running per-entry validation; only the candidate row is checked.
        """
        relations: dict[str, str | None] = {}
        for entry in self.approval_payloads():
            if entry.type == "message":
                for request in entry.data.get("approval_requests", ()):
                    request_id = request.get("request_id")
                    if isinstance(request_id, str):
                        relations.setdefault(request_id, None)
            elif entry.type == "approval_resolution":
                request_id = entry.data.get("request_id")
                if isinstance(request_id, str):
                    relations[request_id] = entry.data.get("decision")
        return _ApprovalValidationOverlay(relations)

    def _reset_incremental_validation_state(self) -> None:
        self._entry_ids: set[str] = set()
        self._delivery_fold = DeliveryFold()

    def _record_physical_entry(self, entry: ConversationEntry) -> None:
        """Record the all-branch physical task-exit id for one appended row.

        One owner for the task-exit id write, which is physical (every branch,
        kept across branch switches) unlike the branch-scoped delivery fold.
        """
        if (
            entry.type == "notification"
            and entry.data.get("kind") == TASK_EXITED_NOTIFICATION_KIND
        ):
            task_id = entry.data.get("task_id")
            if type(task_id) is str and task_id:
                self._task_notification_ids.add(task_id)

    def _rebuild_incremental_validation_state(self) -> None:
        """Build O(1) integrity indexes after a complete validation pass."""
        self._reset_incremental_validation_state()
        for index, ref in enumerate(self._resident_rows.rows):
            if index and index % 64 == 0:
                time.sleep(0.0001)
            self._entry_ids.add(ref.id)
        approvals = _ApprovalValidationOverlay()
        fold = DeliveryFold()
        branch = self._resident_rows.stream_rows(self._resident_rows.branch_refs())
        for index, entry in enumerate(branch):
            if index and index % 64 == 0:
                time.sleep(0.0001)
            fold.apply(entry, approvals)
        self._delivery_fold = fold

    def _accept_incremental_entries(self, entries: list[ConversationEntry]) -> bool:
        """Validate a tail against staged state, then publish it as one update."""
        staged = copy.copy(self)
        staged._resident_rows = self._resident_rows.clone()
        staged._entry_ids = set(self._entry_ids)
        staged._entry_prefix_digest = self._entry_prefix_digest
        staged._delivery_fold = self._delivery_fold.staged()
        staged._task_notification_ids = set(self._task_notification_ids)
        staged._sent_message_receipts = list(self._sent_message_receipts)
        touched_approvals = _ApprovalValidationOverlay.touched(entries)
        # An incremental tail that touches an approval row validates against the
        # authoritative branch overlay (one scan, only when approvals appear),
        # never a full reload: the sidecar is derived and must not gate authority.
        approvals = (
            self._full_branch_approval_overlay()
            if touched_approvals
            else _ApprovalValidationOverlay()
        )
        for index, entry in enumerate(entries):
            # Large off-thread refreshes must release the GIL often enough for
            # the owner event loop to remain responsive.
            if index and index % 64 == 0:
                time.sleep(0.001)
            if not staged._accept_incremental_entry(entry, approvals):
                return False

        self._before_incremental_state_install()
        self._entry_ids = staged._entry_ids
        self._entry_prefix_digest = staged._entry_prefix_digest
        self._delivery_fold = staged._delivery_fold
        self._task_notification_ids = staged._task_notification_ids
        self._sent_message_receipts = staged._sent_message_receipts
        # Publish entries last so readers never see rows without matching indexes.
        self._resident_rows.replace(
            staged._resident_rows.stream_rows(staged._resident_rows.rows)
        )
        self._record_client_delivery_entries(entries)
        return True

    def _before_incremental_state_install(self) -> None:
        """Test hook immediately before a staged tail becomes visible."""

    def _accept_incremental_entry(
        self, entry: ConversationEntry, approvals: _ApprovalValidationOverlay
    ) -> bool:
        """Validate and install one linear tail entry, or request a full reload."""
        previous_ref = self._resident_rows.rows[-1] if self._resident_rows.rows else None
        expected_seq = previous_ref.seq + 1 if previous_ref is not None else 1
        if entry.seq != expected_seq:
            raise ConversationIntegrityError(
                f"non-monotonic conversation sequence at {entry.id}"
            )
        if entry.id in self._entry_ids:
            raise ConversationIntegrityError(f"duplicate conversation id: {entry.id}")
        if previous_ref is not None and entry.parent_id is None:
            raise ConversationIntegrityError(
                f"conversation entry {entry.id} is an orphaned root"
            )
        if entry.parent_id is not None and entry.parent_id not in self._entry_ids:
            raise ConversationIntegrityError(
                f"missing prior parent {entry.parent_id} for {entry.id}"
            )
        # A non-tip parent changes the active branch. It is uncommon and its
        # integrity state is deliberately rebuilt by the complete validator.
        if previous_ref is not None and entry.parent_id != previous_ref.id:
            return False
        validated_payload = self._validate_entry_payload(entry)
        if entry.type == "fork":
            self._validate_fork_entry(entry)
        self._apply_to_working_fold(
            entry,
            approvals,
            validated_payload if isinstance(validated_payload, Message) else None,
        )
        self._resident_rows.append(entry)
        self._record_entry_prefix(entry)
        self._entry_ids.add(entry.id)
        self._sent_message_receipts.extend(sent_message_receipts(entry))
        self._record_physical_entry(entry)
        return True

    def _install_validated_linear_entries(
        self,
        entries: list[ConversationEntry],
        *,
        start_offset: int | None = None,
    ) -> None:
        """Install worker-validated tail rows and publish the entry list once."""
        previous = self._resident_rows.rows[-1] if self._resident_rows.rows else None
        fold = self._delivery_fold.staged()
        for entry in entries:
            expected_seq = previous.seq + 1 if previous is not None else 1
            expected_parent = previous.id if previous is not None else None
            assert entry.seq == expected_seq and entry.parent_id == expected_parent
            assert entry.id not in self._entry_ids
            fold.apply(entry)
            self._entry_ids.add(entry.id)
            self._record_physical_entry(entry)
            previous = entry
        self._delivery_fold = fold
        self._resident_rows.extend(entries, start_offset=start_offset)

    def _apply_to_working_fold(
        self,
        entry: ConversationEntry,
        approvals: _ApprovalValidationOverlay | None = None,
        message: Message | None = None,
    ) -> None:
        """Apply one active-branch entry in place to the current working fold.

        The caller must first set ``self._delivery_fold`` to a working fold it
        owns (a fresh fold on rebuild or branch select, a ``staged()`` copy on an
        incremental tail); this writes into that fold in place. It must never run
        against an installed fold, which is immutable.
        """
        if message is None:
            self._delivery_fold.apply(entry, approvals)
        else:
            self._delivery_fold.apply_validated(entry, approvals, message)
