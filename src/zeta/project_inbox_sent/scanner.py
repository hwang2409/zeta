"""Polling seam for project inbox notifications."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..project_registry import ProjectRegistry
from . import SentLedgerOutcome, SentMessageTracker

_Signature = tuple[int, tuple[tuple[str, int], ...]]


class ProjectInboxScanner:
    """Poll incoming directories and sender-owned sent-message records."""

    def __init__(
        self,
        registry: ProjectRegistry,
        project_id: str,
        *,
        sessions_root: Path,
        session_id: str | None = None,
        tracker: SentMessageTracker | None = None,
    ) -> None:
        if tracker is not None and tracker.session_id != session_id:
            raise ValueError("sent-message tracker belongs to another session")
        from ..project_inbox import ProjectInbox

        self.inbox = ProjectInbox(
            registry,
            sessions_root=sessions_root,
            sent_tracker=tracker,
            observed_epoch=tracker.observed_epoch if tracker is not None else 0,
        )
        self.project_id = project_id
        self.session_id = session_id
        self.tracker = tracker or (
            SentMessageTracker(sessions_root, session_id)
            if session_id is not None
            else None
        )
        self._activation_sent_outcome: SentLedgerOutcome[None] | None = None
        self._fingerprint: tuple[tuple[int, int, int, int] | None, ...] | None = None
        self._signature: _Signature | None = None
        self._recovered = False

    def poll(self) -> tuple[tuple[str, ...] | None, tuple[dict[str, Any], ...]]:
        """Return one complete incoming and sent snapshot for a supervisor."""
        if not self._recovered:
            if self.tracker is not None:
                self.inbox.recover_sent_intents(self.tracker)
            self._recovered = True
        if self.inbox._generation_enabled and not (
            self.tracker is not None and self.tracker.broken_cached
        ):
            return self._poll_by_generation()
        message_ids = self.scan()
        sent = self.scan_sent()
        return message_ids, sent

    def _poll_by_generation(
        self,
    ) -> tuple[tuple[str, ...] | None, tuple[dict[str, Any], ...]]:
        """Skip the whole scan when no inbox generation moved since last poll."""
        signature = self._generation_signature()
        if self._signature is not None and signature == self._signature:
            return None, ()
        message_ids = self.inbox.new_ids(self.project_id, session_id=self.session_id)
        sent = self.scan_sent()
        # Commit the pre-scan generation; a mutation during the scan only causes
        # one extra conservative rescan on the next poll.
        self._signature = signature
        return message_ids, sent

    def _generation_signature(self) -> _Signature:
        own = self.inbox._project_generation(self.project_id)
        targets = (
            self.tracker.tracked_targets() if self.tracker is not None else set()
        )
        return own, tuple(
            (target, self.inbox._project_generation(target))
            for target in sorted(targets)
        )

    def scan(self) -> tuple[str, ...] | None:
        current = self._directory_fingerprint()
        if self._fingerprint is not None and current == self._fingerprint:
            return None
        message_ids = self.inbox.new_ids(self.project_id, session_id=self.session_id)
        self._fingerprint = (
            current
            if any(item is not None for item in current)
            else self._directory_fingerprint()
        )
        return message_ids

    def scan_sent(self) -> tuple[dict[str, Any], ...]:
        """Read only sender-tracked targets; empty tracking has no inbox I/O."""
        if self.tracker is None or self.session_id is None:
            return ()
        if not self.tracker.broken_cached and not self.tracker._pending:
            return ()
        source = self.inbox._resolve_project(self.project_id)
        outcome = self.tracker.sent_view(
            self.inbox, source.project_id, self.tracker._receipts
        )
        return tuple(
            item
            for item in outcome.result
            if "done" not in item.get("reported", [])
        )

    def mark_reported(self, statuses: Iterator[tuple[str, str]]) -> None:
        if self.tracker is not None:
            self.tracker.mark_reported(statuses)

    def _directory_fingerprint(self) -> tuple[tuple[int, int, int, int] | None, ...]:
        inbox = self.inbox.registry.root / self.project_id / "inbox"
        result: list[tuple[int, int, int, int] | None] = []
        for name in ("new", "claimed"):
            try:
                info = (inbox / name).stat(follow_symlinks=False)
            except FileNotFoundError:
                result.append(None)
            else:
                result.append(
                    (info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size)
                )
        return tuple(result)

