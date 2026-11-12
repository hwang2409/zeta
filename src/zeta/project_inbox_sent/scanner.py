"""Polling seam for project inbox notifications."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from ..project_registry import ProjectRegistry
from . import SentLedgerOutcome, SentMessageTracker


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
            registry, sessions_root=sessions_root, sent_tracker=tracker
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
