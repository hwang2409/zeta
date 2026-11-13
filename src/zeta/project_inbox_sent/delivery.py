"""The sender side of project-inbox delivery: durable send, recovery, sent view.

``ProjectInbox`` owns one ``SentDelivery``. These methods orchestrate the inbox's
own recipient primitives (``_outgoing``, ``_directories``, ``_publish_outgoing``,
``_directory_handles``, ``_read_directory``, ``_lock``, ``_read_tracked_message``),
so the inbox is passed in as the collaborator and reached through ``self._inbox``,
the same way ``ProjectInboxScanner`` holds its inbox; the coupling stays visible
rather than hidden in a mixin.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from ..core.session_files import SessionError, reference_publication
from ..project_errors import ProjectNotFoundError
from ..storage_format import StorageFeature, StorageFormatError, feature_enabled
from . import SentLedgerOutcome, SentMessageTracker

if TYPE_CHECKING:
    from ..project_inbox import ProjectInbox


class SentDelivery:
    """Deliver tracked sends, recover durable intents, and resolve the sent view."""

    def __init__(self, inbox: ProjectInbox) -> None:
        self._inbox = inbox

    def deliver_tracked(
        self, *, on_visible: Callable[[], None] | None = None, **kwargs: Any
    ) -> tuple[str, SentLedgerOutcome[None]]:
        from ..project_inbox import InboxConflictError, InboxError

        inbox = self._inbox
        session, target, message_id, to_session, body, record = inbox._outgoing(**kwargs)
        # At the durable-intent epoch a pending record precedes the recipient
        # publish, so a crash in between recovers to exactly one delivery. The
        # post-publish record() call marks it published; below the epoch it
        # writes the schema-v1 row after publish, exactly as before.
        tracker = self.tracker_for(session)
        durable = feature_enabled(
            inbox.observed_epoch, StorageFeature.INBOX_DURABLE_INTENT
        )
        lease = inbox.registry._storage_lease
        # The reference gate holds storage authority while this message (a
        # reference to one or two sessions) is published and recorded, so the
        # retention reaper cannot delete a referenced session mid-publish.
        with reference_publication(
            inbox.registry.root.parent,
            lease,
            referenced_ids=(session, to_session) if to_session else (session,),
        ):
            # Acquiring the recipient inbox authority can fail with a retryable
            # InboxBusyError or a plain unsafe InboxError. Let those propagate
            # before any durable intent exists, so the caller sees the real
            # retryable/unsafe status and a retry cannot duplicate a message. The
            # durable intent is written while the lock is held, so a crash
            # between the intent and the publish still recovers to one delivery.
            with inbox._directories(target, create=True) as dirs:
                if durable:
                    # A broken sender ledger saves no intent, so there is nothing
                    # to retry; drop the durable promise and let a failed publish
                    # surface the plain error below, so the model may resend.
                    durable = (
                        tracker.record_intent(
                            record, _storage_lease=lease
                        ).health.state
                        != "broken"
                    )
                try:
                    inbox._publish_outgoing(dirs, message_id, body, record)
                except InboxConflictError:
                    raise
                except (InboxError, OSError, SessionError, StorageFormatError) as exc:
                    if durable:
                        # The intent is already durable; a later poll re-publishes
                        # it. Tell the caller so the model does not duplicate it.
                        raise InboxError(
                            "inbox message saved; the recipient could not be "
                            "reached now and delivery will be retried "
                            "automatically. Do not resend this message."
                        ) from exc
                    raise
            if on_visible is not None:
                try:
                    on_visible()
                except Exception:  # noqa: BLE001,S110
                    # The durable record stays authoritative if its wake fails.
                    pass
            outcome = tracker.record(message_id, target, _storage_lease=lease)
        return message_id, outcome

    def recover_sent_intents(self, tracker: SentMessageTracker) -> list[str]:
        """Re-publish each durable intent whose recipient publish is not durable.

        Every intent is isolated: one that cannot be resolved or published (a
        missing spill, a conflict, an OSError) is reported and skipped rather
        than blocking the recovery of the others or the incoming scan. The
        returned reasons let the poller surface degraded health.
        """
        from ..project_inbox import InboxError

        inbox = self._inbox
        failures: list[str] = []
        lease = inbox.registry._storage_lease
        # shortcut: an intent whose target can never succeed (a removed project,
        # or a caller-supplied id that conflicts) retries every poll for the
        # session's life with degraded health; there is no abandonment path.
        # Upgrade to a dead-letter/abandon path if that case becomes common.
        for envelope in tracker.pending_intents():
            target = envelope["to_project"]
            message_id = envelope["id"]
            sender = envelope["from"]["session"]
            to_session = envelope.get("to_session")
            try:
                body = tracker.resolve_pending_body(message_id, envelope["body"])
                record = dict(envelope)
                record["body"] = body
                # The reference gate holds storage authority while the re-publish
                # and its record run, exactly as the live send does, so the
                # retention reaper cannot delete the referenced recipient session
                # mid-publish.
                with reference_publication(
                    inbox.registry.root.parent,
                    lease,
                    referenced_ids=(sender, to_session) if to_session else (sender,),
                ):
                    with inbox._directories(target, create=True) as dirs:
                        inbox._publish_outgoing(dirs, message_id, body, record)
                    tracker.record(message_id, target, _storage_lease=lease)
            except (OSError, SessionError, InboxError, ValueError) as exc:
                failures.append(f"{message_id}: {exc}")
        return failures

    def tracker_for(self, session: str) -> SentMessageTracker:
        inbox = self._inbox
        if inbox.sent_tracker is not None and inbox.sent_tracker.session_id == session:
            return inbox.sent_tracker
        return SentMessageTracker(
            inbox.sessions_root, session, observed_epoch=inbox.observed_epoch
        )

    def broken_project_rows(
        self,
        project_id: str,
        source_project: str,
        source_session: str,
        receipt_map: dict[str, set[str]],
    ) -> list[dict[str, Any]]:
        from ..project_inbox import _InboxNotFound

        inbox = self._inbox
        invalid: list[dict[str, str]] = []
        rows: list[dict[str, Any]] = []
        def read(project_fd: int) -> None:
            try:
                with inbox._directory_handles(project_fd, create=False) as dirs:
                    for status, directory_fd in zip(("new", "claimed", "done"), dirs[:3]):
                        for record in inbox._read_directory(
                            directory_fd,
                            dirs[3],
                            status=status,
                            invalid=invalid,
                            resolve_body=False,
                        ):
                            if record.get("from") == {
                                "project": source_project,
                                "session": source_session,
                            }:
                                rows.append(self._broken_row(record, status, project_id, project_name, source_session, receipt_map))
            except _InboxNotFound:
                return
        # Identity ranks before the inbox lock, so resolve the name first.
        try:
            project_name = inbox.registry.show_project(project_id).name
        except ProjectNotFoundError:
            project_name = project_id
        with inbox._lock(project_id, exclusive=False) as project_fd:
            read(project_fd)
        return rows

    def _broken_row(
        self,
        record: dict[str, Any],
        status: str,
        project_id: str,
        project_name: str,
        source_session: str,
        receipt_map: dict[str, set[str]],
    ) -> dict[str, Any]:
        row = {
            "id": record["id"],
            "status": status,
            "to_project": project_id,
            "to_project_name": project_name[:200],
            "from_session": source_session,
            "kind": record["kind"],
            "title": record["title"][:200],
            "created_at": record["created_at"][:64],
            "reported": [
                name for name in ("claimed", "done") if name in receipt_map.get(record["id"], set())
            ],
        }
        if status in {"claimed", "done"}:
            row["claimed_at"] = str(record.get("claimed_at") or "")[:64]
            row["claimer_session"] = record.get("claimer_session")
        if status == "done":
            row.update(
                done_at=str(record.get("done_at") or "")[:64],
                outcome=str(record.get("outcome") or "")[:500],
                reply_id=record.get("reply_id"),
            )
        return row

    def resolve_tracked_sent(
        self,
        source_project: str,
        source_session: str,
        tracked: tuple[dict[str, Any], ...],
    ) -> list[dict[str, Any]]:
        inbox = self._inbox
        sent: list[dict[str, Any]] = []
        for item in tracked:
            found = inbox._read_tracked_message(
                source_project,
                source_session,
                item["target_project"],
                item["id"],
            )
            if found is None:
                continue
            status, record = found
            try:
                target_name = inbox.registry.show_project(item["target_project"]).name
            except ProjectNotFoundError:
                target_name = item["target_project"]
            result: dict[str, Any] = {
                "id": record["id"],
                "status": status,
                "to_project": str(record["to_project"])[:200],
                "to_project_name": target_name[:200],
                "from_session": source_session,
                "kind": record["kind"],
                "title": record["title"][:200],
                "created_at": record["created_at"][:64],
                "reported": list(item["reported"]),
            }
            if status in {"claimed", "done"}:
                result["claimed_at"] = str(record.get("claimed_at") or "")[:64]
                result["claimer_session"] = record.get("claimer_session")
            if status == "done":
                result["done_at"] = str(record.get("done_at") or "")[:64]
                result["outcome"] = str(record.get("outcome") or "")[:500]
                result["reply_id"] = record.get("reply_id")
            sent.append(result)
        sent.sort(key=lambda item: (item["created_at"], item["id"]), reverse=True)
        return sent
