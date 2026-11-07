"""Durable, batched terminal transitions for attention records."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import secrets
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

from .attention_records import AttentionRecord, AttentionStore
from .project_inbox import ProjectInbox
from .runtime.project_registry_state import (
    ProjectRegistryState,
    available_project_registry,
)

_MAX_REPLY_BYTES = 32_000

ReplyStatus = Literal["applied", "duplicate", "conflict", "already-resolved"]


@dataclass(frozen=True, slots=True)
class ReplyItemOutcome:
    attention_id: str
    status: ReplyStatus


@dataclass(frozen=True, slots=True)
class ReplyOutcome:
    """Complete result of one public attention-reply operation."""

    items: tuple[ReplyItemOutcome, ...] = ()
    staged: tuple[tuple[str, str], ...] = ()
    published: bool = False
    message_id: str | None = None

    @property
    def already_resolved(self) -> bool:
        return bool(self.items) and all(
            item.status == "already-resolved" for item in self.items
        )


class ReplyConflict(ValueError):
    """A terminal conflict, including publication completed during recovery."""

    def __init__(self, message: str, outcome: ReplyOutcome) -> None:
        super().__init__(message)
        self.outcome = outcome


class AttentionReplies:
    """The only owner of terminal attention transitions for one source session."""

    def __init__(
        self,
        home: Path,
        source_session_id: str,
        *,
        registry_state: ProjectRegistryState | None = None,
    ) -> None:
        self.home = Path(home)
        self.source_session_id = source_session_id
        self.session_dir = self.home / "sessions" / source_session_id
        self.attention_dir = self.session_dir / "attention"
        self.staged_dir = self.attention_dir / "staged"
        self.registry_state = registry_state

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.attention_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.attention_dir / ".resolution.lock", os.O_RDWR | os.O_CREAT, 0o600
        )
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def recover(self) -> ReplyOutcome:
        with self._lock():
            return self._recover_locked()

    def resolve_local(self, attention_id: str, reply: str) -> ReplyOutcome:
        reply = self._validate_reply(reply)
        with self._lock():
            recovered = self._recover_locked()
            self._reject_staged(attention_id, recovered)
            record = self._record(attention_id)
            self._check_session(record)
            if record.status == "resolved":
                if record.decision != reply:
                    self._raise_conflict(
                        "attention item is already resolved with a different reply",
                        (attention_id,),
                        recovered,
                    )
                return self._outcome(
                    recovered,
                    items=(ReplyItemOutcome(attention_id, "already-resolved"),),
                )
            AttentionStore(self.session_dir).replace(
                replace(record, status="resolved", resolved_at=_now(), decision=reply)
            )
            return self._outcome(
                recovered, items=(ReplyItemOutcome(attention_id, "applied"),)
            )

    def _recover_locked(self) -> ReplyOutcome:
        published = False
        recovered_ids: list[str] = []
        journals = sorted(self.attention_dir.glob("sending-*-*"))
        for journal in journals:
            self._validate_journal(journal)
        for journal in journals:
            recovered_ids.extend(
                path.name
                for path in journal.iterdir()
                if path.is_file() and not path.name.startswith(".")
            )
            self._send_journal(journal)
            self._finish_journal_locked(journal)
            published = True
        for path in self.attention_dir.glob("done-*"):
            if path.is_dir():
                self._cleanup(path)
                self._sync_directory()
        for path in self.attention_dir.glob(".batch-*"):
            if path.is_dir():
                self._cleanup(path)
                self._sync_directory()
        staged = []
        if self.staged_dir.is_dir():
            for path in sorted(self.staged_dir.iterdir()):
                if path.name.startswith("."):
                    path.unlink()
                    self._sync_directory(self.staged_dir)
                elif path.is_file():
                    staged.append((path.name, path.read_text(encoding="utf-8")))
        return ReplyOutcome(
            items=tuple(
                ReplyItemOutcome(item_id, "applied") for item_id in recovered_ids
            ),
            staged=tuple(staged),
            published=published,
        )

    def stage(
        self,
        attention_id: str,
        reply: str,
        *,
        flush_ids: set[str] | None = None,
        from_session: str | None = None,
    ) -> ReplyOutcome:
        reply = self._validate_reply(reply)
        with self._lock():
            recovered = self._recover_locked()
            record = self._record(attention_id)
            self._check_session(record)
            if record.status == "resolved":
                if record.decision != reply:
                    self._raise_conflict(
                        "attention item is already resolved with a different reply",
                        (attention_id,),
                        recovered,
                    )
                staged_path = self.staged_dir / attention_id
                if staged_path.exists():
                    if staged_path.read_text(encoding="utf-8") != reply:
                        self._raise_staged_conflict(attention_id, recovered)
                    staged_path.unlink()
                    self._sync_directory(self.staged_dir)
                return ReplyOutcome(
                    items=(ReplyItemOutcome(attention_id, "already-resolved"),),
                    staged=self._staged_locked(),
                    published=recovered.published,
                )
            self.staged_dir.mkdir(parents=True, exist_ok=True)
            path = self.staged_dir / attention_id
            status: ReplyStatus = "applied"
            if path.exists():
                if path.read_text(encoding="utf-8") != reply:
                    self._raise_staged_conflict(attention_id, recovered)
                status = "duplicate"
            else:
                tmp = self.staged_dir / f".{attention_id}.{secrets.token_hex(8)}"
                tmp.write_text(reply, encoding="utf-8")
                os.chmod(tmp, 0o600)
                self._sync_file(tmp)
                os.replace(tmp, path)
                self._sync_directory(self.staged_dir)
            staged = self._staged_locked()
            if flush_ids is not None and flush_ids <= {
                item_id for item_id, _ in staged
            }:
                if from_session is None:
                    raise ValueError(
                        "from_session is required when staged replies can flush"
                    )
                journal = self._prepare_locked(
                    list(staged), source=self.staged_dir, from_session=from_session
                )
                message_id = journal.name.rsplit("-", 1)[-1]
                self._send_journal(journal)
                self._finish_journal_locked(journal)
                return ReplyOutcome(
                    items=(ReplyItemOutcome(attention_id, status),),
                    published=True,
                    message_id=message_id,
                )
            return ReplyOutcome(
                items=(ReplyItemOutcome(attention_id, status),),
                staged=staged,
                published=recovered.published,
            )

    def flush(self, from_session: str) -> ReplyOutcome:
        with self._lock():
            recovered = self._recover_locked()
            outcomes: list[ReplyItemOutcome] = []
            open_pairs: list[tuple[str, str]] = []
            for item_id, reply in recovered.staged:
                record = self._record(item_id)
                self._check_session(record)
                if record.status == "resolved":
                    if record.decision != reply:
                        self._raise_conflict(
                            "attention item is already resolved with a different reply",
                            (item_id,),
                            recovered,
                        )
                    outcomes.append(ReplyItemOutcome(item_id, "already-resolved"))
                    (self.staged_dir / item_id).unlink()
                    self._sync_directory(self.staged_dir)
                else:
                    outcomes.append(ReplyItemOutcome(item_id, "applied"))
                    open_pairs.append((item_id, reply))
            if not open_pairs:
                return ReplyOutcome(
                    items=tuple(outcomes),
                    staged=self._staged_locked(),
                    published=recovered.published,
                )
            journal = self._prepare_locked(
                open_pairs, source=self.staged_dir, from_session=from_session
            )
            message_id = journal.name.rsplit("-", 1)[-1]
            self._send_journal(journal)
            self._finish_journal_locked(journal)
            return ReplyOutcome(
                items=tuple(outcomes),
                published=True,
                message_id=message_id,
            )

    def reply(self, attention_id: str, reply: str, from_session: str) -> ReplyOutcome:
        reply = self._validate_reply(reply)
        with self._lock():
            recovered = self._recover_locked()
            self._reject_staged(attention_id, recovered)
            record = self._record(attention_id)
            self._check_session(record)
            message_id = self._message_id([(attention_id, reply)])
            if record.status == "resolved":
                if record.decision != reply:
                    self._raise_conflict(
                        "attention item is already resolved with a different reply",
                        (attention_id,),
                        recovered,
                    )
                return self._outcome(
                    recovered,
                    items=(ReplyItemOutcome(attention_id, "already-resolved"),),
                    message_id=message_id,
                )
            journal = self._prepare_locked(
                [(attention_id, reply)], from_session=from_session
            )
            self._send_journal(journal)
            self._finish_journal_locked(journal)
            return ReplyOutcome(
                items=(ReplyItemOutcome(attention_id, "applied"),),
                published=True,
                message_id=message_id,
            )

    def dismiss(self, attention_id: str) -> ReplyOutcome:
        with self._lock():
            recovered = self._recover_locked()
            self._reject_staged(attention_id, recovered)
            record = self._record(attention_id)
            self._check_session(record)
            if record.status == "resolved":
                return self._outcome(
                    recovered,
                    items=(ReplyItemOutcome(attention_id, "already-resolved"),),
                )
            if record.kind == "decision":
                self._raise_conflict(
                    "decision attention items require a reply",
                    (attention_id,),
                    recovered,
                )
            AttentionStore(self.session_dir).replace(
                replace(record, status="resolved", resolved_at=_now(), decision=None)
            )
            return self._outcome(
                recovered, items=(ReplyItemOutcome(attention_id, "applied"),)
            )

    def stage_and_flush(
        self, replies: list[tuple[str, str]], from_session: str
    ) -> ReplyOutcome:
        with self._lock():
            recovered = self._recover_locked()
            staged = dict(recovered.staged)
            conflicting = tuple(item_id for item_id, _ in replies if item_id in staged)
            if conflicting:
                self._raise_staged_conflict(conflicting[0], recovered, conflicting)
            pairs = [
                (item_id, self._validate_reply(reply)) for item_id, reply in replies
            ]
            outcomes: list[ReplyItemOutcome] = []
            open_pairs: list[tuple[str, str]] = []
            for item_id, reply in pairs:
                record = self._record(item_id)
                self._check_session(record)
                if record.status == "resolved":
                    if record.decision != reply:
                        self._raise_conflict(
                            "attention item is already resolved with a different reply",
                            (item_id,),
                            recovered,
                        )
                    outcomes.append(ReplyItemOutcome(item_id, "already-resolved"))
                else:
                    outcomes.append(ReplyItemOutcome(item_id, "applied"))
                    open_pairs.append((item_id, reply))
            if not open_pairs:
                return self._outcome(recovered, items=tuple(outcomes))
            journal = self._prepare_locked(open_pairs, from_session=from_session)
            message_id = journal.name.rsplit("-", 1)[-1]
            self._send_journal(journal)
            self._finish_journal_locked(journal)
            return ReplyOutcome(
                items=tuple(outcomes),
                staged=recovered.staged,
                published=True,
                message_id=message_id,
            )

    def _finish_journal_locked(self, journal: Path) -> None:
        done = journal.with_name(journal.name.replace("sending-", "done-", 1))
        os.replace(journal, done)
        self._sync_directory()
        self._cleanup(done)
        self._sync_directory()

    def _staged_locked(self) -> tuple[tuple[str, str], ...]:
        if not self.staged_dir.is_dir():
            return ()
        return tuple(
            (path.name, path.read_text(encoding="utf-8"))
            for path in sorted(self.staged_dir.iterdir())
            if path.is_file() and not path.name.startswith(".")
        )

    def _reject_staged(self, attention_id: str, recovered: ReplyOutcome) -> None:
        if any(item_id == attention_id for item_id, _ in recovered.staged):
            self._raise_staged_conflict(attention_id, recovered)

    def _raise_staged_conflict(
        self,
        attention_id: str,
        recovered: ReplyOutcome,
        item_ids: tuple[str, ...] | None = None,
    ) -> None:
        self._raise_conflict(
            f"attention {attention_id} has a staged reply in a TUI decisions popup; "
            "finish or exit that popup before acting on it here",
            item_ids or (attention_id,),
            recovered,
        )

    @staticmethod
    def _raise_conflict(
        message: str, item_ids: tuple[str, ...], recovered: ReplyOutcome
    ) -> None:
        raise ReplyConflict(
            message,
            ReplyOutcome(
                items=tuple(
                    ReplyItemOutcome(item_id, "conflict") for item_id in item_ids
                ),
                staged=recovered.staged,
                published=recovered.published,
            ),
        )

    @staticmethod
    def _outcome(
        recovered: ReplyOutcome,
        *,
        items: tuple[ReplyItemOutcome, ...] = (),
        message_id: str | None = None,
    ) -> ReplyOutcome:
        return ReplyOutcome(
            items=items,
            staged=recovered.staged,
            published=recovered.published,
            message_id=message_id,
        )

    def _record(self, attention_id: str) -> AttentionRecord:
        return AttentionStore(self.session_dir).get(attention_id)

    def _check_session(self, record: AttentionRecord) -> None:
        if record.session_id != self.source_session_id:
            raise ValueError("attention record does not belong to this session")

    def _check_open(self, record: AttentionRecord) -> None:
        self._check_session(record)
        if record.status == "resolved":
            raise ValueError("attention item is already resolved")

    @staticmethod
    def _validate_reply(reply: str) -> str:
        if (
            not isinstance(reply, str)
            or not reply.strip()
            or len(reply.encode()) > _MAX_REPLY_BYTES
        ):
            raise ValueError("reply must be nonempty and within the size limit")
        return reply.strip()

    def _prepare_locked(
        self,
        pairs: list[tuple[str, str]],
        *,
        from_session: str,
        source: Path | None = None,
    ) -> Path:
        for item_id, reply in pairs:
            self._check_open(self._record(item_id))
        sequence = 0
        for path in self.attention_dir.glob("sending-*"):
            try:
                sequence = max(sequence, int(path.name.split("-", 2)[1]))
            except (ValueError, IndexError):
                raise ValueError("invalid attention journal sequence")
        for path in self.attention_dir.glob("done-*"):
            try:
                sequence = max(sequence, int(path.name.split("-", 2)[1]))
            except (ValueError, IndexError):
                raise ValueError("invalid attention journal sequence")
        message_id = self._message_id(pairs)
        if not pairs:
            raise ValueError("attention batch must not be empty")
        if source is None:
            source = self.attention_dir / f".batch-{secrets.token_hex(8)}"
            source.mkdir(mode=0o700)
            for item_id, reply in pairs:
                path = source / item_id
                path.write_text(reply, encoding="utf-8")
                os.chmod(path, 0o600)
                self._sync_file(path)
        sender_path = source / ".sender"
        sender_path.write_text(from_session, encoding="utf-8")
        os.chmod(sender_path, 0o600)
        self._sync_file(sender_path)
        self._sync_directory(source)
        journal = self.attention_dir / f"sending-{sequence + 1:020d}-{message_id}"
        os.replace(source, journal)
        self._sync_directory()
        return journal

    @staticmethod
    def _sync_file(path: Path) -> None:
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _sync_directory(self, directory: Path | None = None) -> None:
        directory = directory or self.attention_dir
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _journal_pairs(self, journal: Path) -> list[tuple[str, str]]:
        return [
            (path.name, path.read_text(encoding="utf-8"))
            for path in sorted(journal.iterdir())
            if not path.name.startswith(".")
        ]

    def _validate_journal(self, journal: Path) -> None:
        pairs = self._journal_pairs(journal)
        if not pairs:
            raise ValueError("attention journal is empty")
        for item_id, reply in pairs:
            record = self._record(item_id)
            if record.status == "resolved" and record.decision != reply:
                raise ValueError(
                    "attention journal conflicts with an already resolved reply"
                )
            if record.status not in {"open", "resolved"}:
                raise ValueError(
                    "attention journal contains an invalid terminal record"
                )

    def _send_journal(self, journal: Path) -> None:
        self._validate_journal(journal)
        pairs = self._journal_pairs(journal)
        if not pairs:
            raise ValueError("attention journal is empty")
        state = self.registry_state or available_project_registry(
            self.home / "projects"
        )
        if state.registry is None:
            raise ValueError("project registry is unavailable")
        inbox = ProjectInbox(state.registry, sessions_root=self.home / "sessions")
        records = [self._record(item_id) for item_id, _ in pairs]
        if any(record.project_id is None for record in records):
            raise ValueError("attention record has no project")
        sender = (journal / ".sender").read_text(encoding="utf-8")
        first = records[0]
        label = "Decision" if first.kind == "decision" else "Reply"
        sections = []
        for record, (_, reply) in zip(records, pairs):
            item_label = "Decision" if record.kind == "decision" else "Reply"
            sections.append(f"{item_label}: {reply}")
        inbox = ProjectInbox(
            (
                self.registry_state
                or available_project_registry(self.home / "projects")
            ).registry,
            sessions_root=self.home / "sessions",
        )
        inbox.send(
            from_project=first.project_id,
            from_session=sender,
            to_project=first.project_id,
            to_session=first.session_id,
            kind="reply",
            title=f"{label}: {first.title}"
            if len(records) == 1
            else f"{label} batch ({len(records)})",
            body=f"User relayed from session {sender}. Check whether the situation has changed before acting.\\n\\n"
            + "\\n\\n".join(sections),
            message_id=self._message_id(pairs),
        )
        for record, (_, reply) in zip(records, pairs):
            current = self._record(record.id)
            if current.status == "open":
                AttentionStore(self.session_dir).replace(
                    replace(
                        current, status="resolved", resolved_at=_now(), decision=reply
                    )
                )

    @staticmethod
    def _message_id(pairs: list[tuple[str, str]]) -> str:
        return hashlib.sha256(
            json.dumps(sorted(pairs), separators=(",", ":")).encode()
        ).hexdigest()[:32]

    @staticmethod
    def _cleanup(path: Path) -> None:
        for child in path.iterdir():
            child.unlink()
        path.rmdir()


def _now() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
