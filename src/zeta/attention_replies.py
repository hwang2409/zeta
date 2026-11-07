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

from .attention_records import AttentionRecord, AttentionStore
from .project_inbox import ProjectInbox
from .runtime.project_registry_state import (
    ProjectRegistryState,
    available_project_registry,
)

_MAX_REPLY_BYTES = 32_000

@dataclass(frozen=True, slots=True)
class RecoveryResult:
    staged: tuple[tuple[str, str], ...]
    published: bool = False

class AttentionReplies:
    """The only owner of terminal attention transitions for one source session."""
    def __init__(self, home: Path, source_session_id: str, *, registry_state: ProjectRegistryState | None = None):
        self.home = Path(home)
        self.source_session_id = source_session_id
        self.session_dir = self.home / "sessions" / source_session_id
        self.attention_dir = self.session_dir / "attention"
        self.staged_dir = self.attention_dir / "staged"
        self.registry_state = registry_state

    @contextmanager
    def _lock(self) -> Iterator[None]:
        self.attention_dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.attention_dir / ".resolution.lock", os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def recover(self) -> RecoveryResult:
        with self._lock():
            return self._recover_locked()

    def _recover_locked(self) -> RecoveryResult:
        published = False
        journals = sorted(self.attention_dir.glob("sending-*-*"))
        for journal in journals:
            self._send_journal(journal)
            done = journal.with_name(journal.name.replace("sending-", "done-", 1))
            os.replace(journal, done)
            published = True
            done.rmdir()
        for path in self.attention_dir.glob("done-*"):
            if path.is_dir():
                self._cleanup(path)
        for path in self.attention_dir.glob(".batch-*"):
            if path.is_dir():
                self._cleanup(path)
        staged = []
        if self.staged_dir.is_dir():
            for path in sorted(self.staged_dir.iterdir()):
                if path.is_file():
                    staged.append((path.name, path.read_text(encoding="utf-8")))
        return RecoveryResult(tuple(staged), published)

    def stage(self, attention_id: str, reply: str) -> None:
        reply = self._validate_reply(reply)
        with self._lock():
            self._recover_locked()
            record = self._record(attention_id)
            self._check_open(record)
            self.staged_dir.mkdir(parents=True, exist_ok=True)
            path = self.staged_dir / attention_id
            if path.exists():
                if path.read_text(encoding="utf-8") != reply:
                    raise ValueError(
                        f"attention {attention_id} has a staged reply in a TUI decisions popup; "
                        "finish or exit that popup before acting on it here"
                    )
                return
            tmp = self.staged_dir / f".{attention_id}.{secrets.token_hex(8)}"
            tmp.write_text(reply, encoding="utf-8")
            os.chmod(tmp, 0o600)
            os.replace(tmp, path)

    def flush(self, from_session: str) -> RecoveryResult:
        with self._lock():
            result = self._recover_locked()
            pairs = list(result.staged)
            if not pairs:
                return result
            journal = self._prepare_locked(pairs)
            for attention_id, _ in pairs:
                (self.staged_dir / attention_id).unlink()
            self._send_journal(journal, from_session=from_session)
            done = journal.with_name(journal.name.replace("sending-", "done-", 1))
            os.replace(journal, done)
            self._cleanup(done)
            return RecoveryResult((), True)

    def reply(self, attention_id: str, reply: str, from_session: str) -> tuple[str, bool]:
        reply = self._validate_reply(reply)
        with self._lock():
            result = self._recover_locked()
            if any(item_id == attention_id for item_id, _ in result.staged):
                raise ValueError(
                    f"attention {attention_id} has a staged reply in a TUI decisions popup; "
                    "finish or exit that popup before acting on it here"
                )
            record = self._record(attention_id)
            if record.status == "resolved":
                if record.decision != reply:
                    raise ValueError("attention item is already resolved with a different reply")
                return self._message_id([(attention_id, reply)]), True
            journal = self._prepare_locked([(attention_id, reply)])
            message_id = journal.name.rsplit("-", 1)[-1]
            self._send_journal(journal, from_session=from_session)
            done = journal.with_name(journal.name.replace("sending-", "done-", 1))
            os.replace(journal, done)
            self._cleanup(done)
            return message_id, False

    def dismiss(self, attention_id: str) -> bool:
        with self._lock():
            result = self._recover_locked()
            if any(item_id == attention_id for item_id, _ in result.staged):
                raise ValueError(
                    f"attention {attention_id} has a staged reply in a TUI decisions popup; "
                    "finish or exit that popup before acting on it here"
                )
            record = self._record(attention_id)
            if record.status == "resolved":
                return True
            if record.kind == "decision":
                raise ValueError("decision attention items require a reply")
            AttentionStore(self.session_dir).replace(replace(record, status="resolved", resolved_at=_now(), decision=None))
            return False

    def stage_and_flush(self, replies: list[tuple[str, str]], from_session: str) -> RecoveryResult:
        with self._lock():
            result = self._recover_locked()
            staged = dict(result.staged)
            if any(item_id in staged for item_id, _ in replies):
                item_id = next(item_id for item_id, _ in replies if item_id in staged)
                raise ValueError(
                    f"attention {item_id} has a staged reply in a TUI decisions popup; "
                    "finish or exit that popup before acting on it here"
                )
            pairs = [(item_id, self._validate_reply(reply)) for item_id, reply in replies]
            for item_id, _ in pairs:
                self._check_open(self._record(item_id))
            journal = self._prepare_locked(pairs)
            self._send_journal(journal, from_session=from_session)
            done = journal.with_name(journal.name.replace("sending-", "done-", 1))
            os.replace(journal, done)
            self._cleanup(done)
            return RecoveryResult(tuple(result.staged), True)

    def _record(self, attention_id: str) -> AttentionRecord:
        return AttentionStore(self.session_dir).get(attention_id)

    def _check_open(self, record: AttentionRecord) -> None:
        if record.session_id != self.source_session_id:
            raise ValueError("attention record does not belong to this session")
        if record.status == "resolved":
            raise ValueError("attention item is already resolved")

    @staticmethod
    def _validate_reply(reply: str) -> str:
        if not isinstance(reply, str) or not reply.strip() or len(reply.encode()) > _MAX_REPLY_BYTES:
            raise ValueError("reply must be nonempty and within the size limit")
        return reply.strip()

    def _prepare_locked(self, pairs: list[tuple[str, str]]) -> Path:
        for item_id, reply in pairs:
            self._check_open(self._record(item_id))
        sequence = 0
        for path in self.attention_dir.glob("sending-*"):
            try: sequence = max(sequence, int(path.name.split("-", 2)[1]))
            except (ValueError, IndexError): raise ValueError("invalid attention journal sequence")
        for path in self.attention_dir.glob("done-*"):
            try: sequence = max(sequence, int(path.name.split("-", 2)[1]))
            except (ValueError, IndexError): raise ValueError("invalid attention journal sequence")
        message_id = self._message_id(pairs)
        private = self.attention_dir / f".batch-{secrets.token_hex(8)}"
        if not pairs:
            raise ValueError("attention batch must not be empty")
        private.mkdir(mode=0o700)
        for item_id, reply in pairs:
            path = private / item_id
            path.write_text(reply, encoding="utf-8")
            os.chmod(path, 0o600)
        self._sync_directory(private)
        journal = self.attention_dir / f"sending-{sequence + 1:020d}-{message_id}"
        os.replace(private, journal)
        self._sync_directory()
        return journal

    def _sync_directory(self, directory: Path | None = None) -> None:
        directory = directory or self.attention_dir
        fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def _send_journal(self, journal: Path, *, from_session: str | None = None) -> None:
        pairs = [(path.name, path.read_text(encoding="utf-8")) for path in sorted(journal.iterdir())]
        if not pairs:
            raise ValueError("attention journal is empty")
        state = self.registry_state or available_project_registry(self.home / "projects")
        if state.registry is None:
            raise ValueError("project registry is unavailable")
        inbox = ProjectInbox(state.registry, sessions_root=self.home / "sessions")
        records = [self._record(item_id) for item_id, _ in pairs]
        if any(record.project_id is None for record in records):
            raise ValueError("attention record has no project")
        sender = from_session or self.source_session_id
        first = records[0]
        label = "Decision" if first.kind == "decision" else "Reply"
        sections = []
        for record, (_, reply) in zip(records, pairs):
            item_label = "Decision" if record.kind == "decision" else "Reply"
            sections.append(f"{item_label}: {reply}")
        inbox = ProjectInbox(
            (self.registry_state or available_project_registry(self.home / "projects")).registry,
            sessions_root=self.home / "sessions",
        )
        inbox.send(
            from_project=first.project_id,
            from_session=sender,
            to_project=first.project_id,
            to_session=first.session_id,
            kind="reply",
            title=f"{label}: {first.title}" if len(records) == 1 else f"{label} batch ({len(records)})",
            body=f"User relayed from session {sender}. Check whether the situation has changed before acting.\\n\\n" + "\\n\\n".join(sections),
            message_id=self._message_id(pairs),
        )
        for record, (_, reply) in zip(records, pairs):
            current = self._record(record.id)
            if current.status == "open":
                AttentionStore(self.session_dir).replace(replace(current, status="resolved", resolved_at=_now(), decision=reply))

    @staticmethod
    def _message_id(pairs: list[tuple[str, str]]) -> str:
        return hashlib.sha256(json.dumps(sorted(pairs), separators=(",", ":")).encode()).hexdigest()[:32]

    @staticmethod
    def _cleanup(path: Path) -> None:
        for child in path.iterdir(): child.unlink()
        path.rmdir()

def _now() -> str:
    from datetime import UTC, datetime
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")
