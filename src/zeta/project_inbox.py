"""Private file-based inboxes between projects in one Zeta home."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import logging
import os
import re
import stat
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .core.checkpoints import ConversationEntry
from .core.session_files import (
    SessionError,
    atomic_publish_file,
    child_directory,
    open_session_file,
    read_session_file,
    session_root,
    write_session_json,
)
from .project_errors import ProjectNotFoundError
from .project_registry import Project, ProjectRegistry, ProjectRegistryError
from .protocol.types import Message, is_passive_harness_message
from .session_liveness import session_is_live

SCHEMA_VERSION = 1
BODY_SPILL_BYTES = 64 * 1024
DONE_HISTORY_LIMIT = 100
SENT_HISTORY_LIMIT = 512
SENT_PAGE_LIMIT = 100
_SENT_TRACKING_FILE = "project-inbox-sent.json"
SENT_STATUS_EVENT = "project_inbox_sent_status"
KINDS = frozenset({"bug_report", "change_request", "question", "info", "reply"})
LOCAL_ORIGIN = "local"
_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_FILE_BYTES = 10 * 1024 * 1024
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 100_000
_MAX_INVALID_REASON_BYTES = 512
_MAX_INVALID_REPORTS_PER_STATUS = 100
_MAX_LOGGED_INVALID = 1_000
_WAKE_CLAIM_SECONDS = 10
_LOG = logging.getLogger(__name__)
_LOGGED_INVALID: OrderedDict[
    tuple[str, int, int, str, str, str, int, int, int], None
] = OrderedDict()


class InboxError(ValueError):
    """An inbox operation or stored message was rejected."""


class _InboxNotFound(FileNotFoundError):
    """The optional top-level inbox directory does not exist."""


def _now() -> str:
    return (
        dt.datetime.now(dt.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


def _text(value: object, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value) or "\x00" in value:
        raise InboxError(
            f"{field} must be a valid {'possibly empty ' if empty else ''}string"
        )
    return value


def _reply_id(message_id: str) -> str:
    return hashlib.sha256(f"project-inbox-reply:{message_id}".encode()).hexdigest()[:32]


def _id(value: object, field: str = "message id") -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise InboxError(f"invalid {field}")
    return value


def _bounded_error(message: str) -> InboxError:
    payload = message.encode("utf-8", errors="replace")
    if len(payload) > _MAX_INVALID_REASON_BYTES:
        message = payload[:_MAX_INVALID_REASON_BYTES].decode("utf-8", errors="ignore")
    return InboxError(message)


def _validate_json_shape(value: object) -> None:
    stack = [(value, 0)]
    nodes = 0
    while stack:
        current, depth = stack.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise InboxError("inbox message structure is too large")
        if not isinstance(current, (dict, list)):
            continue
        if depth > _MAX_JSON_DEPTH:
            raise InboxError("inbox message nesting is too deep")
        children = current.values() if isinstance(current, dict) else current
        for child in children:
            stack.append((child, depth + 1))


class SentMessageTracker:
    """Own the bounded sent-message ledger and its in-memory pending set."""

    def __init__(self, sessions_root: Path, session_id: str) -> None:
        self.sessions_root = Path(sessions_root)
        self.session_id = _id(session_id, "session id")
        self._lock = threading.RLock()
        self._records = self._load()
        self._pending = {
            record["id"] for record in self._records if "done" not in record["reported"]
        }

    @contextmanager
    def _session_directory(self, *, create: bool) -> Iterator[int]:
        with (
            session_root(self.sessions_root, create=create) as root_fd,
            child_directory(root_fd, self.session_id, create=create) as session_fd,
        ):
            yield session_fd

    def record(self, message_id: str, project_id: str) -> None:
        message_id = _id(message_id)
        project_id = _text(project_id, "target project")
        with self._lock:
            existing = next(
                (record for record in self._records if record["id"] == message_id),
                None,
            )
            if existing is not None:
                if existing["target_project"] != project_id:
                    raise InboxError("sent-message tracking record conflicts")
                return
            self._records.append(
                {
                    "id": message_id,
                    "target_project": project_id,
                    "created_at": _now(),
                    "reported": [],
                }
            )
            self._records.sort(key=self._chronology)
            self._pending.add(message_id)
            self._prune()
            self._persist()

    def records(self) -> tuple[dict[str, Any], ...]:
        """Return newest first for the read-only ``sent`` action."""
        with self._lock:
            return tuple(dict(record) for record in reversed(self._records))

    def pending_records(self) -> tuple[dict[str, Any], ...]:
        """Return pending records without filesystem access."""
        with self._lock:
            return tuple(
                dict(record)
                for record in self._records
                if record["id"] in self._pending
            )

    def mark_reported(self, statuses: Iterable[tuple[str, str]]) -> None:
        updates = {
            (_id(message_id), status)
            for message_id, status in statuses
            if status in {"claimed", "done"}
        }
        if not updates:
            return
        with self._lock:
            changed = False
            by_id = {record["id"]: record for record in self._records}
            for message_id, status in updates:
                record = by_id.get(message_id)
                if record is None or status in record["reported"]:
                    continue
                record["reported"] = [*record["reported"], status]
                if status == "done":
                    self._pending.discard(message_id)
                changed = True
            if changed:
                self._persist()

    def reconcile(self, entries: Iterable[ConversationEntry]) -> None:
        """Recover receipts from all physical passive-note transcript entries."""
        statuses: list[tuple[str, str]] = []
        for entry in entries:
            if entry.type != "message":
                continue
            raw = entry.data.get("message")
            if not isinstance(raw, dict) or not is_passive_harness_message(raw):
                continue
            try:
                message = Message.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                continue
            if message.metadata.get("zeta_event") != SENT_STATUS_EVENT:
                continue
            raw_statuses = message.metadata.get("sent_statuses")
            if not isinstance(raw_statuses, list):
                continue
            for item in raw_statuses:
                if not isinstance(item, dict):
                    continue
                message_id = item.get("message_id")
                status = item.get("status")
                if (
                    isinstance(message_id, str)
                    and _ID.fullmatch(message_id)
                    and status in {"claimed", "done"}
                ):
                    statuses.append((message_id, status))
        self.mark_reported(statuses)

    @staticmethod
    def _chronology(record: dict[str, Any]) -> tuple[str, str]:
        return record["created_at"], record["id"]

    def _prune(self) -> None:
        while len(self._records) > SENT_HISTORY_LIMIT:
            terminal = next(
                (
                    record
                    for record in self._records
                    if "done" in record["reported"]
                ),
                None,
            )
            removed = terminal or self._records[0]
            self._records.remove(removed)
            self._pending.discard(removed["id"])

    def _load(self) -> list[dict[str, Any]]:
        try:
            with self._session_directory(create=False) as session_fd:
                payload = read_session_file(session_fd, _SENT_TRACKING_FILE)
        except FileNotFoundError:
            return []
        except (OSError, SessionError, ValueError) as exc:
            raise InboxError("sent-message tracking is unsafe or unavailable") from exc
        if len(payload) > _MAX_FILE_BYTES:
            raise InboxError("sent-message tracking ledger is too large")
        try:
            value = json.loads(payload)
        except (ValueError, RecursionError) as exc:
            raise InboxError("invalid sent-message tracking ledger") from exc
        if not isinstance(value, dict) or value.get("schema_version") != 1:
            raise InboxError("invalid sent-message tracking ledger")
        records = value.get("records")
        if not isinstance(records, list):
            raise InboxError("invalid sent-message tracking ledger")
        validated = [self._validate_record(record) for record in records]
        if len({record["id"] for record in validated}) != len(validated):
            raise InboxError("invalid sent-message tracking ledger")
        validated.sort(key=self._chronology)
        return validated

    @staticmethod
    def _validate_record(value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise InboxError("invalid sent-message tracking record")
        message_id = value.get("id")
        target = value.get("target_project")
        created_at = value.get("created_at")
        reported = value.get("reported")
        if (
            not isinstance(message_id, str)
            or _ID.fullmatch(message_id) is None
            or not isinstance(target, str)
            or not target
            or not isinstance(created_at, str)
            or not isinstance(reported, list)
            or any(
                not isinstance(item, str) or item not in {"claimed", "done"}
                for item in reported
            )
            or len(set(reported)) != len(reported)
        ):
            raise InboxError("invalid sent-message tracking record")
        return {
            "id": message_id,
            "target_project": target,
            "created_at": created_at,
            "reported": list(reported),
        }

    def _persist(self) -> None:
        try:
            with self._session_directory(create=True) as session_fd:
                write_session_json(
                    session_fd,
                    _SENT_TRACKING_FILE,
                    {"schema_version": 1, "records": self._records},
                )
        except (OSError, SessionError, ValueError) as exc:
            raise InboxError("sent-message tracking is unsafe or unavailable") from exc


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
        pending = self.tracker.pending_records()
        if not pending:
            return ()
        source = self.inbox._resolve_project(self.project_id)
        return tuple(
            self.inbox._resolve_tracked_sent(source.project_id, self.session_id, pending)
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


class ProjectInbox:
    """Own inbox storage, validation, claiming, completion, and stale recovery."""

    def __init__(
        self,
        registry: ProjectRegistry,
        *,
        sessions_root: Path,
        sent_tracker: SentMessageTracker | None = None,
    ):
        self.registry = registry
        self.sessions_root = Path(sessions_root)
        self.sent_tracker = sent_tracker

    def known_projects(self) -> list[dict[str, str]]:
        return [
            {"id": project.project_id, "name": project.name, "scope": project.scope}
            for project in self.registry.list_projects()
        ]

    def send(
        self,
        *,
        from_project: str,
        from_session: str,
        to_project: str,
        kind: str,
        title: str,
        body: str,
        in_reply_to: str | None = None,
        message_id: str | None = None,
        to_session: str | None = None,
    ) -> str:
        sender = self._resolve_project(from_project)
        target = self._resolve_project(to_project)
        session = _id(from_session, "session id")
        if kind not in KINDS:
            raise InboxError(f"unknown message kind: {kind}")
        title = _text(title, "title")
        body = _text(body, "body", empty=True)
        if in_reply_to is not None:
            in_reply_to = _id(in_reply_to, "in_reply_to")
        message_id = _id(message_id or uuid.uuid4().hex)
        if to_session is not None:
            to_session = _id(to_session, "target session id")
        record: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "id": message_id,
            "origin": LOCAL_ORIGIN,
            "from": {"project": sender.project_id, "session": session},
            "to_project": target.project_id,
            "to_session": to_session,
            "kind": kind,
            "title": title,
            "body": body,
            "in_reply_to": in_reply_to,
            "created_at": _now(),
        }
        with self._directories(target.project_id, create=True) as dirs:
            new_fd, _claimed_fd, _done_fd, bodies_fd = dirs[:4]
            body_bytes = body.encode("utf-8")
            if len(body_bytes) > BODY_SPILL_BYTES:
                body_name = f"{message_id}.txt"
                self._publish_once(bodies_fd, body_name, body_bytes)
                record["body"] = {"file": f"bodies/{body_name}"}
            name = f"{message_id}.json"
            existing = None
            for directory_fd in (new_fd, _claimed_fd, _done_fd):
                try:
                    existing = self._read_record(directory_fd, name, bodies_fd)
                except FileNotFoundError:
                    continue
                break
            if existing is None:
                self._publish_once(new_fd, name, self._encode(record))
            else:
                immutable_fields = {
                    "schema_version",
                    "id",
                    "origin",
                    "from",
                    "to_project",
                    "to_session",
                    "kind",
                    "title",
                    "body",
                    "in_reply_to",
                }
                comparable = {
                    key: (
                        existing.get(key, LOCAL_ORIGIN)
                        if key == "origin"
                        else existing.get(key)
                        if key == "to_session"
                        else existing[key]
                    )
                    for key in immutable_fields
                }
                expected = {key: record[key] for key in immutable_fields}
                # Decoding resolves spilled bodies, so compare against the caller body.
                expected["body"] = body
                if comparable != expected:
                    raise InboxError("message id already exists with different content")
        tracker = (
            self.sent_tracker
            if self.sent_tracker is not None and self.sent_tracker.session_id == session
            else SentMessageTracker(self.sessions_root, session)
        )
        tracker.record(message_id, target.project_id)
        return message_id

    def list(
        self, project: str, *, session_id: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        target = self._resolve_project(project)
        invalid: list[dict[str, str]] = []
        with self._directories(target.project_id, create=True) as dirs:
            self._recover_stale(*dirs[:3], bodies_fd=dirs[3], invalid=invalid)
            result = {
                "new": self._read_directory(
                    dirs[0], dirs[3], status="new", invalid=invalid
                ),
                "claimed": self._read_directory(
                    dirs[1], dirs[3], status="claimed", invalid=invalid
                ),
                "done": self._read_directory(
                    dirs[2], dirs[3], status="done", invalid=invalid
                ),
                "invalid": invalid,
            }
        if session_id is not None:
            session_id = _id(session_id, "session id")
            for status in ("new", "claimed", "done"):
                result[status] = [
                    record
                    for record in result[status]
                    if record.get("to_session") in {None, session_id}
                ]
        result["done"].sort(key=lambda item: item.get("done_at", ""), reverse=True)
        return result

    def sent(
        self,
        project: str,
        *,
        session_id: str,
        offset: int = 0,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Return one bounded page resolved from this sender session's tracked IDs."""
        source = self._resolve_project(project)
        session_id = _id(session_id, "session id")
        if type(offset) is not int or offset < 0:
            raise InboxError("offset must be a nonnegative integer")
        if type(limit) is not int or not 1 <= limit <= SENT_PAGE_LIMIT:
            raise InboxError(f"limit must be an integer from 1 to {SENT_PAGE_LIMIT}")

        tracker = (
            self.sent_tracker
            if self.sent_tracker is not None
            and self.sent_tracker.session_id == session_id
            else SentMessageTracker(self.sessions_root, session_id)
        )
        tracked = tracker.records()
        sent = self._resolve_tracked_sent(source.project_id, session_id, tracked)
        page = sent[offset : offset + limit]
        next_offset = offset + len(page)
        return {
            "messages": page,
            "offset": offset,
            "limit": limit,
            "next_offset": next_offset if next_offset < len(sent) else None,
            "total": len(sent),
            "truncated": False,
        }

    def _resolve_tracked_sent(
        self,
        source_project: str,
        source_session: str,
        tracked: tuple[dict[str, Any], ...],
    ) -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []
        for item in tracked:
            found = self._read_tracked_message(
                source_project,
                source_session,
                item["target_project"],
                item["id"],
            )
            if found is None:
                continue
            status, record = found
            try:
                target_name = self.registry.show_project(item["target_project"]).name
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

    def read(self, project: str) -> dict[str, list[dict[str, Any]]]:
        """Read inbox state without creating storage or recovering claims."""
        target = self._resolve_project(project)

        def read(root_fd: int) -> dict[str, list[dict[str, Any]]]:
            try:
                with self._directory_handles(
                    root_fd, target.project_id, create=False, lock=False
                ) as dirs:
                    invalid: list[dict[str, str]] = []
                    result = {
                        "new": self._read_directory(
                            dirs[0], dirs[3], status="new", invalid=invalid
                        ),
                        "claimed": self._read_directory(
                            dirs[1], dirs[3], status="claimed", invalid=invalid
                        ),
                        "done": self._read_directory(
                            dirs[2], dirs[3], status="done", invalid=invalid
                        ),
                        "invalid": invalid,
                    }
            except _InboxNotFound:
                return {"new": [], "claimed": [], "done": [], "invalid": []}
            result["done"].sort(key=lambda item: item.get("done_at", ""), reverse=True)
            return result

        try:
            return self.registry._read(read)
        except (OSError, ProjectRegistryError, SessionError) as exc:
            raise InboxError("inbox storage is unsafe or unavailable") from exc

    def _read_tracked_message(
        self,
        source_project: str,
        source_session: str,
        target_project: str,
        message_id: str,
    ) -> tuple[str, dict[str, Any]] | None:
        name = f"{_id(message_id)}.json"

        def read(root_fd: int) -> tuple[str, dict[str, Any]] | None:
            try:
                with self._directory_handles(
                    root_fd, target_project, create=False, lock=False
                ) as dirs:
                    for index, status in enumerate(("new", "claimed", "done")):
                        try:
                            record = self._read_record(
                                dirs[index], name, dirs[3], resolve_body=False
                            )
                        except FileNotFoundError:
                            continue
                        sender = record.get("from")
                        if not isinstance(sender, dict) or sender != {
                            "project": source_project,
                            "session": source_session,
                        }:
                            raise InboxError("tracked sent message has another sender")
                        return status, record
                    return None
            except _InboxNotFound:
                return None

        try:
            return self.registry._read(read)
        except (OSError, ProjectRegistryError, SessionError) as exc:
            raise InboxError("inbox storage is unsafe or unavailable") from exc

    def new_ids(
        self, project: str, *, session_id: str | None = None
    ) -> tuple[str, ...]:
        """Return validated new-message IDs without loading spilled bodies."""
        target = self._resolve_project(project)
        self._ensure_directories(target.project_id)
        invalid: list[dict[str, str]] = []
        with self._directories(target.project_id, create=True) as dirs:
            self._recover_stale(*dirs[:3], bodies_fd=dirs[3], invalid=invalid)
            records = self._read_directory(
                dirs[0], dirs[3], status="new", invalid=invalid, resolve_body=False
            )
        if session_id is not None:
            session_id = _id(session_id, "session id")
            records = [
                record
                for record in records
                if record.get("to_session") in {None, session_id}
            ]
        return tuple(record["id"] for record in records)

    def claim_wake(self, project: str, message_ids: tuple[str, ...]) -> bool:
        """Atomically claim one short-lived model wake for an inbox change."""
        target = self._resolve_project(project)
        if not message_ids:
            return False
        digest = hashlib.sha256("\n".join(message_ids).encode()).hexdigest()
        name = f"{digest}.wake"
        now_ns = time.time_ns()
        self._ensure_directories(target.project_id)
        with self._directories(target.project_id, create=True) as dirs:
            wake_fd = dirs[4]
            for existing in os.listdir(wake_fd):
                info = os.stat(existing, dir_fd=wake_fd, follow_symlinks=False)
                if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                    raise InboxError("unsafe inbox wake claim")
                if now_ns - info.st_mtime_ns > _WAKE_CLAIM_SECONDS * 1_000_000_000:
                    os.unlink(existing, dir_fd=wake_fd)
            try:
                fd = open_session_file(
                    wake_fd, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                )
            except FileExistsError:
                return False
            with os.fdopen(fd, "wb") as stream:
                stream.write(b"wake\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.fsync(wake_fd)
            return True

    def claim(
        self, project: str, message_id: str, session_id: str
    ) -> dict[str, Any] | None:
        target = self._resolve_project(project)
        message_id = _id(message_id)
        session_id = _id(session_id, "session id")
        name = f"{message_id}.json"
        with self._directories(target.project_id, create=True) as dirs:
            new_fd, claimed_fd, _done_fd, bodies_fd = dirs[:4]
            try:
                record = self._read_record(new_fd, name, bodies_fd, resolve_body=False)
                target_session = record.get("to_session")
                if target_session is not None and target_session != session_id:
                    return None
                os.rename(name, name, src_dir_fd=new_fd, dst_dir_fd=claimed_fd)
            except FileNotFoundError:
                return None
            except InboxError as exc:
                self._report_invalid([], new_fd, "new", name, exc)
                return None
            except OSError as exc:
                if exc.errno in {2, 17}:
                    return None
                raise InboxError("could not claim message") from exc
            record["claimer_session"] = session_id
            record["claimed_at"] = _now()
            atomic_publish_file(
                claimed_fd, name, self._encode(record), sync_directory=True
            )
            return self._read_record(claimed_fd, name, bodies_fd)

    def done(
        self,
        project: str,
        message_id: str,
        session_id: str,
        outcome: str,
        *,
        reply: str | None = None,
    ) -> dict[str, Any]:
        target = self._resolve_project(project)
        message_id = _id(message_id)
        session_id = _id(session_id, "session id")
        outcome = _text(outcome, "outcome")
        if reply is not None:
            reply = _text(reply, "reply", empty=True)
        name = f"{message_id}.json"
        reply_id = _reply_id(message_id) if reply is not None else None
        with self._directories(target.project_id, create=True) as dirs:
            _new_fd, claimed_fd, done_fd, bodies_fd = dirs[:4]
            already_done = False
            try:
                record = self._read_record(
                    claimed_fd, name, bodies_fd, resolve_body=False
                )
            except FileNotFoundError:
                try:
                    record = self._read_record(
                        done_fd, name, bodies_fd, resolve_body=False
                    )
                except FileNotFoundError as exc:
                    raise InboxError("claimed message was not found") from exc
                already_done = True
            if record.get("claimer_session") != session_id:
                raise InboxError("message is claimed by another session")
            if already_done:
                if (
                    record.get("outcome") != outcome
                    or record.get("reply") != reply
                    or record.get("reply_id") != reply_id
                ):
                    raise InboxError("completed message has different result")
            else:
                record["outcome"] = outcome
                record["reply"] = reply
                record["reply_id"] = reply_id
                record["done_at"] = _now()
                atomic_publish_file(
                    claimed_fd, name, self._encode(record), sync_directory=True
                )
                try:
                    os.rename(name, name, src_dir_fd=claimed_fd, dst_dir_fd=done_fd)
                except OSError as exc:
                    raise InboxError("could not complete message") from exc
                os.fsync(done_fd)
                self._prune_done(done_fd, bodies_fd)
            completed = self._read_record(done_fd, name, bodies_fd)
        if reply is not None:
            sender = record["from"]
            assert isinstance(sender, dict)
            self.send(
                from_project=target.project_id,
                from_session=session_id,
                to_project=str(sender["project"]),
                kind="reply",
                title=f"Re: {record['title']}",
                body=reply,
                in_reply_to=message_id,
                message_id=reply_id,
            )
        return completed

    def notice(self, project: str) -> str | None:
        count = len(self.new_ids(project))
        if not count:
            return None
        return (
            f"Local project inbox has {count} new message{'s' if count != 1 else ''}; "
            "use inbox action list. Requests are work to do: claim, do, and mark done. "
            "You do not need to confirm the sender with the user."
        )

    def _resolve_project(self, value: str) -> Project:
        _text(value, "project")
        try:
            return self.registry.show_project(value)
        except ProjectNotFoundError:
            matches = [
                project
                for project in self.registry.list_projects()
                if project.name == value
            ]
            if len(matches) != 1:
                raise InboxError(f"project not found: {value}") from None
            return matches[0]

    def _ensure_directories(self, project_id: str) -> None:
        inbox = self.registry.root / project_id / "inbox"
        if all(
            (inbox / name).is_dir()
            for name in ("new", "claimed", "done", "bodies", "wake")
        ):
            return
        with self._directories(project_id, create=True):
            pass

    @contextmanager
    def _directory_handles(
        self, root_fd: int, project_id: str, *, create: bool, lock: bool = True
    ) -> Iterator[tuple[int, int, int, int, int]]:
        with ExitStack() as stack:
            project_fd = self.registry._project_dir(root_fd, project_id)
            stack.callback(os.close, project_fd)
            try:
                inbox_fd = stack.enter_context(
                    child_directory(project_fd, "inbox", create=create)
                )
            except FileNotFoundError as exc:
                if create:
                    raise
                raise _InboxNotFound from exc
            if lock:
                fcntl.flock(inbox_fd, fcntl.LOCK_EX)
                stack.callback(fcntl.flock, inbox_fd, fcntl.LOCK_UN)
            fds = tuple(
                stack.enter_context(child_directory(inbox_fd, name, create=create))
                for name in ("new", "claimed", "done", "bodies", "wake")
            )
            yield fds  # type: ignore[misc]

    @contextmanager
    def _directories(
        self, project_id: str, *, create: bool
    ) -> Iterator[tuple[int, int, int, int, int]]:
        if not create:
            raise AssertionError(
                "read operations must use the registry read transaction"
            )
        try:
            with (
                self.registry._locked(write=True) as root_fd,
                self._directory_handles(root_fd, project_id, create=True) as fds,
            ):
                yield fds
        except InboxError:
            raise
        except (OSError, ProjectRegistryError, SessionError) as exc:
            raise InboxError("inbox storage is unsafe or unavailable") from exc

    @staticmethod
    def _encode(record: dict[str, Any]) -> bytes:
        return (
            json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

    @staticmethod
    def _publish_once(directory_fd: int, name: str, data: bytes) -> None:
        try:
            fd = open_session_file(
                directory_fd, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL
            )
        except FileExistsError:
            with os.fdopen(
                open_session_file(directory_fd, name, os.O_RDONLY), "rb"
            ) as stream:
                if stream.read() != data:
                    raise InboxError(
                        "existing inbox file has different content"
                    ) from None
            return
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory_fd)

    def _read_record(
        self,
        directory_fd: int,
        name: str,
        bodies_fd: int,
        *,
        resolve_body: bool = True,
    ) -> dict[str, Any]:
        """Load one record while containing every file-specific failure."""
        try:
            return self._load_record(
                directory_fd, name, bodies_fd, resolve_body=resolve_body
            )
        except FileNotFoundError:
            raise
        except Exception as exc:
            if isinstance(exc, InboxError):
                error = _bounded_error(str(exc))
            else:
                error = _bounded_error(f"malformed inbox message: {name}")
            raise error from exc

    def _load_record(
        self,
        directory_fd: int,
        name: str,
        bodies_fd: int,
        *,
        resolve_body: bool,
    ) -> dict[str, Any]:
        fd = open_session_file(directory_fd, name, os.O_RDONLY)
        with os.fdopen(fd, "rb") as stream:
            if os.fstat(stream.fileno()).st_size > _MAX_FILE_BYTES:
                raise InboxError(f"inbox message is too large: {name}")
            data = stream.read(_MAX_FILE_BYTES + 1)
        if len(data) > _MAX_FILE_BYTES:
            raise InboxError(f"inbox message is too large: {name}")
        value = json.loads(data)
        _validate_json_shape(value)
        if not isinstance(value, dict):
            raise InboxError(f"unknown inbox schema: {name}")
        schema_version = value.get("schema_version")
        if (
            not isinstance(schema_version, int)
            or isinstance(schema_version, bool)
            or schema_version != SCHEMA_VERSION
        ):
            raise InboxError(f"unknown inbox schema: {name}")
        message_id = _id(value.get("id"))
        if name != f"{message_id}.json":
            raise InboxError("message id does not match filename")
        required = {
            "schema_version",
            "id",
            "from",
            "to_project",
            "kind",
            "title",
            "body",
            "in_reply_to",
            "created_at",
        }
        if not required.issubset(value):
            raise InboxError(f"invalid inbox message fields: {name}")
        origin = value.setdefault("origin", LOCAL_ORIGIN)
        _text(origin, "message origin")
        if value.get("to_session") is not None:
            _id(value["to_session"], "target session")
        sender = value["from"]
        if not isinstance(sender, dict) or not {"project", "session"}.issubset(sender):
            raise InboxError("invalid message sender")
        _text(sender.get("project"), "sender project")
        _id(sender.get("session"), "sender session")
        _text(value.get("to_project"), "target project")
        if value.get("kind") not in KINDS:
            raise InboxError("invalid message kind")
        _text(value.get("title"), "title")
        _text(value.get("created_at"), "created_at")
        if "claimer_session" in value:
            _id(value["claimer_session"], "claimer session")
        for field in ("claimed_at", "recovery_note", "outcome", "done_at"):
            if field in value:
                _text(value[field], field)
        if "reply" in value and value["reply"] is not None:
            _text(value["reply"], "reply", empty=True)
        if "reply_id" in value and value["reply_id"] is not None:
            _id(value["reply_id"], "reply id")
        if value.get("in_reply_to") is not None:
            _id(value["in_reply_to"], "in_reply_to")
        body = value["body"]
        if isinstance(body, dict):
            if body != {"file": f"bodies/{message_id}.txt"}:
                raise InboxError("invalid body reference")
            try:
                with os.fdopen(
                    open_session_file(bodies_fd, f"{message_id}.txt", os.O_RDONLY),
                    "rb",
                ) as stream:
                    if resolve_body:
                        value["body"] = stream.read().decode("utf-8")
            except (OSError, SessionError, UnicodeDecodeError) as exc:
                raise InboxError("unsafe or missing message body") from exc
        else:
            _text(body, "body", empty=True)
        return value

    def _read_directory(
        self,
        directory_fd: int,
        bodies_fd: int,
        *,
        status: str,
        invalid: list[dict[str, str]],
        resolve_body: bool = True,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        records = []
        names = os.listdir(directory_fd)
        if limit is not None:
            names.sort(
                key=lambda name: (
                    os.stat(
                        name, dir_fd=directory_fd, follow_symlinks=False
                    ).st_mtime_ns
                ),
                reverse=True,
            )
            names = names[:limit]
        else:
            names.sort()
        for name in names:
            try:
                if not name.endswith(".json"):
                    raise InboxError(f"unexpected inbox file: {name}")
                record = self._read_record(
                    directory_fd, name, bodies_fd, resolve_body=resolve_body
                )
            except InboxError as exc:
                self._report_invalid(invalid, directory_fd, status, name, exc)
                continue
            records.append(record)
        return records

    def _report_invalid(
        self,
        invalid: list[dict[str, str]],
        directory_fd: int,
        status: str,
        name: str,
        error: InboxError,
    ) -> None:
        reason = str(error)
        entry = {"filename": name, "reason": reason, "status": status}
        if (
            entry not in invalid
            and sum(item["status"] == status for item in invalid)
            < _MAX_INVALID_REPORTS_PER_STATUS
        ):
            invalid.append(entry)

        directory = os.fstat(directory_fd)
        try:
            file_state = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
            file_identity = (
                file_state.st_ino,
                file_state.st_size,
                file_state.st_mtime_ns,
            )
        except OSError:
            file_identity = (0, 0, 0)
        log_key = (
            os.fspath(self.registry.root),
            directory.st_dev,
            directory.st_ino,
            status,
            name,
            reason,
            *file_identity,
        )
        if log_key in _LOGGED_INVALID:
            _LOGGED_INVALID.move_to_end(log_key)
            return
        _LOGGED_INVALID[log_key] = None
        if len(_LOGGED_INVALID) > _MAX_LOGGED_INVALID:
            _LOGGED_INVALID.popitem(last=False)
        _LOG.warning(
            "Skipping invalid project inbox message %s/%s: %s",
            status,
            name,
            reason,
        )

    def _session_alive(self, session_id: str) -> bool:
        return session_is_live(self.sessions_root / session_id)

    def _recover_stale(
        self,
        new_fd: int,
        claimed_fd: int,
        done_fd: int,
        *,
        bodies_fd: int,
        invalid: list[dict[str, str]],
    ) -> None:
        del done_fd
        for name in list(os.listdir(claimed_fd)):
            try:
                record = self._read_record(
                    claimed_fd, name, bodies_fd, resolve_body=False
                )
            except InboxError as exc:
                self._report_invalid(invalid, claimed_fd, "claimed", name, exc)
                continue
            claimer = record.get("claimer_session")
            if isinstance(claimer, str) and self._session_alive(claimer):
                continue
            record.pop("claimer_session", None)
            claimed_at = record.pop("claimed_at", None)
            record["recovery_note"] = (
                f"Returned from stale claim{f' made at {claimed_at}' if claimed_at else ''}."
            )
            atomic_publish_file(
                claimed_fd, name, self._encode(record), sync_directory=True
            )
            os.rename(name, name, src_dir_fd=claimed_fd, dst_dir_fd=new_fd)
            os.fsync(new_fd)

    def _prune_done(self, done_fd: int, bodies_fd: int) -> None:
        invalid: list[dict[str, str]] = []
        valid_names = []
        for name in os.listdir(done_fd):
            if not name.endswith(".json"):
                continue
            try:
                self._read_record(done_fd, name, bodies_fd, resolve_body=False)
            except InboxError as exc:
                self._report_invalid(invalid, done_fd, "done", name, exc)
                continue
            valid_names.append(name)
        names = sorted(
            valid_names,
            key=lambda name: (
                os.stat(name, dir_fd=done_fd, follow_symlinks=False).st_mtime_ns
            ),
            reverse=True,
        )
        for name in names[DONE_HISTORY_LIMIT:]:
            info = os.stat(name, dir_fd=done_fd, follow_symlinks=False)
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise InboxError("unsafe done history file")
            os.unlink(name, dir_fd=done_fd)
            body_name = f"{name.removesuffix('.json')}.txt"
            try:
                body_info = os.stat(body_name, dir_fd=bodies_fd, follow_symlinks=False)
            except FileNotFoundError:
                continue
            if not stat.S_ISREG(body_info.st_mode) or body_info.st_nlink != 1:
                raise InboxError("unsafe done history body")
            os.unlink(body_name, dir_fd=bodies_fd)
