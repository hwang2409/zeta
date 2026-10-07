"""Private file-based inboxes between projects in one Zeta home."""

from __future__ import annotations

import datetime as dt
import fcntl
import hashlib
import json
import os
import re
import stat
import time
import uuid
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Any

from .core.session_files import (
    SessionError,
    atomic_publish_file,
    child_directory,
    open_session_file,
)
from .core.session_liveness import session_is_live
from .project_errors import ProjectNotFoundError
from .project_registry import Project, ProjectRegistry, ProjectRegistryError

SCHEMA_VERSION = 1
BODY_SPILL_BYTES = 64 * 1024
DONE_HISTORY_LIMIT = 100
KINDS = frozenset({"bug_report", "change_request", "question", "info", "reply"})
LOCAL_ORIGIN = "local"
_ID = re.compile(r"[0-9a-f]{32}\Z")
_MAX_FILE_BYTES = 10 * 1024 * 1024
_WAKE_CLAIM_SECONDS = 10


class InboxError(ValueError):
    """An inbox operation or stored message was rejected."""


class _InboxNotFound(FileNotFoundError):
    """The optional top-level inbox directory does not exist."""


def _now() -> str:
    return dt.datetime.now(dt.UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def _text(value: object, field: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value) or "\x00" in value:
        raise InboxError(f"{field} must be a valid {'possibly empty ' if empty else ''}string")
    return value


def _reply_id(message_id: str) -> str:
    return hashlib.sha256(f"project-inbox-reply:{message_id}".encode()).hexdigest()[:32]


def _id(value: object, field: str = "message id") -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value):
        raise InboxError(f"invalid {field}")
    return value


class ProjectInboxScanner:
    """Skip validated inbox scans while watched directories are unchanged."""

    def __init__(
        self,
        registry: ProjectRegistry,
        project_id: str,
        *,
        sessions_root: Path,
        session_id: str | None = None,
    ) -> None:
        self.inbox = ProjectInbox(registry, sessions_root=sessions_root)
        self.project_id = project_id
        self.session_id = session_id
        self._fingerprint: tuple[tuple[int, int, int, int] | None, ...] | None = None

    def scan(self) -> tuple[str, ...] | None:
        current = self._directory_fingerprint()
        if self._fingerprint is not None and current == self._fingerprint:
            return None
        message_ids = self.inbox.new_ids(self.project_id, session_id=self.session_id)
        self._fingerprint = (
            current if any(item is not None for item in current) else self._directory_fingerprint()
        )
        return message_ids

    def _directory_fingerprint(self) -> tuple[tuple[int, int, int, int] | None, ...]:
        inbox = self.inbox.registry.root / self.project_id / "inbox"
        result: list[tuple[int, int, int, int] | None] = []
        for name in ("new", "claimed"):
            try:
                info = (inbox / name).stat(follow_symlinks=False)
            except FileNotFoundError:
                result.append(None)
            else:
                result.append((info.st_dev, info.st_ino, info.st_mtime_ns, info.st_size))
        return tuple(result)


class ProjectInbox:
    """Own inbox storage, validation, claiming, completion, and stale recovery."""

    def __init__(self, registry: ProjectRegistry, *, sessions_root: Path):
        self.registry = registry
        self.sessions_root = Path(sessions_root)

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
            return message_id

    def list(
        self, project: str, *, session_id: str | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        target = self._resolve_project(project)
        with self._directories(target.project_id, create=True) as dirs:
            self._recover_stale(*dirs[:3], bodies_fd=dirs[3])
            result = {
                "new": self._read_directory(dirs[0], dirs[3]),
                "claimed": self._read_directory(dirs[1], dirs[3]),
                "done": self._read_directory(dirs[2], dirs[3]),
            }
        if session_id is not None:
            session_id = _id(session_id, "session id")
            result = {
                status: [
                    record
                    for record in records
                    if record.get("to_session") in {None, session_id}
                ]
                for status, records in result.items()
            }
        result["done"].sort(key=lambda item: item.get("done_at", ""), reverse=True)
        return result

    def read(self, project: str) -> dict[str, list[dict[str, Any]]]:
        """Read inbox state without creating storage or recovering claims."""
        target = self._resolve_project(project)

        def read(root_fd: int) -> dict[str, list[dict[str, Any]]]:
            try:
                with self._directory_handles(
                    root_fd, target.project_id, create=False
                ) as dirs:
                    result = {
                        "new": self._read_directory(dirs[0], dirs[3]),
                        "claimed": self._read_directory(dirs[1], dirs[3]),
                        "done": self._read_directory(dirs[2], dirs[3]),
                    }
            except _InboxNotFound:
                return {"new": [], "claimed": [], "done": []}
            result["done"].sort(
                key=lambda item: item.get("done_at", ""), reverse=True
            )
            return result

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
        with self._directories(target.project_id, create=True) as dirs:
            self._recover_stale(*dirs[:3], bodies_fd=dirs[3])
            records = self._read_directory(dirs[0], dirs[3], resolve_body=False)
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

    def claim(self, project: str, message_id: str, session_id: str) -> dict[str, Any] | None:
        target = self._resolve_project(project)
        message_id = _id(message_id)
        session_id = _id(session_id, "session id")
        name = f"{message_id}.json"
        with self._directories(target.project_id, create=True) as dirs:
            new_fd, claimed_fd, _done_fd, bodies_fd = dirs[:4]
            try:
                record = self._read_record(
                    new_fd, name, bodies_fd, resolve_body=False
                )
                target_session = record.get("to_session")
                if target_session is not None and target_session != session_id:
                    return None
                os.rename(name, name, src_dir_fd=new_fd, dst_dir_fd=claimed_fd)
            except FileNotFoundError:
                return None
            except OSError as exc:
                if exc.errno in {2, 17}:
                    return None
                raise InboxError("could not claim message") from exc
            record["claimer_session"] = session_id
            record["claimed_at"] = _now()
            atomic_publish_file(claimed_fd, name, self._encode(record), sync_directory=True)
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
            matches = [project for project in self.registry.list_projects() if project.name == value]
            if len(matches) != 1:
                raise InboxError(f"project not found: {value}") from None
            return matches[0]

    def _ensure_directories(self, project_id: str) -> None:
        inbox = self.registry.root / project_id / "inbox"
        if all((inbox / name).is_dir() for name in ("new", "claimed", "done", "bodies", "wake")):
            return
        with self._directories(project_id, create=True):
            pass

    @contextmanager
    def _directory_handles(
        self, root_fd: int, project_id: str, *, create: bool
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
            raise AssertionError("read operations must use the registry read transaction")
        try:
            with self.registry._locked(write=True) as root_fd, self._directory_handles(
                root_fd, project_id, create=True
            ) as fds:
                yield fds
        except InboxError:
            raise
        except (OSError, ProjectRegistryError, SessionError) as exc:
            raise InboxError("inbox storage is unsafe or unavailable") from exc

    @staticmethod
    def _encode(record: dict[str, Any]) -> bytes:
        return (json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n").encode()

    @staticmethod
    def _publish_once(directory_fd: int, name: str, data: bytes) -> None:
        try:
            fd = open_session_file(directory_fd, name, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            with os.fdopen(open_session_file(directory_fd, name, os.O_RDONLY), "rb") as stream:
                if stream.read() != data:
                    raise InboxError("existing inbox file has different content") from None
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
        try:
            fd = open_session_file(directory_fd, name, os.O_RDONLY)
        except FileNotFoundError:
            raise
        except (OSError, SessionError) as exc:
            raise InboxError(f"unsafe inbox message: {name}") from exc
        try:
            with os.fdopen(fd, "rb") as stream:
                data = stream.read(_MAX_FILE_BYTES + 1)
            if len(data) > _MAX_FILE_BYTES:
                raise InboxError(f"inbox message is too large: {name}")
            value = json.loads(data)
        except InboxError:
            raise
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
            raise InboxError(f"malformed inbox message: {name}") from exc
        if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
            raise InboxError(f"unknown inbox schema: {name}")
        message_id = _id(value.get("id"))
        if name != f"{message_id}.json":
            raise InboxError("message id does not match filename")
        required = {"schema_version", "id", "from", "to_project", "kind", "title", "body", "in_reply_to", "created_at"}
        allowed = required | {
            "to_session",
            "origin",
            "claimer_session",
            "claimed_at",
            "recovery_note",
            "outcome",
            "reply",
            "reply_id",
            "done_at",
        }
        if not required.issubset(value) or not set(value).issubset(allowed):
            raise InboxError(f"invalid inbox message fields: {name}")
        origin = value.setdefault("origin", LOCAL_ORIGIN)
        _text(origin, "message origin")
        if value.get("to_session") is not None:
            _id(value["to_session"], "target session id")
        sender = value["from"]
        if not isinstance(sender, dict) or set(sender) != {"project", "session"}:
            raise InboxError("invalid message sender")
        _text(sender.get("project"), "sender project")
        _id(sender.get("session"), "sender session")
        _text(value.get("to_project"), "target project")
        if value.get("kind") not in KINDS:
            raise InboxError("invalid message kind")
        _text(value.get("title"), "title")
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
                    open_session_file(
                        bodies_fd, f"{message_id}.txt", os.O_RDONLY
                    ),
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
        self, directory_fd: int, bodies_fd: int, *, resolve_body: bool = True
    ) -> list[dict[str, Any]]:
        records = []
        for name in sorted(os.listdir(directory_fd)):
            if not name.endswith(".json"):
                raise InboxError(f"unexpected inbox file: {name}")
            records.append(
                self._read_record(
                    directory_fd, name, bodies_fd, resolve_body=resolve_body
                )
            )
        return records

    def _session_alive(self, session_id: str) -> bool:
        return session_is_live(self.sessions_root / session_id)

    def _recover_stale(self, new_fd: int, claimed_fd: int, done_fd: int, *, bodies_fd: int) -> None:
        del done_fd
        for name in list(os.listdir(claimed_fd)):
            record = self._read_record(
                claimed_fd, name, bodies_fd, resolve_body=False
            )
            claimer = record.get("claimer_session")
            if isinstance(claimer, str) and self._session_alive(claimer):
                continue
            record.pop("claimer_session", None)
            claimed_at = record.pop("claimed_at", None)
            record["recovery_note"] = f"Returned from stale claim{f' made at {claimed_at}' if claimed_at else ''}."
            atomic_publish_file(claimed_fd, name, self._encode(record), sync_directory=True)
            os.rename(name, name, src_dir_fd=claimed_fd, dst_dir_fd=new_fd)
            os.fsync(new_fd)

    @staticmethod
    def _prune_done(done_fd: int, bodies_fd: int) -> None:
        names = sorted(
            (name for name in os.listdir(done_fd) if name.endswith(".json")),
            key=lambda name: os.stat(name, dir_fd=done_fd, follow_symlinks=False).st_mtime_ns,
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
