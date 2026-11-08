from __future__ import annotations

import json
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .. import project_inbox as _owner
from ..core.session_files import (
    SessionError,
    atomic_publish_file,
    child_directory,
    read_session_file,
    session_root,
)
from ..project_inbox import InboxError, ProjectInbox
from ..storage_format import open_writer, require_closed_variant


@dataclass(frozen=True)
class _Missing:
    path: Path


@dataclass(frozen=True)
class _Loaded:
    path: Path
    records: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class _Broken:
    path: Path
    reason: str


class SentMessageTracker:
    """Own the bounded sent ledger and its read-time broken projection."""

    def __init__(self, sessions_root: Path, session_id: str) -> None:
        self.sessions_root = Path(sessions_root)
        self.session_id = _owner._id(session_id, "session id")
        self.path = self.sessions_root / self.session_id / _owner._SENT_TRACKING_FILE
        self._lock = threading.RLock()
        self._deleted_ids: set[str] = set()
        self._receipts: tuple[tuple[str, str], ...] = ()
        self._state = self._load_state()
        self._records = (
            list(self._state.records) if isinstance(self._state, _Loaded) else []
        )
        self._pending = {
            record["id"] for record in self._records if "done" not in record["reported"]
        }

    @property
    def broken_cached(self) -> bool:
        return isinstance(self._state, _Broken)

    @property
    def broken(self) -> bool:
        with self._lock:
            self._refresh()
            return isinstance(self._state, _Broken)

    def health(self) -> dict[str, Any]:
        with self._lock:
            self._refresh()
            if not isinstance(self._state, _Broken):
                return {"state": "healthy", "history_complete": True}
            return {
                "state": "broken",
                "reason": self._state.reason,
                "ledger": str(self._state.path.relative_to(self.sessions_root.parent)),
                "history_complete": False,
            }

    def health_text(self) -> str:
        health = self.health()
        if health["state"] != "broken":
            return ""
        return (
            f"sent history is incomplete ({health['reason']}) at {health['ledger']}; messages sent or replied "
            "to while this ledger is broken are delivered, but they are not added to "
            "the ledger. Removing it loses those messages from sent history; repairing "
            "it loses them unless the repaired ledger includes their IDs. Later "
            "claimed/done updates for a lost ID will not be discovered. The file was "
            "not changed."
        )

    def _refresh(self) -> None:
        state = self._load_state()
        self._state = state
        if isinstance(state, _Loaded):
            self._records = list(state.records)
            self._pending = {
                r["id"] for r in self._records if "done" not in r["reported"]
            }
        elif isinstance(state, _Missing):
            self._records = []
            self._pending = set()
        else:
            self._records = []
            self._pending = set()

    @contextmanager
    def _session_directory(self, *, create: bool) -> Iterator[int]:
        with (
            session_root(self.sessions_root, create=create) as root_fd,
            child_directory(root_fd, self.session_id, create=create) as session_fd,
        ):
            yield session_fd

    def record(self, message_id: str, project_id: str) -> None:
        message_id = _owner._id(message_id)
        project_id = _owner._text(project_id, "target project")
        with self._lock:
            self._refresh()
            if isinstance(self._state, _Broken):
                return
            existing = next((r for r in self._records if r["id"] == message_id), None)
            if existing is not None:
                if existing["target_project"] != project_id:
                    raise InboxError("sent-message tracking record conflicts")
                return
            record = {
                "id": message_id,
                "target_project": project_id,
                "created_at": _owner._now(),
                "reported": [],
            }
            self._records.append(record)
            self._records.sort(key=self._chronology)
            self._pending.add(message_id)
            self._prune()
            self._persist((record,), self._deleted_ids)

    def records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._refresh()
            return tuple(dict(record) for record in reversed(self._records))

    def pending_records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._refresh()
            return tuple(dict(r) for r in self._records if r["id"] in self._pending)

    def sent_view(
        self,
        inbox: ProjectInbox,
        source_project: str,
        receipts: Iterable[tuple[str, str]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        with self._lock:
            self._receipts = tuple(receipts)
            self._refresh()
            state = self._state
            tracked = tuple(dict(r) for r in reversed(self._records))
        if isinstance(state, _Broken):
            return inbox._projected_broken_sent(
                source_project, self.session_id, self._receipts
            ), self.health()
        return inbox._resolve_tracked_sent(
            source_project, self.session_id, tracked
        ), self.health()

    def mark_reported(self, statuses: Iterable[tuple[str, str]]) -> None:
        updates = {
            (_owner._id(mid), status)
            for mid, status in statuses
            if status in {"claimed", "done"}
        }
        if not updates:
            return
        with self._lock:
            self._receipts = tuple(set(self._receipts).union(updates))
            self._refresh()
            if isinstance(self._state, _Broken):
                return
            by_id = {r["id"]: r for r in self._records}
            changed = []
            for mid, status in updates:
                record = by_id.get(mid)
                if record is not None and status not in record["reported"]:
                    record["reported"].append(status)
                    changed.append(record)
                    if status == "done":
                        self._pending.discard(mid)
            if changed:
                self._persist(tuple(changed), ())

    def reconcile(self, receipts: Iterable[tuple[str, str]]) -> None:
        self._receipts = tuple(receipts)
        self.mark_reported(self._receipts)

    @staticmethod
    def _chronology(record: dict[str, Any]) -> tuple[str, str]:
        return record["created_at"], record["id"]

    def _prune(self) -> None:
        while len(self._records) > _owner.SENT_HISTORY_LIMIT:
            terminal = next((r for r in self._records if "done" in r["reported"]), None)
            removed = terminal or self._records[0]
            self._records.remove(removed)
            self._pending.discard(removed["id"])
            self._deleted_ids.add(removed["id"])

    def _load_state(self) -> _Missing | _Loaded | _Broken:
        path = self.path
        try:
            with self._session_directory(create=False) as session_fd:
                payload = read_session_file(session_fd, _owner._SENT_TRACKING_FILE)
        except FileNotFoundError:
            return _Missing(path)
        except (OSError, SessionError, ValueError):
            return _Broken(path, "unsafe_or_unavailable")
        if len(payload) > _owner._MAX_FILE_BYTES:
            return _Broken(path, "too_large")
        try:
            value = json.loads(payload)
        except (ValueError, RecursionError):
            return _Broken(path, "invalid_json")
        try:
            version = value.get("schema_version") if isinstance(value, dict) else None
            require_closed_variant("sent_message_ledger.schema_version", version)
        except (ValueError, AttributeError):
            return _Broken(path, "unsupported_schema")
        try:
            records = value.get("records")
            if not isinstance(records, list):
                raise InboxError("invalid")
            validated = [self._validate_record(r) for r in records]
            if len({r["id"] for r in validated}) != len(validated):
                raise InboxError("duplicate")
        except (InboxError, AttributeError, TypeError):
            return _Broken(path, "invalid_records")
        validated.sort(key=self._chronology)
        return _Loaded(path, tuple(validated))

    @staticmethod
    def _validate_record(value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise InboxError("invalid sent-message tracking record")
        mid, target, created, reported = (
            value.get(k) for k in ("id", "target_project", "created_at", "reported")
        )
        if (
            not isinstance(mid, str)
            or _owner._ID.fullmatch(mid) is None
            or not isinstance(target, str)
            or not target
            or not isinstance(created, str)
            or not isinstance(reported, list)
            or any(not isinstance(x, str) for x in reported)
            or len(set(reported)) != len(reported)
        ):
            raise InboxError("invalid sent-message tracking record")
        try:
            for status in reported:
                require_closed_variant("sent_message_ledger.reported_status", status)
        except ValueError as exc:
            raise InboxError("invalid sent-message tracking record") from exc
        return {
            "id": mid,
            "target_project": target,
            "created_at": created,
            "reported": list(reported),
        }

    def _validate_current(self, raw: object) -> None:
        if not isinstance(raw, dict) or raw.get("schema_version") != 1:
            raise ValueError("invalid sent-message tracking ledger")
        records = raw.get("records")
        if not isinstance(records, list):
            raise TypeError("invalid sent-message tracking ledger")
        validated = [self._validate_record(r) for r in records]
        if len({r["id"] for r in validated}) != len(validated):
            raise ValueError("invalid sent-message tracking ledger")

    def _persist(
        self, upserts: Iterable[dict[str, Any]], delete_ids: Iterable[str]
    ) -> None:
        try:
            with (
                open_writer(self.sessions_root.parent) as lease,
                self._session_directory(create=True) as session_fd,
            ):
                lease.publish_records(
                    session_fd,
                    _owner._SENT_TRACKING_FILE,
                    upserts,
                    identity="id",
                    delete_ids=delete_ids,
                    path=("records",),
                    initial={"schema_version": 1, "records": []},
                    validate=self._validate_current,
                    publish=lambda fd, name, data: atomic_publish_file(
                        fd, name, data, sync_directory=True
                    ),
                )
            self._deleted_ids.clear()
        except (ValueError, TypeError):
            self._state = self._load_state()
            self._records = []
            self._pending = set()
            self._deleted_ids.clear()
        except (OSError, SessionError) as exc:
            raise InboxError("sent-message tracking is unsafe or unavailable") from exc
