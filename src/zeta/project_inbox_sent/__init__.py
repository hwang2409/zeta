from __future__ import annotations

import datetime as dt
import json
import re
import threading
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..core.session_files import (
    SessionError,
    atomic_publish_file,
    child_directory,
    read_session_file,
    session_root,
)
from ..storage_format import open_writer, require_closed_variant

if TYPE_CHECKING:
    from ..project_inbox import ProjectInbox


_ID = re.compile(r"[0-9a-f]{32}\Z")
_SENT_TRACKING_FILE = "project-inbox-sent.json"
_MAX_FILE_BYTES = 10 * 1024 * 1024
SENT_HISTORY_LIMIT = 512


def _id(value: object, field: str = "message id") -> str:
    if not isinstance(value, str) or _ID.fullmatch(value) is None:
        raise ValueError(f"invalid {field}")
    return value


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError(f"{field} must be a valid string")
    return value


def _now() -> str:
    return (
        dt.datetime.now(dt.UTC)
        .isoformat(timespec="microseconds")
        .replace("+00:00", "Z")
    )


@dataclass(frozen=True)
class SentLedgerHealth:
    state: str
    history_complete: bool
    reason: str | None = None
    ledger: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "state": self.state,
            "history_complete": self.history_complete,
        }
        if self.reason is not None:
            result["reason"] = self.reason
        if self.ledger is not None:
            result["ledger"] = self.ledger
        return result


@dataclass(frozen=True)
class SentLedgerOutcome[T]:
    result: T
    health: SentLedgerHealth
    warning: str


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


class _BrokenLedger(ValueError):
    pass


class SentMessageTracker:
    """Own the bounded sent ledger and its read-time broken projection."""

    def __init__(self, sessions_root: Path, session_id: str) -> None:
        self.sessions_root = Path(sessions_root)
        self.session_id = _id(session_id, "session id")
        self.path = self.sessions_root / self.session_id / _SENT_TRACKING_FILE
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
        return self.health().health.state == "broken"

    def health(self) -> SentLedgerOutcome[None]:
        with self._lock:
            self._refresh()
            return self._outcome(None)

    def _outcome[T](self, result: T) -> SentLedgerOutcome[T]:
        if not isinstance(self._state, _Broken):
            return SentLedgerOutcome(result, SentLedgerHealth("healthy", True), "")
        health = SentLedgerHealth(
            "broken",
            False,
            self._state.reason,
            str(self._state.path.relative_to(self.sessions_root.parent)),
        )
        warning = (
            f"sent history is incomplete ({health.reason}) at {health.ledger}; messages sent or replied "
            "to while this ledger is broken are delivered, but they are not added to "
            "the ledger. Removing it loses those messages from sent history; repairing "
            "it loses them unless the repaired ledger includes their IDs. Later "
            "claimed/done updates for a lost ID will not be discovered. The file was "
            "not changed."
        )
        return SentLedgerOutcome(result, health, warning)

    def _refresh(self) -> None:
        self._adopt_state(self._load_state())

    def _adopt_state(self, state: _Missing | _Loaded | _Broken) -> None:
        self._state = state
        if isinstance(state, _Loaded):
            self._records = list(state.records)
            self._pending = {
                r["id"] for r in self._records if "done" not in r["reported"]
            }
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

    def record(self, message_id: str, project_id: str) -> SentLedgerOutcome[None]:
        message_id = _id(message_id)
        project_id = _text(project_id, "target project")
        with self._lock:
            self._refresh()
            if isinstance(self._state, _Broken):
                return self._outcome(None)
            existing = next((r for r in self._records if r["id"] == message_id), None)
            if existing is not None:
                if existing["target_project"] != project_id:
                    raise ValueError("sent-message tracking record conflicts")
                return self._outcome(None)
            record = {
                "id": message_id,
                "target_project": project_id,
                "created_at": _now(),
                "reported": [],
            }
            self._records.append(record)
            self._records.sort(key=self._chronology)
            self._pending.add(message_id)
            self._prune()
            return self._persist((record,), self._deleted_ids)

    def records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._refresh()
            return tuple(dict(record) for record in reversed(self._records))

    def sent_view(
        self,
        inbox: ProjectInbox,
        source_project: str,
        receipts: Iterable[tuple[str, str]],
    ) -> SentLedgerOutcome[list[dict[str, Any]]]:
        with self._lock:
            self._receipts = tuple(receipts)
            self._refresh()
            tracked = tuple(dict(r) for r in reversed(self._records))
            if isinstance(self._state, _Broken):
                result = self._projected_broken_sent(inbox, source_project)
            else:
                result = inbox._resolve_tracked_sent(
                    source_project, self.session_id, tracked
                )
            return self._outcome(result)

    def _projected_broken_sent(
        self, inbox: ProjectInbox, source_project: str
    ) -> list[dict[str, Any]]:
        receipt_map: dict[str, set[str]] = {}
        for message_id, status in self._receipts:
            receipt_map.setdefault(message_id, set()).add(status)
        rows: list[dict[str, Any]] = []
        for target in sorted(
            inbox.registry.list_projects(), key=lambda project: project.project_id
        ):
            rows.extend(
                inbox._broken_project_rows(
                    target.project_id,
                    source_project,
                    self.session_id,
                    receipt_map,
                )
            )
        rows.sort(key=lambda item: (item["created_at"], item["id"]))
        while len(rows) > SENT_HISTORY_LIMIT:
            terminal = next(
                (row for row in rows if "done" in row["reported"]), None
            )
            rows.remove(terminal or rows[0])
        return list(reversed(rows))

    def mark_reported(
        self, statuses: Iterable[tuple[str, str]]
    ) -> SentLedgerOutcome[None]:
        updates = {
            (_id(mid), status)
            for mid, status in statuses
            if status in {"claimed", "done"}
        }
        with self._lock:
            self._receipts = tuple(set(self._receipts).union(updates))
            self._refresh()
            if isinstance(self._state, _Broken) or not updates:
                return self._outcome(None)
            by_id = {r["id"]: r for r in self._records}
            changed: dict[str, dict[str, Any]] = {}
            for mid, status in updates:
                record = by_id.get(mid)
                if record is not None and status not in record["reported"]:
                    record["reported"].append(status)
                    changed[mid] = record
                    if status == "done":
                        self._pending.discard(mid)
            if changed:
                return self._persist(tuple(changed.values()), ())
            return self._outcome(None)

    def reconcile(
        self, receipts: Iterable[tuple[str, str]]
    ) -> SentLedgerOutcome[None]:
        self._receipts = tuple(receipts)
        return self.mark_reported(self._receipts)

    @staticmethod
    def _chronology(record: dict[str, Any]) -> tuple[str, str]:
        return record["created_at"], record["id"]

    def _prune(self) -> None:
        while len(self._records) > SENT_HISTORY_LIMIT:
            terminal = next((r for r in self._records if "done" in r["reported"]), None)
            removed = terminal or self._records[0]
            self._records.remove(removed)
            self._pending.discard(removed["id"])
            self._deleted_ids.add(removed["id"])

    def _load_state(self) -> _Missing | _Loaded | _Broken:
        path = self.path
        try:
            with self._session_directory(create=False) as session_fd:
                payload = read_session_file(session_fd, _SENT_TRACKING_FILE)
        except FileNotFoundError:
            return _Missing(path)
        except (OSError, SessionError, ValueError):
            return _Broken(path, "unsafe_or_unavailable")
        if len(payload) > _MAX_FILE_BYTES:
            return _Broken(path, "too_large")
        try:
            value = json.loads(payload)
        except (ValueError, RecursionError):
            return _Broken(path, "invalid_json")
        return self._classify_value(value)

    def _classify_value(self, value: object) -> _Loaded | _Broken:
        path = self.path
        try:
            version = value.get("schema_version") if isinstance(value, dict) else None
            require_closed_variant("sent_message_ledger.schema_version", version)
        except (ValueError, AttributeError):
            return _Broken(path, "unsupported_schema")
        try:
            records = value.get("records")
            if not isinstance(records, list):
                raise TypeError("invalid")
            validated = [self._validate_record(r) for r in records]
            if len({r["id"] for r in validated}) != len(validated):
                raise ValueError("duplicate")
        except (ValueError, AttributeError, TypeError):
            return _Broken(path, "invalid_records")
        validated.sort(key=self._chronology)
        return _Loaded(path, tuple(validated))

    @staticmethod
    def _validate_record(value: object) -> dict[str, Any]:
        if not isinstance(value, dict):
            raise TypeError("invalid sent-message tracking record")
        mid, target, created, reported = (
            value.get(k) for k in ("id", "target_project", "created_at", "reported")
        )
        if (
            not isinstance(mid, str)
            or _ID.fullmatch(mid) is None
            or not isinstance(target, str)
            or not target
            or not isinstance(created, str)
            or not isinstance(reported, list)
            or any(not isinstance(x, str) for x in reported)
            or len(set(reported)) != len(reported)
        ):
            raise ValueError("invalid sent-message tracking record")
        try:
            for status in reported:
                require_closed_variant("sent_message_ledger.reported_status", status)
        except ValueError as exc:
            raise ValueError("invalid sent-message tracking record") from exc
        return {
            "id": mid,
            "target_project": target,
            "created_at": created,
            "reported": list(reported),
        }

    def _persist(
        self, upserts: Iterable[dict[str, Any]], delete_ids: Iterable[str]
    ) -> SentLedgerOutcome[None]:
        operation_outcome: SentLedgerOutcome[None] | None = None

        def guard_current(raw: object) -> None:
            nonlocal operation_outcome
            state = self._classify_value(raw)
            if isinstance(state, _Broken):
                self._adopt_state(state)
                operation_outcome = self._outcome(None)
                raise _BrokenLedger

        def broken_invalid_json() -> None:
            nonlocal operation_outcome
            self._adopt_state(_Broken(self.path, "invalid_json"))
            operation_outcome = self._outcome(None)

        def publish(fd: int, name: str, data: bytes) -> None:
            nonlocal operation_outcome
            atomic_publish_file(fd, name, data, sync_directory=True)
            operation_outcome = self._outcome(None)

        try:
            with (
                open_writer(self.sessions_root.parent) as lease,
                self._session_directory(create=True) as session_fd,
            ):
                try:
                    lease.publish_records(
                        session_fd,
                        _SENT_TRACKING_FILE,
                        upserts,
                        identity="id",
                        delete_ids=delete_ids,
                        path=("records",),
                        initial={"schema_version": 1, "records": []},
                        validate=guard_current,
                        on_invalid_json=broken_invalid_json,
                        publish=publish,
                    )
                except _BrokenLedger:
                    pass
            if operation_outcome is None:
                raise ValueError("sent-message tracking outcome is unavailable")
            self._deleted_ids.clear()
            return operation_outcome
        except (OSError, SessionError, ValueError, TypeError) as exc:
            if operation_outcome is not None:
                self._deleted_ids.clear()
                return operation_outcome
            raise ValueError("sent-message tracking is unsafe or unavailable") from exc
