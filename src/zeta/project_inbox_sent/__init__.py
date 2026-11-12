from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
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
from ..storage_format import (
    ReplaceSubtree,
    SetValue,
    StorageFeature,
    StorageFormatError,
    feature_enabled,
    open_writer,
    require_closed_variant,
)
from . import _v2

if TYPE_CHECKING:
    from ..project_inbox import ProjectInbox


_ID = re.compile(r"[0-9a-f]{32}\Z")
_SENT_TRACKING_FILE = "project-inbox-sent.json"
_MAX_FILE_BYTES = 10 * 1024 * 1024
SENT_HISTORY_LIMIT = 512
# Mirrors project_inbox.BODY_SPILL_BYTES; a larger durable intent body spills to
# a sender-owned file rather than inflating the bounded ledger.
_SENDER_SPILL_BYTES = 64 * 1024


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

    def __init__(
        self, sessions_root: Path, session_id: str, *, observed_epoch: int = 0
    ) -> None:
        self.sessions_root = Path(sessions_root)
        self.session_id = _id(session_id, "session id")
        self.observed_epoch = observed_epoch
        self._at_epoch = feature_enabled(
            observed_epoch, StorageFeature.INBOX_DURABLE_INTENT
        )
        self.path = self.sessions_root / self.session_id / _SENT_TRACKING_FILE
        self._lock = threading.RLock()
        self._deleted_ids: set[str] = set()
        self._intents: dict[str, tuple[str, str]] = {}
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

    def record_intent(self, envelope: dict[str, Any]) -> SentLedgerOutcome[None]:
        """Durably record a sender intent before the recipient publish.

        At or above the durable-intent epoch this writes a pending schema-v2
        record that carries the full recipient envelope, so a crash before the
        recipient publish recovers to exactly one delivered message. Below the
        epoch it only retains the in-process intent and writes nothing.
        """
        message_id = _id(envelope["id"])
        target = _text(envelope["to_project"], "target project")
        created = _text(envelope["created_at"], "created_at")
        with self._lock:
            self._refresh()
            if isinstance(self._state, _Broken):
                return self._outcome(None)
            existing = next((r for r in self._records if r["id"] == message_id), None)
            if existing is not None:
                if existing["target_project"] != target:
                    raise ValueError("sent-message tracking record conflicts")
                self._intents[message_id] = (target, created)
                return self._outcome(None)
            if not self._at_epoch:
                self._intents[message_id] = (target, created)
                return self._outcome(None)
            record = {
                "id": message_id,
                "created_at": created,
                "target_project": target,
                "publication": "pending",
                "envelope": self._spill_envelope(envelope),
                "reported": [],
            }
            self._records.append(record)
            self._records.sort(key=self._chronology)
            self._pending.add(message_id)
            self._prune()
            return self._persist((record,), self._deleted_ids)

    def mark_published(
        self, message_id: str, target_project: str
    ) -> SentLedgerOutcome[None]:
        """Mark the recipient publish durable for a previously recorded intent."""
        message_id = _id(message_id)
        target_project = _text(target_project, "target project")
        with self._lock:
            self._refresh()
            if isinstance(self._state, _Broken):
                self._intents.pop(message_id, None)
                return self._outcome(None)
            if not self._at_epoch:
                created = self._intents.pop(message_id, (target_project, _now()))[1]
                return self._record_published_v1(message_id, target_project, created)
            self._intents.pop(message_id, None)
            record = next((r for r in self._records if r["id"] == message_id), None)
            if record is None:
                return self._outcome(None)
            changed = record.get("publication") != "published"
            record["publication"] = "published"
            envelope = record.get("envelope")
            if envelope is not None and isinstance(envelope.get("body"), dict) and (
                "sha256" in envelope["body"]
            ):
                envelope["body"] = {"file": _v2.recipient_body_name(message_id)}
                self._delete_spill(message_id)
                changed = True
            if changed:
                return self._persist((record,), self._deleted_ids)
            return self._outcome(None)

    def pending_intents(self) -> list[dict[str, Any]]:
        """Return recipient envelopes whose publish is not yet durable."""
        with self._lock:
            self._refresh()
            if not self._at_epoch or isinstance(self._state, _Broken):
                return []
            result: list[dict[str, Any]] = []
            for record in self._records:
                if record.get("publication") != "pending":
                    continue
                envelope = record.get("envelope")
                if envelope is None:
                    continue
                resolved = dict(envelope)
                resolved["body"] = self._resolve_pending_body(
                    record["id"], envelope["body"]
                )
                result.append(resolved)
            return result

    def _record_published_v1(
        self, message_id: str, target_project: str, created: str
    ) -> SentLedgerOutcome[None]:
        existing = next((r for r in self._records if r["id"] == message_id), None)
        if existing is not None:
            if existing["target_project"] != target_project:
                raise ValueError("sent-message tracking record conflicts")
            return self._outcome(None)
        record = {
            "id": message_id,
            "target_project": target_project,
            "created_at": created,
            "reported": [],
        }
        self._records.append(record)
        self._records.sort(key=self._chronology)
        self._pending.add(message_id)
        self._prune()
        return self._persist((record,), self._deleted_ids)

    def _spill_envelope(self, envelope: dict[str, Any]) -> dict[str, Any]:
        body = envelope["body"]
        stored = dict(envelope)
        if isinstance(body, str):
            data = body.encode("utf-8")
            if len(data) > _SENDER_SPILL_BYTES:
                self._write_spill(envelope["id"], data)
                stored["body"] = {
                    "file": _v2.sender_spill_name(envelope["id"]),
                    "length": len(data),
                    "sha256": hashlib.sha256(data).hexdigest(),
                }
        return stored

    def _resolve_pending_body(
        self, message_id: str, body: object
    ) -> str:
        if isinstance(body, str):
            return body
        if isinstance(body, dict) and "sha256" in body:
            return self._read_spill(
                message_id, body["length"], body["sha256"]
            ).decode("utf-8")
        raise ValueError("pending sender intent body is not resolvable")

    def _write_spill(self, message_id: str, data: bytes) -> None:
        with (
            self._session_directory(create=True) as session_fd,
            child_directory(
                session_fd, _v2.SENDER_SPILL_DIR, create=True
            ) as spill_fd,
        ):
            atomic_publish_file(
                spill_fd, f"{message_id}.txt", data, sync_directory=True
            )

    def _read_spill(self, message_id: str, length: int, sha256: str) -> bytes:
        with (
            self._session_directory(create=False) as session_fd,
            child_directory(
                session_fd, _v2.SENDER_SPILL_DIR, create=False
            ) as spill_fd,
        ):
            data = read_session_file(spill_fd, f"{message_id}.txt")
        if len(data) != length or hashlib.sha256(data).hexdigest() != sha256:
            raise ValueError("sender spill does not match its reference")
        return data

    def _delete_spill(self, message_id: str) -> None:
        try:
            with (
                self._session_directory(create=False) as session_fd,
                child_directory(
                    session_fd, _v2.SENDER_SPILL_DIR, create=False
                ) as spill_fd,
            ):
                os.unlink(f"{message_id}.txt", dir_fd=spill_fd)
        except (FileNotFoundError, SessionError):
            pass

    @staticmethod
    def _to_v2(record: dict[str, Any]) -> dict[str, Any]:
        if "publication" in record:
            return record
        # A pre-epoch v1 row is already published; it carries no envelope.
        return {
            "id": record["id"],
            "created_at": record["created_at"],
            "target_project": record["target_project"],
            "publication": "published",
            "reported": list(record["reported"]),
        }

    def records(self) -> tuple[dict[str, Any], ...]:
        with self._lock:
            self._refresh()
            return tuple(dict(record) for record in reversed(self._records))

    def tracked_targets(self) -> set[str]:
        """Distinct target projects of non-terminal tracked records."""
        with self._lock:
            return {
                record["target_project"]
                for record in self._records
                if "done" not in record["reported"]
            }

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
            if terminal is None:
                return
            self._records.remove(terminal)
            self._pending.discard(terminal["id"])
            self._deleted_ids.add(terminal["id"])

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
        if version == _v2.V2_SCHEMA_VERSION:
            try:
                decoded = _v2.decode_root(value)
            except (ValueError, TypeError):
                return _Broken(path, "invalid_records")
            records = list(decoded["records"])
            records.sort(key=self._chronology)
            return _Loaded(path, tuple(records))
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
        if self._at_epoch:
            return self._persist_v2()
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

    def _persist_v2(self) -> SentLedgerOutcome[None]:
        """Publish the full schema-v2 root, converting any pre-epoch v1 rows.

        v2 is a strict shape, so the whole record set is rewritten under the
        shape lock rather than patched; the per-session sender ledger has one
        writer, so there is no concurrent upsert to preserve.
        """
        records = [self._to_v2(record) for record in self._records]
        try:
            with (
                open_writer(self.sessions_root.parent) as lease,
                self._session_directory(create=True) as session_fd,
            ):
                lease.publish_json(
                    session_fd,
                    _SENT_TRACKING_FILE,
                    (
                        SetValue(("schema_version",), _v2.V2_SCHEMA_VERSION),
                        ReplaceSubtree(("records",), records),
                    ),
                    publish=lambda fd, name, data: atomic_publish_file(
                        fd, name, data, sync_directory=True
                    ),
                    initial={"schema_version": _v2.V2_SCHEMA_VERSION, "records": []},
                )
            self._deleted_ids.clear()
            return self._outcome(None)
        except StorageFormatError:
            self._adopt_state(_Broken(self.path, "invalid_json"))
            return self._outcome(None)
        except (OSError, SessionError, ValueError, TypeError) as exc:
            raise ValueError("sent-message tracking is unsafe or unavailable") from exc
