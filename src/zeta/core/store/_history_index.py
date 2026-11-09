"""serialized disposable index for bounded session history reads."""

from __future__ import annotations

import json
import os
import sqlite3
import stat
import threading
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

from ...protocol.types import is_passive_harness_message, is_visible_harness_notice
from ..checkpoints import ConversationEntry


class HistoryIndexError(RuntimeError):
    """base class for derived history index failures."""


class HistoryIndexClosedError(HistoryIndexError):
    """the index has been closed and cannot be reopened."""


class HistoryIndexUnavailableError(HistoryIndexError):
    """the disposable index is missing, corrupt, or unavailable."""


class HistoryIndexThreadSafetyError(HistoryIndexError):
    """sqlite cannot safely be used by the store's executor threads."""

    def __init__(self, observed: int, *, required: int = 3) -> None:
        super().__init__(f"sqlite threadsafety {observed}; required {required}")
        self.observed = observed
        self.required = required


class StaleHistorySnapshot(HistoryIndexError):
    """the authoritative transcript changed during a history request."""


@dataclass(frozen=True, slots=True)
class IndexedHistoryPage:
    groups: tuple[tuple[ConversationEntry, ...], ...]
    start: int
    total: int
    approval_calls: Mapping[str, object]
    endpoint: str | None


class HistoryIndex:
    """A serialized leaf owner for one store family's derived history."""

    PAGE_LIMIT = 128
    PAGE_BYTES = 1024 * 1024
    SCHEMA_VERSION = 2
    _NAME = "history-index.sqlite3"

    def __init__(
        self,
        session_dir: Path,
        directory_fd: int,
        *,
        observer: bool = False,
    ) -> None:
        self.path = session_dir / self._NAME
        self._directory_fd = directory_fd
        self._session_name = session_dir.name
        self._observer = observer
        self._lock = threading.Lock()
        self._closed = False
        self._available = False

    @property
    def available(self) -> bool:
        return self._available

    def _check_directory(self) -> None:
        pinned = os.fstat(self._directory_fd)
        parent_fd = os.open("..", os.O_RDONLY | os.O_DIRECTORY, dir_fd=self._directory_fd)
        try:
            named = os.stat(self._session_name, dir_fd=parent_fd, follow_symlinks=False)
        except OSError as exc:
            raise HistoryIndexUnavailableError("session directory identity changed") from exc
        finally:
            os.close(parent_fd)
        if (named.st_dev, named.st_ino) != (pinned.st_dev, pinned.st_ino):
            raise HistoryIndexUnavailableError("session directory identity changed")

    def _exists(self) -> bool:
        try:
            info = os.stat(self._NAME, dir_fd=self._directory_fd, follow_symlinks=False)
        except FileNotFoundError:
            return False
        return stat.S_ISREG(info.st_mode)

    def _connection(self, *, write: bool) -> sqlite3.Connection:
        if sqlite3.threadsafety != 3:
            raise HistoryIndexThreadSafetyError(sqlite3.threadsafety)
        self._check_directory()
        read_only = self._observer or not write
        if (self._observer and write) or (read_only and not self._exists()):
            raise HistoryIndexUnavailableError("observer index is unavailable")
        uri = f"file:{self.path}?mode=ro" if read_only else str(self.path)
        connection = sqlite3.connect(
            uri,
            uri=read_only,
            timeout=0 if read_only else 5,
            check_same_thread=False,
        )
        try:
            self._check_directory()
            if read_only:
                connection.execute("PRAGMA busy_timeout=0")
            else:
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA foreign_keys=ON")
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _value(value: object) -> str:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _row(entry: object) -> tuple[int, str, str | None, str, str, str]:
        if isinstance(entry, Mapping):
            value = entry
            return (
                value["seq"], value["id"], value.get("parent_id"),
                value.get("lane", "main"), value["type"], HistoryIndex._value(value["data"]),
            )
        return (
            entry.seq, entry.id, entry.parent_id, entry.lane, entry.type,
            HistoryIndex._value(entry.data),
        )

    @staticmethod
    def _facts(value: object) -> dict[str, object]:
        if isinstance(value, Mapping):
            return dict(value)
        return {
            key: getattr(value, key)
            for key in (
                "st_dev", "st_ino", "byte_boundary", "last_seq", "last_id",
                "last_parent", "prefix_digest", "reset_generation",
            )
            if hasattr(value, key)
        }

    @staticmethod
    def _starts_next_group(entry_type: str, data: Mapping[str, object]) -> bool:
        if entry_type == "message":
            message = data.get("message")
            return not (
                isinstance(message, dict)
                and is_passive_harness_message(message)
                and not is_visible_harness_notice(message)
            )
        return entry_type == "notification" or (
            entry_type == "warning" and data.get("kind") == "transcript_recovery"
        )

    @staticmethod
    def _belongs_in_history(entry_type: str, data: Mapping[str, object]) -> bool:
        return HistoryIndex._starts_next_group(entry_type, data) or entry_type in {
            "client_delivery", "approval_resolution", "approval_history", "turn_end"
        }

    def _create_schema(self, connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, parent_id TEXT, "
            "lane TEXT NOT NULL, type TEXT NOT NULL, data BLOB NOT NULL, "
            "history_group INTEGER)"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS entries_parent ON entries(parent_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS entries_type ON entries(type)")
        connection.execute("CREATE INDEX IF NOT EXISTS entries_history_group ON entries(history_group,seq)")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS approval_calls ("
            "request_id TEXT PRIMARY KEY, tool_call BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS accepted_messages ("
            "delivery_id TEXT PRIMARY KEY, message BLOB NOT NULL)"
        )

    def _stored_facts(self, connection: sqlite3.Connection) -> dict[str, object] | None:
        row = connection.execute("SELECT value FROM meta WHERE key='facts'").fetchone()
        return self._facts(json.loads(row[0])) if row is not None else None

    def _set_meta(
        self, connection: sqlite3.Connection, facts: object, *, total: int
    ) -> None:
        payload = self._facts(facts)
        payload.update(schema_version=self.SCHEMA_VERSION, history_groups=total)
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('facts',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (self._value(payload),),
        )

    def _check_open(self) -> None:
        if self._closed:
            raise HistoryIndexClosedError("history index is closed")

    def _remove_file(self) -> None:
        try:
            os.unlink(self._NAME, dir_fd=self._directory_fd)
        except FileNotFoundError:
            pass

    def _rebuild(
        self, source_facts: object, validated_rows: Iterable[object]
    ) -> None:
        rows = iter(validated_rows)
        connection: sqlite3.Connection | None = None
        try:
            self._remove_file()
            connection = self._connection(write=True)
            self._create_schema(connection)
            group = -1
            with connection:
                for entry in rows:
                    seq, entry_id, parent_id, lane, entry_type, encoded = self._row(entry)
                    data = json.loads(encoded)
                    if self._starts_next_group(entry_type, data):
                        group += 1
                    history_group = group if self._belongs_in_history(entry_type, data) else None
                    connection.execute(
                        "INSERT INTO entries(seq,id,parent_id,lane,type,data,history_group) "
                        "VALUES(?,?,?,?,?,?,?)",
                        (seq, entry_id, parent_id, lane, entry_type, encoded, history_group),
                    )
                    if entry_type == "message":
                        for approval in data.get("approval_requests", ()):
                            connection.execute(
                                "INSERT OR REPLACE INTO approval_calls(request_id,tool_call) VALUES(?,?)",
                                (approval["request_id"], self._value(approval["tool_call"])),
                            )
                    if entry_type == "client_delivery" and isinstance(data.get("accepted_message"), dict):
                        connection.execute(
                            "INSERT OR REPLACE INTO accepted_messages(delivery_id,message) VALUES(?,?)",
                            (data["delivery_id"], self._value(data["accepted_message"])),
                        )
                self._set_meta(connection, source_facts, total=group + 1)
            self._available = True
        finally:
            if connection is not None:
                connection.close()

    def sync_validated(self, source_facts: object, validated_rows: Iterable[object]) -> None:
        """Make the disposable index match validated authority without failing it."""
        with self._lock:
            self._check_open()
            if self._observer:
                self._available = self._exists()
                return
            rows = tuple(validated_rows)
            try:
                connection = self._connection(write=True)
                try:
                    self._create_schema(connection)
                    stored = self._stored_facts(connection)
                    expected = self._facts(source_facts)
                    if stored is not None and all(
                        stored.get(key) == value for key, value in expected.items()
                    ) and stored.get("schema_version") == self.SCHEMA_VERSION:
                        self._available = True
                        return
                finally:
                    connection.close()
                self._rebuild(source_facts, rows)
            except (HistoryIndexError, OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
                self._available = False
                try:
                    self._rebuild(source_facts, rows)
                except (HistoryIndexError, OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
                    self._available = False

    def apply_committed(self, facts: object, rows: Iterable[object]) -> None:
        """Commit an immutable post-fsync delta before resident publication."""
        entries = tuple(rows)
        if not entries:
            return
        with self._lock:
            self._check_open()
            if self._observer:
                raise HistoryIndexUnavailableError("observer index is read-only")
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connection(write=True)
                expected = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(seq),0) + 1 FROM entries"
                    ).fetchone()[0]
                )
                if expected != entries[0].seq:
                    raise HistoryIndexUnavailableError("history index delta is not contiguous")
                current_group = int(
                    connection.execute(
                        "SELECT COALESCE(MAX(history_group),-1) FROM entries"
                    ).fetchone()[0]
                )
                projected: list[tuple[object, int | None]] = []
                for entry in entries:
                    data = dict(entry.data)
                    if self._starts_next_group(entry.type, data):
                        current_group += 1
                    group = (
                        current_group
                        if current_group >= 0 and self._belongs_in_history(entry.type, data)
                        else None
                    )
                    projected.append((entry, group))
                with connection:
                    connection.executemany(
                        "INSERT INTO entries(seq,id,parent_id,lane,type,data,history_group) "
                        "VALUES(?,?,?,?,?,?,?)",
                        ((*self._row(entry), group) for entry, group in projected),
                    )
                    for entry in entries:
                        data = dict(entry.data)
                        if entry.type == "message":
                            for approval in data.get("approval_requests", ()):
                                connection.execute(
                                    "INSERT OR REPLACE INTO approval_calls(request_id,tool_call) VALUES(?,?)",
                                    (approval["request_id"], self._value(approval["tool_call"])),
                                )
                        accepted = data.get("accepted_message")
                        if entry.type == "client_delivery" and isinstance(accepted, dict):
                            connection.execute(
                                "INSERT OR REPLACE INTO accepted_messages(delivery_id,message) VALUES(?,?)",
                                (data["delivery_id"], self._value(accepted)),
                            )
                    self._set_meta(connection, facts, total=current_group + 1)
                self._available = True
            except HistoryIndexError:
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                raise HistoryIndexUnavailableError("history index commit failed") from exc
            finally:
                if connection is not None:
                    connection.close()

    def page(self, request: object, g0: object) -> IndexedHistoryPage:
        """Return at most one bounded group page; the caller performs g1."""
        with self._lock:
            self._check_open()
            if not self._available and not self._observer:
                raise HistoryIndexUnavailableError("history index is unavailable")
            connection: sqlite3.Connection | None = None
            try:
                request = request if isinstance(request, Mapping) else vars(request)
                connection = self._connection(write=False)
                stored = self._stored_facts(connection)
                requested = self._facts(g0)
                if stored is None or any(stored.get(key) != value for key, value in requested.items()):
                    self._available = False
                    raise StaleHistorySnapshot
                current_total = int(stored.get("history_groups", 0))
                total = int(request.get("end", current_total))
                if total < 0 or total > current_total:
                    raise StaleHistorySnapshot
                if request.get("backward"):
                    raw_before = request.get("before")
                    before = total + 1 if raw_before is None else int(raw_before)
                    start = max(0, before - self.PAGE_LIMIT)
                    stop = before
                else:
                    start = int(request.get("offset", 0))
                    stop = min(total + 1, start + self.PAGE_LIMIT)
                if start < 0 or stop < start or stop > total + 1:
                    raise ValueError("invalid history page")
                raw = connection.execute(
                    "SELECT seq,id,parent_id,lane,type,data,history_group FROM entries "
                    "WHERE history_group>=? AND history_group<? ORDER BY seq",
                    (start, stop),
                ).fetchall()
                groups: list[list[ConversationEntry]] = [[] for _ in range(stop - start)]
                request_ids: set[str] = set()
                for seq, entry_id, parent_id, lane, entry_type, data_json, group in raw:
                    data = json.loads(data_json)
                    if entry_type == "approval_resolution":
                        request_ids.add(data["request_id"])
                    if entry_type == "client_delivery" and data.get("status") in {"delivered", "dropped"}:
                        accepted = connection.execute(
                            "SELECT message FROM accepted_messages WHERE delivery_id=?",
                            (data.get("delivery_id"),),
                        ).fetchone()
                        if accepted is not None:
                            data = {**data, "_accepted_message": json.loads(accepted[0])}
                    groups[int(group) - start].append(
                        ConversationEntry(seq, entry_id, parent_id, lane, entry_type, data)
                    )
                approval_calls = {
                    request_id: json.loads(tool_call)
                    for request_id, tool_call in connection.execute(
                        f"SELECT request_id,tool_call FROM approval_calls WHERE request_id IN ({','.join('?' for _ in request_ids)})",
                        tuple(request_ids),
                    ).fetchall()
                } if request_ids else {}
                endpoint_row = connection.execute(
                    "SELECT id FROM entries WHERE history_group=? AND type IN "
                    "('message','notification','warning') ORDER BY seq DESC LIMIT 1",
                    (total - 1,),
                ).fetchone() if total else None
                return IndexedHistoryPage(
                    tuple(tuple(group) for group in groups), start, total,
                    approval_calls, endpoint_row[0] if endpoint_row else None,
                )
            except HistoryIndexError:
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HistoryIndexUnavailableError("history index page failed") from exc
            finally:
                if connection is not None:
                    connection.close()

    def _mark_unavailable(self) -> None:
        self._available = False

    def close(self) -> None:
        with self._lock:
            self._closed = True
