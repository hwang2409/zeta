"""serialized disposable index for bounded session history reads."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ...agent.receipt import encode_json
from ...protocol.types import is_passive_harness_message, is_visible_harness_notice
from ..checkpoints import ConversationEntry
from ..transcript_recovery import recovery_status

_HISTORY_PAGE_ATTEMPTS = 16

if TYPE_CHECKING:
    from ._store import ConversationStore


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


class InvalidHistoryPageRequest(HistoryIndexError):
    """the caller supplied a cursor outside the indexed history."""


@dataclass(frozen=True, slots=True)
class IndexedHistoryPage:
    groups: tuple[tuple[ConversationEntry, ...], ...]
    start: int
    total: int
    approval_calls: Mapping[str, object]
    endpoint: str | None
    repair_generation: str | None


class HistoryIndex:
    """A serialized leaf owner for one store family's derived history."""

    PAGE_LIMIT = 128
    PAGE_BYTES = 1024 * 1024
    SCHEMA_VERSION = 3
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
                "last_parent", "prefix_digest", "reset_generation", "repair_generation",
            )
            if hasattr(value, key)
        }

    @staticmethod
    def _starts_next_group(entry_type: str, data: Mapping[str, object]) -> bool:
        if entry_type == "message":
            message = data.get("message")
            return isinstance(message, dict) and not (
                is_passive_harness_message(message)
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
            group = 0
            with connection:
                for entry in rows:
                    seq, entry_id, parent_id, lane, entry_type, encoded = self._row(entry)
                    data = json.loads(encoded)
                    history_group = (
                        group if self._belongs_in_history(entry_type, data) else None
                    )
                    if self._starts_next_group(entry_type, data):
                        group += 1
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
                self._set_meta(connection, source_facts, total=group)
            self._available = True
        finally:
            if connection is not None:
                connection.close()

    @staticmethod
    def _entry_digest(previous: bytes, entry: object) -> bytes:
        value = entry if isinstance(entry, Mapping) else entry.to_dict()
        return hashlib.sha256(previous + encode_json(value) + b"\n").digest()

    def _suffix_start(
        self,
        stored: Mapping[str, object],
        expected: Mapping[str, object],
        rows: Sequence[object],
    ) -> int | None:
        if stored.get("schema_version") != self.SCHEMA_VERSION or any(
            stored.get(key) != expected.get(key)
            for key in ("st_dev", "st_ino", "repair_generation")
        ):
            return None
        stored_boundary = stored.get("byte_boundary")
        expected_boundary = expected.get("byte_boundary")
        if (
            not isinstance(stored_boundary, int)
            or not isinstance(expected_boundary, int)
            or stored_boundary > expected_boundary
        ):
            return None
        start = stored.get("last_seq")
        if not isinstance(start, int) or not 0 <= start <= len(rows):
            return None
        boundary = rows[start - 1] if start else None
        if stored.get("last_id") != (self._row(boundary)[1] if boundary else None):
            return None
        if stored.get("last_parent") != (self._row(boundary)[2] if boundary else None):
            return None
        digest_hex = stored.get("prefix_digest")
        if not isinstance(digest_hex, str):
            return None
        try:
            digest = bytes.fromhex(digest_hex)
        except ValueError:
            return None
        suffix = rows[start:]
        if not suffix:
            return None
        parent_id = stored.get("last_id")
        for entry in suffix:
            row = self._row(entry)
            if row[2] != parent_id:
                return None
            parent_id = row[1]
            digest = self._entry_digest(digest, entry)
        return start if digest.hex() == expected.get("prefix_digest") else None

    def _project_entries(
        self, entries: tuple[object, ...], total: int
    ) -> tuple[list[tuple[object, int | None]], int]:
        projected: list[tuple[object, int | None]] = []
        for entry in entries:
            data = dict(entry.data)
            group = total if self._belongs_in_history(entry.type, data) else None
            projected.append((entry, group))
            if self._starts_next_group(entry.type, data):
                total += 1
        return projected, total

    def _write_correlations(
        self, connection: sqlite3.Connection, entries: tuple[object, ...]
    ) -> None:
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

    def _append_entries(
        self,
        connection: sqlite3.Connection,
        facts: object,
        entries: tuple[object, ...],
    ) -> None:
        if not entries:
            return
        expected_seq = int(
            connection.execute("SELECT COALESCE(MAX(seq),0) + 1 FROM entries").fetchone()[0]
        )
        if expected_seq != self._row(entries[0])[0]:
            raise HistoryIndexUnavailableError("history index delta is not contiguous")
        stored = self._stored_facts(connection) or {}
        projected, total = self._project_entries(
            entries, int(stored.get("history_groups", 0))
        )
        with connection:
            connection.executemany(
                "INSERT INTO entries(seq,id,parent_id,lane,type,data,history_group) "
                "VALUES(?,?,?,?,?,?,?)",
                ((*self._row(entry), group) for entry, group in projected),
            )
            self._write_correlations(connection, entries)
            self._set_meta(connection, facts, total=total)

    def catch_up_validated(
        self, source_facts: object, validated_rows: Sequence[object]
    ) -> bool:
        """Append only rows after a matching indexed boundary."""
        with self._lock:
            self._check_open()
            if self._observer:
                return False
            expected = self._facts(source_facts)
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connection(write=True)
                self._create_schema(connection)
                stored = self._stored_facts(connection)
                if stored is None:
                    return False
                if all(
                    stored.get(key) == value for key, value in expected.items()
                ) and stored.get("schema_version") == self.SCHEMA_VERSION:
                    self._available = True
                    return True
                suffix_start = self._suffix_start(stored, expected, validated_rows)
                if suffix_start is None:
                    return False
                self._append_entries(
                    connection, source_facts, tuple(validated_rows[suffix_start:])
                )
                self._available = True
                return True
            except (HistoryIndexError, OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError):
                self._available = False
                return False
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
            expected = self._facts(source_facts)
            try:
                connection = self._connection(write=True)
                try:
                    self._create_schema(connection)
                    stored = self._stored_facts(connection)
                    if stored is not None and all(
                        stored.get(key) == value for key, value in expected.items()
                    ) and stored.get("schema_version") == self.SCHEMA_VERSION:
                        self._available = True
                        return
                    suffix_start = (
                        self._suffix_start(stored, expected, rows)
                        if stored is not None else None
                    )
                    if suffix_start is not None:
                        self._append_entries(connection, source_facts, rows[suffix_start:])
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
                self._append_entries(connection, facts, entries)
                self._available = True
            except HistoryIndexError:
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                raise HistoryIndexUnavailableError("history index commit failed") from exc
            finally:
                if connection is not None:
                    connection.close()

    def _page_bounds(
        self, request: Mapping[str, object], total: int
    ) -> tuple[int, int]:
        if request.get("backward"):
            raw_before = request.get("before")
            if raw_before is None:
                before = total + 1
            else:
                before = int(raw_before)
                if before < 0 or before > total:
                    raise InvalidHistoryPageRequest("before must not exceed history end")
            return max(0, before - self.PAGE_LIMIT), before
        start = int(request.get("offset", 0))
        if start < 0:
            raise InvalidHistoryPageRequest("offset must be nonnegative")
        if start > total:
            return start, start
        return start, min(total + 1, start + self.PAGE_LIMIT)

    def _read_groups(
        self, connection: sqlite3.Connection, start: int, stop: int
    ) -> tuple[tuple[tuple[ConversationEntry, ...], ...], set[str]]:
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
        return tuple(tuple(group) for group in groups), request_ids

    def _approval_calls(
        self, connection: sqlite3.Connection, request_ids: set[str]
    ) -> dict[str, object]:
        if not request_ids:
            return {}
        placeholders = ",".join("?" for _ in request_ids)
        return {
            request_id: json.loads(tool_call)
            for request_id, tool_call in connection.execute(
                f"SELECT request_id,tool_call FROM approval_calls WHERE request_id IN ({placeholders})",
                tuple(request_ids),
            ).fetchall()
        }

    def _query_page(
        self,
        connection: sqlite3.Connection,
        request: Mapping[str, object],
        total: int,
        stored_generation: object,
    ) -> IndexedHistoryPage:
        start, stop = self._page_bounds(request, total)
        groups, request_ids = self._read_groups(connection, start, stop)
        endpoint_row = connection.execute(
            "SELECT id FROM entries WHERE history_group=? AND type IN "
            "('message','notification','warning') ORDER BY seq DESC LIMIT 1",
            (total - 1,),
        ).fetchone() if total else None
        return IndexedHistoryPage(
            groups,
            start,
            total,
            self._approval_calls(connection, request_ids),
            endpoint_row[0] if endpoint_row else None,
            stored_generation if isinstance(stored_generation, str) else None,
        )

    def page(self, request: object, g0: object) -> IndexedHistoryPage:
        """Return at most one bounded group page; the caller performs g1."""
        with self._lock:
            self._check_open()
            if not self._available and not self._observer:
                raise HistoryIndexUnavailableError("history index is unavailable")
            connection: sqlite3.Connection | None = None
            try:
                page_request = request if isinstance(request, Mapping) else vars(request)
                connection = self._connection(write=False)
                stored = self._stored_facts(connection)
                requested = self._facts(g0)
                if stored is None or any(
                    stored.get(key) != value for key, value in requested.items()
                ):
                    self._available = False
                    raise StaleHistorySnapshot
                current_total = int(stored.get("history_groups", 0))
                total = int(page_request.get("end", current_total))
                if total < 0 or total > current_total:
                    raise StaleHistorySnapshot
                return self._query_page(
                    connection, page_request, total, stored.get("repair_generation")
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


def _rebuild_history_index(store: ConversationStore) -> None:
    writer = store._new_persistent_durable_writer()
    try:
        with writer._append_lock():
            facts = store._history_index_facts()
            if store._history_index.catch_up_validated(facts, store._entries):
                return
            store._sync_history_index(store.active_branch_snapshot())
    finally:
        writer.close()


def _read_stable_page(
    store: ConversationStore,
    index: HistoryIndex,
    request: Mapping[str, object],
) -> IndexedHistoryPage:
    for attempt in range(_HISTORY_PAGE_ATTEMPTS):
        if not index.available:
            _rebuild_history_index(store)
        facts = store._history_index_facts()
        try:
            page = index.page(request, facts)
        except StaleHistorySnapshot:
            if "end" in request or attempt == _HISTORY_PAGE_ATTEMPTS - 1:
                raise
            index._mark_unavailable()
            continue
        except HistoryIndexUnavailableError as exc:
            if store._history_index_facts() != facts:
                raise StaleHistorySnapshot from exc
            if store._read_only or attempt == _HISTORY_PAGE_ATTEMPTS - 1:
                raise
            index._mark_unavailable()
            continue
        if store._history_index_facts() == facts:
            return page
        if "end" in request or attempt == _HISTORY_PAGE_ATTEMPTS - 1:
            raise StaleHistorySnapshot
    raise HistoryIndexUnavailableError("history index is unavailable")


def _overlay_resident_entries(
    store: ConversationStore, page: IndexedHistoryPage
) -> IndexedHistoryPage:
    def resident_entry(entry: ConversationEntry) -> ConversationEntry:
        if entry.seq > len(store._entries):
            return entry
        resident = store._entries[entry.seq - 1]
        if resident.id != entry.id:
            return entry
        accepted = entry.data.get("_accepted_message")
        if accepted is None:
            return resident
        return replace(
            resident,
            data={**resident.data, "_accepted_message": accepted},
        )

    groups = tuple(
        tuple(resident_entry(entry) for entry in group)
        for group in page.groups
    )
    return replace(page, groups=groups)


class HistoryPageStoreMixin:
    def _refresh_repair_generation(self: ConversationStore) -> None:
        record = recovery_status(self.directory_fd, self._repair_target).latest_committed
        self._repair_generation = record.quarantine_id if record is not None else None

    def _initialize_history_index(
        self: ConversationStore, *, observer: bool
    ) -> None:
        if self._history_index is None:
            self._history_index = HistoryIndex(
                self.session_dir, self.directory_fd, observer=observer
            )
        if not self._history_index_owner or observer:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            self._history_index._mark_unavailable()
            return
        try:
            self._sync_history_index(self.active_branch_snapshot())
        except (OSError, HistoryIndexError):
            self._history_index._mark_unavailable()

    def _history_index_facts(self: ConversationStore) -> dict[str, object]:
        stat = os.stat(
            "conversation.jsonl", dir_fd=self.directory_fd, follow_symlinks=False
        )
        last = self._entries[-1] if self._entries else None
        return {
            "st_dev": stat.st_dev,
            "st_ino": stat.st_ino,
            "byte_boundary": stat.st_size,
            "last_seq": last.seq if last else 0,
            "last_id": last.id if last else None,
            "last_parent": last.parent_id if last else None,
            "prefix_digest": self._entry_prefix_digest.hex(),
            "repair_generation": self._repair_generation,
        }

    def _sync_history_index(
        self: ConversationStore, entries: Iterable[ConversationEntry]
    ) -> None:
        if self._history_index is None:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._history_index.sync_validated(self._history_index_facts(), entries)
        else:
            self._history_index._mark_unavailable()

    def _apply_history_index_entries(
        self: ConversationStore,
        entries: tuple[ConversationEntry, ...],
        *,
        leave_stale_on_loop: bool = False,
    ) -> None:
        if self._history_index is None or not entries:
            return
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            if not leave_stale_on_loop:
                self._history_index._mark_unavailable()
            return
        if not self._history_index.available:
            self._sync_history_index(self.active_branch_snapshot())
        digest = self._entry_prefix_digest
        for entry in entries:
            digest = self._next_entry_prefix_digest(
                digest, encode_json(entry.to_dict()) + b"\n"
            )
        facts = self._history_index_facts()
        last = entries[-1]
        facts.update(
            last_seq=last.seq,
            last_id=last.id,
            last_parent=last.parent_id,
            prefix_digest=digest.hex(),
        )
        try:
            self._history_index.apply_committed(facts, entries)
        except Exception:  # noqa: BLE001 -- derived state cannot fail authority
            self._history_index._mark_unavailable()

    @property
    def repair_generation(self: ConversationStore) -> str | None:
        return self._repair_generation

    def history_index_page(
        self: ConversationStore, request: Mapping[str, object]
    ) -> IndexedHistoryPage | None:
        """Read one stable derived page, or defer when no sidecar exists."""
        index = self._history_index
        if self._closed:
            return None
        if index is None or (self._read_only and not index.available):
            return None
        return _overlay_resident_entries(
            self, _read_stable_page(self, index, request)
        )
