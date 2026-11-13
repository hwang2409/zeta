"""serialized disposable index for bounded session history reads."""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import sqlite3
import stat
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from ...agent.receipt import encode_json
from ...protocol.types import (
    ToolCall,
    is_passive_harness_message,
    is_visible_harness_notice,
)
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
class IndexedApprovalState:
    request_entry_id: str
    tool_call: Mapping[str, object]
    decision: str | None


@dataclass(frozen=True, slots=True)
class IndexedApprovalResolution:
    data: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class IndexedPendingApprovalPage:
    items: tuple[tuple[str, Mapping[str, object]], ...]
    next_seq: int | None


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
    AUTO_INDEX_MIN_BYTES = 1024 * 1024
    SCHEMA_VERSION = 5
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
                # The transcript is authoritative and repairs interrupted cache writes.
                connection.execute("PRAGMA journal_mode=OFF")
                connection.execute("PRAGMA synchronous=OFF")
                connection.execute("PRAGMA foreign_keys=ON")
            return connection
        except BaseException:
            connection.close()
            raise

    @staticmethod
    def _value(value: object) -> str:
        return json.dumps(value, separators=(",", ":"), sort_keys=True)

    @staticmethod
    def _value_ordered(value: object) -> str:
        # Approval-resolution data is read back and reserialized for a
        # byte-identical transcript row, so its key order must be preserved.
        return json.dumps(value, separators=(",", ":"))

    @staticmethod
    def _row(entry: object) -> tuple[int, str, str | None, str, str, str]:
        if isinstance(entry, Mapping):
            value = entry
            return (
                value["seq"], value["id"], value.get("parent_id"),
                value.get("lane", "main"), value["type"], HistoryIndex._value(value["data"]),
            )
        return (
            entry.seq,
            entry.id,
            entry.parent_id,
            entry.lane,
            entry.type,
            HistoryIndex._value(getattr(entry, "data", None)),
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
        connection.execute(
            "CREATE TABLE IF NOT EXISTS active_approval_requests ("
            "request_id TEXT PRIMARY KEY, request_seq INTEGER NOT NULL, "
            "request_entry_id TEXT NOT NULL, tool_call BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS active_approval_resolutions ("
            "request_id TEXT PRIMARY KEY, decision TEXT NOT NULL, data BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS warm_rows ("
            "seq INTEGER PRIMARY KEY, byte_offset INTEGER NOT NULL, "
            "byte_length INTEGER NOT NULL, facts BLOB)"
        )

    def _stored_facts(self, connection: sqlite3.Connection) -> dict[str, object] | None:
        row = connection.execute("SELECT value FROM meta WHERE key='facts'").fetchone()
        return self._facts(json.loads(row[0])) if row is not None else None

    def warm_snapshot(
        self, stat: os.stat_result
    ) -> tuple[list[tuple[object, ...]], list[dict[str, object]], dict[str, object]] | None:
        """Return a compact unchanged message transcript, or require full validation."""
        with self._lock:
            try:
                connection = self._connection(write=False)
            except (HistoryIndexError, OSError, sqlite3.Error):
                return None
            try:
                facts = self._stored_facts(connection)
                expected = {
                    "st_dev": stat.st_dev,
                    "st_ino": stat.st_ino,
                    "byte_boundary": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                    "ctime_ns": stat.st_ctime_ns,
                    "schema_version": self.SCHEMA_VERSION,
                }
                if (
                    facts is None
                    or type(facts.get("cwd")) is not str
                    or not facts["cwd"]
                    or type(facts.get("prefix_fingerprint")) is not str
                    or any(facts.get(key) != value for key, value in expected.items())
                ):
                    return None
                if len(bytes.fromhex(facts["prefix_fingerprint"])) != 16:
                    return None
                try:
                    expected_digest = bytes.fromhex(str(facts.get("prefix_digest")))
                    if len(expected_digest) != 32:
                        return None
                except ValueError:
                    return None
                digest = b""
                indexed_entries = connection.execute(
                    "SELECT seq,id,parent_id,lane,type,data FROM entries ORDER BY seq"
                )
                for seq, entry_id, parent_id, lane, entry_type, data in indexed_entries:
                    encoded_line = b"".join(
                        (
                            b'{"data":',
                            data.encode(),
                            b',"id":',
                            encode_json(entry_id),
                            b',"lane":',
                            encode_json(lane),
                            b',"parent_id":',
                            encode_json(parent_id),
                            b',"seq":',
                            encode_json(seq),
                            b',"type":',
                            encode_json(entry_type),
                            b"}\n",
                        )
                    )
                    digest = hashlib.sha256(digest + encoded_line).digest()
                if digest != expected_digest:
                    return None
                rows = connection.execute(
                    "SELECT e.seq,e.id,e.parent_id,e.lane,e.type,w.byte_offset,"
                    "w.byte_length,w.facts FROM entries e JOIN warm_rows w USING(seq) "
                    "ORDER BY e.seq"
                ).fetchall()
                if len(rows) != facts.get("last_seq") or any(
                    row[4] not in {"message", "compaction"} for row in rows
                ):
                    return None
                unsupported_message = connection.execute(
                    "SELECT 1 FROM entries WHERE type='message' AND ("
                    "COALESCE(json_array_length(json_extract(data,'$.message.content')),0) != 1 "
                    "OR json_extract(data,'$.message.content[0].type') != 'text' "
                    "OR EXISTS (SELECT 1 FROM json_each(json_extract(data,'$.message.metadata')) "
                    "WHERE key != 'zeta.origin')) LIMIT 1"
                ).fetchone()
                if unsupported_message is not None:
                    return None
                payload_rows = connection.execute(
                    "SELECT seq,id,parent_id,lane,type,data FROM entries "
                    "WHERE type='compaction' OR seq>? ORDER BY seq",
                    (max(0, len(rows) - 1024),),
                ).fetchall()
                payloads = [
                    {
                        "seq": row[0],
                        "id": row[1],
                        "parent_id": row[2],
                        "lane": row[3],
                        "type": row[4],
                        "data": json.loads(row[5]),
                    }
                    for row in payload_rows
                ]
                return rows, payloads, facts
            except (HistoryIndexError, OSError, sqlite3.Error, ValueError, TypeError):
                return None
            finally:
                connection.close()

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

    @staticmethod
    def _active_ids(rows: Sequence[object]) -> set[str]:
        by_id = {HistoryIndex._row(entry)[1]: entry for entry in rows}
        active: set[str] = set()
        current = rows[-1] if rows else None
        while current is not None:
            row = HistoryIndex._row(current)
            if row[1] in active:
                break
            active.add(row[1])
            current = by_id.get(row[2]) if row[2] else None
        return active

    @staticmethod
    def _warm_facts(value: object) -> object:
        if value is None or isinstance(value, (str, int, float, bool, list, tuple, dict)):
            return value
        if is_dataclass(value):
            return {"kind": type(value).__name__, "value": asdict(value)}
        raise TypeError(f"unsupported warm row facts: {type(value).__name__}")

    def _rebuild(
        self,
        source_facts: object,
        row_refs: Sequence[object],
        validated_rows: Iterable[object],
    ) -> None:
        active_ids = self._active_ids(row_refs)
        connection: sqlite3.Connection | None = None
        try:
            self._remove_file()
            connection = self._connection(write=True)
            self._create_schema(connection)
            for index in ("entries_parent", "entries_type", "entries_history_group"):
                connection.execute(f"DROP INDEX {index}")
            group = 0
            with connection:
                count = 0
                entry_batch: list[tuple[object, ...]] = []
                warm_batch: list[tuple[object, ...]] = []
                for ref, entry in zip(row_refs, validated_rows, strict=True):
                    count += 1
                    seq, entry_id, parent_id, lane, entry_type, encoded = self._row(entry)
                    data = (
                        entry["data"]
                        if isinstance(entry, Mapping)
                        else json.loads(encoded)
                    )
                    active = entry_id in active_ids
                    history_group = (
                        group if active and self._belongs_in_history(entry_type, data) else None
                    )
                    if active and self._starts_next_group(entry_type, data):
                        group += 1
                    entry_batch.append(
                        (seq, entry_id, parent_id, lane, entry_type, encoded, history_group)
                    )
                    warm_batch.append(
                        (
                            seq,
                            int(getattr(ref, "offset", 0)),
                            int(getattr(ref, "length", 0)),
                            self._value(self._warm_facts(getattr(ref, "facts", None))),
                        )
                    )
                    if len(entry_batch) == 2048:
                        connection.executemany(
                            "INSERT INTO entries(seq,id,parent_id,lane,type,data,history_group) "
                            "VALUES(?,?,?,?,?,?,?)",
                            entry_batch,
                        )
                        connection.executemany(
                            "INSERT INTO warm_rows(seq,byte_offset,byte_length,facts) "
                            "VALUES(?,?,?,?)",
                            warm_batch,
                        )
                        entry_batch.clear()
                        warm_batch.clear()
                    if active and entry_type == "message":
                        for approval in data.get("approval_requests", ()):
                            connection.execute(
                                "INSERT OR REPLACE INTO approval_calls(request_id,tool_call) VALUES(?,?)",
                                (approval["request_id"], self._value(approval["tool_call"])),
                            )
                            connection.execute(
                                "INSERT INTO active_approval_requests(request_id,request_seq,request_entry_id,tool_call) VALUES(?,?,?,?)",
                                (approval["request_id"], seq, entry_id, self._value(approval["tool_call"])),
                            )
                    if active and entry_type == "approval_resolution":
                        connection.execute(
                            "INSERT INTO active_approval_resolutions(request_id,decision,data) VALUES(?,?,?)",
                            (data["request_id"], data["decision"], self._value_ordered(data)),
                        )
                    if active and entry_type == "client_delivery" and isinstance(data.get("accepted_message"), dict):
                        connection.execute(
                            "INSERT OR REPLACE INTO accepted_messages(delivery_id,message) VALUES(?,?)",
                            (data["delivery_id"], self._value(data["accepted_message"])),
                        )
                connection.executemany(
                    "INSERT INTO entries(seq,id,parent_id,lane,type,data,history_group) "
                    "VALUES(?,?,?,?,?,?,?)",
                    entry_batch,
                )
                connection.executemany(
                    "INSERT INTO warm_rows(seq,byte_offset,byte_length,facts) "
                    "VALUES(?,?,?,?)",
                    warm_batch,
                )
                connection.execute("CREATE INDEX entries_parent ON entries(parent_id)")
                connection.execute("CREATE INDEX entries_type ON entries(type)")
                connection.execute(
                    "CREATE INDEX entries_history_group ON entries(history_group,seq)"
                )
                self._set_meta(connection, source_facts, total=group)
                connection.execute(
                    "INSERT INTO meta(key,value) VALUES('startup_projection',?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (
                        self._value(
                            {
                                "active_head": self._row(row_refs[-1])[1] if row_refs else None,
                                "row_count": count,
                                "resident_tail_start": max(1, count - 1023) if count else 0,
                            }
                        ),
                    ),
                )
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
                    connection.execute(
                        "INSERT INTO active_approval_requests(request_id,request_seq,request_entry_id,tool_call) VALUES(?,?,?,?)",
                        (
                            approval["request_id"],
                            entry.seq,
                            entry.id,
                            self._value(approval["tool_call"]),
                        ),
                    )
            if entry.type == "approval_resolution":
                connection.execute(
                    "INSERT INTO active_approval_resolutions(request_id,decision,data) VALUES(?,?,?)",
                    (data["request_id"], data["decision"], self._value_ordered(data)),
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
            from ._rows import row_facts

            lengths = [len(encode_json(entry.to_dict())) + 1 for entry in entries]
            offset = int(self._facts(facts)["byte_boundary"]) - sum(lengths)
            warm_rows = []
            for entry, length in zip(entries, lengths, strict=True):
                warm_rows.append(
                    (
                        entry.seq,
                        offset,
                        length,
                        self._value(self._warm_facts(row_facts(entry))),
                    )
                )
                offset += length
            connection.executemany(
                "INSERT INTO warm_rows(seq,byte_offset,byte_length,facts) VALUES(?,?,?,?)",
                warm_rows,
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

    def sync_validated(
        self,
        source_facts: object,
        row_refs: Iterable[object],
        validated_rows: Iterable[object] | Callable[[], Iterable[object]] | None = None,
    ) -> None:
        """Make the disposable index match validated authority without retaining it."""
        with self._lock:
            self._check_open()
            if self._observer:
                self._available = self._exists()
                return
            refs = tuple(row_refs)
            def rows() -> Iterable[object]:
                if validated_rows is None:
                    return refs
                return validated_rows() if callable(validated_rows) else validated_rows
            expected = self._facts(source_facts)
            try:
                connection = self._connection(write=True)
                try:
                    self._create_schema(connection)
                    stored = self._stored_facts(connection)
                    warm_count = connection.execute(
                        "SELECT COUNT(*) FROM warm_rows"
                    ).fetchone()[0]
                    if (
                        stored is not None
                        and all(
                            stored.get(key) == value for key, value in expected.items()
                        )
                        and stored.get("schema_version") == self.SCHEMA_VERSION
                        and warm_count == len(refs)
                    ):
                        self._available = True
                        return
                finally:
                    connection.close()
                self._rebuild(source_facts, refs, rows())
            except (
                HistoryIndexError,
                OSError,
                sqlite3.Error,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ):
                self._available = False
                try:
                    self._rebuild(source_facts, refs, rows())
                except (
                    HistoryIndexError,
                    OSError,
                    sqlite3.Error,
                    TypeError,
                    ValueError,
                    json.JSONDecodeError,
                ):
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

    def _approval_query(
        self,
        facts: object,
        query: str,
        parameters: tuple[object, ...] = (),
    ) -> list[tuple[object, ...]]:
        with self._lock:
            self._check_open()
            if not self._available and not self._observer:
                raise HistoryIndexUnavailableError("history index is unavailable")
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connection(write=False)
                stored = self._stored_facts(connection)
                expected = self._facts(facts)
                if stored is None or any(
                    stored.get(key) != value for key, value in expected.items()
                ):
                    # A facts mismatch around an approval read is transient: the
                    # caller (_decide_approval) falls back to the resident
                    # projection. Do not clear _available; only the history-page
                    # path owns the sidecar rebuild, so clearing it here would
                    # make the "never marks it unavailable" docstring false.
                    raise StaleHistorySnapshot
                return connection.execute(query, parameters).fetchall()
            except HistoryIndexError:
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HistoryIndexUnavailableError("approval index query failed") from exc
            finally:
                if connection is not None:
                    connection.close()

    def approval_state(
        self, facts: object, request_id: str
    ) -> IndexedApprovalState | None:
        rows = self._approval_query(
            facts,
            "SELECT r.request_entry_id,r.tool_call,s.decision FROM active_approval_requests r "
            "LEFT JOIN active_approval_resolutions s USING(request_id) "
            "WHERE r.request_id=?",
            (request_id,),
        )
        if not rows:
            return None
        request_entry_id, tool_call, decision = rows[0]
        return IndexedApprovalState(request_entry_id, json.loads(tool_call), decision)

    def approval_resolution(
        self, facts: object, request_id: str
    ) -> IndexedApprovalResolution | None:
        rows = self._approval_query(
            facts,
            "SELECT data FROM active_approval_resolutions WHERE request_id=?",
            (request_id,),
        )
        return (
            None
            if not rows
            else IndexedApprovalResolution(json.loads(rows[0][0]))
        )

    def pending_approval_page(
        self, facts: object, after_seq: int, limit: int = PAGE_LIMIT
    ) -> IndexedPendingApprovalPage:
        if not 1 <= limit <= self.PAGE_LIMIT:
            raise InvalidHistoryPageRequest("invalid pending approval page limit")
        # ``limit`` bounds distinct message seqs, not rows: a page returns every
        # pending request of the first ``limit`` pending messages after
        # ``after_seq``. The cursor is a seq with a strict ``>``, so ending a
        # page inside a message would skip its remaining requests forever.
        rows = self._approval_query(
            facts,
            "SELECT r.request_id,r.request_seq,r.tool_call "
            "FROM active_approval_requests r "
            "LEFT JOIN active_approval_resolutions s USING(request_id) "
            "WHERE s.request_id IS NULL AND r.request_seq IN ("
            "  SELECT seq FROM ("
            "    SELECT DISTINCT r2.request_seq AS seq "
            "    FROM active_approval_requests r2 "
            "    LEFT JOIN active_approval_resolutions s2 ON s2.request_id=r2.request_id "
            "    WHERE s2.request_id IS NULL AND r2.request_seq>? "
            "    ORDER BY seq LIMIT ?"
            "  )"
            ") ORDER BY r.request_seq",
            (after_seq, limit),
        )
        seqs = {row[1] for row in rows}
        return IndexedPendingApprovalPage(
            tuple((request_id, json.loads(tool_call)) for request_id, _seq, tool_call in rows),
            int(rows[-1][1]) if len(seqs) == limit else None,
        )

    def _mark_unavailable(self) -> None:
        self._available = False

    def close(self) -> None:
        with self._lock:
            self._closed = True


def _rebuild_history_index(store: ConversationStore) -> dict[str, object]:
    if store._read_only:
        observer = type(store)(
            store.root_dir,
            session_id=store.session_id,
            _must_exist=True,
            _read_only=True,
            _history_index=store._history_index,
        )
        try:
            with observer._append_lock(observer=True):
                facts = observer._history_index_facts()
                if not store._history_index.catch_up_validated(
                    facts, observer._resident_rows.entries
                ):
                    observer._sync_history_index(observer._resident_rows.entries)
                return facts
        finally:
            observer.close()
    writer = store._new_persistent_durable_writer()
    try:
        with writer._append_lock():
            facts = store._history_index_facts()
            if not store._history_index.catch_up_validated(facts, store._resident_rows.entries):
                store._sync_history_index(store._resident_rows.entries)
            return facts
    finally:
        writer.close()


def _read_stable_page(
    store: ConversationStore,
    index: HistoryIndex,
    request: Mapping[str, object],
) -> IndexedHistoryPage | None:
    attempts = _HISTORY_PAGE_ATTEMPTS if "end" in request else 1
    for attempt in range(attempts):
        try:
            if not index.available:
                _rebuild_history_index(store)
            facts = store._history_index_facts()
            page = index.page(request, facts)
        except (StaleHistorySnapshot, HistoryIndexUnavailableError):
            index._mark_unavailable()
            if store._read_only or attempt == attempts - 1:
                return None
            continue
        if store._history_index_facts() == facts:
            return page
        if store._read_only or attempt == attempts - 1:
            return None
        index._mark_unavailable()
    return None


def _overlay_resident_entries(
    store: ConversationStore, page: IndexedHistoryPage
) -> IndexedHistoryPage:
    def resident_entry(entry: ConversationEntry) -> ConversationEntry:
        if entry.seq > len(store._resident_rows.rows):
            return entry
        resident = store._resident_rows.entries[entry.seq - 1]
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


def _decide_approval(
    store: ConversationStore, index_query: object, resident: object
) -> object:
    """Answer an approval read from the index when its facts match, else from
    the resident active branch.

    The seam never rebuilds the sidecar and never marks it unavailable: a
    closed, read-only, or index-less store, an unavailable index (any on-loop
    append marks it so), or a facts change around the query all fall back to
    ``resident``, the O(1) memoized in-memory projection. The async caller has
    already run ``_sync_log_without_waiting`` so resident is current. The
    sidecar rebuild belongs to the history-page path (``_read_stable_page``),
    not to an approval read on the hot loop.
    """
    index = store._history_index
    if store._closed or store._read_only or index is None or not index.available:
        return resident()
    facts = store._history_index_facts()
    try:
        result = index_query(index, facts)
    except (StaleHistorySnapshot, HistoryIndexUnavailableError, OSError):
        return resident()
    if store._history_index_facts() == facts:
        return result
    return resident()


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
            self._sync_history_index(
                self.entry(ref) for ref in self.branch_refs()
            )
        except (OSError, HistoryIndexError):
            self._history_index._mark_unavailable()

    def _history_index_facts(self: ConversationStore) -> dict[str, object]:
        stat = os.stat(
            "conversation.jsonl", dir_fd=self.directory_fd, follow_symlinks=False
        )
        last = self._resident_rows.entries[-1] if self._resident_rows.entries else None
        return {
            "cwd": self.cwd,
            "st_dev": stat.st_dev,
            "st_ino": stat.st_ino,
            "byte_boundary": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "ctime_ns": stat.st_ctime_ns,
            "last_seq": last.seq if last else 0,
            "last_id": last.id if last else None,
            "last_parent": last.parent_id if last else None,
            "prefix_digest": self._entry_prefix_digest.hex(),
            "prefix_fingerprint": self._log_prefix_fingerprint.hex(),
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
            facts = self._history_index_facts()
            if (
                facts["byte_boundary"] < self._history_index.AUTO_INDEX_MIN_BYTES
                and not self._history_index._exists()
            ):
                self._history_index._mark_unavailable()
                return
            refs = self._resident_rows.resident_refs()
            self._history_index.sync_validated(
                facts,
                refs,
                lambda: self._resident_rows.stream_validated_rows(refs),
            )
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
            self._sync_history_index(
                self.entry(ref) for ref in self.branch_refs()
            )
            if not self._history_index.available:
                return
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

    async def approval_state(
        self, request_id: str
    ) -> tuple[ToolCall, str | None] | None:
        if not self._closed:
            self._sync_log_without_waiting()

        def index_query(index: HistoryIndex, facts: Mapping[str, object]) -> object:
            result = index.approval_state(facts, request_id)
            return (
                None
                if result is None
                else (ToolCall.from_dict(dict(result.tool_call)), result.decision)
            )

        return await self._to_thread_store(
            lambda: _decide_approval(
                self,
                index_query,
                lambda: self.approval_state_now(request_id),
            )
        )

    async def approval_resolution(self, request_id: str) -> Mapping[str, str] | None:
        if not self._closed:
            self._sync_log_without_waiting()

        def index_query(index: HistoryIndex, facts: Mapping[str, object]) -> object:
            result = index.approval_resolution(facts, request_id)
            return None if result is None else result.data

        return await self._to_thread_store(
            lambda: _decide_approval(
                self,
                index_query,
                lambda: self.approval_resolution_now(request_id),
            )
        )

    async def pending_approval_page(
        self, after_seq: int = 0, limit: int = HistoryIndex.PAGE_LIMIT
    ) -> IndexedPendingApprovalPage:
        if not self._closed:
            self._sync_log_without_waiting()
        return await self._to_thread_store(
            lambda: _decide_approval(
                self,
                lambda index, facts: index.pending_approval_page(
                    facts, after_seq, limit
                ),
                lambda: self._resident_pending_approval_page(after_seq, limit),
            )
        )

    def history_index_page(
        self: ConversationStore, request: Mapping[str, object]
    ) -> IndexedHistoryPage | None:
        """Read one stable derived page, or defer when no sidecar exists."""
        index = self._history_index
        if self._closed:
            return None
        if index is None or (self._read_only and not index.available):
            return None
        page = _read_stable_page(self, index, request)
        return _overlay_resident_entries(self, page) if page is not None else None
