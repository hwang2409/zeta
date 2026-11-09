"""serialized disposable index for bounded session history reads."""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


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


class HistoryIndex:
    """A serialized leaf owner for one store family's derived history."""

    PAGE_LIMIT = 128
    PAGE_BYTES = 1024 * 1024
    SCHEMA_VERSION = 1

    def __init__(self, session_dir: Path, *, observer: bool = False) -> None:
        self.path = session_dir / "history-index.sqlite3"
        self._observer = observer
        self._lock = threading.Lock()
        self._closed = False

    @property
    def available(self) -> bool:
        return self.path.is_file()

    def _connection(self, *, write: bool) -> sqlite3.Connection:
        if sqlite3.threadsafety != 3:
            raise HistoryIndexThreadSafetyError(sqlite3.threadsafety)
        if self._observer:
            if write or not self.path.is_file():
                raise HistoryIndexUnavailableError("observer index is unavailable")
            connection = sqlite3.connect(
                f"file:{self.path}?mode=ro",
                uri=True,
                timeout=0,
                check_same_thread=False,
            )
            connection.execute("PRAGMA busy_timeout=0")
            return connection
        connection = sqlite3.connect(
            self.path, timeout=5, check_same_thread=False
        )
        connection.execute("PRAGMA journal_mode=DELETE")
        connection.execute("PRAGMA synchronous=FULL")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value BLOB NOT NULL)"
        )
        connection.execute(
            "CREATE TABLE IF NOT EXISTS entries ("
            "seq INTEGER PRIMARY KEY, id TEXT NOT NULL UNIQUE, parent_id TEXT, "
            "lane TEXT NOT NULL, type TEXT NOT NULL, data BLOB NOT NULL)"
        )
        connection.execute("CREATE INDEX IF NOT EXISTS entries_parent ON entries(parent_id)")
        connection.execute("CREATE INDEX IF NOT EXISTS entries_type ON entries(type)")
        return connection

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
            for key in ("st_dev", "st_ino", "byte_boundary", "last_seq", "last_id", "last_parent", "prefix_digest", "reset_generation")
            if hasattr(value, key)
        }

    def _set_meta(self, connection: sqlite3.Connection, facts: object) -> None:
        payload = self._facts(facts)
        payload["schema_version"] = self.SCHEMA_VERSION
        connection.execute(
            "INSERT INTO meta(key,value) VALUES('facts',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (self._value(payload),),
        )

    def _check_open(self) -> None:
        if self._closed:
            raise HistoryIndexClosedError("history index is closed")

    def sync_validated(self, source_facts: object, validated_rows: Iterable[object]) -> None:
        """Rebuild or append validated rows to match authoritative source facts."""
        with self._lock:
            self._check_open()
            if self._observer:
                raise HistoryIndexUnavailableError("observer index is read-only")
            rows = iter(validated_rows)
            connection: sqlite3.Connection | None = None
            try:
                connection = self._connection(write=True)
                prior = connection.execute(
                    "SELECT value FROM meta WHERE key='facts'"
                ).fetchone()
                facts = self._facts(source_facts)
                first = next(rows, None)
                expected = int(connection.execute("SELECT COALESCE(MAX(seq),0) FROM entries").fetchone()[0]) + 1
                append = (
                    first is not None
                    and prior is not None
                    and expected == self._row(first)[0]
                    and self._facts(json.loads(prior[0])).get("st_dev") == facts.get("st_dev")
                    and self._facts(json.loads(prior[0])).get("st_ino") == facts.get("st_ino")
                )
                with connection:
                    if not append:
                        connection.execute("DELETE FROM entries")
                    batch: list[tuple[int, str, str | None, str, str, str]] = []
                    if first is not None:
                        batch.append(self._row(first))
                    for entry in rows:
                        batch.append(self._row(entry))
                        if len(batch) >= 128:
                            connection.executemany("INSERT INTO entries(seq,id,parent_id,lane,type,data) VALUES(?,?,?,?,?,?)", batch)
                            batch.clear()
                    if batch:
                        connection.executemany("INSERT INTO entries(seq,id,parent_id,lane,type,data) VALUES(?,?,?,?,?,?)", batch)
                    self._set_meta(connection, source_facts)
            except HistoryIndexError:
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
                raise HistoryIndexUnavailableError("history index sync failed") from exc
            finally:
                if connection is not None:
                    connection.close()

    def apply_committed(self, facts: object) -> None:
        """Apply an immutable post-fsync delta without consulting the store."""
        payload = self._facts(facts)
        rows = payload.pop("rows", ())
        self.sync_validated(payload, rows)

    def page(self, request: object, g0: object) -> list[dict[str, Any]]:
        """Return one bounded page; callers perform the authoritative g1 check."""
        with self._lock:
            self._check_open()
            connection: sqlite3.Connection | None = None
            try:
                request = request if isinstance(request, Mapping) else vars(request)
                after = int(request.get("after", request.get("offset", 0)))
                limit = min(int(request.get("limit", self.PAGE_LIMIT)), self.PAGE_LIMIT)
                if after < 0 or limit < 0:
                    raise ValueError("invalid history page")
                connection = self._connection(write=False)
                meta = connection.execute(
                    "SELECT value FROM meta WHERE key='facts'"
                ).fetchone()
                stored_facts = self._facts(json.loads(meta[0])) if meta is not None else None
                requested_facts = self._facts(g0)
                if stored_facts is None or any(
                    stored_facts.get(key) != value
                    for key, value in requested_facts.items()
                ):
                    raise StaleHistorySnapshot
                rows = connection.execute(
                    "SELECT seq,id,parent_id,lane,type,data FROM entries WHERE seq>? ORDER BY seq LIMIT ?",
                    (after, limit),
                ).fetchall()
                result: list[dict[str, Any]] = []
                size = 0
                for seq, entry_id, parent_id, lane, entry_type, data in rows:
                    row = {"seq": seq, "id": entry_id, "parent_id": parent_id, "lane": lane, "type": entry_type, "data": json.loads(data)}
                    encoded = len(self._value(row).encode())
                    if result and size + encoded > self.PAGE_BYTES:
                        break
                    result.append(row)
                    size += encoded
                return result
            except HistoryIndexError:
                raise
            except (OSError, sqlite3.Error, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise HistoryIndexUnavailableError("history index page failed") from exc
            finally:
                if connection is not None:
                    connection.close()

    def close(self) -> None:
        with self._lock:
            self._closed = True


def transcript_generation(path: Path, repair_generation: int = 0) -> tuple[int, int, int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, repair_generation
