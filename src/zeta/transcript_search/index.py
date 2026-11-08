"""Deep project transcript index: ingestion, lifecycle, and lexical search."""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import sqlite3
import stat
import tempfile
import threading
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from zeta.core.checkpoints import ConversationEntry, active_branch
from zeta.core.store._log import PersistedAppend
from zeta.transcript_search.source import (
    _Cursor,
    _prefix_digest,
    _read_transcript,
)

logger = logging.getLogger(__name__)

from zeta.transcript_search.render import (
    MAX_UNIT_BYTES,  # noqa: F401 - compatibility re-export
    TranscriptUnit,
    _completed_assistant,
    _message,
    render_transcript_units,
)

SCHEMA_VERSION = 5
SANITIZER_VERSION = 2
INDEX_FILENAME = "transcript-index.sqlite3"
MAX_ACTIVE_TAIL_BYTES = 1024 * 1024
MAX_META_BYTES = 1024 * 1024
_MAX_APPEND_PREPARE_ATTEMPTS = 3
_REQUIRED_SAFE_OPEN_NAMES = ("O_NOFOLLOW", "O_NONBLOCK", "O_CLOEXEC")
_TOKEN = re.compile(r"[\w.-]+", re.UNICODE)
_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}\Z")
_PROJECT_LOCKS: dict[tuple[Path, str], threading.RLock] = {}
_PROJECT_LOCKS_GUARD = threading.Lock()


def is_indexable_top_level_project_session(
    *,
    agent_depth: int,
    project_id: str | None,
    parent_session_id: str | None,
    session_dir: Path,
) -> bool:
    """Return whether one session belongs in ordinary project transcript search."""

    if agent_depth != 0 or project_id is None or parent_session_id is not None:
        return False
    try:
        (Path(session_dir) / "attention_fork.json").lstat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


class TranscriptIndexError(RuntimeError):
    """A recoverable derived-index failure."""


class TranscriptIndexUnavailable(TranscriptIndexError):
    """The disposable index must be rebuilt before it can serve reads."""


@dataclass(frozen=True, slots=True)
class TranscriptSource:
    session_id: str
    session_dir: Path
    expected_project_id: str | None = None
    append_receipts: tuple[PersistedAppend, ...] | None = None

    @property
    def conversation_path(self) -> Path:
        return self.session_dir / "conversation.jsonl"


@dataclass(frozen=True, slots=True)
class _StoredCursor:
    byte_offset: int
    last_seq: int
    source_device: int
    source_inode: int
    source_mtime_ns: int
    source_ctime_ns: int
    source_size: int
    incremental_ready: bool
    last_entry_id: str | None
    active_tail_json: str
    prefix_digest: str | None

    def as_cursor(self) -> _Cursor:
        return _Cursor(
            self.byte_offset,
            self.last_seq,
            self.source_device,
            self.source_inode,
            self.source_mtime_ns,
            self.source_ctime_ns,
            self.source_size,
            self.incremental_ready,
            self.last_entry_id,
            tuple(json.loads(self.active_tail_json)),
            self.prefix_digest,
        )


@dataclass(frozen=True, slots=True)
class _CursorSnapshot:
    expected: _StoredCursor | None
    read_from: _Cursor | None


@dataclass(frozen=True, slots=True)
class _PreparedAppend:
    source: TranscriptSource
    expected_cursor: _StoredCursor | None
    full: bool
    units: tuple[TranscriptUnit, ...]
    next_cursor: _StoredCursor


@dataclass(frozen=True, slots=True)
class SearchHit:
    unit: TranscriptUnit
    score: float
    match: str

    @property
    def unit_id(self) -> str:
        return self.unit.unit_id


@dataclass(frozen=True, slots=True)
class IndexStatus:
    project_id: str
    schema_version: int
    sanitizer_version: int
    generation: int
    unit_count: int
    session_count: int
    cursors: dict[str, int]
    path: Path
    size_bytes: int
    ready: bool
    detail: str | None = None


@dataclass(frozen=True, slots=True)
class EvalResult:
    query_count: int
    recall_at: dict[int, float]
    mrr: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_count": self.query_count,
            "recall_at": {str(key): value for key, value in self.recall_at.items()},
            "mrr": self.mrr,
        }


class TranscriptIndex:
    """Own one project's complete derived index behind a small interface."""

    def __init__(self, project_dir: Path, project_id: str) -> None:
        if not _PROJECT_ID.fullmatch(project_id):
            raise ValueError("invalid project id")
        self.project_dir = Path(project_dir)
        self.project_id = project_id
        self.path = self.project_dir / INDEX_FILENAME
        self.lock_path = self.project_dir / ".transcript-index.lock"
        try:
            self.project_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            with self._write_guard():
                if not self.path.exists():
                    with self._temporary_database() as temporary_path:
                        self._initialize(temporary_path)
                        self._publish(temporary_path)
                else:
                    self._migrate_schema()
        except (OSError, sqlite3.Error) as exc:
            raise TranscriptIndexError(
                f"could not initialize transcript index: {exc}"
            ) from exc

    def append(self, source: TranscriptSource) -> IndexStatus:
        """Incrementally ingest one source without holding the guard while reading."""

        for _ in range(_MAX_APPEND_PREPARE_ATTEMPTS):
            snapshot = self._cursor_snapshot(source.session_id)
            prepared = self._prepare_append(source, snapshot)
            status = self._commit_prepared(prepared)
            if status is not None:
                return status
        raise TranscriptIndexError(
            "transcript changed during index refresh; retry required"
        )

    def append_units(
        self,
        session_id: str,
        units: Sequence[TranscriptUnit],
        *,
        cursor: int,
    ) -> None:
        """Store trusted units for evaluation and focused index tests."""

        if not session_id or type(cursor) is not int or cursor < 0:
            raise ValueError("invalid session cursor")
        if any(
            unit.project_id != self.project_id or unit.session_id != session_id
            for unit in units
        ):
            raise ValueError("transcript unit belongs to another project or session")
        try:
            with self._write_guard(), self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                stored = connection.execute(
                    "SELECT last_seq FROM session_cursors WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                if stored is not None and int(stored[0]) > cursor:
                    connection.rollback()
                    return
                self._replace_units(connection, session_id, units)
                connection.execute(
                    """INSERT INTO session_cursors(
                        session_id, byte_offset, last_seq, source_device,
                        source_inode, source_mtime_ns, source_ctime_ns,
                        source_size, incremental_ready, last_entry_id, active_tail_json,
                        prefix_digest
                    ) VALUES (?, 0, ?, 0, 0, 0, 0, 0, 1, NULL, '[]', NULL)
                    ON CONFLICT(session_id) DO UPDATE SET last_seq=excluded.last_seq""",
                    (session_id, cursor),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise TranscriptIndexError(
                f"could not update transcript index: {exc}"
            ) from exc

    def rebuild(self, sources: Iterable[TranscriptSource]) -> IndexStatus:
        """Build a fresh generation and atomically publish it."""

        sources = tuple(sources)
        with self._write_guard():
            generation = self._generation_best_effort() + 1
            try:
                with self._temporary_database() as temporary_path:
                    replacement = object.__new__(TranscriptIndex)
                    replacement.project_dir = self.project_dir
                    replacement.project_id = self.project_id
                    replacement.path = temporary_path
                    replacement.lock_path = self.lock_path
                    replacement._initialize(temporary_path)
                    with closing(replacement._connect()) as connection:
                        connection.execute(
                            "UPDATE metadata SET value = ? WHERE key = 'generation'",
                            (str(generation),),
                        )
                        connection.commit()
                    for source in sources:
                        replacement._append_locked(source, force_full=True)
                    self._publish(temporary_path)
                    return self._status()
            except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
                raise TranscriptIndexError(
                    f"could not rebuild transcript index: {exc}"
                ) from exc

    def delete_session(self, session_id: str) -> None:
        with self._write_guard():
            try:
                self._require_ready()
                with closing(self._connect()) as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute(
                        "DELETE FROM units WHERE session_id = ?", (session_id,)
                    )
                    connection.execute(
                        "DELETE FROM session_cursors WHERE session_id = ?",
                        (session_id,),
                    )
                    connection.commit()
            except sqlite3.Error as exc:
                raise TranscriptIndexError(
                    f"could not remove indexed session: {exc}"
                ) from exc

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        """Rank all-term matches first, then rare-term partial matches."""

        with self._read_guard():
            return self._search(query, limit=limit)

    def _search(self, query: str, *, limit: int) -> tuple[SearchHit, ...]:
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("search limit must be between 1 and 100")
        self._require_ready()
        terms = list(dict.fromkeys(_tokens(query)))
        if not terms:
            return ()
        try:
            with closing(self._connect()) as connection:
                complete = self._search_query(
                    connection, " AND ".join(_quote(term) for term in terms), limit
                )
                hits = [self._hit(row, "all") for row in complete]
                if len(hits) >= limit:
                    return tuple(hits)
                counts = Counter()
                for term in terms:
                    row = connection.execute(
                        "SELECT count(*) FROM unit_fts WHERE unit_fts MATCH ?",
                        (_quote(term),),
                    ).fetchone()
                    counts[term] = int(row[0]) if row else 0
                distinctive = [
                    term
                    for term, count in sorted(
                        counts.items(), key=lambda item: (item[1] or 10**12, item[0])
                    )
                    if count
                ][: max(1, (len(terms) + 1) // 2)]
                if not distinctive:
                    return tuple(hits)
                partial = self._search_query(
                    connection,
                    " OR ".join(_quote(term) for term in distinctive),
                    limit * 3,
                )
                seen = {hit.unit_id for hit in hits}
                for row in partial:
                    hit = self._hit(row, "partial")
                    if hit.unit_id not in seen:
                        hits.append(hit)
                        seen.add(hit.unit_id)
                        if len(hits) == limit:
                            break
                return tuple(hits)
        except sqlite3.Error as exc:
            raise TranscriptIndexError(
                f"could not search transcript index: {exc}"
            ) from exc

    def status(self) -> IndexStatus:
        with self._read_guard():
            return self._status()

    def _status(self) -> IndexStatus:
        version, sanitizer, generation, detail = self._version_state()
        ready = version == SCHEMA_VERSION and sanitizer == SANITIZER_VERSION
        unit_count = 0
        cursors: dict[str, int] = {}
        if ready:
            try:
                with closing(self._connect()) as connection:
                    unit_count = int(
                        connection.execute("SELECT count(*) FROM units").fetchone()[0]
                    )
                    cursors = {
                        str(session_id): int(last_seq)
                        for session_id, last_seq in connection.execute(
                            "SELECT session_id, last_seq FROM session_cursors ORDER BY session_id"
                        )
                    }
            except sqlite3.Error as exc:
                ready = False
                detail = f"index is unreadable and requires rebuild: {exc}"
        return IndexStatus(
            project_id=self.project_id,
            schema_version=version,
            sanitizer_version=sanitizer,
            generation=generation,
            unit_count=unit_count,
            session_count=len(cursors),
            cursors=cursors,
            path=self.path,
            size_bytes=self.path.stat().st_size if self.path.exists() else 0,
            ready=ready,
            detail=detail,
        )

    def _append_locked(
        self, source: TranscriptSource, *, force_full: bool = False
    ) -> IndexStatus:
        """Append while an outer exclusive guard is already held (rebuild only)."""

        for _ in range(_MAX_APPEND_PREPARE_ATTEMPTS):
            snapshot = self._cursor_snapshot_locked(source.session_id)
            prepared = self._prepare_append(source, snapshot, force_full=force_full)
            status = self._commit_prepared_locked(prepared)
            if status is not None:
                return status
        raise TranscriptIndexError(
            "transcript changed during index refresh; retry required"
        )

    def _cursor_snapshot(self, session_id: str) -> _CursorSnapshot:
        with self._read_guard():
            self._require_ready()
            return self._cursor_snapshot_locked(session_id)

    def _cursor_snapshot_locked(self, session_id: str) -> _CursorSnapshot:
        try:
            with closing(self._connect()) as connection:
                expected = self._stored_cursor(connection, session_id)
                indexed_seq = int(
                    connection.execute(
                        "SELECT coalesce(max(seq_end), 0) FROM units WHERE session_id = ?",
                        (session_id,),
                    ).fetchone()[0]
                )
            read_from = expected.as_cursor() if expected is not None else None
            if read_from is not None and indexed_seq > read_from.last_seq:
                read_from = None
            return _CursorSnapshot(expected=expected, read_from=read_from)
        except sqlite3.Error as exc:
            raise TranscriptIndexError(
                f"could not refresh transcript index: {exc}"
            ) from exc

    def _prepare_append(
        self,
        source: TranscriptSource,
        snapshot: _CursorSnapshot,
        *,
        force_full: bool = False,
    ) -> _PreparedAppend:
        cursor = None if force_full else snapshot.read_from
        read = _read_transcript(
            source.conversation_path,
            cursor,
            None if force_full else source.append_receipts,
        )
        if (
            not read.full
            and read.rows
            and cursor is not None
            and read.rows[0].get("parent_id") != cursor.last_entry_id
        ):
            read = _read_transcript(source.conversation_path, None)
        prior_tail = () if read.full or cursor is None else cursor.active_tail
        render_rows = prior_tail + read.rows
        units = render_transcript_units(self.project_id, source.session_id, render_rows)
        active_tail = _active_tail(render_rows)
        tail_json = json.dumps(active_tail, separators=(",", ":"), sort_keys=True)
        incremental_ready = len(tail_json.encode("utf-8")) <= MAX_ACTIVE_TAIL_BYTES
        if not incremental_ready:
            tail_json = "[]"
        return _PreparedAppend(
            source=source,
            expected_cursor=snapshot.expected,
            full=read.full,
            units=units,
            next_cursor=_StoredCursor(
                byte_offset=read.byte_offset,
                last_seq=read.last_seq,
                source_device=read.source_device,
                source_inode=read.source_inode,
                source_mtime_ns=read.source_mtime_ns,
                source_ctime_ns=read.source_ctime_ns,
                source_size=read.source_size,
                incremental_ready=incremental_ready,
                last_entry_id=read.last_entry_id,
                active_tail_json=tail_json,
                prefix_digest=read.prefix_digest,
            ),
        )

    def _commit_prepared(self, prepared: _PreparedAppend) -> IndexStatus | None:
        with self._write_guard():
            self._require_ready()
            return self._commit_prepared_locked(prepared)

    def _commit_prepared_locked(self, prepared: _PreparedAppend) -> IndexStatus | None:
        try:
            with closing(self._connect()) as connection:
                connection.execute("BEGIN IMMEDIATE")
                if (
                    self._stored_cursor(connection, prepared.source.session_id)
                    != prepared.expected_cursor
                ):
                    connection.rollback()
                    return None
                if not self._source_is_bound(prepared.source):
                    connection.execute(
                        "DELETE FROM units WHERE session_id = ?",
                        (prepared.source.session_id,),
                    )
                    connection.execute(
                        "DELETE FROM session_cursors WHERE session_id = ?",
                        (prepared.source.session_id,),
                    )
                    connection.commit()
                    return self._status()
                if not self._source_matches_prepared(prepared):
                    connection.rollback()
                    return None
                if prepared.full:
                    self._replace_units(
                        connection, prepared.source.session_id, prepared.units
                    )
                else:
                    self._upsert_units(connection, prepared.units)
                cursor = prepared.next_cursor
                connection.execute(
                    """INSERT INTO session_cursors(
                        session_id, byte_offset, last_seq, source_device,
                        source_inode, source_mtime_ns, source_ctime_ns,
                        source_size, incremental_ready, last_entry_id, active_tail_json,
                        prefix_digest
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        byte_offset=excluded.byte_offset,
                        last_seq=excluded.last_seq,
                        source_device=excluded.source_device,
                        source_inode=excluded.source_inode,
                        source_mtime_ns=excluded.source_mtime_ns,
                        source_ctime_ns=excluded.source_ctime_ns,
                        source_size=excluded.source_size,
                        incremental_ready=excluded.incremental_ready,
                        last_entry_id=excluded.last_entry_id,
                        active_tail_json=excluded.active_tail_json,
                        prefix_digest=excluded.prefix_digest""",
                    (
                        prepared.source.session_id,
                        cursor.byte_offset,
                        cursor.last_seq,
                        cursor.source_device,
                        cursor.source_inode,
                        cursor.source_mtime_ns,
                        cursor.source_ctime_ns,
                        cursor.source_size,
                        cursor.incremental_ready,
                        cursor.last_entry_id,
                        cursor.active_tail_json,
                        cursor.prefix_digest,
                    ),
                )
                if not self._source_matches_prepared(prepared):
                    connection.rollback()
                    return None
                connection.commit()
            return self._status()
        except sqlite3.Error as exc:
            raise TranscriptIndexError(
                f"could not refresh transcript index: {exc}"
            ) from exc

    @staticmethod
    def _safe_open_flags() -> int | None:
        values = [getattr(os, name, None) for name in _REQUIRED_SAFE_OPEN_NAMES]
        if any(type(value) is not int for value in values):
            return None
        return os.O_RDONLY | values[0] | values[1] | values[2]

    def _source_is_bound(self, source: TranscriptSource) -> bool:
        expected = source.expected_project_id
        if expected is None:
            return True
        flags = self._safe_open_flags()
        if flags is None:
            return False
        try:
            fd = os.open(source.session_dir / "meta.json", flags)
        except OSError:
            return False
        try:
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_META_BYTES:
                return False
            payload = bytearray()
            while len(payload) <= MAX_META_BYTES:
                chunk = os.read(fd, MAX_META_BYTES + 1 - len(payload))
                if not chunk:
                    break
                payload.extend(chunk)
            if len(payload) > MAX_META_BYTES:
                return False
            value = json.loads(payload)
        except (OSError, ValueError, TypeError):
            return False
        finally:
            os.close(fd)
        return (
            isinstance(value, dict)
            and value.get("project_id") == expected == self.project_id
        )

    def _source_matches_prepared(self, prepared: _PreparedAppend) -> bool:
        flags = self._safe_open_flags()
        if flags is None:
            return False
        try:
            fd = os.open(prepared.source.conversation_path, flags)
        except OSError:
            return False
        try:
            info = os.fstat(fd)
            cursor = prepared.next_cursor
            if (
                not stat.S_ISREG(info.st_mode)
                or info.st_dev != cursor.source_device
                or info.st_ino != cursor.source_inode
                or info.st_size != cursor.source_size
                or info.st_mtime_ns != cursor.source_mtime_ns
                or info.st_ctime_ns != cursor.source_ctime_ns
                or not 0 <= cursor.byte_offset <= cursor.source_size
            ):
                return False
            return _prefix_digest(fd, cursor.source_size) == cursor.prefix_digest
        except (OSError, ValueError, TypeError):
            return False
        finally:
            os.close(fd)

    @staticmethod
    def _stored_cursor(
        connection: sqlite3.Connection, session_id: str
    ) -> _StoredCursor | None:
        row = connection.execute(
            """SELECT byte_offset, last_seq, source_device, source_inode,
                source_mtime_ns, source_ctime_ns, source_size, incremental_ready,
                last_entry_id, active_tail_json, prefix_digest
            FROM session_cursors WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        active_tail_json = json.dumps(
            json.loads(row[9]), separators=(",", ":"), sort_keys=True
        )
        return _StoredCursor(
            *(int(row[index]) for index in range(7)),
            bool(row[7]),
            str(row[8]) if row[8] is not None else None,
            active_tail_json,
            str(row[10]) if row[10] is not None else None,
        )

    @staticmethod
    def _cursor(connection: sqlite3.Connection, session_id: str) -> _Cursor | None:
        stored = TranscriptIndex._stored_cursor(connection, session_id)
        return stored.as_cursor() if stored is not None else None

    @staticmethod
    def _replace_units(
        connection: sqlite3.Connection,
        session_id: str,
        units: Sequence[TranscriptUnit],
    ) -> None:
        connection.execute("DELETE FROM units WHERE session_id = ?", (session_id,))
        connection.executemany(
            """INSERT INTO units(
                unit_id, turn_id, project_id, session_id, seq_start, seq_end,
                started_at, ended_at, origin, kind, chunk_index, chunk_count, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    unit.unit_id,
                    unit.turn_id,
                    unit.project_id,
                    unit.session_id,
                    unit.seq_start,
                    unit.seq_end,
                    unit.started_at,
                    unit.ended_at,
                    unit.origin,
                    unit.kind,
                    unit.chunk_index,
                    unit.chunk_count,
                    unit.text,
                )
                for unit in units
            ],
        )

    @staticmethod
    def _upsert_units(
        connection: sqlite3.Connection, units: Sequence[TranscriptUnit]
    ) -> None:
        connection.executemany(
            """INSERT OR REPLACE INTO units(
                unit_id, turn_id, project_id, session_id, seq_start, seq_end,
                started_at, ended_at, origin, kind, chunk_index, chunk_count, text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    unit.unit_id,
                    unit.turn_id,
                    unit.project_id,
                    unit.session_id,
                    unit.seq_start,
                    unit.seq_end,
                    unit.started_at,
                    unit.ended_at,
                    unit.origin,
                    unit.kind,
                    unit.chunk_index,
                    unit.chunk_count,
                    unit.text,
                )
                for unit in units
            ],
        )

    def _search_query(
        self, connection: sqlite3.Connection, expression: str, limit: int
    ) -> list[sqlite3.Row]:
        return connection.execute(
            """SELECT units.*, bm25(unit_fts) AS rank
            FROM unit_fts JOIN units ON units.id = unit_fts.rowid
            WHERE unit_fts MATCH ? ORDER BY rank, units.unit_id LIMIT ?""",
            (expression, limit),
        ).fetchall()

    @staticmethod
    def _hit(row: sqlite3.Row, match: str) -> SearchHit:
        unit = TranscriptUnit(
            schema_version=1,
            unit_id=row["unit_id"],
            turn_id=row["turn_id"],
            project_id=row["project_id"],
            session_id=row["session_id"],
            seq_start=row["seq_start"],
            seq_end=row["seq_end"],
            started_at=row["started_at"],
            ended_at=row["ended_at"],
            origin=row["origin"],
            kind=row["kind"],
            chunk_index=row["chunk_index"],
            chunk_count=row["chunk_count"],
            text=row["text"],
        )
        return SearchHit(unit, float(row["rank"]), match)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    @contextmanager
    def _temporary_database(self) -> Iterator[Path]:
        fd, temporary = tempfile.mkstemp(
            dir=self.project_dir, prefix=".transcript-index.", suffix=".sqlite3"
        )
        os.close(fd)
        path = Path(temporary)
        path.unlink()
        try:
            yield path
        finally:
            path.unlink(missing_ok=True)
            _remove_sqlite_sidecars(path)

    def _prepare_publication(self, temporary_path: Path) -> None:
        with closing(sqlite3.connect(temporary_path)) as connection:
            connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
        with closing(sqlite3.connect(temporary_path)) as connection:
            connection.execute("PRAGMA journal_mode=DELETE").fetchone()
        os.chmod(temporary_path, 0o600)

    def _publish(self, temporary_path: Path) -> None:
        self._prepare_publication(temporary_path)
        _remove_sqlite_sidecars(self.path)
        os.replace(temporary_path, self.path)
        with closing(self._connect()) as connection:
            connection.execute("PRAGMA journal_mode=WAL")

    @staticmethod
    def _initialize(path: Path) -> None:
        connection = sqlite3.connect(path)
        try:
            connection.executescript(_SCHEMA)
            connection.executemany(
                "INSERT OR REPLACE INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("sanitizer_version", str(SANITIZER_VERSION)),
                    ("generation", "0"),
                ),
            )
            connection.commit()
        finally:
            connection.close()
        os.chmod(path, 0o600)

    def _migrate_schema(self) -> None:
        try:
            connection = sqlite3.connect(self.path)
            with closing(connection), connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
                if metadata.get("schema_version") != "4":
                    return
                columns = {
                    str(row[1])
                    for row in connection.execute("PRAGMA table_info(session_cursors)")
                }
                if "prefix_digest" not in columns:
                    connection.execute(
                        "ALTER TABLE session_cursors ADD COLUMN prefix_digest TEXT"
                    )
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'schema_version'",
                    (str(SCHEMA_VERSION),),
                )
        except sqlite3.DatabaseError:
            return

    def _version_state(self) -> tuple[int, int, int, str | None]:
        try:
            with closing(
                sqlite3.connect(f"file:{self.path}?mode=ro", uri=True)
            ) as connection:
                metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            version = int(metadata.get("schema_version", -1))
            sanitizer = int(metadata.get("sanitizer_version", -1))
            generation = int(metadata.get("generation", 0))
        except (OSError, sqlite3.Error, TypeError, ValueError) as exc:
            return -1, -1, 0, f"index is unreadable and requires rebuild: {exc}"
        if version != SCHEMA_VERSION or sanitizer != SANITIZER_VERSION:
            return (
                version,
                sanitizer,
                generation,
                "index versions changed and require rebuild",
            )
        return version, sanitizer, generation, None

    def _require_ready(self) -> None:
        status = self._status()
        if not status.ready:
            raise TranscriptIndexUnavailable(
                status.detail or "transcript index requires rebuild"
            )

    def _generation_best_effort(self) -> int:
        return self._version_state()[2]

    def _read_guard(self):
        return _ProjectFileGuard(
            self.project_dir, self.project_id, self.lock_path, fcntl.LOCK_SH
        )

    def _write_guard(self):
        return _ProjectFileGuard(
            self.project_dir, self.project_id, self.lock_path, fcntl.LOCK_EX
        )


class _ProjectFileGuard:
    def __init__(
        self, project_dir: Path, project_id: str, lock_path: Path, mode: int
    ) -> None:
        key = (project_dir.resolve(), project_id)
        with _PROJECT_LOCKS_GUARD:
            self.thread_lock = _PROJECT_LOCKS.setdefault(key, threading.RLock())
        self.lock_path = lock_path
        self.mode = mode
        self.fd = -1

    def __enter__(self) -> Self:
        self.thread_lock.acquire()
        try:
            self.fd = os.open(
                self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
            )
            fcntl.flock(self.fd, self.mode)
            return self
        except BaseException:
            if self.fd >= 0:
                os.close(self.fd)
            self.thread_lock.release()
            raise

    def __exit__(self, exc_type, exc, traceback) -> None:
        try:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
        finally:
            self.thread_lock.release()


def evaluate_manifest(manifest_path: Path, units_path: Path) -> EvalResult:
    """Evaluate the frozen corpus through the production FTS5 search path."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    units = [
        TranscriptUnit.from_dict(json.loads(line))
        for line in units_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    project_ids = {unit.project_id for unit in units}
    if len(project_ids) != 1:
        raise ValueError("evaluation units must belong to one project")
    project_id = project_ids.pop()
    rankings: list[list[str]] = []
    targets: list[set[str]] = []
    with tempfile.TemporaryDirectory() as directory:
        index = TranscriptIndex(Path(directory) / project_id, project_id)
        by_session: dict[str, list[TranscriptUnit]] = {}
        for unit in units:
            by_session.setdefault(unit.session_id, []).append(unit)
        for session_id, session_units in by_session.items():
            index.append_units(
                session_id,
                session_units,
                cursor=max(unit.seq_end for unit in session_units),
            )
        for case in manifest["queries"]:
            rankings.append(
                [hit.unit_id for hit in index.search(case["query"], limit=10)]
            )
            targets.append(set(case["target_unit_ids"]))
    recall = {
        k: sum(
            bool(set(ranking[:k]) & target)
            for ranking, target in zip(rankings, targets, strict=True)
        )
        / len(targets)
        for k in (1, 5, 10)
    }
    reciprocal = []
    for ranking, target in zip(rankings, targets, strict=True):
        rank = next(
            (index for index, unit_id in enumerate(ranking, 1) if unit_id in target),
            None,
        )
        reciprocal.append(1.0 / rank if rank else 0.0)
    return EvalResult(len(targets), recall, sum(reciprocal) / len(reciprocal))


def delete_indexed_session(
    projects_root: Path, project_id: str, session_id: str
) -> None:
    """Best-effort coordinated cleanup for deletion and reassignment."""

    try:
        TranscriptIndex(projects_root / project_id, project_id).delete_session(
            session_id
        )
    except (OSError, sqlite3.Error, TranscriptIndexError, ValueError) as exc:
        logger.warning("could not remove transcript index session: %s", exc)


def _active_tail(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], ...]:
    """Return only the active, incomplete turn needed by the next append."""

    entries = [
        ConversationEntry.from_dict(row)
        for row in rows
        if type(row.get("seq")) is int and row.get("type") != "header"
    ]
    active = [entry.to_dict() for entry in active_branch(entries)]
    turn_start: int | None = None
    completed = False
    for index, row in enumerate(active):
        message = _message(row)
        if message is None:
            continue
        role = message.get("role")
        if role == "user":
            if turn_start is None or completed:
                turn_start = index
                completed = False
        elif role == "assistant" and turn_start is not None:
            completed = _completed_assistant(message)
    if turn_start is None or completed:
        return ()
    return tuple(active[turn_start:])


def _remove_sqlite_sidecars(path: Path) -> None:
    for suffix in ("-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)


def _tokens(value: str) -> list[str]:
    return [match.group().casefold() for match in _TOKEN.finditer(value)][:64]


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


_SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=FULL;
CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL) WITHOUT ROWID;
CREATE TABLE session_cursors (
    session_id TEXT PRIMARY KEY,
    byte_offset INTEGER NOT NULL CHECK(byte_offset >= 0),
    last_seq INTEGER NOT NULL CHECK(last_seq >= 0),
    source_device INTEGER NOT NULL,
    source_inode INTEGER NOT NULL,
    source_mtime_ns INTEGER NOT NULL,
    source_ctime_ns INTEGER NOT NULL,
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    incremental_ready INTEGER NOT NULL CHECK(incremental_ready IN (0, 1)),
    last_entry_id TEXT,
    active_tail_json TEXT NOT NULL,
    prefix_digest TEXT
) WITHOUT ROWID;
CREATE TABLE units (
    id INTEGER PRIMARY KEY,
    unit_id TEXT NOT NULL UNIQUE,
    turn_id TEXT NOT NULL,
    project_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    seq_start INTEGER NOT NULL,
    seq_end INTEGER NOT NULL,
    started_at TEXT,
    ended_at TEXT,
    origin TEXT NOT NULL,
    kind TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    chunk_count INTEGER NOT NULL,
    text TEXT NOT NULL
);
CREATE INDEX units_session ON units(session_id);
CREATE VIRTUAL TABLE unit_fts USING fts5(
    text, content='units', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER units_ai AFTER INSERT ON units BEGIN
    INSERT INTO unit_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER units_ad AFTER DELETE ON units BEGIN
    INSERT INTO unit_fts(unit_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER units_au AFTER UPDATE ON units BEGIN
    INSERT INTO unit_fts(unit_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO unit_fts(rowid, text) VALUES (new.id, new.text);
END;
"""
