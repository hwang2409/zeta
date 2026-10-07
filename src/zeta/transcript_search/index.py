"""Disposable, project-scoped SQLite FTS5 transcript index."""

from __future__ import annotations

import json
import os
import re
import sqlite3
import stat
import tempfile
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .units import TranscriptUnit, render_transcript_units

SCHEMA_VERSION = 1
SANITIZER_VERSION = 1
INDEX_FILENAME = "transcript-index.sqlite3"
MAX_TRANSCRIPT_BYTES = 2 * 1024 * 1024 * 1024
MAX_ROW_BYTES = 32 * 1024 * 1024
_TOKEN = re.compile(r"[\w.-]+", re.UNICODE)
_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}\Z")


@dataclass(frozen=True, slots=True)
class TranscriptSource:
    session_id: str
    session_dir: Path

    @property
    def conversation_path(self) -> Path:
        return self.session_dir / "conversation.jsonl"


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
    generation: int
    unit_count: int
    session_count: int
    cursors: dict[str, int]
    path: Path
    size_bytes: int


class TranscriptIndex:
    """Own one project's complete lexical index behind a small interface."""

    def __init__(self, project_dir: Path, project_id: str) -> None:
        if not _PROJECT_ID.fullmatch(project_id):
            raise ValueError("invalid project id")
        self.project_dir = Path(project_dir)
        self.project_id = project_id
        self.path = self.project_dir / INDEX_FILENAME
        self.project_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        self._initialize(self.path)

    def append(self, source: TranscriptSource) -> IndexStatus:
        """Idempotently replace one session projection and advance its cursor."""

        rows = _read_transcript(source.conversation_path)
        units = render_transcript_units(self.project_id, source.session_id, rows)
        cursor = max((unit.seq_end for unit in units), default=0)
        self.append_units(source.session_id, units, cursor=cursor)
        return self.status()

    def append_units(
        self,
        session_id: str,
        units: Sequence[TranscriptUnit],
        *,
        cursor: int,
    ) -> None:
        """Atomically replace one session's units and cursor."""

        if not session_id or type(cursor) is not int or cursor < 0:
            raise ValueError("invalid session cursor")
        if any(
            unit.project_id != self.project_id or unit.session_id != session_id
            for unit in units
        ):
            raise ValueError("transcript unit belongs to another project or session")
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
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
            connection.execute(
                """INSERT INTO session_cursors(session_id, last_seq)
                VALUES (?, ?) ON CONFLICT(session_id) DO UPDATE SET last_seq=excluded.last_seq""",
                (session_id, cursor),
            )
            connection.commit()

    def rebuild(self, sources: Iterable[TranscriptSource]) -> IndexStatus:
        """Build a new generation and atomically publish it."""

        generation = self.status().generation + 1
        fd, temporary = tempfile.mkstemp(
            dir=self.project_dir, prefix=".transcript-index.", suffix=".sqlite3"
        )
        os.close(fd)
        temporary_path = Path(temporary)
        try:
            temporary_path.unlink()
            replacement = object.__new__(TranscriptIndex)
            replacement.project_dir = self.project_dir
            replacement.project_id = self.project_id
            replacement.path = temporary_path
            replacement._initialize(temporary_path)
            with replacement._connect() as connection:
                connection.execute(
                    "UPDATE metadata SET value = ? WHERE key = 'generation'",
                    (str(generation),),
                )
                connection.commit()
            for source in sources:
                replacement.append(source)
            os.chmod(temporary_path, 0o600)
            os.replace(temporary_path, self.path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return self.status()

    def delete_session(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM units WHERE session_id = ?", (session_id,))
            connection.execute(
                "DELETE FROM session_cursors WHERE session_id = ?", (session_id,)
            )
            connection.commit()

    def search(self, query: str, *, limit: int = 10) -> tuple[SearchHit, ...]:
        """Rank all-term matches first, then rare-term partial matches."""

        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("search limit must be between 1 and 100")
        terms = list(dict.fromkeys(_tokens(query)))
        if not terms:
            return ()
        with self._connect() as connection:
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
                for term, count in sorted(counts.items(), key=lambda item: (item[1] or 10**12, item[0]))
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

    def status(self) -> IndexStatus:
        with self._connect() as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            unit_count = int(connection.execute("SELECT count(*) FROM units").fetchone()[0])
            cursors = {
                str(session_id): int(last_seq)
                for session_id, last_seq in connection.execute(
                    "SELECT session_id, last_seq FROM session_cursors ORDER BY session_id"
                )
            }
        return IndexStatus(
            project_id=self.project_id,
            schema_version=int(metadata["schema_version"]),
            generation=int(metadata["generation"]),
            unit_count=unit_count,
            session_count=len(cursors),
            cursors=cursors,
            path=self.path,
            size_bytes=self.path.stat().st_size,
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

    def _hit(self, row: sqlite3.Row, match: str) -> SearchHit:
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
        return connection

    def _initialize(self, path: Path) -> None:
        with sqlite3.connect(path) as connection:
            connection.executescript(_SCHEMA)
            existing = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema_version'"
            ).fetchone()
            if existing is not None and int(existing[0]) != SCHEMA_VERSION:
                raise RuntimeError("transcript index schema requires rebuild")
            connection.executemany(
                "INSERT OR IGNORE INTO metadata(key, value) VALUES (?, ?)",
                (
                    ("schema_version", str(SCHEMA_VERSION)),
                    ("sanitizer_version", str(SANITIZER_VERSION)),
                    ("generation", "0"),
                ),
            )
            connection.commit()
        os.chmod(path, 0o600)


def rebuild_project_index(
    index: TranscriptIndex, sources: Iterable[TranscriptSource]
) -> IndexStatus:
    return index.rebuild(tuple(sources))


def _read_transcript(path: Path) -> list[dict[str, Any]]:
    flags = os.O_RDONLY | os.O_NOFOLLOW
    fd = os.open(path, flags)
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_TRANSCRIPT_BYTES:
            raise ValueError("transcript is not a bounded regular file")
        rows: list[dict[str, Any]] = []
        with os.fdopen(fd, "rb", closefd=False) as handle:
            while True:
                line = handle.readline(MAX_ROW_BYTES + 1)
                if not line:
                    break
                if len(line) > MAX_ROW_BYTES:
                    raise ValueError("transcript row exceeds size limit")
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("transcript row is not an object")
                rows.append(value)
        return rows
    finally:
        os.close(fd)


def _tokens(value: str) -> list[str]:
    return [match.group().casefold() for match in _TOKEN.finditer(value)][:64]


def _quote(term: str) -> str:
    return '"' + term.replace('"', '""') + '"'


_SCHEMA = """
PRAGMA journal_mode=DELETE;
PRAGMA synchronous=FULL;
CREATE TABLE IF NOT EXISTS metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS session_cursors (
    session_id TEXT PRIMARY KEY,
    last_seq INTEGER NOT NULL CHECK(last_seq >= 0)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS units (
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
CREATE INDEX IF NOT EXISTS units_session ON units(session_id);
CREATE VIRTUAL TABLE IF NOT EXISTS unit_fts USING fts5(
    text, content='units', content_rowid='id', tokenize='unicode61'
);
CREATE TRIGGER IF NOT EXISTS units_ai AFTER INSERT ON units BEGIN
    INSERT INTO unit_fts(rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS units_ad AFTER DELETE ON units BEGIN
    INSERT INTO unit_fts(unit_fts, rowid, text) VALUES('delete', old.id, old.text);
END;
CREATE TRIGGER IF NOT EXISTS units_au AFTER UPDATE ON units BEGIN
    INSERT INTO unit_fts(unit_fts, rowid, text) VALUES('delete', old.id, old.text);
    INSERT INTO unit_fts(rowid, text) VALUES (new.id, new.text);
END;
"""
