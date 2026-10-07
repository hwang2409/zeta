"""Deep project transcript index: ingestion, lifecycle, and lexical search."""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import weakref
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Self

from zeta.core.checkpoints import ConversationEntry, active_branch
from zeta.memory.safety import redact_secrets
from zeta.transcript_search.source import (
    _SAFE_ARGUMENTS,
    MAX_DIAGNOSTIC_CHARS,
    _Cursor,
    _read_transcript,
)

logger = logging.getLogger(__name__)

MAX_UNIT_BYTES = 12 * 1024
MAX_TOOL_DIGEST_BYTES = 4 * 1024
_UNIT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class TranscriptUnit:
    """One bounded search document with a trusted source range."""

    schema_version: int
    unit_id: str
    turn_id: str
    project_id: str
    session_id: str
    seq_start: int
    seq_end: int
    started_at: str | None
    ended_at: str | None
    origin: str
    kind: Literal["turn", "child_report"]
    chunk_index: int
    chunk_count: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TranscriptUnit:
        return cls(**dict(value))


@dataclass(slots=True)
class _Turn:
    rows: list[Mapping[str, Any]]
    human: list[str]
    inputs: list[str]
    assistant: list[str]
    tools: list[str]
    completed: bool = False


def render_transcript_units(
    project_id: str,
    session_id: str,
    rows: Iterable[Mapping[str, Any]],
) -> tuple[TranscriptUnit, ...]:
    """Render completed turns and parent-visible child reports deterministically."""

    entries = sorted(
        (
            ConversationEntry.from_dict(row)
            for row in rows
            if type(row.get("seq")) is int and row.get("type") != "header"
        ),
        key=lambda entry: entry.seq,
    )
    ordered = [entry.to_dict() for entry in active_branch(entries)]
    units: list[TranscriptUnit] = []
    turn: _Turn | None = None
    call_names: dict[str, str] = {}
    for row in ordered:
        report = _child_report(row)
        if report is not None:
            units.extend(_report_units(project_id, session_id, row, report))
            continue
        message = _message(row)
        if message is None:
            continue
        role = message.get("role")
        if role == "user":
            if turn is not None and turn.completed:
                units.extend(_turn_units(project_id, session_id, turn))
                turn = None
            if turn is None:
                turn = _Turn([], [], [], [], [])
            turn.rows.append(row)
            text = _message_text(message)
            if text:
                if _is_human(message, row):
                    turn.human.append(text)
                else:
                    turn.inputs.append(text)
            continue
        if turn is None:
            continue
        turn.rows.append(row)
        if role == "assistant":
            text = _message_text(message)
            if text and message.get("metadata", {}).get("response_state") != "synthetic":
                turn.assistant.append(text)
            for call in _tool_calls(message):
                call_id = call.get("id")
                name = call.get("name")
                if isinstance(call_id, str) and isinstance(name, str):
                    call_names[call_id] = name
                turn.tools.append(_tool_call_digest(call))
            turn.completed = _completed_assistant(message)
        elif role == "tool_result":
            result = message.get("tool_result")
            if isinstance(result, Mapping):
                turn.tools.append(_tool_result_digest(result, call_names))
    if turn is not None and turn.completed:
        units.extend(_turn_units(project_id, session_id, turn))
    return tuple(sorted(units, key=lambda unit: (unit.seq_start, unit.kind, unit.chunk_index)))


def _message(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    if row.get("type") != "message":
        return None
    data = row.get("data")
    message = data.get("message") if isinstance(data, Mapping) else None
    return message if isinstance(message, Mapping) else None


def _message_text(message: Mapping[str, Any]) -> str:
    content = message.get("content")
    if not isinstance(content, list):
        return ""
    values = [
        block["text"]
        for block in content
        if isinstance(block, Mapping)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
    ]
    return "\n".join(value.strip() for value in values if value.strip())


def _is_human(message: Mapping[str, Any], row: Mapping[str, Any]) -> bool:
    metadata = message.get("metadata")
    origin = metadata.get("zeta.origin") if isinstance(metadata, Mapping) else None
    return origin == "user"


def _completed_assistant(message: Mapping[str, Any]) -> bool:
    metadata = message.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    state = metadata.get("response_state")
    if state is not None:
        return state == "completed"
    return not metadata.get("turn_failed")


def _tool_calls(message: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    calls: list[Mapping[str, Any]] = []
    content = message.get("content")
    if isinstance(content, list):
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "tool_use":
                continue
            call = block.get("tool_call")
            if isinstance(call, Mapping):
                calls.append(call)
    return calls


def _tool_call_digest(call: Mapping[str, Any]) -> str:
    name = str(call.get("name") or "unknown")
    arguments = call.get("arguments")
    fields: list[str] = []
    if isinstance(arguments, Mapping):
        keys = sorted(str(key) for key in arguments)
        fields.append("arguments=" + ",".join(keys[:24]))
        for key in _SAFE_ARGUMENTS:
            value = arguments.get(key)
            if isinstance(value, str) and value:
                fields.append(f"{key}={_bounded_text(redact_secrets(value), 320)}")
    return _bounded_text(f"Tool {name} call: {'; '.join(fields)}", 720)


def _tool_result_digest(result: Mapping[str, Any], names: Mapping[str, str]) -> str:
    call_id = result.get("tool_call_id")
    name = names.get(call_id, "unknown") if isinstance(call_id, str) else "unknown"
    status = "error" if result.get("is_error") else "success"
    content = result.get("content")
    text = content if isinstance(content, str) else ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    diagnostic = ""
    if lines:
        selected = [lines[0]] if len(lines) == 1 else [lines[0], lines[-1]]
        diagnostic = "; diagnostics=" + " | ".join(
            _bounded_text(line, MAX_DIAGNOSTIC_CHARS) for line in selected
        )
    projected_size = result.get("source_bytes")
    size = (
        projected_size
        if type(projected_size) is int and projected_size >= 0
        else len(content.encode("utf-8")) if isinstance(content, str) else 0
    )
    return _bounded_text(
        f"Tool {name} result: {status}; bytes={size}{diagnostic}", 720
    )


def _child_report(row: Mapping[str, Any]) -> tuple[str, str, str] | None:
    if row.get("type") != "notification":
        return None
    data = row.get("data")
    if not isinstance(data, Mapping) or data.get("kind") != "agent_completion":
        return None
    text = data.get("text")
    if not isinstance(text, str) or not text:
        return None
    return (
        str(data.get("description") or "child agent"),
        str(data.get("status") or "unknown"),
        text,
    )


def _turn_units(project_id: str, session_id: str, turn: _Turn) -> list[TranscriptUnit]:
    sections: list[str] = []
    sections.extend(f"Human: {text}" for text in turn.human)
    sections.extend(f"Input: {text}" for text in turn.inputs)
    sections.extend(f"Agent: {text}" for text in turn.assistant)
    if turn.tools:
        digest = _bounded_utf8("\n".join(turn.tools), MAX_TOOL_DIGEST_BYTES)
        sections.append("Tool activity:\n" + digest)
    origin = "human" if turn.human else "unverified_input"
    return _make_units(project_id, session_id, turn.rows, origin, "turn", "\n\n".join(sections))


def _report_units(
    project_id: str,
    session_id: str,
    row: Mapping[str, Any],
    report: tuple[str, str, str],
) -> list[TranscriptUnit]:
    description, status, text = report
    rendered = f"Child report ({description}, {status}):\n{text}"
    return _make_units(
        project_id, session_id, [row], "parent_visible_child", "child_report", rendered
    )


def _make_units(
    project_id: str,
    session_id: str,
    rows: list[Mapping[str, Any]],
    origin: str,
    kind: Literal["turn", "child_report"],
    text: str,
) -> list[TranscriptUnit]:
    text = redact_secrets(text)
    if not text.strip():
        return []
    seq_start = int(rows[0]["seq"])
    seq_end = int(rows[-1]["seq"])
    turn_id = _stable_id(session_id, kind, seq_start, seq_end)
    chunks = _split_utf8(text, MAX_UNIT_BYTES)
    timestamps = [_timestamp(row) for row in rows]
    timestamps = [value for value in timestamps if value is not None]
    base = TranscriptUnit(
        schema_version=_UNIT_SCHEMA_VERSION,
        unit_id="",
        turn_id=turn_id,
        project_id=project_id,
        session_id=session_id,
        seq_start=seq_start,
        seq_end=seq_end,
        started_at=timestamps[0] if timestamps else None,
        ended_at=timestamps[-1] if timestamps else None,
        origin=origin,
        kind=kind,
        chunk_index=0,
        chunk_count=len(chunks),
        text="",
    )
    return [
        replace(
            base,
            unit_id=_stable_id(turn_id, str(index)),
            chunk_index=index,
            text=chunk,
        )
        for index, chunk in enumerate(chunks)
    ]


def _timestamp(row: Mapping[str, Any]) -> str | None:
    data = row.get("data")
    if not isinstance(data, Mapping):
        return None
    value = data.get("created_at") or data.get("timestamp")
    return value if isinstance(value, str) else None


def _stable_id(*parts: str | int) -> str:
    source = "\0".join(str(part) for part in parts).encode()
    return hashlib.sha256(source).hexdigest()[:32]


def _bounded_text(value: str, max_chars: int) -> str:
    clean = re.sub(r"\s+", " ", value).strip()
    return clean if len(clean) <= max_chars else clean[: max_chars - 1] + "…"


def _bounded_utf8(value: str, max_bytes: int) -> str:
    encoded = value.encode("utf-8")
    if len(encoded) <= max_bytes:
        return value
    return encoded[: max_bytes - 3].decode("utf-8", errors="ignore") + "…"


def _split_utf8(value: str, max_bytes: int) -> list[str]:
    remaining = value.strip()
    chunks: list[str] = []
    while len(remaining.encode("utf-8")) > max_bytes:
        encoded = remaining.encode("utf-8")
        prefix = encoded[:max_bytes].decode("utf-8", errors="ignore")
        split = max(prefix.rfind("\n\n"), prefix.rfind("\n"), prefix.rfind(" "))
        if split < max_bytes // 2:
            split = len(prefix)
        chunk = prefix[:split].rstrip()
        chunks.append(chunk)
        remaining = remaining[len(prefix[:split]) :].lstrip()
    if remaining:
        chunks.append(remaining)
    return chunks or [""]


def dump_units(units: Iterable[TranscriptUnit]) -> str:
    """Serialize units as stable JSONL for fixtures and diagnostics."""

    return "".join(json.dumps(unit.to_dict(), sort_keys=True) + "\n" for unit in units)


SCHEMA_VERSION = 3
SANITIZER_VERSION = 2
INDEX_FILENAME = "transcript-index.sqlite3"
MAX_ACTIVE_TAIL_BYTES = 1024 * 1024
_TOKEN = re.compile(r"[\w.-]+", re.UNICODE)
_PROJECT_ID = re.compile(r"p_[0-9a-f]{32}\Z")
_PROJECT_LOCKS: dict[tuple[Path, str], threading.RLock] = {}
_PROJECT_LOCKS_GUARD = threading.Lock()


class TranscriptIndexError(RuntimeError):
    """A recoverable derived-index failure."""


class TranscriptIndexUnavailable(TranscriptIndexError):
    """The disposable index must be rebuilt before it can serve reads."""


@dataclass(frozen=True, slots=True)
class TranscriptSource:
    session_id: str
    session_dir: Path
    expected_project_id: str | None = None

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
    sanitizer_version: int
    generation: int
    unit_count: int
    session_count: int
    cursors: dict[str, int]
    path: Path
    size_bytes: int
    ready: bool
    detail: str | None = None


@dataclass(slots=True)
class _RefreshState:
    pending: dict[str, TranscriptSource] = field(default_factory=dict)
    waiters: list[asyncio.Future[None]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None


_REFRESH_STATES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[Path, str], _RefreshState]
] = weakref.WeakKeyDictionary()


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
            if not self.path.exists():
                self._initialize(self.path)
        except (OSError, sqlite3.Error) as exc:
            raise TranscriptIndexError(f"could not initialize transcript index: {exc}") from exc

    def append(self, source: TranscriptSource) -> IndexStatus:
        """Incrementally ingest one source without rereading committed bytes."""

        with self._write_guard():
            self._require_ready()
            return self._append_locked(source)

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
                        source_inode, source_mtime_ns, source_size,
                        prefix_fingerprint, last_entry_id, active_tail_json
                    ) VALUES (?, 0, ?, 0, 0, 0, 0, '', NULL, '[]')
                    ON CONFLICT(session_id) DO UPDATE SET last_seq=excluded.last_seq""",
                    (session_id, cursor),
                )
                connection.commit()
        except sqlite3.Error as exc:
            raise TranscriptIndexError(f"could not update transcript index: {exc}") from exc

    def rebuild(self, sources: Iterable[TranscriptSource]) -> IndexStatus:
        """Build a fresh generation and atomically publish it."""

        sources = tuple(sources)
        with self._write_guard():
            generation = self._generation_best_effort() + 1
            fd, temporary = tempfile.mkstemp(
                dir=self.project_dir, prefix=".transcript-index.", suffix=".sqlite3"
            )
            os.close(fd)
            temporary_path = Path(temporary)
            temporary_path.unlink()
            try:
                replacement = object.__new__(TranscriptIndex)
                replacement.project_dir = self.project_dir
                replacement.project_id = self.project_id
                replacement.path = temporary_path
                replacement.lock_path = self.lock_path
                replacement._initialize(temporary_path)
                with replacement._connect() as connection:
                    connection.execute(
                        "UPDATE metadata SET value = ? WHERE key = 'generation'",
                        (str(generation),),
                    )
                    connection.commit()
                for source in sources:
                    replacement._append_locked(source, force_full=True)
                with replacement._connect() as connection:
                    connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                os.chmod(temporary_path, 0o600)
                _remove_sqlite_sidecars(self.path)
                os.replace(temporary_path, self.path)
                with self._connect() as connection:
                    connection.execute("PRAGMA journal_mode=WAL")
                return self._status()
            except (OSError, sqlite3.Error, ValueError, TypeError) as exc:
                raise TranscriptIndexError(f"could not rebuild transcript index: {exc}") from exc
            finally:
                temporary_path.unlink(missing_ok=True)
                Path(f"{temporary_path}-wal").unlink(missing_ok=True)
                Path(f"{temporary_path}-shm").unlink(missing_ok=True)

    def delete_session(self, session_id: str) -> None:
        with self._write_guard():
            try:
                self._require_ready()
                with self._connect() as connection:
                    connection.execute("BEGIN IMMEDIATE")
                    connection.execute("DELETE FROM units WHERE session_id = ?", (session_id,))
                    connection.execute("DELETE FROM session_cursors WHERE session_id = ?", (session_id,))
                    connection.commit()
            except sqlite3.Error as exc:
                raise TranscriptIndexError(f"could not remove indexed session: {exc}") from exc

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
            raise TranscriptIndexError(f"could not search transcript index: {exc}") from exc

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
                with self._connect() as connection:
                    unit_count = int(connection.execute("SELECT count(*) FROM units").fetchone()[0])
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
        try:
            with self._connect() as connection:
                cursor = self._cursor(connection, source.session_id)
                indexed_seq = connection.execute(
                    "SELECT coalesce(max(seq_end), 0) FROM units WHERE session_id = ?",
                    (source.session_id,),
                ).fetchone()[0]
            if cursor is not None and int(indexed_seq) > cursor.last_seq:
                cursor = None
            read = _read_transcript(
                source.conversation_path, None if force_full else cursor
            )
            if (
                not read.full
                and read.rows
                and read.rows[0].get("parent_id") != cursor.last_entry_id
            ):
                read = _read_transcript(source.conversation_path, None)
            prior_tail = () if read.full or cursor is None else cursor.active_tail
            render_rows = prior_tail + read.rows
            units = render_transcript_units(
                self.project_id, source.session_id, render_rows
            )
            active_tail = _active_tail(render_rows)
            tail_json = json.dumps(
                active_tail, separators=(",", ":"), sort_keys=True
            )
            incremental_ready = len(tail_json.encode("utf-8")) <= MAX_ACTIVE_TAIL_BYTES
            if not incremental_ready:
                active_tail = ()
                tail_json = "[]"
            with self._connect() as connection:
                connection.execute("BEGIN IMMEDIATE")
                stored = self._cursor(connection, source.session_id)
                if not read.full and stored != cursor:
                    connection.rollback()
                    return self._status()
                if not self._source_is_bound(source):
                    connection.rollback()
                    self._delete_session_rows(source.session_id)
                    return self._status()
                if read.full:
                    self._replace_units(connection, source.session_id, units)
                else:
                    self._upsert_units(connection, units)
                connection.execute(
                    """INSERT INTO session_cursors(
                        session_id, byte_offset, last_seq, source_device,
                        source_inode, source_mtime_ns, source_size,
                        prefix_fingerprint, last_entry_id, active_tail_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(session_id) DO UPDATE SET
                        byte_offset=excluded.byte_offset,
                        last_seq=excluded.last_seq,
                        source_device=excluded.source_device,
                        source_inode=excluded.source_inode,
                        source_mtime_ns=excluded.source_mtime_ns,
                        source_size=excluded.source_size,
                        prefix_fingerprint=excluded.prefix_fingerprint,
                        last_entry_id=excluded.last_entry_id,
                        active_tail_json=excluded.active_tail_json""",
                    (
                        source.session_id,
                        read.byte_offset,
                        read.last_seq,
                        read.source_device,
                        read.source_inode,
                        read.source_mtime_ns,
                        read.source_size,
                        read.prefix_fingerprint if incremental_ready else "",
                        read.last_entry_id,
                        tail_json,
                    ),
                )
                connection.commit()
            return self._status()
        except sqlite3.Error as exc:
            raise TranscriptIndexError(f"could not refresh transcript index: {exc}") from exc

    def _delete_session_rows(self, session_id: str) -> None:
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute("DELETE FROM units WHERE session_id = ?", (session_id,))
            connection.execute("DELETE FROM session_cursors WHERE session_id = ?", (session_id,))
            connection.commit()

    def _source_is_bound(self, source: TranscriptSource) -> bool:
        expected = source.expected_project_id
        if expected is None:
            return True
        try:
            payload = (source.session_dir / "meta.json").read_bytes()
            if len(payload) > 1024 * 1024:
                return False
            value = json.loads(payload)
        except (OSError, ValueError):
            return False
        return isinstance(value, dict) and value.get("project_id") == expected == self.project_id

    @staticmethod
    def _cursor(connection: sqlite3.Connection, session_id: str) -> _Cursor | None:
        row = connection.execute(
            """SELECT byte_offset, last_seq, source_device, source_inode,
                source_mtime_ns, source_size, prefix_fingerprint,
                last_entry_id, active_tail_json
            FROM session_cursors WHERE session_id = ?""",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return _Cursor(
            *(int(row[index]) for index in range(6)),
            str(row[6]),
            str(row[7]) if row[7] is not None else None,
            tuple(json.loads(row[8])),
        )

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

    @staticmethod
    def _initialize(path: Path) -> None:
        with sqlite3.connect(path) as connection:
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
        os.chmod(path, 0o600)

    def _version_state(self) -> tuple[int, int, int, str | None]:
        try:
            with sqlite3.connect(f"file:{self.path}?mode=ro", uri=True) as connection:
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
            raise TranscriptIndexUnavailable(status.detail or "transcript index requires rebuild")

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
            self.fd = os.open(self.lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
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


async def refresh_transcript_index(
    projects_root: Path,
    project_id: str,
    session_id: str,
    session_dir: Path,
) -> None:
    """Coalesce project refreshes and perform every filesystem operation off-loop."""

    loop = asyncio.get_running_loop()
    key = (Path(projects_root).resolve(), project_id)
    states = _REFRESH_STATES.setdefault(loop, {})
    state = states.setdefault(key, _RefreshState())
    state.pending[session_id] = TranscriptSource(session_id, session_dir, project_id)
    waiter = loop.create_future()
    state.waiters.append(waiter)
    if state.task is None:
        state.task = loop.create_task(_drain_refreshes(key, state, states))
    await waiter


async def _drain_refreshes(
    key: tuple[Path, str],
    state: _RefreshState,
    states: dict[tuple[Path, str], _RefreshState],
) -> None:
    error: BaseException | None = None
    try:
        while state.pending:
            pending = tuple(state.pending.values())
            state.pending.clear()
            await asyncio.to_thread(_refresh_sources, key[0], key[1], pending)
    except (OSError, sqlite3.Error, TranscriptIndexError, TypeError, ValueError) as exc:
        logger.warning("could not refresh transcript index: %s", exc)
        error = None
    finally:
        waiters, state.waiters = state.waiters, []
        state.task = None
        states.pop(key, None)
        for waiter in waiters:
            if not waiter.done():
                if error is None:
                    waiter.set_result(None)
                else:
                    waiter.set_exception(error)


def _refresh_sources(
    projects_root: Path,
    project_id: str,
    sources: Sequence[TranscriptSource],
) -> None:
    index = TranscriptIndex(projects_root / project_id, project_id)
    for source in sources:
        index.append(source)


def delete_indexed_session(
    projects_root: Path, project_id: str, session_id: str
) -> None:
    """Best-effort coordinated cleanup for deletion and reassignment."""

    try:
        TranscriptIndex(projects_root / project_id, project_id).delete_session(session_id)
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
    source_size INTEGER NOT NULL CHECK(source_size >= 0),
    prefix_fingerprint TEXT NOT NULL,
    last_entry_id TEXT,
    active_tail_json TEXT NOT NULL
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
