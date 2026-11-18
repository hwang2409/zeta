"""The viewer database: exact slice-1 schema, one writer thread, and read-only queries.

Only the writer thread writes. Reads use a separate read-only connection; both run off the
event loop. An OTLP batch is acknowledged only after its transaction commits.
"""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import os
import queue
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .receiver import EVENT_COLUMNS, SPAN_COLUMNS, root_session_id

SCHEMA_VERSION = 1
INGEST_QUEUE_CAPACITY = 64
GRACEFUL_COMMIT_TIMEOUT_SECONDS = 5.0
BUSY_TIMEOUT_MS = 5_000
DEFAULT_PAGE_SIZE = 50
MAX_PAGE_SIZE = 100

SCHEMA = """
CREATE TABLE viewer_meta (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    schema_version INTEGER NOT NULL CHECK (schema_version >= 1),
    committed_sequence INTEGER NOT NULL DEFAULT 0 CHECK (committed_sequence >= 0)
) STRICT;

CREATE TABLE spans (
    trace_id BLOB NOT NULL CHECK (length(trace_id) = 16),
    span_id BLOB NOT NULL CHECK (length(span_id) = 8),
    parent_span_id BLOB CHECK (parent_span_id IS NULL OR length(parent_span_id) = 8),
    operation TEXT NOT NULL CHECK (operation IN ('invoke_agent','turn','chat','execute_tool','context_compact','serve','mcp')),
    kind INTEGER NOT NULL CHECK (kind BETWEEN 0 AND 5),
    start_ns INTEGER NOT NULL CHECK (start_ns >= 0),
    end_ns INTEGER NOT NULL CHECK (end_ns >= start_ns),
    duration_ns INTEGER NOT NULL CHECK (duration_ns = end_ns - start_ns),
    status TEXT NOT NULL CHECK (status IN ('unset','ok','error')),
    error_type TEXT,
    session_id TEXT,
    root_session_id TEXT,
    service_version TEXT NOT NULL,
    scope_version TEXT NOT NULL,
    agent_name TEXT,
    agent_depth INTEGER CHECK (agent_depth IS NULL OR agent_depth >= 0),
    background INTEGER CHECK (background IS NULL OR background IN (0,1)),
    turn_number INTEGER CHECK (turn_number IS NULL OR turn_number >= 0),
    notification INTEGER CHECK (notification IS NULL OR notification IN (0,1)),
    provider TEXT,
    request_model TEXT,
    response_model TEXT,
    stream INTEGER CHECK (stream IS NULL OR stream IN (0,1)),
    input_tokens INTEGER CHECK (input_tokens IS NULL OR input_tokens >= 0),
    output_tokens INTEGER CHECK (output_tokens IS NULL OR output_tokens >= 0),
    total_tokens INTEGER CHECK (total_tokens IS NULL OR total_tokens >= 0),
    finish_reason TEXT,
    tool_name TEXT,
    tool_type TEXT,
    tool_call_id TEXT,
    tool_argument_bytes INTEGER CHECK (tool_argument_bytes IS NULL OR tool_argument_bytes >= 0),
    tool_result_bytes INTEGER CHECK (tool_result_bytes IS NULL OR tool_result_bytes >= 0),
    tool_count INTEGER CHECK (tool_count IS NULL OR tool_count >= 0),
    emergency INTEGER CHECK (emergency IS NULL OR emergency IN (0,1)),
    tokens_before INTEGER CHECK (tokens_before IS NULL OR tokens_before >= 0),
    tokens_after INTEGER CHECK (tokens_after IS NULL OR tokens_after >= 0),
    rpc_method TEXT,
    mcp_method TEXT,
    mcp_server TEXT,
    mcp_tool TEXT,
    mcp_prompt TEXT,
    mcp_protocol TEXT,
    request_id TEXT,
    PRIMARY KEY (trace_id, span_id)
) STRICT;

CREATE TABLE span_events (
    trace_id BLOB NOT NULL CHECK (length(trace_id) = 16),
    span_id BLOB NOT NULL CHECK (length(span_id) = 8),
    event_index INTEGER NOT NULL CHECK (event_index >= 0),
    name TEXT NOT NULL CHECK (name IN ('gen_ai.client.retry','zeta.approval.requested','zeta.approval.resolved','zeta.serve.invalid_propagation')),
    time_ns INTEGER NOT NULL CHECK (time_ns >= 0),
    retry_attempt INTEGER CHECK (retry_attempt IS NULL OR retry_attempt >= 0),
    retry_reason TEXT,
    retry_delay INTEGER CHECK (retry_delay IS NULL OR retry_delay >= 0),
    retry_after INTEGER CHECK (retry_after IS NULL OR retry_after IN (0,1)),
    decision TEXT,
    scope TEXT,
    duration_ns INTEGER CHECK (duration_ns IS NULL OR duration_ns >= 0),
    PRIMARY KEY (trace_id, span_id, event_index),
    FOREIGN KEY (trace_id, span_id) REFERENCES spans(trace_id, span_id) ON DELETE CASCADE
) STRICT;

CREATE TABLE traces (
    trace_id BLOB PRIMARY KEY CHECK (length(trace_id) = 16),
    first_ns INTEGER NOT NULL CHECK (first_ns >= 0),
    last_ns INTEGER NOT NULL CHECK (last_ns >= first_ns),
    root_span_id BLOB CHECK (root_span_id IS NULL OR length(root_span_id) = 8),
    root_session_id TEXT,
    status TEXT NOT NULL CHECK (status IN ('unset','ok','error')),
    span_count INTEGER NOT NULL DEFAULT 0 CHECK (span_count >= 0),
    error_count INTEGER NOT NULL DEFAULT 0 CHECK (error_count >= 0)
) STRICT;

CREATE TABLE sessions (
    root_session_id TEXT PRIMARY KEY,
    first_ns INTEGER NOT NULL CHECK (first_ns >= 0),
    last_ns INTEGER NOT NULL CHECK (last_ns >= first_ns),
    last_trace_id BLOB NOT NULL CHECK (length(last_trace_id) = 16),
    span_count INTEGER NOT NULL DEFAULT 0 CHECK (span_count >= 0),
    turn_count INTEGER NOT NULL DEFAULT 0 CHECK (turn_count >= 0),
    model_count INTEGER NOT NULL DEFAULT 0 CHECK (model_count >= 0),
    tool_count INTEGER NOT NULL DEFAULT 0 CHECK (tool_count >= 0),
    error_count INTEGER NOT NULL DEFAULT 0 CHECK (error_count >= 0),
    input_tokens INTEGER NOT NULL DEFAULT 0 CHECK (input_tokens >= 0),
    output_tokens INTEGER NOT NULL DEFAULT 0 CHECK (output_tokens >= 0),
    total_tokens INTEGER NOT NULL DEFAULT 0 CHECK (total_tokens >= 0),
    profile TEXT,
    model TEXT,
    last_span_id BLOB NOT NULL CHECK (length(last_span_id) = 8),
    profile_last_ns INTEGER CHECK (profile_last_ns IS NULL OR profile_last_ns >= 0),
    profile_trace_id BLOB CHECK (profile_trace_id IS NULL OR length(profile_trace_id) = 16),
    profile_span_id BLOB CHECK (profile_span_id IS NULL OR length(profile_span_id) = 8),
    model_last_ns INTEGER CHECK (model_last_ns IS NULL OR model_last_ns >= 0),
    model_trace_id BLOB CHECK (model_trace_id IS NULL OR length(model_trace_id) = 16),
    model_span_id BLOB CHECK (model_span_id IS NULL OR length(model_span_id) = 8),
    CHECK ((profile IS NULL) = (profile_last_ns IS NULL)),
    CHECK ((profile IS NULL) = (profile_trace_id IS NULL)),
    CHECK ((profile IS NULL) = (profile_span_id IS NULL)),
    CHECK ((model IS NULL) = (model_last_ns IS NULL)),
    CHECK ((model IS NULL) = (model_trace_id IS NULL)),
    CHECK ((model IS NULL) = (model_span_id IS NULL))
) STRICT;

CREATE INDEX spans_trace_start ON spans(trace_id, start_ns);
CREATE INDEX spans_parent ON spans(trace_id, parent_span_id);
CREATE INDEX spans_session_start ON spans(root_session_id, start_ns);
CREATE INDEX spans_provider_model_start ON spans(provider, request_model, start_ns);
CREATE INDEX spans_tool_start ON spans(tool_name, start_ns);
CREATE INDEX spans_error_start ON spans(start_ns) WHERE status = 'error';
CREATE INDEX span_events_span_time ON span_events(trace_id, span_id, time_ns);
CREATE INDEX span_events_name_time ON span_events(name, time_ns);
CREATE INDEX traces_last ON traces(last_ns DESC);
CREATE INDEX traces_session_last ON traces(root_session_id, last_ns DESC);
CREATE INDEX traces_status_last ON traces(status, last_ns DESC);
CREATE INDEX sessions_last ON sessions(last_ns DESC);
CREATE INDEX sessions_profile_last ON sessions(profile, last_ns DESC);
CREATE INDEX sessions_model_last ON sessions(model, last_ns DESC);
"""

_TRACE_UPSERT = """
INSERT INTO traces (
    trace_id, first_ns, last_ns, root_span_id, root_session_id,
    status, span_count, error_count
) VALUES (
    :trace_id, :first_ns, :last_ns, :root_span_id, :root_session_id,
    :status, :span_count, :error_count
)
ON CONFLICT(trace_id) DO UPDATE SET
    first_ns = MIN(traces.first_ns, excluded.first_ns),
    last_ns = MAX(traces.last_ns, excluded.last_ns),
    root_span_id = COALESCE(traces.root_span_id, excluded.root_span_id),
    root_session_id = COALESCE(traces.root_session_id, excluded.root_session_id),
    status = CASE
        WHEN traces.status = 'error' OR excluded.status = 'error' THEN 'error'
        WHEN traces.status = 'ok' OR excluded.status = 'ok' THEN 'ok'
        ELSE 'unset'
    END,
    span_count = traces.span_count + excluded.span_count,
    error_count = traces.error_count + excluded.error_count;
"""

_SESSION_UPSERT = """
INSERT INTO sessions (
    root_session_id, first_ns, last_ns, last_trace_id,
    span_count, turn_count, model_count, tool_count, error_count,
    input_tokens, output_tokens, total_tokens,
    profile, model, last_span_id,
    profile_last_ns, profile_trace_id, profile_span_id,
    model_last_ns, model_trace_id, model_span_id
) VALUES (
    :root_session_id, :first_ns, :last_ns, :last_trace_id,
    :span_count, :turn_count, :model_count, :tool_count, :error_count,
    :input_tokens, :output_tokens, :total_tokens,
    :profile, :model, :last_span_id,
    :profile_last_ns, :profile_trace_id, :profile_span_id,
    :model_last_ns, :model_trace_id, :model_span_id
)
ON CONFLICT(root_session_id) DO UPDATE SET
    first_ns = MIN(sessions.first_ns, excluded.first_ns),
    last_ns = MAX(sessions.last_ns, excluded.last_ns),
    last_trace_id = CASE
        WHEN (excluded.last_ns, excluded.last_trace_id, excluded.last_span_id) >
        (sessions.last_ns, sessions.last_trace_id, sessions.last_span_id)
        THEN excluded.last_trace_id ELSE sessions.last_trace_id
    END,
    last_span_id = CASE
        WHEN (excluded.last_ns, excluded.last_trace_id, excluded.last_span_id) >
        (sessions.last_ns, sessions.last_trace_id, sessions.last_span_id)
        THEN excluded.last_span_id ELSE sessions.last_span_id
    END,
    span_count = sessions.span_count + excluded.span_count,
    turn_count = sessions.turn_count + excluded.turn_count,
    model_count = sessions.model_count + excluded.model_count,
    tool_count = sessions.tool_count + excluded.tool_count,
    error_count = sessions.error_count + excluded.error_count,
    input_tokens = sessions.input_tokens + excluded.input_tokens,
    output_tokens = sessions.output_tokens + excluded.output_tokens,
    total_tokens = sessions.total_tokens + excluded.total_tokens,
    profile = CASE
        WHEN excluded.profile IS NOT NULL AND (
            sessions.profile IS NULL OR
            (excluded.profile_last_ns, excluded.profile_trace_id, excluded.profile_span_id) >
            (sessions.profile_last_ns, sessions.profile_trace_id, sessions.profile_span_id)
        ) THEN excluded.profile ELSE sessions.profile
    END,
    profile_last_ns = CASE
        WHEN excluded.profile IS NOT NULL AND (
            sessions.profile IS NULL OR
            (excluded.profile_last_ns, excluded.profile_trace_id, excluded.profile_span_id) >
            (sessions.profile_last_ns, sessions.profile_trace_id, sessions.profile_span_id)
        ) THEN excluded.profile_last_ns ELSE sessions.profile_last_ns
    END,
    profile_trace_id = CASE
        WHEN excluded.profile IS NOT NULL AND (
            sessions.profile IS NULL OR
            (excluded.profile_last_ns, excluded.profile_trace_id, excluded.profile_span_id) >
            (sessions.profile_last_ns, sessions.profile_trace_id, sessions.profile_span_id)
        ) THEN excluded.profile_trace_id ELSE sessions.profile_trace_id
    END,
    profile_span_id = CASE
        WHEN excluded.profile IS NOT NULL AND (
            sessions.profile IS NULL OR
            (excluded.profile_last_ns, excluded.profile_trace_id, excluded.profile_span_id) >
            (sessions.profile_last_ns, sessions.profile_trace_id, sessions.profile_span_id)
        ) THEN excluded.profile_span_id ELSE sessions.profile_span_id
    END,
    model = CASE
        WHEN excluded.model IS NOT NULL AND (
            sessions.model IS NULL OR
            (excluded.model_last_ns, excluded.model_trace_id, excluded.model_span_id) >
            (sessions.model_last_ns, sessions.model_trace_id, sessions.model_span_id)
        ) THEN excluded.model ELSE sessions.model
    END,
    model_last_ns = CASE
        WHEN excluded.model IS NOT NULL AND (
            sessions.model IS NULL OR
            (excluded.model_last_ns, excluded.model_trace_id, excluded.model_span_id) >
            (sessions.model_last_ns, sessions.model_trace_id, sessions.model_span_id)
        ) THEN excluded.model_last_ns ELSE sessions.model_last_ns
    END,
    model_trace_id = CASE
        WHEN excluded.model IS NOT NULL AND (
            sessions.model IS NULL OR
            (excluded.model_last_ns, excluded.model_trace_id, excluded.model_span_id) >
            (sessions.model_last_ns, sessions.model_trace_id, sessions.model_span_id)
        ) THEN excluded.model_trace_id ELSE sessions.model_trace_id
    END,
    model_span_id = CASE
        WHEN excluded.model IS NOT NULL AND (
            sessions.model IS NULL OR
            (excluded.model_last_ns, excluded.model_trace_id, excluded.model_span_id) >
            (sessions.model_last_ns, sessions.model_trace_id, sessions.model_span_id)
        ) THEN excluded.model_span_id ELSE sessions.model_span_id
    END;
"""

_SPAN_INSERT = (
    f"INSERT INTO spans ({', '.join(SPAN_COLUMNS)}) VALUES ({', '.join(':' + name for name in SPAN_COLUMNS)}) "
    "ON CONFLICT(trace_id, span_id) DO NOTHING"
)
_EVENT_INSERT = f"INSERT INTO span_events ({', '.join(EVENT_COLUMNS)}) VALUES ({', '.join(':' + name for name in EVENT_COLUMNS)})"
_SESSION_KEYS = (
    "root_session_id", "first_ns", "last_ns", "last_trace_id", "span_count", "turn_count", "model_count",
    "tool_count", "error_count", "input_tokens", "output_tokens", "total_tokens", "profile", "model",
)
_TRACE_KEYS = ("trace_id", "first_ns", "last_ns", "root_span_id", "root_session_id", "status", "span_count", "error_count")
_TRACE_SPAN_KEYS = (
    "span_id", "parent_span_id", "operation", "kind", "start_ns", "end_ns", "duration_ns", "status", "error_type",
    "session_id", "agent_name", "provider", "request_model", "response_model", "tool_name", "tool_type",
    "input_tokens", "output_tokens", "total_tokens",
)
_EVENT_KEYS = (
    "event_index", "name", "time_ns", "retry_attempt", "retry_reason", "retry_delay", "retry_after", "decision",
    "scope", "duration_ns",
)
_HEX_KEYS = frozenset({"trace_id", "span_id", "parent_span_id", "root_span_id", "last_trace_id"})


def project_span(span: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The trace candidate and optional session candidate of one newly inserted span."""
    root = root_session_id(span["session_id"])
    operation = span["operation"]
    error_count = 1 if span["status"] == "error" else 0
    trace = {
        "trace_id": span["trace_id"],
        "first_ns": span["start_ns"],
        "last_ns": span["end_ns"],
        "root_span_id": span["span_id"] if span["parent_span_id"] is None else None,
        "root_session_id": root,
        "status": span["status"],
        "span_count": 1,
        "error_count": error_count,
    }
    if root is None:
        return trace, None
    is_chat = operation == "chat"
    profile = span["agent_name"] if operation == "invoke_agent" else None
    model = None
    if is_chat:
        model = span["response_model"] if span["response_model"] is not None else span["request_model"]
    session = {
        "root_session_id": root,
        "first_ns": span["start_ns"],
        "last_ns": span["end_ns"],
        "last_trace_id": span["trace_id"],
        "last_span_id": span["span_id"],
        "span_count": 1,
        "turn_count": int(operation == "turn"),
        "model_count": int(is_chat),
        "tool_count": int(operation == "execute_tool"),
        "error_count": error_count,
        "input_tokens": (span["input_tokens"] or 0) if is_chat else 0,
        "output_tokens": (span["output_tokens"] or 0) if is_chat else 0,
        "total_tokens": (span["total_tokens"] or 0) if is_chat else 0,
        "profile": profile,
        "model": model,
    }
    for name, value in (("profile", profile), ("model", model)):
        latest = value is not None
        session[f"{name}_last_ns"] = span["end_ns"] if latest else None
        session[f"{name}_trace_id"] = span["trace_id"] if latest else None
        session[f"{name}_span_id"] = span["span_id"] if latest else None
    return trace, session


def _connect(path: Path, **options: Any) -> sqlite3.Connection:
    connection = sqlite3.connect(path, check_same_thread=False, **options)
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def open_database(path: Path) -> None:
    """Create or migrate the database; the caller holds the viewer lock. Fails closed."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    os.close(descriptor)
    os.chmod(path, 0o600)
    connection = _connect(path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if version == SCHEMA_VERSION:
            return
        if version != 0:
            raise RuntimeError(f"telemetry database schema {version} is not supported (expected {SCHEMA_VERSION})")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        try:
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    connection.execute(statement)
            connection.execute("INSERT INTO viewer_meta (singleton, schema_version) VALUES (1, ?)", (SCHEMA_VERSION,))
            connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
    finally:
        connection.close()


class Writer:
    """One thread and one connection; a bounded queue of batches feeds it."""

    def __init__(self, path: Path) -> None:
        self._connection = _connect(path)
        self._queue: queue.Queue[tuple[list[dict[str, Any]], Callable[[object], None]] | None] = queue.Queue(
            INGEST_QUEUE_CAPACITY
        )
        self._thread = threading.Thread(target=self._run, name="zeta-otel-viewer-writer", daemon=True)
        # the last committed sequence; set only after its transaction commits
        self.committed_sequence = self._connection.execute("SELECT committed_sequence FROM viewer_meta").fetchone()[0]

    def start(self) -> None:
        self._thread.start()

    def submit(self, rows: list[dict[str, Any]]) -> asyncio.Future[int]:
        """Queue one batch; the future resolves after commit. Raises ``queue.Full``."""
        loop = asyncio.get_running_loop()
        future: asyncio.Future[int] = loop.create_future()

        def settle(outcome: object) -> None:
            if future.done():
                return
            if isinstance(outcome, BaseException):
                future.set_exception(outcome)
            else:
                future.set_result(outcome)

        self._queue.put_nowait((rows, lambda outcome: loop.call_soon_threadsafe(settle, outcome)))
        return future

    def close(self) -> None:
        """Let queued batches commit for up to the graceful timeout, then stop."""
        try:
            self._queue.put(None, timeout=GRACEFUL_COMMIT_TIMEOUT_SECONDS)
        except queue.Full:
            return
        self._thread.join(GRACEFUL_COMMIT_TIMEOUT_SECONDS)

    def _run(self) -> None:
        try:
            while (item := self._queue.get()) is not None:
                rows, reply = item
                try:
                    outcome: object = self._commit(rows)
                except Exception as exc:  # noqa: BLE001 - a failed batch fails its request; the writer keeps running
                    outcome = exc
                reply(outcome)
        finally:
            self._connection.close()

    def _commit(self, rows: list[dict[str, Any]]) -> int:
        connection = self._connection
        connection.execute("BEGIN IMMEDIATE")
        try:
            inserted = 0
            for row in rows:
                connection.execute(_SPAN_INSERT, row)
                if connection.execute("SELECT changes()").fetchone()[0] != 1:
                    continue  # a duplicate span changes nothing
                inserted += 1
                connection.executemany(_EVENT_INSERT, row["events"])
                trace, session = project_span(row)
                connection.execute(_TRACE_UPSERT, trace)
                if session is not None:
                    connection.execute(_SESSION_UPSERT, session)
            if inserted:
                connection.execute("UPDATE viewer_meta SET committed_sequence = committed_sequence + 1 WHERE singleton = 1")
            sequence = connection.execute("SELECT committed_sequence FROM viewer_meta").fetchone()[0]
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        self.committed_sequence = sequence
        return sequence


def parse_page(limit: str | None, cursor: str | None) -> tuple[int, tuple[int, str] | None]:
    """Validate ``/api/sessions`` paging; raises ``ValueError`` for anything malformed."""
    size = DEFAULT_PAGE_SIZE
    if limit is not None:
        if not limit.isascii() or not limit.isdigit() or not 1 <= int(limit) <= MAX_PAGE_SIZE:
            raise ValueError("invalid limit")
        size = int(limit)
    if cursor is None:
        return size, None
    if not cursor or "=" in cursor:
        raise ValueError("invalid cursor")
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise ValueError("invalid cursor") from None
    if (
        not isinstance(value, list)
        or len(value) != 2
        or type(value[0]) is not int
        or value[0] < 0
        or not isinstance(value[1], str)
    ):
        raise ValueError("invalid cursor")
    return size, (value[0], value[1])


def _encode_cursor(last_ns: int, root: str) -> str:
    text = json.dumps([last_ns, root], separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def _json_row(row: sqlite3.Row, keys: tuple[str, ...]) -> dict[str, Any]:
    return {key: row[key].hex() if key in _HEX_KEYS and row[key] is not None else row[key] for key in keys}


class Reader:
    """A read-only connection; callers run it off the event loop."""

    def __init__(self, path: Path) -> None:
        self._connection = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, check_same_thread=False)
        self._connection.row_factory = sqlite3.Row
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def sessions(self, limit: int, after: tuple[int, str] | None) -> dict[str, Any]:
        query = f"SELECT {', '.join(_SESSION_KEYS)} FROM sessions"
        parameters: tuple[object, ...] = ()
        if after is not None:
            query += " WHERE last_ns < ? OR (last_ns = ? AND root_session_id > ?)"
            parameters = (after[0], after[0], after[1])
        query += " ORDER BY last_ns DESC, root_session_id ASC LIMIT ?"
        with self._lock:
            rows = self._connection.execute(query, (*parameters, limit + 1)).fetchall()
        page = [_json_row(row, _SESSION_KEYS) for row in rows[:limit]]
        next_cursor = _encode_cursor(page[-1]["last_ns"], page[-1]["root_session_id"]) if len(rows) > limit else None
        return {"sessions": page, "next_cursor": next_cursor}

    def trace(self, trace_id: str) -> dict[str, Any] | None:
        key = bytes.fromhex(trace_id)
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN")
            try:
                trace = connection.execute(f"SELECT {', '.join(_TRACE_KEYS)} FROM traces WHERE trace_id = ?", (key,)).fetchone()
                spans = connection.execute(
                    f"SELECT {', '.join(_TRACE_SPAN_KEYS)} FROM spans WHERE trace_id = ? ORDER BY start_ns, span_id", (key,)
                ).fetchall()
                events = connection.execute(
                    f"SELECT span_id, {', '.join(_EVENT_KEYS)} FROM span_events WHERE trace_id = ? "
                    "ORDER BY time_ns, event_index",
                    (key,),
                ).fetchall()
            finally:
                connection.rollback()
        if trace is None:
            return None
        by_span: dict[bytes, list[dict[str, Any]]] = {}
        for event in events:
            by_span.setdefault(event["span_id"], []).append(_json_row(event, _EVENT_KEYS))
        return {
            "trace": _json_row(trace, _TRACE_KEYS),
            "spans": [{**_json_row(span, _TRACE_SPAN_KEYS), "events": by_span.get(span["span_id"], [])} for span in spans],
        }
