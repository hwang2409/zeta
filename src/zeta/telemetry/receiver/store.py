"""The telemetry database: its schema, forward migrations, and its one writer thread.

Only the receiver's writer thread writes. ``zeta serve`` reads through ``reader.py`` with a
read-only connection. An OTLP batch is acknowledged only after its transaction commits.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import sqlite3
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .projector import EVENT_COLUMNS, SPAN_COLUMNS, root_session_id

SCHEMA_VERSION = 3
RAW_DAYS = 14
ROLLUP_DAYS = 90
MAX_BYTES = 1 << 30
PRUNE_CHUNK = 1_000
INGEST_QUEUE_CAPACITY = 64
GRACEFUL_COMMIT_TIMEOUT_SECONDS = 5.0
BUSY_TIMEOUT_MS = 5_000

# the schema-1 tables; a new database also runs every entry of MIGRATIONS, so both paths reach one shape
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

CREATE INDEX spans_trace_start ON spans(trace_id, start_ns, span_id);
CREATE INDEX spans_parent ON spans(trace_id, parent_span_id);
CREATE INDEX spans_session_start ON spans(root_session_id, start_ns);
CREATE INDEX spans_provider_model_start ON spans(provider, request_model, start_ns);
CREATE INDEX spans_tool_start ON spans(tool_name, start_ns);
CREATE INDEX spans_error_start ON spans(start_ns) WHERE status = 'error';
CREATE INDEX span_events_span_time ON span_events(trace_id, span_id, time_ns);
CREATE INDEX span_events_name_time ON span_events(name, time_ns);
CREATE INDEX traces_first ON traces(first_ns);
CREATE INDEX traces_last ON traces(last_ns DESC);
CREATE INDEX traces_session_last ON traces(root_session_id, last_ns DESC);
CREATE INDEX traces_status_last ON traces(status, last_ns DESC);
CREATE INDEX sessions_last ON sessions(last_ns DESC, root_session_id DESC);
CREATE INDEX sessions_profile_last ON sessions(profile, last_ns DESC, root_session_id DESC);
CREATE INDEX sessions_model_last ON sessions(model, last_ns DESC, root_session_id DESC);
"""

# v3 replaces the unreleased rollup table with family-specific projections.
V3_SCHEMA = (  # noqa: SIM905
"""CREATE TABLE span_fingerprints(trace_id BLOB NOT NULL, span_id BLOB NOT NULL, seen_ns INTEGER NOT NULL CHECK(seen_ns>=0), PRIMARY KEY(trace_id,span_id)) STRICT;
CREATE INDEX span_fingerprints_seen ON span_fingerprints(seen_ns,trace_id,span_id);
CREATE TABLE hourly_series(hour_ns INTEGER NOT NULL, group_by TEXT NOT NULL CHECK(group_by IN('none','model','tool','operation')), group_key TEXT NOT NULL, spans INTEGER NOT NULL CHECK(spans>=0), errors INTEGER NOT NULL CHECK(errors>=0), model_calls INTEGER NOT NULL CHECK(model_calls>=0), tool_calls INTEGER NOT NULL CHECK(tool_calls>=0), input_tokens INTEGER NOT NULL CHECK(input_tokens>=0), output_tokens INTEGER NOT NULL CHECK(output_tokens>=0), total_tokens INTEGER NOT NULL CHECK(total_tokens>=0), duration_ns INTEGER NOT NULL CHECK(duration_ns>=0), PRIMARY KEY(hour_ns,group_by,group_key)) STRICT;
CREATE TABLE hourly_distinct(hour_ns INTEGER NOT NULL, group_by TEXT NOT NULL CHECK(group_by IN('none','model','tool','operation')), group_key TEXT NOT NULL, identity_kind TEXT NOT NULL CHECK(identity_kind IN('session','trace')), identity_key BLOB NOT NULL, PRIMARY KEY(hour_ns,group_by,group_key,identity_kind,identity_key)) STRICT;
CREATE TABLE hourly_models(hour_ns INTEGER NOT NULL, model_key TEXT NOT NULL, provider TEXT, model TEXT NOT NULL, calls INTEGER NOT NULL CHECK(calls>=0), errors INTEGER NOT NULL CHECK(errors>=0), duration_ns INTEGER NOT NULL CHECK(duration_ns>=0), input_tokens INTEGER NOT NULL CHECK(input_tokens>=0), output_tokens INTEGER NOT NULL CHECK(output_tokens>=0), total_tokens INTEGER NOT NULL CHECK(total_tokens>=0), last_ns INTEGER NOT NULL CHECK(last_ns>=0), PRIMARY KEY(hour_ns,model_key)) STRICT;
CREATE TABLE hourly_tools(hour_ns INTEGER NOT NULL, tool_key TEXT NOT NULL, tool_name TEXT NOT NULL, tool_type TEXT, calls INTEGER NOT NULL CHECK(calls>=0), errors INTEGER NOT NULL CHECK(errors>=0), duration_ns INTEGER NOT NULL CHECK(duration_ns>=0), argument_bytes INTEGER NOT NULL CHECK(argument_bytes>=0), result_bytes INTEGER NOT NULL CHECK(result_bytes>=0), last_ns INTEGER NOT NULL CHECK(last_ns>=0), PRIMARY KEY(hour_ns,tool_key)) STRICT;
CREATE TABLE hourly_latency(hour_ns INTEGER NOT NULL, family TEXT NOT NULL CHECK(family IN('model','tool')), family_key TEXT NOT NULL, bucket INTEGER NOT NULL CHECK(bucket BETWEEN 0 AND 63), calls INTEGER NOT NULL CHECK(calls>=0), PRIMARY KEY(hour_ns,family,family_key,bucket)) STRICT;
CREATE INDEX hourly_distinct_range ON hourly_distinct(group_by,group_key,hour_ns,identity_kind);
CREATE INDEX hourly_models_range ON hourly_models(hour_ns,provider,model);
CREATE INDEX hourly_tools_range ON hourly_tools(hour_ns,tool_name,tool_type);
CREATE INDEX hourly_latency_range ON hourly_latency(family,family_key,hour_ns,bucket);""").split(';')

MIGRATIONS_2 = (

        "ALTER TABLE spans ADD COLUMN error_code TEXT",
        "ALTER TABLE spans ADD COLUMN run_id TEXT",
        "ALTER TABLE spans ADD COLUMN parent_run_id TEXT",
        "ALTER TABLE spans ADD COLUMN outcome TEXT",
        "ALTER TABLE spans ADD COLUMN cache_read_tokens INTEGER CHECK (cache_read_tokens IS NULL OR cache_read_tokens >= 0)",
        "ALTER TABLE spans ADD COLUMN cache_write_tokens INTEGER CHECK (cache_write_tokens IS NULL OR cache_write_tokens >= 0)",
        "ALTER TABLE spans ADD COLUMN ttft_ns INTEGER CHECK (ttft_ns IS NULL OR ttft_ns >= 0)",
        "ALTER TABLE spans ADD COLUMN retry_count INTEGER CHECK (retry_count IS NULL OR retry_count >= 0)",

)
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

_FINGERPRINT_INSERT = "INSERT INTO span_fingerprints (trace_id, span_id, seen_ns) VALUES (?, ?, ?) ON CONFLICT(trace_id, span_id) DO NOTHING"

_SPAN_INSERT = (
    f"INSERT INTO spans ({', '.join(SPAN_COLUMNS)}) VALUES ({', '.join(':' + name for name in SPAN_COLUMNS)}) "
    "ON CONFLICT(trace_id, span_id) DO NOTHING"
)
_EVENT_INSERT = f"INSERT INTO span_events ({', '.join(EVENT_COLUMNS)}) VALUES ({', '.join(':' + name for name in EVENT_COLUMNS)})"
def project_span(span: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """The trace candidate and optional session candidate of one newly inserted span."""
    root = root_session_id(span["session_id"])
    error_count = 1 if span["status"] == "error" else 0
    trace = _trace_candidate(span, root, error_count)
    return trace, None if root is None else _session_candidate(span, root, error_count)


def _trace_candidate(span: dict[str, Any], root: str | None, error_count: int) -> dict[str, Any]:
    return {
        "trace_id": span["trace_id"],
        "first_ns": span["start_ns"],
        "last_ns": span["end_ns"],
        "root_span_id": span["span_id"] if span["parent_span_id"] is None else None,
        "root_session_id": root,
        "status": span["status"],
        "span_count": 1,
        "error_count": error_count,
    }


def _session_candidate(span: dict[str, Any], root: str, error_count: int) -> dict[str, Any]:
    operation = span["operation"]
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
    return session



def _database_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in (path, Path(f"{path}-wal"), Path(f"{path}-shm")) if p.exists())


def _connect(path: Path, **options: Any) -> sqlite3.Connection:
    connection = sqlite3.connect(path, check_same_thread=False, **options)
    connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA synchronous = NORMAL")
    return connection


def _set_v3_tables(connection: sqlite3.Connection) -> None:
    for statement in V3_SCHEMA:
        if statement.strip():
            connection.execute(statement)


def _row_dict(row: sqlite3.Row) -> dict[str, Any]:
    result = dict(row)
    result.setdefault("events", [])
    return result


def _model_key(row: dict[str, Any]) -> str:
    return json.dumps([row.get("provider"), row.get("response_model") or row.get("request_model")], separators=(",", ":"))


def _tool_key(row: dict[str, Any]) -> str:
    return json.dumps([row.get("tool_name"), row.get("tool_type")], separators=(",", ":"))


def _upsert_series(connection: sqlite3.Connection, row: dict[str, Any], hour: int, group: str, key: str) -> None:
    model = row["operation"] == "chat" and (row.get("response_model") or row.get("request_model")) is not None
    tool = row["operation"] == "execute_tool" and row.get("tool_name") is not None
    connection.execute("""INSERT INTO hourly_series VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(hour_ns,group_by,group_key) DO UPDATE SET spans=spans+1,errors=errors+excluded.errors,model_calls=model_calls+excluded.model_calls,tool_calls=tool_calls+excluded.tool_calls,input_tokens=input_tokens+excluded.input_tokens,output_tokens=output_tokens+excluded.output_tokens,total_tokens=total_tokens+excluded.total_tokens,duration_ns=duration_ns+excluded.duration_ns""", (hour, group, key, 1, int(row["status"] == "error"), int(model), int(tool), row.get("input_tokens") or 0 if model else 0, row.get("output_tokens") or 0 if model else 0, row.get("total_tokens") or 0 if model else 0, row["duration_ns"]))
    for kind, identity in (("trace", row["trace_id"]), ("session", row.get("root_session_id") or row.get("session_id"))):
        if identity is not None:
            connection.execute("INSERT OR IGNORE INTO hourly_distinct VALUES(?,?,?,?,?)", (hour, group, key, kind, identity.encode() if isinstance(identity, str) else identity))


def _project_rollup(connection: sqlite3.Connection, row: dict[str, Any]) -> None:
    hour = row["start_ns"] // 3_600_000_000_000 * 3_600_000_000_000
    _upsert_series(connection, row, hour, "none", "")
    _upsert_series(connection, row, hour, "operation", row["operation"])
    model = row["operation"] == "chat" and (row.get("response_model") or row.get("request_model")) is not None
    tool = row["operation"] == "execute_tool" and row.get("tool_name") is not None
    if model:
        key = _model_key(row)
        connection.execute("""INSERT INTO hourly_models VALUES(?,?,?,?,?,?,?,?,?,?,?) ON CONFLICT(hour_ns,model_key) DO UPDATE SET calls=calls+1,errors=errors+excluded.errors,duration_ns=duration_ns+excluded.duration_ns,input_tokens=input_tokens+excluded.input_tokens,output_tokens=output_tokens+excluded.output_tokens,total_tokens=total_tokens+excluded.total_tokens,last_ns=MAX(last_ns,excluded.last_ns)""", (hour,key,row.get("provider"),row.get("response_model") or row.get("request_model"),1,int(row["status"]=="error"),row["duration_ns"],row.get("input_tokens") or 0,row.get("output_tokens") or 0,row.get("total_tokens") or 0,row["end_ns"]))
        bucket = _histogram_bucket(row["duration_ns"])
        connection.execute("INSERT INTO hourly_latency VALUES(?,?,?,?,1) ON CONFLICT(hour_ns,family,family_key,bucket) DO UPDATE SET calls=calls+1", (hour,"model",key,bucket))
        _upsert_series(connection, row, hour, "model", key)
    if tool:
        key = _tool_key(row)
        connection.execute("""INSERT INTO hourly_tools VALUES(?,?,?,?,?,?,?,?,?,?) ON CONFLICT(hour_ns,tool_key) DO UPDATE SET calls=calls+1,errors=errors+excluded.errors,duration_ns=duration_ns+excluded.duration_ns,argument_bytes=argument_bytes+excluded.argument_bytes,result_bytes=result_bytes+excluded.result_bytes,last_ns=MAX(last_ns,excluded.last_ns)""", (hour,key,row["tool_name"],row.get("tool_type"),1,int(row["status"]=="error"),row["duration_ns"],row.get("tool_argument_bytes") or 0,row.get("tool_result_bytes") or 0,row["end_ns"]))
        bucket = _histogram_bucket(row["duration_ns"])
        connection.execute("INSERT INTO hourly_latency VALUES(?,?,?,?,1) ON CONFLICT(hour_ns,family,family_key,bucket) DO UPDATE SET calls=calls+1", (hour,"tool",key,bucket))
        _upsert_series(connection, row, hour, "tool", key)


def open_database(path: Path) -> None:
    """Create or migrate the database to ``SCHEMA_VERSION``; the receiver's writer thread calls it before serving."""
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    os.close(descriptor); os.chmod(path, 0o600)
    connection = _connect(path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if not 0 <= version <= SCHEMA_VERSION: raise RuntimeError(f"telemetry database schema {version} is not supported (expected {SCHEMA_VERSION})")
        _enable_incremental_vacuum(connection)
        if version == SCHEMA_VERSION: return
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("BEGIN IMMEDIATE")
        try:
            if version == 0:
                for statement in SCHEMA.split(";"):
                    if statement.strip(): connection.execute(statement)
                connection.execute("INSERT INTO viewer_meta (singleton,schema_version) VALUES (1,1)"); version = 1
            if version == 1:
                for statement in MIGRATIONS_2:
                    connection.execute(statement)
            _migrate_v3(connection)
            connection.commit()
        except BaseException:
            connection.rollback(); raise
    finally: connection.close()


def _enable_incremental_vacuum(connection: sqlite3.Connection) -> None:
    """Size reclaim needs ``auto_vacuum=INCREMENTAL``; an existing file only changes mode through ``VACUUM``."""
    if connection.execute("PRAGMA auto_vacuum").fetchone()[0] == 2: return
    connection.execute("PRAGMA auto_vacuum = INCREMENTAL")
    try: connection.execute("VACUUM")
    except sqlite3.OperationalError:  # disk full: keep auto_vacuum=0; the size stall rule still keeps the rows
        print("zeta otel receiver: could not enable incremental vacuum; size reclaim is limited", file=sys.stderr)


def _migrate_v3(connection: sqlite3.Connection) -> None:
    """Create the v3 tables, backfill fingerprints and rollups from every raw span, and publish v3."""
    _set_v3_tables(connection)
    connection.row_factory = sqlite3.Row
    for raw in connection.execute("SELECT * FROM spans"):
        row = _row_dict(raw)
        connection.execute("INSERT OR IGNORE INTO span_fingerprints VALUES(?,?,?)", (row["trace_id"],row["span_id"],row["start_ns"]))
        _project_rollup(connection, row)
    connection.execute("UPDATE viewer_meta SET schema_version=3 WHERE singleton=1")
    connection.execute("PRAGMA user_version=3")


def _histogram_bucket(duration_ns: int) -> int:
    return 0 if duration_ns == 0 else min(63, duration_ns.bit_length())


DAY_NS = 86_400 * 10**9
HOUR_NS = 3_600 * 10**9
STARTUP_SECONDS = 2.0
VACUUM_PAGES = 1_000
_ROLLUP_TABLES = (
    ("hourly_series", "hour_ns,group_by,group_key"), ("hourly_distinct", "hour_ns,group_by,group_key,identity_kind,identity_key"),
    ("hourly_models", "hour_ns,model_key"), ("hourly_tools", "hour_ns,tool_key"), ("hourly_latency", "hour_ns,family,family_key,bucket"),
)
# (family, table, age column, oldest-first order); the same order picks age and size victims
_RETENTION_ORDER = (
    ("raw", "spans", "start_ns", "start_ns,trace_id,span_id"),
    ("fingerprints", "span_fingerprints", "seen_ns", "seen_ns,trace_id,span_id"),
    *(("rollups", table, "hour_ns", order) for table, order in _ROLLUP_TABLES),
)
# an orphan trace has no spans left; an orphan session has no traces left
_ORPHAN_DELETE = {
    "traces": "DELETE FROM traces WHERE trace_id=?1 AND NOT EXISTS(SELECT 1 FROM spans WHERE trace_id=?1)",
    "sessions": "DELETE FROM sessions WHERE root_session_id=?1 AND NOT EXISTS(SELECT 1 FROM traces WHERE root_session_id=?1)",
}
_SWEEP_START: dict[str, bytes | str] = {"traces": b"", "sessions": ""}
_SWEEP_KEY = {"traces": "trace_id", "sessions": "root_session_id"}


class Writer:
    """The receiver's one writer thread: it migrates, catches up retention, then commits ingest batches."""

    def __init__(self, path: Path, *, clock_ns: Callable[[], int] = time.time_ns) -> None:
        self._path = path; self._clock_ns = clock_ns
        self._connection: sqlite3.Connection
        self._queue: queue.Queue[tuple[list[dict[str, Any]], Callable[[object], None]] | None] = queue.Queue(INGEST_QUEUE_CAPACITY)
        self._thread = threading.Thread(target=self._run, name="zeta-otel-receiver-writer", daemon=True)
        self._ready = threading.Event(); self._startup_error: BaseException | None = None
        self._stalled_bytes: int | None = None  # in memory only: a restart may try one size victim again
        self._sweep = dict(_SWEEP_START)
        self.committed_sequence = 0

    def start(self) -> None:
        """Start the writer thread and block until it has migrated and caught up; call it off the event loop."""
        self._thread.start(); self._ready.wait()
        if self._startup_error is not None: raise self._startup_error

    def submit(self, rows: list[dict[str, Any]]) -> asyncio.Future[int]:
        loop = asyncio.get_running_loop(); future: asyncio.Future[int] = loop.create_future()
        def settle(outcome: object) -> None:
            if future.done(): return
            future.set_exception(outcome) if isinstance(outcome, BaseException) else future.set_result(outcome)
        self._queue.put_nowait((rows, lambda outcome: loop.call_soon_threadsafe(settle, outcome))); return future
    def close(self) -> None:
        try: self._queue.put(None, timeout=GRACEFUL_COMMIT_TIMEOUT_SECONDS)
        except queue.Full: return
        self._thread.join(GRACEFUL_COMMIT_TIMEOUT_SECONDS)

    def _catch_up(self) -> None:
        self.committed_sequence = self._connection.execute("SELECT committed_sequence FROM viewer_meta").fetchone()[0]
        deadline = time.monotonic() + STARTUP_SECONDS
        while time.monotonic() < deadline:  # leftover catch-up continues with each ingest batch
            result = self.maintenance_step(self._clock_ns())
            if result["caught_up"] or result["stalled"]: break

    def _run(self) -> None:
        try:
            open_database(self._path)
            self._connection = _connect(self._path)
            try: self._catch_up()
            except BaseException: self._connection.close(); raise
        except BaseException as exc:  # noqa: BLE001 - start() re-raises it to the lifespan
            self._startup_error = exc; return
        finally: self._ready.set()
        try:
            while (item := self._queue.get()) is not None:
                rows, reply = item
                try: outcome = self._commit(rows)
                except Exception as exc:  # noqa: BLE001
                    outcome = exc
                reply(outcome)
        finally: self._connection.close()

    def _commit(self, rows: list[dict[str, Any]]) -> int:
        c = self._connection; c.execute("BEGIN IMMEDIATE")
        try:
            inserted = 0
            for row in rows:
                c.execute(_FINGERPRINT_INSERT, (row["trace_id"],row["span_id"],row["start_ns"]))
                if c.execute("SELECT changes()").fetchone()[0] != 1: continue
                c.execute(_SPAN_INSERT, row)
                if c.execute("SELECT changes()").fetchone()[0] != 1: continue
                inserted += 1; c.executemany(_EVENT_INSERT, row["events"])
                trace, session = project_span(row); c.execute(_TRACE_UPSERT, trace)
                if session is not None: c.execute(_SESSION_UPSERT, session)
                _project_rollup(c, row)
            if inserted: c.execute("UPDATE viewer_meta SET committed_sequence=committed_sequence+1 WHERE singleton=1")
            sequence = c.execute("SELECT committed_sequence FROM viewer_meta").fetchone()[0]; c.commit()
        except BaseException: c.rollback(); raise
        self.committed_sequence = sequence; self.maintenance_step(self._clock_ns()); return sequence

    def _reclaim(self) -> int:
        """Fixed checkpoint and vacuum work; returns db+WAL+SHM bytes."""
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        self._connection.execute(f"PRAGMA incremental_vacuum({VACUUM_PAGES})")
        self._connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        return _database_bytes(self._path)

    def _delete_chunk(self, table: str, where: str, args: tuple[Any, ...], order: str) -> int:
        """Delete up to ``PRUNE_CHUNK`` oldest rows in one transaction, with the traces and sessions they orphan."""
        c = self._connection; c.execute("BEGIN IMMEDIATE")
        try:
            columns = "rowid,trace_id,session_id" if table == "spans" else "rowid"
            victims = c.execute(f"SELECT {columns} FROM {table} WHERE {where} ORDER BY {order} LIMIT ?", (*args, PRUNE_CHUNK)).fetchall()
            c.executemany(f"DELETE FROM {table} WHERE rowid=?", [victim[:1] for victim in victims])
            if table == "spans":
                c.executemany(_ORPHAN_DELETE["traces"], {(victim[1],) for victim in victims})
                c.executemany(_ORPHAN_DELETE["sessions"], {(root,) for victim in victims if (root := root_session_id(victim[2]))})
            c.commit()
        except BaseException: c.rollback(); raise
        return len(victims)

    def _sweep_orphans(self) -> int:
        """Delete leftover orphans, examining at most ``PRUNE_CHUNK`` keys of each projection from a cursor."""
        c = self._connection; changes = c.total_changes; c.execute("BEGIN IMMEDIATE")
        try:
            for table, key in _SWEEP_KEY.items():
                keys = c.execute(f"SELECT {key} FROM {table} WHERE {key} > ? ORDER BY {key} LIMIT ?", (self._sweep[table], PRUNE_CHUNK)).fetchall()
                c.executemany(_ORPHAN_DELETE[table], keys)
                self._sweep[table] = keys[-1][0] if len(keys) == PRUNE_CHUNK else _SWEEP_START[table]
            c.commit()
        except BaseException: c.rollback(); raise
        return c.total_changes - changes

    def maintenance_step(self, now_ns: int) -> dict[str, Any]:
        """One bounded retention step; the only code that deletes for age or size. It never loops.

        It deletes one age chunk from the oldest expired family, sweeps one chunk of orphans, then, over the cap,
        deletes one size chunk and stalls if that did not reduce db+WAL+SHM bytes. A stalled writer deletes
        no further size victim until the measured size grows past the stalled size.
        """
        before = _database_bytes(self._path)
        raw_cutoff = now_ns - RAW_DAYS * DAY_NS; fingerprint_cutoff = now_ns - ROLLUP_DAYS * DAY_NS
        cutoffs = {"raw": raw_cutoff, "fingerprints": fingerprint_cutoff, "rollups": fingerprint_cutoff // HOUR_NS * HOUR_NS}
        age_family, age_rows = None, 0
        for family, table, column, order in _RETENTION_ORDER:
            if age_rows := self._delete_chunk(table, f"{column} < ?", (cutoffs[family],), order):
                age_family = family; break
        orphans = self._sweep_orphans()
        measured = after = self._reclaim()
        size_family, size_rows, stalled = None, 0, False
        if measured <= MAX_BYTES:
            self._stalled_bytes = None
        elif self._stalled_bytes is not None and measured <= self._stalled_bytes:
            stalled = True
        else:
            for family, table, _column, order in _RETENTION_ORDER:
                if size_rows := self._delete_chunk(table, "1", (), order):
                    size_family = family; break
            after = self._reclaim()
            stalled = not size_rows or after >= measured
            self._stalled_bytes = after if stalled else None
        over = after > MAX_BYTES
        return {"age_family": age_family, "age_rows": age_rows, "size_family": size_family, "size_rows": size_rows,
                "bytes_before": before, "bytes_after": after, "caught_up": age_family is None and not orphans and not over,
                "over_cap": over, "stalled": stalled}
