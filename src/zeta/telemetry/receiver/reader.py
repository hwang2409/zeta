"""Read-only projections of the telemetry database for ``zeta serve`` (``telemetry_read_v1``).

Every function here blocks; serve calls them in its bounded read executor. The database is
opened with ``mode=ro`` and is never created, migrated, or locked for writing: the receiver
is its only writer. Each read is one SQLite snapshot. Lists page through ``page.keyset_page``.
Aggregates read hourly rollups and raw spans split by ``_split``. Results use the exact v1 keys in
``docs/serve-protocol.md``.
"""

from __future__ import annotations

import re
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...read_page import InvalidParams, Key, resume
from ..runtime import _SESSION_ID
from .page import keyset_page
from .paths import MAX_BYTES, RAW_DAYS, ROLLUP_DAYS, database_path, used_bytes
from .process import OTLP_URL, receiver_ready
from .store import (
    BUSY_TIMEOUT_MS,
    DAY_NS,
    HOUR_NS,
    SCHEMA_VERSION,
    _histogram_bucket,
)

DEFAULT_RANGE_NS = 24 * 3600 * 10**9
MAX_FILTER_BYTES = 256
MAX_POINTS = 2_208  # overview and series points over all pages: 92 days of hours
MAX_SERIES_KEYS = 20
MAX_AGENTS = 1_000
_TRACE_ID = re.compile(r"[0-9a-f]{32}\Z")
_ROOT_SESSION_ID = re.compile(r"[0-9a-f]{32}\Z")
_OPERATIONS = ("invoke_agent", "turn", "chat", "execute_tool", "context_compact", "serve", "mcp")
_METRICS = (
    "sessions", "traces", "spans", "errors", "model_calls", "tool_calls", "input_tokens", "output_tokens",
    "total_tokens", "duration_ns",
)
_BUCKETS = {"hour": HOUR_NS, "day": DAY_NS}

_SESSION_COLUMNS = (
    "root_session_id", "first_ns", "last_ns", "last_trace_id", "profile", "model", "span_count", "turn_count",
    "model_count", "tool_count", "error_count", "input_tokens", "output_tokens", "total_tokens",
)
_TRACE_COLUMNS = ("trace_id", "first_ns", "last_ns", "root_span_id", "root_session_id", "status", "span_count", "error_count")
_SPAN_COLUMNS = (
    "span_id", "parent_span_id", "operation", "start_ns", "end_ns", "duration_ns", "status", "error_type",
    "error_code", "session_id", "agent_name", "agent_depth", "background", "run_id", "parent_run_id",
    "turn_number", "notification", "outcome", "provider", "request_model", "response_model", "stream",
    "input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens", "finish_reason",
    "ttft_ns", "retry_count", "tool_name", "tool_type", "tool_call_id", "tool_argument_bytes", "tool_result_bytes",
    "tool_count", "emergency", "tokens_before", "tokens_after", "rpc_method", "mcp_method", "mcp_server", "mcp_tool",
    "mcp_prompt", "mcp_protocol", "request_id",
)
_EVENT_COLUMNS = (
    "name", "time_ns", "retry_attempt", "retry_reason", "retry_delay", "retry_after", "decision", "scope", "duration_ns",
)
_ERROR_COLUMNS = (
    "trace_id", "span_id", "session_id", "time_ns", "operation", "error_type", "duration_ns", "model", "tool_name",
    "agent_name", "mcp_server", "mcp_method", "mcp_tool",
)
_AGENT_COLUMNS = ("session_id", "parent_session_id", "agent_name", "depth", "background", "first_ns", "last_ns", "status")
_HEX = frozenset({"trace_id", "span_id", "parent_span_id", "root_span_id", "last_trace_id"})
_FLAGS = frozenset({"background", "notification", "stream", "emergency", "retry_after"})

# raw-span projections; they match the rollup projector in store.py. canonical keys are compact JSON
# arrays: the projector admits only ASCII identifiers, so SQLite's json_array equals store.py's json.dumps.
_MODEL = "COALESCE(response_model, request_model)"
_IS_MODEL = f"(operation = 'chat' AND {_MODEL} IS NOT NULL)"
_IS_TOOL = "(operation = 'execute_tool' AND tool_name IS NOT NULL)"
# spans.session_id is a valid conversation id or "_invalid"; spans.root_session_id is never written
_ROOT = "CASE WHEN length(session_id) >= 32 THEN substr(session_id, 1, 32) END"
_MODEL_KEY = f"json_array(provider, {_MODEL})"
_TOOL_KEY = "json_array(tool_name, tool_type)"
_GROUPS = {"none": ("''", "1"), "operation": ("operation", "1"), "model": (_MODEL_KEY, _IS_MODEL), "tool": (_TOOL_KEY, _IS_TOOL)}


class Unavailable(Exception):
    """No database file: telemetry is off or the receiver has not created it."""


class TraceNotFound(LookupError):
    def __init__(self, trace_id: str) -> None:
        super().__init__(trace_id)
        self.trace_id = trace_id


class SchemaMismatch(Exception):
    """The database's schema version is not the one this zeta reads."""


Fits = Callable[[dict[str, Any]], bool]


def status(home: Path, endpoint: str | None, live: dict[str, Any]) -> dict[str, Any]:
    """``TelemetryStatus``; ``endpoint`` is the configured ``[otel].endpoint``, ``live`` the serve process's
    receiver and export snapshot. Without a supervised receiver, this checks the receiver's health itself.

    Never raises for a missing database: enabled means configured and a database exists.
    """
    try:
        oldest = _snapshot(home, _oldest)
    except Unavailable:
        oldest = None
    return _status(home, endpoint, live, oldest)


def sessions(home: Path, params: dict[str, Any], fits: Fits) -> dict[str, Any]:
    """``Page<Session>`` ordered by ``last_ns desc, session_id desc``."""
    _only(params, {"limit", "cursor", "start_ns", "end_ns", "status", "profile", "model"})
    limit = _limit(params.get("limit"), 50, 200)
    status_filter = params.get("status", "all")
    if status_filter not in ("all", "error"):
        raise InvalidParams("status must be 'all' or 'error'")
    profile, model = _filter(params.get("profile"), "profile"), _filter(params.get("model"), "model")
    explicit = [_ns(params.get("start_ns"), "start_ns"), _ns(params.get("end_ns"), "end_ns")]
    order = (Key("last_ns", int, descending=True), Key("root_session_id", str, descending=True))
    start = resume(params.get("cursor"), ["telemetry_sessions", status_filter, profile, model, *explicit], _range(*explicit), order)
    query = f"SELECT {', '.join(_SESSION_COLUMNS)} FROM sessions WHERE last_ns >= ? AND first_ns < ?"
    values: list[object] = list(start.pinned)
    if status_filter == "error":
        query += " AND error_count > 0"
    for column, value in (("profile", profile), ("model", model)):
        if value is not None:
            query += f" AND {column} = ?"
            values.append(value)
    as_of = time.time_ns()
    return _snapshot(home, lambda connection: keyset_page(
        connection, query, values, order, start, limit=limit, fits=fits,
        shape=lambda items, cursor: {"items": items, "next_cursor": cursor, "as_of_ns": as_of},
        project=lambda rows: [_session(row) for row in rows],
    ))


def trace(home: Path, params: dict[str, Any], fits: Fits) -> dict[str, Any]:
    """``{"trace", "spans", "next_cursor"}``: spans ordered by ``start_ns, span_id``."""
    _only(params, {"trace_id", "cursor", "limit"})
    trace_id = _hex_id(params.get("trace_id"), "trace_id")
    limit = _limit(params.get("limit"), 500, 500)
    order = (Key("start_ns", int), Key("span_id", bytes))
    start = resume(params.get("cursor"), ["telemetry_trace", trace_id], [], order)
    key = bytes.fromhex(trace_id)

    def read(connection: sqlite3.Connection) -> dict[str, Any]:
        row = connection.execute(f"SELECT {', '.join(_TRACE_COLUMNS)} FROM traces WHERE trace_id = ?", (key,)).fetchone()
        if row is None:
            raise TraceNotFound(trace_id)
        summary = _project(row, _TRACE_COLUMNS)

        def spans(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
            events: dict[bytes, list[dict[str, Any]]] = {}
            if rows:
                for event in connection.execute(
                    f"SELECT span_id, {', '.join(_EVENT_COLUMNS)} FROM span_events WHERE trace_id = ? "
                    f"AND span_id IN ({', '.join('?' * len(rows))}) ORDER BY span_id, time_ns, event_index",
                    (key, *(row["span_id"] for row in rows)),
                ):
                    events.setdefault(event["span_id"], []).append(_project(event, _EVENT_COLUMNS))
            return [{**_project(row, _SPAN_COLUMNS), "events": events.get(row["span_id"], [])} for row in rows]

        return keyset_page(
            connection, f"SELECT {', '.join(_SPAN_COLUMNS)} FROM spans WHERE trace_id = ?", [key], order, start,
            limit=limit, fits=fits, project=spans,
            shape=lambda items, cursor: {"trace": summary, "spans": items, "next_cursor": cursor},
        )

    return _snapshot(home, read)


def overview(home: Path, params: dict[str, Any], fits: Fits, endpoint: str | None, live: dict[str, Any]) -> dict[str, Any]:
    """Status, range totals, model/tool key counts, and one page of ``bucket`` points ordered by start."""
    _only(params, {"start_ns", "end_ns", "bucket", "limit", "cursor"})
    bucket = params.get("bucket", "hour")
    if not isinstance(bucket, str) or bucket not in _BUCKETS:
        raise InvalidParams("bucket must be 'hour' or 'day'")
    limit = _limit(params.get("limit"), 200, 200)
    explicit = [_ns(params.get("start_ns"), "start_ns"), _ns(params.get("end_ns"), "end_ns")]
    order = (Key("bucket", int),)
    start = resume(params.get("cursor"), ["telemetry_overview", bucket, *explicit], _range(*explicit), order)
    begin, end = start.pinned
    size = _BUCKETS[bucket]

    def read(connection: sqlite3.Connection) -> dict[str, Any]:
        split = _split(connection, begin, end)
        sql, args = _series(split, "none", size)
        totals_sql, totals_args = _series(split, "none", None)
        row = connection.execute(totals_sql + " SELECT * FROM series", totals_args).fetchone()
        totals = _point(row, begin, end) if row else _point(None, begin, end)
        head = {
            "status": _status(home, endpoint, live, _oldest(connection)),
            "totals": totals,
            "active_models": _active(connection, split, "model"),
            "active_tools": _active(connection, split, "tool"),
        }
        return keyset_page(
            connection, sql + " SELECT * FROM series", args, order, start, limit=limit, fits=fits, total=MAX_POINTS,
            project=lambda rows: [_point(row, row["bucket"], row["bucket"] + size) for row in rows],
            shape=lambda items, cursor: {
                "status": head["status"], "totals": head["totals"], "series": items,
                "active_models": head["active_models"], "active_tools": head["active_tools"], "next_cursor": cursor,
            },
        )

    return _snapshot(home, read)


def models(home: Path, params: dict[str, Any], fits: Fits) -> dict[str, Any]:
    """``Page`` of model rows ordered by ``calls desc`` then canonical key."""
    return _family(home, params, fits, "model")


def tools(home: Path, params: dict[str, Any], fits: Fits) -> dict[str, Any]:
    """``Page`` of tool rows ordered by ``calls desc`` then canonical key."""
    return _family(home, params, fits, "tool")


def errors(home: Path, params: dict[str, Any], fits: Fits) -> dict[str, Any]:
    """Failed raw spans, newest first, and error classes across raw spans and rollups."""
    _only(params, {"start_ns", "end_ns", "operation", "limit", "cursor"})
    operation = params.get("operation")
    if operation is not None and (not isinstance(operation, str) or operation not in _OPERATIONS):
        raise InvalidParams("invalid operation")
    limit = _limit(params.get("limit"), 50, 200)
    explicit = [_ns(params.get("start_ns"), "start_ns"), _ns(params.get("end_ns"), "end_ns")]
    order = (Key("time_ns", int, descending=True), Key("trace_id", bytes, descending=True), Key("span_id", bytes, descending=True))
    start = resume(params.get("cursor"), ["telemetry_errors", operation, *explicit], _range(*explicit), order)
    query = (
        "SELECT trace_id, span_id, session_id, start_ns AS time_ns, operation, error_type, duration_ns, "
        f"{_MODEL} AS model, tool_name, agent_name, mcp_server, mcp_method, mcp_tool FROM spans "
        "WHERE status = 'error' AND start_ns >= ? AND start_ns < ?"
    )
    values: list[object] = list(start.pinned)
    if operation is not None:
        query += " AND operation = ?"
        values.append(operation)
    as_of = time.time_ns()

    def read(connection: sqlite3.Connection) -> dict[str, Any]:
        rows = connection.execute(*_error_classes_query(_split(connection, *start.pinned), operation)).fetchall()
        classes = [{"error_type": row["error_type"], "count": row["count"]} for row in rows]
        if rows and (remaining := rows[0]["total"] - sum(item["count"] for item in classes)):
            classes.append({"error_type": "other", "count": remaining})
        return keyset_page(
            connection, query, values, order, start, limit=limit, fits=fits,
            project=lambda rows: [_project(row, _ERROR_COLUMNS) for row in rows],
            shape=lambda items, cursor: {"items": items, "next_cursor": cursor, "as_of_ns": as_of, "classes": classes},
        )

    return _snapshot(home, read)


def _error_classes_query(split: tuple[int, int, int, int], operation: str | None) -> tuple[str, tuple[object, ...]]:
    hour_from, hour_to, raw_from, raw_to = split
    clause, filter_args = ("", ()) if operation is None else (" AND operation = ?", (operation,))
    return (
        f"""SELECT error_type, SUM(calls) AS count, SUM(SUM(calls)) OVER () AS total FROM (
            SELECT error_type, COUNT(*) AS calls FROM spans
            WHERE status = 'error' AND start_ns >= ? AND start_ns < ?{clause} GROUP BY error_type
            UNION ALL
            SELECT json_extract(class_key, '$') AS error_type, SUM(calls) FROM hourly_errors
            WHERE hour_ns >= ? AND hour_ns < ?{clause} GROUP BY class_key
        ) GROUP BY error_type ORDER BY count DESC, error_type LIMIT 50""",
        (raw_from, raw_to, *filter_args, hour_from, hour_to, *filter_args),
    )


def series(home: Path, params: dict[str, Any], fits: Fits) -> dict[str, Any]:
    """Points per key and bucket for the top 20 keys by ``metric``; ordered by key, then start."""
    _only(params, {"metric", "start_ns", "end_ns", "bucket", "group_by", "limit", "cursor"})
    metric, bucket, group = params.get("metric"), params.get("bucket"), params.get("group_by")
    if not all(isinstance(value, str) for value in (metric, bucket, group)) or metric not in _METRICS or bucket not in _BUCKETS or group not in _GROUPS:
        raise InvalidParams("invalid metric, bucket, or group_by")
    explicit = [_ns(params.get("start_ns"), "start_ns"), _ns(params.get("end_ns"), "end_ns")]
    if None in explicit:
        raise InvalidParams("start_ns and end_ns are required")
    limit = _limit(params.get("limit"), 200, 200)
    order = (Key("key", str), Key("bucket", int))
    start = resume(params.get("cursor"), ["telemetry_series", metric, bucket, group, *explicit], _range(*explicit), order)
    size = _BUCKETS[bucket]

    def read(connection: sqlite3.Connection) -> dict[str, Any]:
        sql, args = _series(_split(connection, *start.pinned), group, size)
        top = f"SELECT key FROM series GROUP BY key ORDER BY SUM({metric}) DESC, key LIMIT {MAX_SERIES_KEYS}"
        return keyset_page(
            connection, f"{sql} SELECT * FROM series WHERE key IN ({top})", args, order, start,
            limit=limit, fits=fits, total=MAX_POINTS,
            project=lambda rows: [
                {"key": None if group == "none" else row["key"], "point": _point(row, row["bucket"], row["bucket"] + size)}
                for row in rows
            ],
            shape=lambda items, cursor: {"points": items, "next_cursor": cursor},
        )

    return _snapshot(home, read)


def agent_tree(home: Path, params: dict[str, Any], fits: Fits) -> dict[str, Any]:
    """One root session's trace ids, one trace's agents, or one agent's span ids in that trace."""
    _only(params, {"session_id", "collection", "trace_id", "agent_session_id", "limit", "cursor"})
    session_id, collection = params.get("session_id"), params.get("collection")
    if not isinstance(session_id, str) or not _ROOT_SESSION_ID.match(session_id):
        raise InvalidParams("session_id must be a 32-hex root session id")
    trace_id, agent = params.get("trace_id"), params.get("agent_session_id")
    needs = {"traces": (False, False), "agents": (True, False), "span_ids": (True, True)}
    if not isinstance(collection, str) or collection not in needs or needs[collection] != (trace_id is not None, agent is not None):
        raise InvalidParams("collection needs exactly its selectors")
    if trace_id is not None:
        _hex_id(trace_id, "trace_id")
    if agent is not None and not (isinstance(agent, str) and _SESSION_ID.match(agent) and agent[:32] == session_id):
        raise InvalidParams("agent_session_id must be a session id under session_id")
    limit = _limit(params.get("limit"), 200, 200)
    scope = ["telemetry_agent_tree", session_id, collection, trace_id, agent]
    total = None
    if collection == "traces":
        order = (Key("first_ns", int), Key("trace_id", bytes))
        query, args = "SELECT trace_id, first_ns FROM traces WHERE root_session_id = ?", [session_id]
        project: Callable[[list[sqlite3.Row]], list[Any]] = lambda rows: [row["trace_id"].hex() for row in rows]
    elif collection == "agents":
        order, total = (Key("first_ns", int), Key("session_id", str)), MAX_AGENTS
        query = (
            "SELECT session_id, COALESCE(MAX(CASE WHEN operation = 'invoke_agent' THEN agent_name END), MAX(agent_name)) "
            "AS agent_name, COALESCE(MAX(agent_depth), length(session_id) - length(replace(session_id, ':', ''))) AS depth, "
            "MAX(background) AS background, MIN(start_ns) AS first_ns, MAX(end_ns) AS last_ns, "
            "CASE WHEN SUM(status = 'error') > 0 THEN 'error' WHEN SUM(status = 'ok') > 0 THEN 'ok' ELSE 'unset' END AS status "
            f"FROM spans WHERE trace_id = ? AND {_ROOT} = ? GROUP BY session_id"
        )
        args = [bytes.fromhex(trace_id), session_id]
        project = lambda rows: [_agent(row) for row in rows]
    else:
        order = (Key("start_ns", int), Key("span_id", bytes))
        query = "SELECT span_id, start_ns FROM spans WHERE trace_id = ? AND session_id = ?"
        args = [bytes.fromhex(trace_id), agent]
        project = lambda rows: [row["span_id"].hex() for row in rows]
    start = resume(params.get("cursor"), scope, [], order)
    return _snapshot(home, lambda connection: keyset_page(
        connection, query, args, order, start, limit=limit, fits=fits, total=total, project=project,
        shape=lambda items, cursor: {
            "session_id": session_id, "trace_id": trace_id, "agent_session_id": agent, "items": items, "next_cursor": cursor,
        },
    ))


def _split(connection: sqlite3.Connection, start: int, end: int) -> tuple[int, int, int, int]:
    """The retention seam: ``(hour_from, hour_to, raw_from, raw_to)`` for the range ``[start, end)``.

    ``raw_cutoff`` is the first hour boundary after both the 14-day raw age limit and the oldest
    retained span, so raw spans are complete from it on. Hours that start before it come whole from
    the rollups (a range start inside such an hour widens to the hour start); spans that start at or
    after it come from raw. The cutoff is an hour boundary, so no span is counted twice.
    """
    first_hour = start // HOUR_NS * HOUR_NS
    oldest = connection.execute("SELECT MIN(start_ns) FROM spans").fetchone()[0]
    if oldest is None:
        return first_hour, max(first_hour, end), end, end
    cutoff = (max(time.time_ns() - RAW_DAYS * DAY_NS, oldest) // HOUR_NS + 1) * HOUR_NS
    raw_from = max(start, cutoff)
    return first_hour, max(first_hour, min(end, cutoff)), raw_from, max(raw_from, end)


def _series(split: tuple[int, int, int, int], group: str, size: int | None) -> tuple[str, list[object]]:
    """A ``WITH`` clause defining ``series(key, bucket, <SeriesPoint measures>)`` for ``group``.

    ``size`` is the bucket width; ``None`` puts the whole range in bucket 0. Distinct sessions and
    traces count the union of rollup and raw identities, never a sum of per-hour counts.
    """
    hour_from, hour_to, raw_from, raw_to = split
    key, admitted = _GROUPS[group]
    raw_bucket, hour_bucket = ("0", "0") if size is None else (f"start_ns / {size} * {size}", f"hour_ns / {size} * {size}")
    sql = f"""WITH raw AS (
  SELECT {key} AS key, {raw_bucket} AS bucket, {_IS_MODEL} AS is_model, {_IS_TOOL} AS is_tool, status,
    input_tokens, output_tokens, total_tokens, duration_ns, trace_id, {_ROOT} AS sid
  FROM spans WHERE start_ns >= ? AND start_ns < ? AND {admitted}
), facts AS (
  SELECT key, bucket, COUNT(*) AS spans, SUM(status = 'error') AS errors, SUM(is_model) AS model_calls,
    SUM(is_tool) AS tool_calls, SUM(CASE WHEN is_model THEN COALESCE(input_tokens, 0) ELSE 0 END) AS input_tokens,
    SUM(CASE WHEN is_model THEN COALESCE(output_tokens, 0) ELSE 0 END) AS output_tokens,
    SUM(CASE WHEN is_model THEN COALESCE(total_tokens, 0) ELSE 0 END) AS total_tokens, SUM(duration_ns) AS duration_ns
  FROM raw GROUP BY key, bucket
  UNION ALL
  SELECT group_key, {hour_bucket}, spans, errors, model_calls, tool_calls, input_tokens, output_tokens, total_tokens,
    duration_ns
  FROM hourly_series WHERE group_by = ? AND hour_ns >= ? AND hour_ns < ?
), ids AS (
  SELECT key, bucket, 'trace' AS kind, trace_id AS id FROM raw
  UNION SELECT key, bucket, 'session', CAST(sid AS BLOB) FROM raw WHERE sid IS NOT NULL
  UNION SELECT group_key, {hour_bucket}, identity_kind, identity_key FROM hourly_distinct
    WHERE +group_by = ? AND hour_ns >= ? AND hour_ns < ?  -- unary +: seek the hour range on the primary key
), counted AS (
  SELECT key, bucket, SUM(kind = 'session') AS sessions, SUM(kind = 'trace') AS traces FROM ids GROUP BY key, bucket
), series AS (
  SELECT f.key AS key, f.bucket AS bucket, COALESCE(c.sessions, 0) AS sessions, COALESCE(c.traces, 0) AS traces,
    f.spans, f.errors, f.model_calls, f.tool_calls, f.input_tokens, f.output_tokens, f.total_tokens, f.duration_ns
  FROM (SELECT key, bucket, SUM(spans) AS spans, SUM(errors) AS errors, SUM(model_calls) AS model_calls,
      SUM(tool_calls) AS tool_calls, SUM(input_tokens) AS input_tokens, SUM(output_tokens) AS output_tokens,
      SUM(total_tokens) AS total_tokens, SUM(duration_ns) AS duration_ns FROM facts GROUP BY key, bucket) f
  LEFT JOIN counted c ON c.key = f.key AND c.bucket = f.bucket
)"""
    return sql, [raw_from, raw_to, group, hour_from, hour_to, group, hour_from, hour_to]


def _point(row: sqlite3.Row | None, start: int, end: int) -> dict[str, Any]:
    return {"start_ns": start, "end_ns": end, **{metric: row[metric] if row else 0 for metric in _METRICS}}


def _active(connection: sqlite3.Connection, split: tuple[int, int, int, int], family: str) -> int:
    """Distinct canonical model or tool keys with calls in the range."""
    hour_from, hour_to, raw_from, raw_to = split
    key, admitted = _GROUPS[family]
    return connection.execute(
        f"SELECT COUNT(*) FROM (SELECT {key} FROM spans WHERE start_ns >= ? AND start_ns < ? AND {admitted} "
        "UNION SELECT group_key FROM hourly_series WHERE hour_ns >= ? AND hour_ns < ? AND group_by = ?)",
        (raw_from, raw_to, hour_from, hour_to, family),
    ).fetchone()[0]


def _family(home: Path, params: dict[str, Any], fits: Fits, family: str) -> dict[str, Any]:
    """Model or tool rows: rollup hours plus raw spans, with p50/p95 from the hourly latency histograms."""
    _only(params, {"start_ns", "end_ns", "limit", "cursor"})
    limit = _limit(params.get("limit"), 50, 200)
    explicit = [_ns(params.get("start_ns"), "start_ns"), _ns(params.get("end_ns"), "end_ns")]
    order = (Key("calls", int, descending=True), Key("key", str))
    start = resume(params.get("cursor"), [f"telemetry_{family}s", *explicit], _range(*explicit), order)
    if family == "model":
        names = ("provider", "model")
        sums = ("input_tokens", "output_tokens", "total_tokens", "cache_read_tokens", "cache_write_tokens")
        raw = (f"SELECT provider, {_MODEL} AS model, 1 AS calls, status = 'error' AS errors, duration_ns, "
               "COALESCE(input_tokens, 0) AS input_tokens, COALESCE(output_tokens, 0) AS output_tokens, "
               "COALESCE(total_tokens, 0) AS total_tokens, COALESCE(cache_read_tokens, 0) AS cache_read_tokens, "
               f"COALESCE(cache_write_tokens, 0) AS cache_write_tokens, end_ns AS last_ns FROM spans WHERE {_IS_MODEL}")
        table = "hourly_models"
    else:
        names, sums = ("tool_name", "tool_type"), ("argument_bytes", "result_bytes")
        raw = ("SELECT tool_name, tool_type, 1 AS calls, status = 'error' AS errors, duration_ns, "
               "COALESCE(tool_argument_bytes, 0) AS argument_bytes, COALESCE(tool_result_bytes, 0) AS result_bytes, "
               f"end_ns AS last_ns FROM spans WHERE {_IS_TOOL}")
        table = "hourly_tools"
    measures = ("calls", "errors", "duration_ns", *sums, "last_ns")
    query = (
        f"SELECT json_array({', '.join(names)}) AS key, {', '.join(names)}, SUM(calls) AS calls, SUM(errors) AS errors, "
        f"SUM(duration_ns) AS duration_ns, {', '.join(f'SUM({name}) AS {name}' for name in sums)}, MAX(last_ns) AS last_ns "
        f"FROM ({raw} AND start_ns >= ? AND start_ns < ? UNION ALL "
        f"SELECT {', '.join(names)}, {', '.join(measures)} FROM {table} WHERE hour_ns >= ? AND hour_ns < ?) "
        f"GROUP BY {', '.join(names)}"
    )
    as_of = time.time_ns()

    def read(connection: sqlite3.Connection) -> dict[str, Any]:
        hour_from, hour_to, raw_from, raw_to = _split(connection, *start.pinned)

        if family == "model":
            connection.create_function("histogram_bucket", 1, _histogram_bucket, deterministic=True)

        def rows_with_quantiles(rows: list[sqlite3.Row]) -> list[dict[str, Any]]:
            # every hour the range touches, so a raw span's hour counts whole
            keys = [row["key"] for row in rows]
            quantiles = _latency_quantiles(connection, family, keys, hour_from, start.pinned[1])
            ttft = (_quantiles(connection.execute(*_ttft_query(keys, (hour_from, hour_to, raw_from, raw_to))))
                    if family == "model" and keys else {})
            return [
                {
                    **{name: row[name] for name in names},
                    "calls": row["calls"], "errors": row["errors"], "duration_ns": row["duration_ns"],
                    "p50_ns": quantiles.get(row["key"], (0, 0))[0], "p95_ns": quantiles.get(row["key"], (0, 0))[1],
                    **{name: row[name] for name in (*sums, "last_ns")},
                    **({"ttft_p50_ns": ttft.get(row["key"], (None, None))[0],
                        "ttft_p95_ns": ttft.get(row["key"], (None, None))[1]} if family == "model" else {}),
                }
                for row in rows
            ]

        return keyset_page(
            connection, query, [raw_from, raw_to, hour_from, hour_to], order, start, limit=limit, fits=fits,
            project=rows_with_quantiles,
            shape=lambda items, cursor: {"items": items, "next_cursor": cursor, "as_of_ns": as_of},
        )

    return _snapshot(home, read)


def _latency_quantiles(connection: sqlite3.Connection, family: str, keys: list[str], hour_from: int, end: int) -> dict[str, tuple[int, int]]:
    if not keys:
        return {}
    return _quantiles(connection.execute(
        f"SELECT family_key, bucket, SUM(calls) FROM hourly_latency WHERE family = ? AND family_key IN ({', '.join('?' * len(keys))}) "
        "AND hour_ns >= ? AND hour_ns < ? GROUP BY family_key, bucket ORDER BY family_key, bucket",
        (family, *keys, hour_from, end),
    ))


def _ttft_query(keys: list[str], split: tuple[int, int, int, int]) -> tuple[str, tuple[object, ...]]:
    hour_from, hour_to, raw_from, raw_to = split
    wanted = ", ".join("?" * len(keys))
    key = f"json_array(provider, {_MODEL})"
    return (
        f"""SELECT model_key, bucket, SUM(calls) FROM (
            SELECT {key} AS model_key, histogram_bucket(ttft_ns) AS bucket, COUNT(*) AS calls
            FROM spans WHERE {_IS_MODEL} AND start_ns >= ? AND start_ns < ? AND ttft_ns IS NOT NULL
                AND {key} IN ({wanted}) GROUP BY model_key, bucket
            UNION ALL
            SELECT model_key, bucket, SUM(calls) FROM hourly_ttft
            WHERE model_key IN ({wanted}) AND hour_ns >= ? AND hour_ns < ? GROUP BY model_key, bucket
        ) GROUP BY model_key, bucket ORDER BY model_key, bucket""",
        (raw_from, raw_to, *keys, *keys, hour_from, hour_to),
    )


def _quantiles(rows: sqlite3.Cursor) -> dict[str, tuple[int, int]]:
    """Nearest rank over merged buckets, ordered by key and bucket; return the bucket upper edge."""
    counts: dict[str, list[tuple[int, int]]] = {}
    for key, bucket, calls in rows:
        counts.setdefault(key, []).append((bucket, calls))

    def rank(buckets: list[tuple[int, int]], wanted: int) -> int:
        seen = 0
        for bucket, calls in buckets:
            seen += calls
            if seen >= wanted:
                return 0 if bucket == 0 else (1 << bucket) - 1
        return 0

    result = {}
    for key, buckets in counts.items():
        calls = sum(count for _, count in buckets)
        result[key] = (rank(buckets, (calls + 1) // 2), rank(buckets, (95 * calls + 99) // 100))
    return result


def _agent(row: sqlite3.Row) -> dict[str, Any]:
    session_id = row["session_id"]
    parent = session_id.rsplit(":", 1)[0] if ":" in session_id else None
    values = {**dict(row), "parent_session_id": parent, "background": None if row["background"] is None else bool(row["background"])}
    return {column: values[column] for column in _AGENT_COLUMNS}


def _oldest(connection: sqlite3.Connection) -> tuple[int | None, int | None]:
    return (
        connection.execute("SELECT MIN(start_ns) FROM spans").fetchone()[0],
        connection.execute("SELECT MIN(hour_ns) FROM hourly_series").fetchone()[0],
    )


def _status(home: Path, endpoint: str | None, live: dict[str, Any], oldest: tuple[int | None, int | None] | None) -> dict[str, Any]:
    """``oldest`` is ``(oldest_raw_ns, oldest_rollup_ns)``, or None when there is no database."""
    used = used_bytes(home)
    return {
        "enabled": oldest is not None and endpoint is not None,
        "receiver": live["receiver"] or ("ready" if receiver_ready(home) else "stopped"),
        "endpoint": None if endpoint is None else "builtin" if endpoint == OTLP_URL else "external",
        "last_export": live["last_export"],
        "last_export_ns": live["last_export_ns"],
        "raw_days": RAW_DAYS,
        "rollup_days": ROLLUP_DAYS,
        "max_bytes": MAX_BYTES,
        "used_bytes": used,
        "oldest_raw_ns": None if oldest is None else oldest[0],
        "oldest_rollup_ns": None if oldest is None else oldest[1],
        "over_cap": used > MAX_BYTES,
    }


def _snapshot[T](home: Path, read: Callable[[sqlite3.Connection], T]) -> T:
    """Run ``read`` in one read transaction after checking the schema. Never creates the file."""
    path = database_path(home)
    try:
        connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, isolation_level=None)
    except sqlite3.OperationalError:
        if path.exists():
            raise
        raise Unavailable from None
    try:
        connection.row_factory = sqlite3.Row
        connection.execute(f"PRAGMA busy_timeout = {BUSY_TIMEOUT_MS}")
        connection.execute("BEGIN")
        try:
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            meta = version == SCHEMA_VERSION and connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'viewer_meta'"
            ).fetchone()
            stored = connection.execute("SELECT schema_version FROM viewer_meta").fetchone() if meta else None
            if stored is None or stored[0] != SCHEMA_VERSION:
                raise SchemaMismatch(f"telemetry schema {version} is not {SCHEMA_VERSION}")
            return read(connection)
        finally:
            connection.rollback()
    finally:
        connection.close()


def _only(params: dict[str, Any], allowed: set[str]) -> None:
    unknown = set(params) - allowed
    if unknown:
        raise InvalidParams(f"unknown params: {', '.join(sorted(unknown))}")


def _limit(value: object, default: int, maximum: int) -> int:
    if value is None:
        return default
    if type(value) is not int or not 1 <= value <= maximum:
        raise InvalidParams(f"limit must be an integer from 1 to {maximum}")
    return value


def _ns(value: object, name: str) -> int | None:
    if value is not None and (type(value) is not int or value < 0):
        raise InvalidParams(f"{name} must be a nonnegative integer or null")
    return value


def _filter(value: object, name: str) -> str | None:
    if value is not None and (not isinstance(value, str) or not 1 <= len(value.encode()) <= MAX_FILTER_BYTES):
        raise InvalidParams(f"{name} must be a 1-{MAX_FILTER_BYTES} byte string or null")
    return value


def _hex_id(value: object, name: str) -> str:
    """A 32-hex trace id, checked before any ``bytes.fromhex``."""
    if not isinstance(value, str) or not _TRACE_ID.match(value):
        raise InvalidParams(f"{name} must be 32 lowercase hex characters")
    return value


def _range(start: int | None, end: int | None) -> list[int]:
    end = time.time_ns() if end is None else end
    start = max(0, end - DEFAULT_RANGE_NS) if start is None else start
    if start >= end:
        raise InvalidParams("start_ns must be before end_ns")
    return [start, end]


def _project(row: sqlite3.Row, columns: tuple[str, ...]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for column in columns:
        value = row[column]
        if value is not None and column in _HEX:
            value = value.hex()
        elif value is not None and column in _FLAGS:
            value = bool(value)
        result[column] = value
    return result


def _session(row: sqlite3.Row) -> dict[str, Any]:
    projected = _project(row, _SESSION_COLUMNS)
    return {"session_id": projected.pop("root_session_id"), **projected}
