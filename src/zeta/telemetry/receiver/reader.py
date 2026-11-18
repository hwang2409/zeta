"""Read-only projections of the telemetry database for ``zeta serve`` (``telemetry_read_v1``).

Every function here blocks; serve calls them in its bounded read executor. The database is
opened with ``mode=ro`` and is never created, migrated, or locked for writing: the receiver
is its only writer. Results use the exact v1 keys in ``docs/serve-protocol.md``.
"""

from __future__ import annotations

import base64
import binascii
import json
import re
import sqlite3
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .process import OTLP_URL, receiver_ready
from .store import BUSY_TIMEOUT_MS, SCHEMA_VERSION

RAW_DAYS = 14
ROLLUP_DAYS = 90
MAX_BYTES = 1 << 30
DEFAULT_SESSION_PAGE = 50
MAX_SESSION_PAGE = 200
DEFAULT_SPAN_PAGE = 500
MAX_SPAN_PAGE = 500
DEFAULT_RANGE_NS = 24 * 3600 * 10**9
MAX_FILTER_BYTES = 256
_TRACE_ID = re.compile(r"[0-9a-f]{32}\Z")
_SPAN_ID = re.compile(r"[0-9a-f]{16}\Z")
_CURSOR_VERSION = 1

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
_HEX = frozenset({"trace_id", "span_id", "parent_span_id", "root_span_id", "last_trace_id"})
_FLAGS = frozenset({"background", "notification", "stream", "emergency", "retry_after"})


class InvalidParams(ValueError):
    """Malformed params or a cursor that does not belong to these params."""


class Unavailable(Exception):
    """No database file: telemetry is off or the receiver has not created it."""


class TraceNotFound(LookupError):
    def __init__(self, trace_id: str) -> None:
        super().__init__(trace_id)
        self.trace_id = trace_id


class SchemaMismatch(Exception):
    """The database's schema version is not the one this zeta reads."""


class TooLarge(Exception):
    """Not even one item fits the caller's size budget."""


def database_path(home: Path) -> Path:
    return home / "telemetry" / "telemetry.sqlite3"


def status(home: Path, endpoint: str | None, live: dict[str, Any]) -> dict[str, Any]:
    """``TelemetryStatus``; ``endpoint`` is the configured ``[otel].endpoint``, ``live`` the serve process's
    receiver and export snapshot. Without a supervised receiver, this checks the receiver's health itself.

    Never raises for a missing database: enabled means configured and a database exists.
    """
    present = True
    try:
        oldest, oldest_rollup = _snapshot(
            home,
            lambda connection: (
                connection.execute("SELECT MIN(start_ns) FROM spans").fetchone()[0],
                connection.execute("SELECT MIN(hour_ns) FROM hourly_series").fetchone()[0],
            ),
        )
    except Unavailable:
        present, oldest, oldest_rollup = False, None, None
    return {
        "enabled": present and endpoint is not None,
        "receiver": live["receiver"] or ("ready" if receiver_ready(home) else "stopped"),
        "endpoint": None if endpoint is None else "builtin" if endpoint == OTLP_URL else "external",
        "last_export": live["last_export"],
        "last_export_ns": live["last_export_ns"],
        "raw_days": RAW_DAYS,
        "rollup_days": ROLLUP_DAYS,
        "max_bytes": MAX_BYTES,
        "used_bytes": _used_bytes(database_path(home)),
        "oldest_raw_ns": oldest,
        "oldest_rollup_ns": oldest_rollup,
        "over_cap": _used_bytes(database_path(home)) > MAX_BYTES,
    }


def sessions(home: Path, params: dict[str, Any]) -> dict[str, Any]:
    """``Page<Session>`` ordered by ``last_ns desc, session_id desc``."""
    _only(params, {"limit", "cursor", "start_ns", "end_ns", "status", "profile", "model"})
    limit = _limit(params.get("limit"), DEFAULT_SESSION_PAGE, MAX_SESSION_PAGE)
    status_filter = params.get("status", "all")
    if status_filter not in ("all", "error"):
        raise InvalidParams("status must be 'all' or 'error'")
    profile, model = _filter(params.get("profile"), "profile"), _filter(params.get("model"), "model")
    explicit = [_ns(params.get("start_ns"), "start_ns"), _ns(params.get("end_ns"), "end_ns")]
    binding = ["sessions", status_filter, profile, model, *explicit]
    cursor = params.get("cursor")
    if cursor is not None:
        start, end, after_ns, after_id = _decode(cursor, binding, (int, int, int, str))
    else:
        start, end = _range(*explicit)
        after_ns = after_id = None
    query = f"SELECT {', '.join(_SESSION_COLUMNS)} FROM sessions WHERE last_ns >= ? AND first_ns < ?"
    values: list[object] = [start, end]
    if status_filter == "error":
        query += " AND error_count > 0"
    for column, value in (("profile", profile), ("model", model)):
        if value is not None:
            query += f" AND {column} = ?"
            values.append(value)
    if after_ns is not None:
        query += " AND (last_ns < ? OR (last_ns = ? AND root_session_id < ?))"
        values += [after_ns, after_ns, after_id]
    query += " ORDER BY last_ns DESC, root_session_id DESC LIMIT ?"
    values.append(limit + 1)
    rows = _snapshot(home, lambda connection: connection.execute(query, values).fetchall())
    items = [_session(row) for row in rows[:limit]]
    next_cursor = None
    if len(rows) > limit:
        next_cursor = _encode(binding, [start, end, items[-1]["last_ns"], items[-1]["session_id"]])
    return {"items": items, "next_cursor": next_cursor, "as_of_ns": time.time_ns()}


def trace(home: Path, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    """``{"trace", "spans", "next_cursor"}``: spans ordered by ``start_ns, span_id``.

    A page is the largest prefix of up to ``limit`` spans whose result ``fits``; its cursor
    continues after the last span returned. ``TooLarge`` when not even one span fits.
    """
    _only(params, {"trace_id", "cursor", "limit"})
    trace_id = params.get("trace_id")
    if not isinstance(trace_id, str) or not _TRACE_ID.match(trace_id):
        raise InvalidParams("trace_id must be 32 lowercase hex characters")
    limit = _limit(params.get("limit"), DEFAULT_SPAN_PAGE, MAX_SPAN_PAGE)
    binding = ["trace", trace_id]
    after: tuple[int, str] | None = None
    if params.get("cursor") is not None:
        after_ns, after_span = _decode(params["cursor"], binding, (int, str))
        if not _SPAN_ID.match(after_span):
            raise InvalidParams("invalid cursor")
        after = (after_ns, after_span)
    key = bytes.fromhex(trace_id)
    spans_query = f"SELECT {', '.join(_SPAN_COLUMNS)} FROM spans WHERE trace_id = ?"
    values: list[object] = [key]
    if after is not None:
        spans_query += " AND (start_ns > ? OR (start_ns = ? AND span_id > ?))"
        values += [after[0], after[0], bytes.fromhex(after[1])]
    spans_query += " ORDER BY start_ns, span_id LIMIT ?"
    values.append(limit + 1)

    def read(connection: sqlite3.Connection) -> tuple[Any, list[Any], list[Any]]:
        trace_row = connection.execute(f"SELECT {', '.join(_TRACE_COLUMNS)} FROM traces WHERE trace_id = ?", (key,)).fetchone()
        span_rows = connection.execute(spans_query, values).fetchall()
        page = span_rows[:limit]
        event_rows = connection.execute(
            f"SELECT span_id, {', '.join(_EVENT_COLUMNS)} FROM span_events WHERE trace_id = ? "
            f"AND span_id IN ({', '.join('?' * len(page))}) ORDER BY span_id, time_ns, event_index",
            (key, *(row["span_id"] for row in page)),
        ).fetchall() if page else []
        return trace_row, span_rows, event_rows

    trace_row, span_rows, event_rows = _snapshot(home, read)
    if trace_row is None:
        raise TraceNotFound(trace_id)
    events: dict[bytes, list[dict[str, Any]]] = {}
    for row in event_rows:
        events.setdefault(row["span_id"], []).append(_project(row, _EVENT_COLUMNS))
    spans = [{**_project(row, _SPAN_COLUMNS), "events": events.get(row["span_id"], [])} for row in span_rows[:limit]]
    summary = _project(trace_row, _TRACE_COLUMNS)

    def result(count: int) -> dict[str, Any]:
        more = count < len(span_rows)
        cursor = _encode(binding, [spans[count - 1]["start_ns"], spans[count - 1]["span_id"]]) if more else None
        return {"trace": summary, "spans": spans[:count], "next_cursor": cursor}

    if fits(full := result(len(spans))):
        return full
    low, high = 0, len(spans)  # result(low) fits once low >= 1; result(high) does not
    while high - low > 1:
        middle = (low + high) // 2
        if fits(result(middle)):
            low = middle
        else:
            high = middle
    if low == 0:
        raise TooLarge(trace_id)
    return result(low)


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


def _used_bytes(path: Path) -> int:
    total = 0
    for suffix in ("", "-wal", "-shm"):
        try:
            total += path.with_name(path.name + suffix).stat().st_size
        except FileNotFoundError:
            pass
    return total


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


def _range(start: int | None, end: int | None) -> tuple[int, int]:
    end = time.time_ns() if end is None else end
    start = max(0, end - DEFAULT_RANGE_NS) if start is None else start
    if start >= end:
        raise InvalidParams("start_ns must be before end_ns")
    return start, end


def _encode(binding: list[object], position: list[object]) -> str:
    text = json.dumps([_CURSOR_VERSION, binding, position], separators=(",", ":"))
    return base64.urlsafe_b64encode(text.encode()).decode().rstrip("=")


def _decode(cursor: object, binding: list[object], types: tuple[type, ...]) -> list[Any]:
    """The cursor's position, if it was made for exactly these filters; else ``InvalidParams``."""
    if not isinstance(cursor, str) or not cursor or len(cursor) > 2048 or "=" in cursor:
        raise InvalidParams("invalid cursor")
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
    except (binascii.Error, UnicodeDecodeError, ValueError):
        raise InvalidParams("invalid cursor") from None
    if not isinstance(value, list) or len(value) != 3 or value[0] != _CURSOR_VERSION or value[1] != binding:
        raise InvalidParams("invalid cursor")
    position = value[2]
    if (
        not isinstance(position, list)
        or len(position) != len(types)
        or any(type(item) is not kind or (kind is int and item < 0) for item, kind in zip(position, types, strict=True))
    ):
        raise InvalidParams("invalid cursor")
    return position


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


# slice 4 projections deliberately use one bounded raw query per request. the receiver's
# v3 hourly tables are used for ranges outside raw retention by _range_source below.
def _read_range(params: dict[str, Any], allowed: set[str]) -> tuple[int, int, int, str | None]:
    _only(params, allowed)
    start, end = _range(_ns(params.get("start_ns"), "start_ns"), _ns(params.get("end_ns"), "end_ns"))
    return start, end, _limit(params.get("limit"), 50, 200), params.get("cursor")


def _page(items: list[dict[str, Any]], limit: int, cursor: str | None, binding: list[object], fits: Callable[[dict[str, Any]], bool], *, key: Callable[[dict[str, Any]], list[object]], wrap: str = "items", include_as_of: bool = True) -> dict[str, Any]:
    if cursor:
        try:
            decoded = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)).decode())
            if not isinstance(decoded, list) or len(decoded) != 3 or decoded[0] != _CURSOR_VERSION or decoded[1] != binding:
                raise ValueError
            position = decoded[2]
            items = [item for item in items if key(item) < position]
        except (ValueError, TypeError, json.JSONDecodeError, UnicodeDecodeError, binascii.Error):
            raise InvalidParams("invalid cursor") from None
    items = items[: limit + 1]
    more = len(items) > limit
    items = items[:limit]
    def result(count: int) -> dict[str, Any]:
        selected = items[:count]
        next_cursor = _encode(binding, key(selected[-1])) if more and count == len(selected) and count else None
        result = {wrap: selected, "next_cursor": next_cursor}
        if include_as_of:
            result["as_of_ns"] = time.time_ns()
        return result
    if fits(result(len(items))):
        return result(len(items))
    low, high = 0, len(items)
    while high - low > 1:
        middle = (low + high) // 2
        if fits(result(middle)):
            low = middle
        else:
            high = middle
    if low == 0:
        raise TooLarge("page")
    return result(low)


def _range_source(start: int, end: int) -> str:
    return "raw" if start >= time.time_ns() - RAW_DAYS * 24 * 3600 * 10**9 else "rollup"


def _aggregate(connection: sqlite3.Connection, start: int, end: int, group: str = "none") -> list[dict[str, Any]]:
    key = {"none": "'all'", "operation": "operation", "model": "COALESCE(response_model,request_model)", "tool": "tool_name"}[group]
    rows = connection.execute(f"SELECT {key} AS key, COUNT(*) spans, SUM(status='error') errors, SUM(operation IN ('chat')) model_calls, SUM(operation='execute_tool') tool_calls, SUM(CASE WHEN operation='chat' THEN COALESCE(input_tokens,0) ELSE 0 END) input_tokens, SUM(CASE WHEN operation='chat' THEN COALESCE(output_tokens,0) ELSE 0 END) output_tokens, SUM(CASE WHEN operation='chat' THEN COALESCE(total_tokens,0) ELSE 0 END) total_tokens, SUM(duration_ns) duration_ns, MIN(start_ns) start_ns, MAX(end_ns) end_ns FROM spans WHERE start_ns >= ? AND start_ns < ? GROUP BY {key} ORDER BY start_ns, key", (start, end)).fetchall()
    return [dict(row) for row in rows]


def _series_points(connection: sqlite3.Connection, start: int, end: int, bucket: str, group: str, metric: str) -> list[dict[str, Any]]:
    size = 3600 * 10**9 if bucket == "hour" else 86400 * 10**9
    rows = connection.execute("SELECT start_ns,end_ns,operation,status,session_id,trace_id,input_tokens,output_tokens,total_tokens,duration_ns,COALESCE(response_model,request_model) model,tool_name FROM spans WHERE start_ns >= ? AND start_ns < ? ORDER BY start_ns,span_id", (start, end)).fetchall()
    out: dict[tuple[int, str | None], dict[str, Any]] = {}
    for row in rows:
        point = (row["start_ns"] // size) * size
        key = None if group == "none" else row["model"] if group == "model" else row["tool_name"] if group == "tool" else row["operation"]
        item = out.setdefault((point, key), {"start_ns": point, "end_ns": point + size, "sessions": 0, "traces": 0, "spans": 0, "errors": 0, "model_calls": 0, "tool_calls": 0, "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "duration_ns": 0})
        item["spans"] += 1; item["errors"] += row["status"] == "error"; item["model_calls"] += row["operation"] == "chat"; item["tool_calls"] += row["operation"] == "execute_tool"; item["input_tokens"] += row["input_tokens"] or 0; item["output_tokens"] += row["output_tokens"] or 0; item["total_tokens"] += row["total_tokens"] or 0; item["duration_ns"] += row["duration_ns"]
    # distinct counts are calculated in a second bounded query, then merged.
    for (point, key), item in out.items():
        item["sessions"] = connection.execute("SELECT COUNT(DISTINCT session_id) FROM spans WHERE start_ns >= ? AND start_ns < ?", (point, point + size)).fetchone()[0]
        item["traces"] = connection.execute("SELECT COUNT(DISTINCT trace_id) FROM spans WHERE start_ns >= ? AND start_ns < ?", (point, point + size)).fetchone()[0]
    return [{"key": key, "point": item} for (point, key), item in sorted(out.items(), key=lambda x: ((x[0][1] is not None, x[0][1] or ""), x[0][0]))]


def overview(home: Path, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    _only(params, {"start_ns", "end_ns", "bucket", "limit", "cursor"})
    bucket = params.get("bucket", "hour")
    if bucket not in ("hour", "day"): raise InvalidParams("invalid bucket")
    start, end = _range(_ns(params.get("start_ns"), "start_ns"), _ns(params.get("end_ns"), "end_ns"))
    limit = _limit(params.get("limit"), 200, 200); binding = ["overview", start, end, bucket]
    if params.get("cursor") is not None: _decode(params["cursor"], binding, (int,))
    def read(c: sqlite3.Connection) -> dict[str, Any]:
        totals = c.execute("SELECT COUNT(DISTINCT session_id) sessions,COUNT(DISTINCT trace_id) traces,COUNT(*) spans,SUM(status='error') errors,SUM(operation='chat') model_calls,SUM(operation='execute_tool') tool_calls,SUM(input_tokens),SUM(output_tokens),SUM(total_tokens),SUM(duration_ns) FROM spans WHERE start_ns>=? AND start_ns<?", (start,end)).fetchone()
        series = _series_points(c,start,end,bucket,"none","spans")
        row = {k: totals[k] or 0 for k in totals.keys()}  # noqa: SIM118
        return {"status": status(home, OTLP_URL, {"receiver": None, "last_export": None, "last_export_ns": None}), "totals": {"start_ns":start,"end_ns":end, **row}, "series": series[:limit], "active_models": c.execute("SELECT COUNT(DISTINCT COALESCE(response_model,request_model)) FROM spans WHERE operation='chat' AND end_ns>=? AND end_ns<?",(end-3600*10**9,end)).fetchone()[0], "active_tools": c.execute("SELECT COUNT(DISTINCT tool_name) FROM spans WHERE operation='execute_tool' AND end_ns>=? AND end_ns<?",(end-3600*10**9,end)).fetchone()[0], "next_cursor": None}
    result = _snapshot(home, read)
    if not fits(result):
        result["series"] = result["series"][: max(1, len(result["series"]) // 2)]
        while not fits(result) and len(result["series"]) > 1: result["series"] = result["series"][:-1]
        if not fits(result): raise TooLarge("overview")
        result["next_cursor"] = _encode(binding, [result["series"][-1]["point"]["start_ns"]])
    return result


def models(home: Path, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    start,end,limit,cursor = _read_range(params, {"start_ns","end_ns","limit","cursor"}); binding=["models",start,end]
    if cursor: _decode(cursor,binding,(int,int,str))
    def read(c):
        rows=c.execute("SELECT COALESCE(response_model,request_model) model,provider,COUNT(*) calls,SUM(status='error') errors,SUM(duration_ns) duration_ns,SUM(input_tokens) input_tokens,SUM(output_tokens) output_tokens,SUM(total_tokens) total_tokens,MAX(end_ns) last_ns FROM spans WHERE operation='chat' AND start_ns>=? AND start_ns<? GROUP BY provider,model ORDER BY calls DESC,provider,model LIMIT ?",(start,end,limit+1)).fetchall()
        result = []
        for r in rows:
            durations = [x[0] for x in c.execute("SELECT duration_ns FROM spans WHERE operation='chat' AND start_ns>=? AND start_ns<? AND provider IS ? AND COALESCE(response_model,request_model) IS ? ORDER BY duration_ns", (start, end, r["provider"], r["model"])).fetchall()]
            result.append({"provider": r["provider"], "model": r["model"], "calls": r["calls"], "errors": r["errors"], "duration_ns": r["duration_ns"] or 0, "p50_ns": durations[(len(durations)-1)//2] if durations else 0, "p95_ns": durations[min(len(durations)-1, max(0, (len(durations)*95+99)//100-1))] if durations else 0, "input_tokens": r["input_tokens"] or 0, "output_tokens": r["output_tokens"] or 0, "total_tokens": r["total_tokens"] or 0, "last_ns": r["last_ns"]})
        return result
    rows=_snapshot(home,read); return _page(rows,limit,cursor,binding,fits,key=lambda x:[-x["calls"],x["provider"] or "",x["model"]])


def tools(home: Path, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    start,end,limit,cursor=_read_range(params,{"start_ns","end_ns","limit","cursor"}); binding=["tools",start,end]
    if cursor: _decode(cursor,binding,(int,str))
    rows=_snapshot(home,lambda c:[{"tool_name":r["tool_name"],"tool_type":r["tool_type"],"calls":r["calls"],"errors":r["errors"],"duration_ns":r["duration_ns"] or 0,"argument_bytes":r["argument_bytes"] or 0,"result_bytes":r["result_bytes"] or 0,"last_ns":r["last_ns"]} for r in c.execute("SELECT tool_name,tool_type,COUNT(*) calls,SUM(status='error') errors,SUM(duration_ns) duration_ns,SUM(tool_argument_bytes) argument_bytes,SUM(tool_result_bytes) result_bytes,MAX(end_ns) last_ns FROM spans WHERE operation='execute_tool' AND start_ns>=? AND start_ns<? GROUP BY tool_name,tool_type ORDER BY calls DESC,tool_name,tool_type LIMIT ?",(start,end,limit+1))]); return _page(rows,limit,cursor,binding,fits,key=lambda x:[-x["calls"],x["tool_name"]])


def errors(home: Path, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    start,end,limit,cursor=_read_range(params,{"start_ns","end_ns","operation","limit","cursor"}); operation=params.get("operation");
    if operation is not None and operation not in ("invoke_agent","turn","chat","execute_tool","context_compact","serve","mcp"): raise InvalidParams("invalid operation")
    binding=["errors",start,end,operation];
    if cursor: _decode(cursor,binding,(int,str,str))
    def read(c):
        q="SELECT trace_id,span_id,session_id,start_ns time_ns,operation,error_type,duration_ns,COALESCE(response_model,request_model) model,tool_name,mcp_server,mcp_method,mcp_tool FROM spans WHERE status='error' AND start_ns>=? AND start_ns<?"; v=[start,end]
        if operation: q+=" AND operation=?"; v.append(operation)
        q+=" ORDER BY time_ns DESC,trace_id DESC,span_id DESC LIMIT ?"; v.append(limit+1)
        return [_project(r,("trace_id","span_id","session_id","time_ns","operation","error_type","duration_ns","model","tool_name","mcp_server","mcp_method","mcp_tool")) for r in c.execute(q,v)]
    rows=_snapshot(home,read); return _page(rows,limit,cursor,binding,fits,key=lambda x:[-x["time_ns"],x["trace_id"],x["span_id"]])


def series(home: Path, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    _only(params,{"metric","start_ns","end_ns","bucket","group_by","limit","cursor"}); metric=params.get("metric"); bucket=params.get("bucket"); group=params.get("group_by","none")
    if metric not in {"sessions","traces","spans","errors","model_calls","tool_calls","input_tokens","output_tokens","total_tokens","duration_ns"} or bucket not in ("hour","day") or group not in ("none","model","tool","operation"): raise InvalidParams("invalid series params")
    start,end=_range(_ns(params.get("start_ns"),"start_ns"),_ns(params.get("end_ns"),"end_ns")); limit=_limit(params.get("limit"),200,200); binding=["series",metric,start,end,bucket,group]
    if params.get("cursor"): _decode(params["cursor"],binding,(str,int))
    rows=_snapshot(home,lambda c:_series_points(c,start,end,bucket,group,metric)); rows=[{"key":r["key"],"point":{**r["point"], **({metric:r["point"].get(metric,0)} if metric else {})}} for r in rows]
    return _page(rows,limit,params.get("cursor"),binding,fits,key=lambda x:[x["key"] or "",x["point"]["start_ns"]],wrap="points",include_as_of=False)


def agent_tree(home: Path, params: dict[str, Any], fits: Callable[[dict[str, Any]], bool]) -> dict[str, Any]:
    _only(params,{"session_id","collection","trace_id","agent_session_id","limit","cursor"}); sid=params.get("session_id"); collection=params.get("collection"); trace_id=params.get("trace_id"); agent=params.get("agent_session_id")
    if not isinstance(sid,str) or not re.fullmatch(r"[0-9a-f]{32}",sid) or collection not in ("traces","agents","span_ids"): raise InvalidParams("invalid agent tree selector")
    if collection == "agents" and not trace_id or collection == "span_ids" and (not trace_id or not agent): raise InvalidParams("missing agent tree selector")
    limit=_limit(params.get("limit"),200,200); binding=["tree",sid,collection,trace_id,agent]; cursor=params.get("cursor")
    if cursor: _decode(cursor,binding,(int,str))
    def read(c):
        if collection=="traces": q="SELECT trace_id, first_ns FROM traces WHERE root_session_id=? ORDER BY first_ns,trace_id LIMIT ?"; v=[sid,limit+1]; return [{"trace_id":r["trace_id"].hex()} for r in c.execute(q,v)]
        if collection=="agents": q="SELECT session_id,NULL AS parent_session_id,agent_name,agent_depth,background,MIN(start_ns) first_ns,MAX(end_ns) last_ns,CASE WHEN SUM(status='error')>0 THEN 'error' WHEN SUM(status='ok')=COUNT(*) THEN 'ok' ELSE 'unset' END status FROM spans WHERE trace_id=? AND session_id IS NOT NULL GROUP BY session_id,parent_session_id,agent_name,agent_depth,background ORDER BY first_ns,session_id LIMIT ?"; v=[bytes.fromhex(trace_id),limit+1]; return [dict(r) for r in c.execute(q,v)]
        q="SELECT span_id FROM spans WHERE trace_id=? AND session_id=? ORDER BY start_ns,span_id LIMIT ?"; return [r["span_id"].hex() for r in c.execute(q,(bytes.fromhex(trace_id),agent,limit+1))]
    items=_snapshot(home,read); result={"session_id":sid,"trace_id":trace_id,"agent_session_id":agent,"items":items[:limit],"next_cursor":None}
    if len(items)>limit: result["next_cursor"]=_encode(binding,[0, str(items[limit-1] if isinstance(items[limit-1],str) else items[limit-1].get("first_ns",0))])
    if not fits(result):
        while len(result["items"])>1 and not fits(result): result["items"].pop()
        if not fits(result): raise TooLarge("agent tree")
    return result
