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
                connection.execute("SELECT MIN(hour_ns) FROM hourly_rollups").fetchone()[0],
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
