"""The complete ``telemetry_read_v1`` contract, one table through a real ``zeta serve`` websocket.

Each row runs against an in-process server so a row can lower the frame budget. Stores are
seeded through the receiver's real writer (and, for redaction, its real OTLP projector).
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from websockets.asyncio.client import connect

from tests.support.owned import await_or_fail
from tests.support.serve_contract import ServeClient
from tests.support.sockets import short_socket_path
from zeta.server import ZetaServer
from zeta.server import protocol as protocol_module
from zeta.server.protocol import MethodDescriptor
from zeta.telemetry.receiver.projector import SPAN_COLUMNS
from zeta.telemetry.receiver.store import Writer, open_database

HOUR = 3_600 * 10**9
DAY = 24 * HOUR
NOW = time.time_ns()
RECENT = (NOW - 3 * HOUR) // HOUR * HOUR + 1_000  # three hours ago: read from raw spans
OLD = NOW - 20 * DAY  # older than raw retention: read from the hourly rollups
RANGE = {"start_ns": NOW - 21 * DAY, "end_ns": NOW + 1}
ROOT, B_ROOT, C_ROOT = "a" * 32, "b" * 32, "c" * 32
TRACE_A, TRACE_B, TRACE_C = "0a" * 16, "0b" * 16, "0c" * 16
SECRET = "SECRET-CONTENT-MARKER"
INVALID = {"code": -32602, "message": "invalid params", "data": {"code": "invalid_params"}}

# telemetry_read_v1, verbatim from design-api.md's method table
TELEMETRY_READ_V1 = (
    "telemetry_status", "telemetry_sessions", "telemetry_trace", "telemetry_overview", "telemetry_models",
    "telemetry_tools", "telemetry_errors", "telemetry_series", "telemetry_agent_tree",
)

# exact v1 shapes from design-api.md (errors gain G8's agent_name and mcp_* fields)
STATUS = {"enabled": 0, "receiver": 0, "endpoint": 0, "last_export": 0, "last_export_ns": 0, "raw_days": 0, "rollup_days": 0,
          "max_bytes": 0, "used_bytes": 0, "oldest_raw_ns": 0, "oldest_rollup_ns": 0, "over_cap": 0}
POINT = {"start_ns": 0, "end_ns": 0, "sessions": 0, "traces": 0, "spans": 0, "errors": 0, "model_calls": 0, "tool_calls": 0,
         "input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "duration_ns": 0}
SESSION = {"session_id": 0, "first_ns": 0, "last_ns": 0, "last_trace_id": 0, "profile": 0, "model": 0, "span_count": 0,
           "turn_count": 0, "model_count": 0, "tool_count": 0, "error_count": 0, "input_tokens": 0, "output_tokens": 0,
           "total_tokens": 0}
MODEL = {"provider": 0, "model": 0, "calls": 0, "errors": 0, "duration_ns": 0, "p50_ns": 0, "p95_ns": 0, "input_tokens": 0,
         "output_tokens": 0, "total_tokens": 0, "last_ns": 0}
TOOL = {"tool_name": 0, "tool_type": 0, "calls": 0, "errors": 0, "duration_ns": 0, "p50_ns": 0, "p95_ns": 0, "argument_bytes": 0,
        "result_bytes": 0, "last_ns": 0}
ERROR = {"trace_id": 0, "span_id": 0, "session_id": 0, "time_ns": 0, "operation": 0, "error_type": 0, "duration_ns": 0,
         "model": 0, "tool_name": 0, "agent_name": 0, "mcp_server": 0, "mcp_method": 0, "mcp_tool": 0}
AGENT = {"session_id": 0, "parent_session_id": 0, "agent_name": 0, "depth": 0, "background": 0, "first_ns": 0, "last_ns": 0,
         "status": 0}


def _page(item: object) -> dict[str, Any]:
    return {"items": [item], "next_cursor": 0, "as_of_ns": 0}


def _tree(item: object) -> dict[str, Any]:
    return {"session_id": 0, "trace_id": 0, "agent_session_id": 0, "items": [item], "next_cursor": 0}


@dataclass(frozen=True)
class Method:
    """One list read: base params, its page collection, its stated order, and its page-limit range."""

    name: str
    params: dict[str, Any]
    collection: str
    order: tuple[tuple[str, bool], ...]  # (field, descending); "" is the item itself
    maximum: int
    shape: dict[str, Any]
    invalid: tuple[dict[str, Any], ...]
    rebind: dict[str, Any]  # another filter or selector: the first page's cursor must not fit it


METHODS = {
    "sessions": Method("telemetry_sessions", {**RANGE}, "items", (("last_ns", True), ("session_id", True)), 200, _page(SESSION),
                       ({"status": "ok"}, {"model": ""}, {"start_ns": 5, "end_ns": 5}, {"start_ns": -1}), {"status": "error"}),
    "trace": Method("telemetry_trace", {"trace_id": TRACE_A}, "spans", (("start_ns", False), ("span_id", False)), 500, {},
                    ({"trace_id": "A" * 32}, {"trace_id": "zz" * 16}, {"trace_id": None}), {"trace_id": TRACE_B}),
    "overview": Method("telemetry_overview", {**RANGE}, "series", (("start_ns", False),), 200,
                       {"status": STATUS, "totals": POINT, "series": [POINT], "active_models": 0, "active_tools": 0, "next_cursor": 0},
                       ({"bucket": "week"}, {"bucket": ["hour"]}, {"start_ns": 9, "end_ns": 3}), {"bucket": "day"}),
    "models": Method("telemetry_models", {**RANGE}, "items", (("calls", True), ("key", False)), 200, _page(MODEL),
                     ({"start_ns": "0"}, {"provider": "p"}), {"start_ns": RANGE["start_ns"] + 1}),
    "tools": Method("telemetry_tools", {**RANGE}, "items", (("calls", True), ("key", False)), 200, _page(TOOL),
                    ({"end_ns": 1.5}, {"tool_name": "t1"}), {"end_ns": RANGE["end_ns"] + 1}),
    "errors": Method("telemetry_errors", {**RANGE}, "items", (("time_ns", True), ("trace_id", True), ("span_id", True)), 200,
                     _page(ERROR), ({"operation": "chatty"}, {"operation": ["mcp"]}, {"status": "error"}), {"operation": "mcp"}),
    "series": Method("telemetry_series", {**RANGE, "metric": "spans", "bucket": "hour", "group_by": "operation"}, "points",
                     (("key", False), ("start_ns", False)), 200, {"points": [{"key": 0, "point": POINT}], "next_cursor": 0},
                     ({"metric": "cost"}, {"group_by": "agent"}, {"start_ns": None}, {"bucket": None}), {"metric": "errors"}),
    "agent_traces": Method("telemetry_agent_tree", {"session_id": ROOT, "collection": "traces"}, "items", (), 200, _tree(""),
                           ({"collection": "spans"}, {"session_id": ROOT + ":1"}, {"trace_id": TRACE_A}), {"session_id": B_ROOT}),
    "agent_agents": Method("telemetry_agent_tree", {"session_id": ROOT, "collection": "agents", "trace_id": TRACE_A}, "items",
                           (("first_ns", False), ("session_id", False)), 200, _tree(AGENT),
                           ({"trace_id": "x" * 32}, {"trace_id": None}, {"agent_session_id": ROOT}), {"trace_id": TRACE_B}),
    "agent_span_ids": Method("telemetry_agent_tree", {"session_id": ROOT, "collection": "span_ids", "trace_id": TRACE_A,
                                                      "agent_session_id": ROOT + ":1"}, "items", (), 200, _tree(""),
                             ({"agent_session_id": B_ROOT + ":1"}, {"agent_session_id": "nothex"}, {"agent_session_id": None}),
                             {"agent_session_id": ROOT + ":2"}),
}
for _method in METHODS.values():
    object.__setattr__(_method, "invalid", (*_method.invalid, {"limit": 0}, {"limit": _method.maximum + 1}, {"limit": True},
                                            {"cursor": "not-a-cursor"}, {"unknown": 1}))


def _span(trace: str, span: int, start: int, *, session: str | None = ROOT, operation: str = "turn", duration: int = 10, **values) -> dict:
    row = dict.fromkeys(SPAN_COLUMNS)
    row.update(
        trace_id=bytes.fromhex(trace), span_id=span.to_bytes(8, "big"), parent_span_id=None, operation=operation, kind=1,
        start_ns=start, end_ns=start + duration, duration_ns=duration, status="ok", session_id=session,
        service_version="test", scope_version="test", events=[],
    )
    row.update(values)
    return row


async def _write(home: Path, rows: list[dict], *, clock_ns: Callable[[], int] = time.time_ns) -> Path:
    """Commit rows through the receiver's real writer and its retention step."""
    path = home / "telemetry" / "telemetry.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    open_database(path)
    writer = Writer(path, clock_ns=clock_ns)
    await asyncio.to_thread(writer.start)
    try:
        for start in range(0, len(rows), 5_000):
            await await_or_fail(writer.submit(rows[start:start + 5_000]), what="telemetry batch commit")
    finally:
        await asyncio.to_thread(writer.close)
    return path


QUANTILE_DURATIONS = (0, 1, 3, 100, 120, 127, 128, 1_000, 5_000, 70_000)


def _contract_rows() -> list[dict]:
    """Ties on every stated order, non-chat spans with model metadata, raw and rollup ranges."""
    a = TRACE_A
    rows = [
        _span(a, 1, RECENT, operation="invoke_agent", agent_name="lead", agent_depth=0, request_model="agent-model"),
        _span(a, 2, RECENT + 10, session=ROOT + ":1", operation="invoke_agent", agent_name="worker", agent_depth=1, background=1),
        _span(a, 3, RECENT + 10, session=ROOT + ":2", operation="invoke_agent", agent_name="worker", agent_depth=1, background=0),
        _span(a, 4, RECENT + 20, session=ROOT + ":1", operation="chat", provider="p", request_model="m1", input_tokens=2, output_tokens=3, total_tokens=5),
        _span(a, 5, RECENT + 20, session=ROOT + ":1", operation="chat", provider="p", request_model="m2", input_tokens=1),
        _span(a, 6, RECENT + 21, operation="chat", provider="p", request_model="m1"),
        _span(a, 7, RECENT + 22, operation="chat", provider="p", response_model="m2", request_model="ignored"),
        _span(a, 8, RECENT + 23, operation="chat", provider=None, request_model="m3"),
        _span(a, 9, RECENT + 24, operation="turn", request_model="turn-model", turn_number=1),
        _span(a, 10, RECENT + 25, operation="execute_tool", tool_name="t1", tool_type="function", tool_argument_bytes=7, tool_result_bytes=11),
        _span(a, 11, RECENT + 25, operation="execute_tool", tool_name="t2", tool_type="mcp"),
        _span(a, 12, RECENT + 26, operation="execute_tool", tool_name="t1", tool_type="function"),
        _span(a, 13, RECENT + 26, operation="execute_tool", tool_name="t2", tool_type="mcp"),
        _span(a, 14, RECENT + 27, operation="execute_tool", tool_name="t3", tool_type=None),
        _span(a, 15, RECENT + 30, operation="mcp", status="error", error_type="McpError", mcp_server="srv", mcp_method="tools/call", mcp_tool="search"),
        _span(a, 16, RECENT + 30, session=ROOT + ":2", operation="execute_tool", status="error", error_type="ToolError", tool_name="t3", agent_name="worker"),
        _span(a, 17, RECENT + 40, operation="chat", status="error", error_type="RateLimit", provider="p", request_model="m3"),
        _span(TRACE_B, 1, RECENT + 50),
        _span(TRACE_C, 1, RECENT + 50),  # ties TRACE_B on first_ns: trace id breaks it
        _span(TRACE_B, 2, RECENT + HOUR),
        _span(TRACE_C, 2, RECENT + 2 * HOUR),
        _span("0d" * 16, 1, RECENT + 60, session=B_ROOT),
        _span("0e" * 16, 1, RECENT + 60, session=C_ROOT),  # ties B_ROOT on last_ns: session id breaks it
        _span("0f" * 16, 1, OLD, operation="chat", provider="p", request_model="m1", input_tokens=4, output_tokens=4, total_tokens=8),
        _span("0f" * 16, 2, OLD + 1, operation="execute_tool", tool_name="t1", tool_type="function"),
        _span("10" * 16, 1, NOW - 13 * DAY, session=B_ROOT),  # the oldest raw span: the cutoff is long before RECENT
    ]
    rows += [
        _span(a, 100 + index, RECENT + 100 + index, operation="chat", provider="q", request_model="mq", duration=duration)
        for index, duration in enumerate(QUANTILE_DURATIONS)
    ]
    rows[0]["events"] = [{**dict.fromkeys(("retry_attempt", "retry_reason", "retry_delay", "retry_after", "decision", "scope", "duration_ns")),
                          "trace_id": rows[0]["trace_id"], "span_id": rows[0]["span_id"], "event_index": 0,
                          "name": "gen_ai.client.retry", "time_ns": RECENT, "retry_attempt": 1}]
    return rows


@asynccontextmanager
async def _serve(home: Path, features: list[str] | None = None) -> AsyncIterator[ServeClient]:
    """An in-process ``zeta serve`` with a negotiated websocket controller."""
    server = ZetaServer(home=home, socket_path=short_socket_path(home), ws_port=0, provider="codex")
    await await_or_fail(server.start(), what="serve start")
    client = None
    try:
        assert server._websocket is not None
        websocket = await await_or_fail(connect(
            f"ws://127.0.0.1:{server._websocket.port}/", subprotocols=["zeta.v1", f"zeta.bearer.{server._websocket.controller_token}"],
            max_size=None,
        ), what="websocket connect")
        client = ServeClient(websocket=websocket)
        hello = await client.request("hello", "hello", {"protocol_version": "1.2", "features": ["telemetry_read_v1"] if features is None else features})
        client.hello = hello["result"]  # type: ignore[attr-defined]
        yield client
    finally:
        if client is not None:
            await client.close()
        await await_or_fail(server.close(), what="serve close")


async def _call(client: ServeClient, method: Method, request_id: str = "read", **params: Any) -> tuple[dict[str, Any], int]:
    raw = await client.request_raw(request_id, method.name, {**method.params, **params})
    reply = json.loads(raw)
    assert "result" in reply, reply
    return reply["result"], len(raw.encode()) + 1  # serve counts a frame with its newline, as on the Unix socket


def _frame_size(result: dict[str, Any]) -> int:
    return protocol_module.FrameCodec().response_size("page", result)


async def _walk(client: ServeClient, method: Method, cursor: str | None = None, **params: Any) -> list[dict[str, Any]]:
    pages = []
    while True:
        page, _ = await _call(client, method, **params, **({"cursor": cursor} if cursor else {}))
        pages.append(page)
        cursor = page["next_cursor"]
        if cursor is None:
            return pages
        assert len(pages) < 5_000, "the cursor does not advance"


def _items(pages: list[dict[str, Any]], method: Method) -> list[Any]:
    return [item for page in pages for item in page[method.collection]]


def _ordered(items: list[Any], method: Method) -> list[Any]:
    """``items`` sorted by the method's stated order, last key first so each pass is stable."""
    def field(item: Any, name: str) -> Any:
        if name == "key" and "key" not in item:  # models and tools: the canonical key
            names = ("provider", "model") if "model" in item else ("tool_name", "tool_type")
            return json.dumps([item[part] for part in names], separators=(",", ":"))
        value = item["point"][name] if "point" in item and name != "key" else item[name]
        return "" if value is None else value

    for name, descending in reversed(method.order):
        items = sorted(items, key=lambda item, name=name: field(item, name), reverse=descending)
    return items


def _skeleton(value: Any) -> Any:
    """Exact nested key order; scalars compare equal."""
    if isinstance(value, dict):
        return [(key, _skeleton(child)) for key, child in value.items()]
    if isinstance(value, list):
        return [_skeleton(value[0])] if value else []
    return None


Row = Callable[[Path, pytest.MonkeyPatch], Awaitable[None]]
ROWS: dict[str, tuple[bool, Row]] = {}  # name -> (reads the shared contract home, row)


def row(name: str, *, shared: bool = False) -> Callable[[Row], Row]:
    def register(function: Row) -> Row:
        ROWS[name] = (shared, function)
        return function
    return register


def _per_method(kind: str, check: Callable[..., Awaitable[None]]) -> None:
    for key, method in METHODS.items():
        ROWS[f"{kind}-{key}"] = (True, lambda root, monkeypatch, method=method: check(root, method, monkeypatch))


@pytest.fixture(scope="module")
def contract_home(tmp_path_factory: pytest.TempPathFactory) -> Path:
    home = tmp_path_factory.mktemp("contract") / "home"
    asyncio.run(_write(home, _contract_rows()))  # the real clock prunes the OLD spans: they live in rollups only
    return home


async def _invalid(root: Path, method: Method, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _serve(root) as client:
        valid, _ = await _call(client, method, limit=1)
        assert valid["next_cursor"] is not None
        replies = [await client.request(f"bad{index}", method.name, {**method.params, **params})
                   for index, params in enumerate((*method.invalid, {"cursor": valid["next_cursor"], **method.rebind},
                                                   {"cursor": valid["next_cursor"] + "="}))]
        assert [reply.get("error") for reply in replies] == [INVALID] * len(replies), [reply for reply in replies]


async def _pages(root: Path, method: Method, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _serve(root) as client:
        whole, _ = await _call(client, method, limit=method.maximum)
        walked = _items(await _walk(client, method, limit=1), method)
        assert len(whole[method.collection]) >= 3 and whole["next_cursor"] is None
        assert walked == whole[method.collection], "walking one item per page lost, repeated, or reordered items"
        if method.order:
            assert walked == _ordered(walked, method)


async def _refused(root: Path, method: Method, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _serve(root, features=[]) as client:
        reply = await client.request("refused", method.name, method.params)
        assert method.name not in client.hello["capabilities"]["requests"]  # type: ignore[attr-defined]
        assert reply["error"]["code"] == -32601


async def _lowered_budget(root: Path, method: Method, monkeypatch: pytest.MonkeyPatch) -> None:
    """Under a budget that fits exactly k items and a cursor, a page asked for k+1 returns those k and continues."""
    async with _serve(root) as client:
        whole, _ = await _call(client, method, limit=method.maximum)
        items = whole[method.collection]
        k = len(items) - 2
        fitting, budget = await _call(client, method, limit=k, request_id="page")
        _, larger = await _call(client, method, limit=k + 1, request_id="page")
        monkeypatch.setattr(protocol_module, "MAX_FRAME_BYTES", budget)
        cut, size = await _call(client, method, limit=k + 1, request_id="page")
        monkeypatch.setattr(protocol_module, "MAX_FRAME_BYTES", larger)
        frame_only, _ = await _call(client, method, limit=method.maximum, request_id="page")
        monkeypatch.undo()
        rest = await _walk(client, method, limit=method.maximum, cursor=cut["next_cursor"])

    assert k >= 1 and larger > budget, "the budget does not cut the page"
    assert size <= budget
    assert cut[method.collection] == items[:k], "the cut page is not the largest fitting ordered prefix"
    assert cut["next_cursor"] == fitting["next_cursor"] is not None, "the frame cut did not continue after its last item"
    assert cut[method.collection] + _items(rest, method) == items, "the continuation after a frame cut lost items"
    if _frame_size(whole) > larger:  # no row past the limit: the frame alone cuts the page, which still continues
        assert frame_only[method.collection] == items[:k + 1] and frame_only["next_cursor"] is not None


async def _shape(root: Path, method: Method, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _serve(root) as client:
        result, _ = await _call(client, method)
        if method.name == "telemetry_trace":
            document = (Path(__file__).parents[2] / "docs" / "serve-protocol.md").read_text()
            expected = json.loads(next(line for line in document.splitlines() if line.startswith('{"trace":{')))
        else:
            expected = method.shape
        assert _skeleton(result) == _skeleton(expected)
        if method.name == "telemetry_agent_tree" and method.params["collection"] == "agents":
            assert {type(item["background"]) for item in result["items"]} <= {bool, type(None)}
            assert any(item["background"] is True for item in result["items"])


_per_method("invalid", _invalid)
_per_method("pages", _pages)
_per_method("refused", _refused)
_per_method("shape", _shape)
_per_method("budget", _lowered_budget)


async def _session_lookup(home: Path, session_id: str, filters: dict[str, Any], *, hit: bool) -> None:
    method = METHODS["sessions"]
    async with _serve(home) as client:
        listed, _ = await _call(client, method)
        reply = await client.request("lookup", method.name, {"session_id": session_id, **filters})
    assert "result" in reply, reply
    page = reply["result"]
    expected = [item for item in listed["items"] if item["session_id"] == session_id]
    assert len(expected) == int(hit)
    assert page == {"items": expected, "next_cursor": None, "as_of_ns": page["as_of_ns"]}


@row("session-id-hit", shared=True)
async def _session_hit(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _session_lookup(root, ROOT, {}, hit=True)


@row("session-id-miss", shared=True)
async def _session_miss(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _session_lookup(root, "f" * 32, {}, hit=False)


@row("session-id-invalid", shared=True)
async def _session_invalid(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _serve(root) as client:
        for invalid in ("A" * 32, ROOT + ":1", None):
            reply = await client.request("invalid", METHODS["sessions"].name, {"session_id": invalid})
            assert reply.get("error") == INVALID


@row("session-id-with-cursor", shared=True)
async def _session_cursor(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    method = METHODS["sessions"]
    async with _serve(root) as client:
        first, _ = await _call(client, method, limit=1)
        assert first["next_cursor"] is not None
        reply = await client.request("invalid", method.name, {"session_id": ROOT, "cursor": first["next_cursor"]})
        assert reply.get("error") == INVALID


@row("session-id-ignores-range-and-filters", shared=True)
async def _session_ignores_filters(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    await _session_lookup(root, B_ROOT, {"start_ns": 1, "end_ns": 1, "status": "error", "profile": "other", "model": "other"}, hit=True)


@row("values-raw-and-rollup", shared=True)
async def _values(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _serve(root) as client:
        models, _ = await _call(client, METHODS["models"])
        tools, _ = await _call(client, METHODS["tools"])
        overview, _ = await _call(client, METHODS["overview"], bucket="day")
        errors, _ = await _call(client, METHODS["errors"], operation="mcp")
        agents, _ = await _call(client, METHODS["agent_agents"])
        traces, _ = await _call(client, METHODS["agent_traces"])
        span_ids, _ = await _call(client, METHODS["agent_span_ids"])
        failed, _ = await _call(client, METHODS["sessions"], status="error")
    assert [item["session_id"] for item in failed["items"]] == [ROOT]
    seed = _contract_rows()
    chat = [row for row in seed if row["operation"] == "chat"]
    by_model = {(item["provider"], item["model"]): item for item in models["items"]}
    assert set(by_model) == {("p", "m1"), ("p", "m2"), (None, "m3"), ("p", "m3"), ("q", "mq")}, "non-chat spans made model rows"
    assert by_model["p", "m1"]["calls"] == 3 and by_model["p", "m1"]["input_tokens"] == 6  # two raw calls plus one rollup call
    assert [(item["tool_name"], item["calls"]) for item in tools["items"]][:3] == [("t1", 3), ("t2", 2), ("t3", 2)]
    assert tools["items"][0]["argument_bytes"] == 7 and tools["items"][0]["result_bytes"] == 11
    totals = overview["totals"]
    assert totals == {
        "start_ns": RANGE["start_ns"], "end_ns": RANGE["end_ns"], "sessions": len({row["session_id"][:32] for row in seed}),
        "traces": len({row["trace_id"] for row in seed}), "spans": len(seed), "errors": sum(row["status"] == "error" for row in seed),
        "model_calls": len(chat), "tool_calls": sum(row["operation"] == "execute_tool" for row in seed),
        "input_tokens": sum(row["input_tokens"] or 0 for row in chat), "output_tokens": sum(row["output_tokens"] or 0 for row in chat),
        "total_tokens": sum(row["total_tokens"] or 0 for row in chat), "duration_ns": sum(row["duration_ns"] for row in seed),
    }, "range totals differ from the seeded spans (raw plus rollup)"
    assert (overview["active_models"], overview["active_tools"]) == (5, 3)
    assert errors["items"] == [{"trace_id": TRACE_A, "span_id": (15).to_bytes(8, "big").hex(), "session_id": ROOT, "time_ns": RECENT + 30,
                                "operation": "mcp", "error_type": "McpError", "duration_ns": 10, "model": None, "tool_name": None,
                                "agent_name": None, "mcp_server": "srv", "mcp_method": "tools/call", "mcp_tool": "search"}]
    assert [(item["session_id"], item["parent_session_id"], item["depth"], item["background"]) for item in agents["items"]] == [
        (ROOT, None, 0, None), (ROOT + ":1", ROOT, 1, True), (ROOT + ":2", ROOT, 1, False)]
    assert traces["items"] == [TRACE_A, TRACE_B, TRACE_C]
    assert span_ids["items"] == [n.to_bytes(8, "big").hex() for n in (2, 4, 5)]


@row("quantiles-within-bound", shared=True)
async def _quantiles(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _serve(root) as client:
        models, _ = await _call(client, METHODS["models"])
    model = next(item for item in models["items"] if item["model"] == "mq")
    ordered = sorted(QUANTILE_DURATIONS)
    for field, fraction in (("p50_ns", 0.5), ("p95_ns", 0.95)):
        true = ordered[-(-len(ordered) * int(fraction * 100) // 100) - 1]  # nearest rank ceil(p*N)
        assert true <= model[field] < max(1, 2 * true), (field, true, model[field])


@row("caps-series-keys")
async def _series_keys(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = root / "keys"
    await _write(home, [_span("1a" * 16, n, RECENT + n, operation="execute_tool", tool_name=f"t{tool:02d}", tool_type="function")
                        for tool in range(25) for n in range(tool * 30, tool * 30 + tool + 1)])
    async with _serve(home) as client:
        pages = await _walk(client, METHODS["series"], group_by="tool", metric="tool_calls", limit=7)
    keys = {point["key"] for page in pages for point in page["points"]}
    assert keys == {json.dumps([f"t{tool:02d}", "function"], separators=(",", ":")) for tool in range(5, 25)}


@row("caps-points-2208")
async def _points(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home, first = root / "points", NOW - 2_300 * HOUR
    await _write(home, [_span("2a" * 16, n, first + n * HOUR) for n in range(2_300)], clock_ns=lambda: first)
    params = {"start_ns": first, "end_ns": first + 2_300 * HOUR}
    async with _serve(home) as client:
        overview = await _walk(client, METHODS["overview"], **params)
        series = await _walk(client, METHODS["series"], **params, group_by="none")
        whole, _ = await _call(client, METHODS["overview"], **params, bucket="day")
    assert sum(len(page["series"]) for page in overview) == 2_208 and len(overview) == 12
    assert [point["start_ns"] for page in overview for point in page["series"]][-1] == first // HOUR * HOUR + 2_207 * HOUR
    assert sum(len(page["points"]) for page in series) == 2_208
    assert whole["totals"]["spans"] == 2_300, "the point cap changed the range totals"


@row("caps-agents-1000")
async def _agents(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = root / "agents"
    await _write(home, [_span("3a" * 16, n, RECENT + n, session=f"{ROOT}:{n}") for n in range(1, 1_006)])
    async with _serve(home) as client:
        pages = await _walk(client, METHODS["agent_agents"], trace_id="3a" * 16)
    assert sum(len(page["items"]) for page in pages) == 1_000 and pages[-1]["next_cursor"] is None


@row("one-mib-results")
async def _one_mib(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Pages of the largest items at the real 1 MiB frame: one full page fits, and a trace page is cut."""
    home, long = root / "large", "x" * 120
    rows = [_span("4a" * 16, n, RECENT + n, session=f"{ROOT}:{n}", operation="chat", provider=f"p{n:03d}{long}",
                  request_model=f"m{n:03d}{long}", status="error", error_type=f"E{long}", agent_name=f"a{long}")
            for n in range(1, 202)]
    rows += [_span("4a" * 16, 300 + n, RECENT + n, operation="execute_tool", tool_name=f"t{n:03d}{long}", tool_type="function")
             for n in range(201)]
    rows += [_span(f"{n:032x}", 1, RECENT + n, session=f"{n:032x}") for n in range(1, 202)]
    big = [_span("ee" * 16, n, RECENT) for n in range(1, 61)]
    for span in big:
        span["events"] = [{**dict.fromkeys(("retry_attempt", "retry_reason", "retry_delay", "retry_after", "decision", "scope", "duration_ns")),
                           "trace_id": span["trace_id"], "span_id": span["span_id"], "event_index": index,
                           "name": "gen_ai.client.retry", "time_ns": RECENT + index, "retry_reason": "r" * 64} for index in range(128)]
    await _write(home, rows + big)
    async with _serve(home) as client:
        for key, extra in (("sessions", {}), ("models", {}), ("tools", {}), ("errors", {}),
                           ("agent_agents", {"trace_id": "4a" * 16}), ("agent_span_ids", {"trace_id": "4a" * 16, "agent_session_id": ROOT})):
            method = METHODS[key]
            page, size = await _call(client, method, limit=200, **extra)
            assert size <= protocol_module.MAX_FRAME_BYTES and page["next_cursor"] is not None, key
            assert len(page[method.collection]) == 200, key
        pages = await _walk(client, METHODS["trace"], trace_id="ee" * 16)
    assert len(pages) > 1 and 0 < len(pages[0]["spans"]) < 60, "a trace page larger than 1 MiB was not cut"
    assert [span["span_id"] for span in _items(pages, METHODS["trace"])] == [n.to_bytes(8, "big").hex() for n in range(1, 61)]


@row("rollup-after-raw-prune")
async def _prune(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Values for a range older than raw retention do not change when the raw spans are pruned."""
    home, old = root / "prune", NOW - 15 * DAY
    path = await _write(home, [
        _span("5a" * 16, 1, old, operation="chat", provider="p", request_model="m", input_tokens=2, output_tokens=3, total_tokens=5),
        _span("5a" * 16, 2, old + 1, operation="execute_tool", tool_name="search", tool_type="mcp", tool_argument_bytes=7),
    ], clock_ns=lambda: old + DAY)
    params = {"start_ns": old - HOUR, "end_ns": old + HOUR}
    reads = {"overview": {**params}, "models": {**params}, "tools": {**params},
             "series": {**params, "metric": "spans", "bucket": "hour", "group_by": "none"}}

    async def read_all() -> dict[str, Any]:
        async with _serve(home) as client:
            results = {}
            for key, values in reads.items():
                method = METHODS[key]
                raw = await client.request_raw(key, method.name, values)
                results[key] = json.loads(raw).get("result")
            return results

    before = await read_all()
    await _write(home, [], clock_ns=time.time_ns)  # the receiver's startup maintenance prunes the 15-day-old spans
    after = await read_all()
    import sqlite3
    connection = sqlite3.connect(path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM spans").fetchone()[0] == 0
    finally:
        connection.close()
    for result in (*before.values(), *after.values()):
        result.pop("as_of_ns", None)
        result.pop("status", None)
    assert after == before, "a range older than raw retention lost its values when raw spans were pruned"
    assert after["overview"]["totals"]["spans"] == 2 and after["models"]["items"][0]["calls"] == 1


@row("crossing-cutoff-counts-once")
async def _crossing(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = root / "crossing"
    boundary = (NOW - 14 * DAY) // HOUR * HOUR
    starts = [boundary + offset * (HOUR // 4) for offset in range(-8, 9)]  # two hours either side of the raw cutoff
    await _write(home, [_span("6a" * 16, n, start, session=f"{n:032x}") for n, start in enumerate(starts, 1)],
                 clock_ns=lambda: boundary)  # nothing is pruned: raw and rollups both hold every span
    params = {"start_ns": boundary - 3 * HOUR, "end_ns": boundary + 3 * HOUR}
    async with _serve(home) as client:
        overview, _ = await _call(client, METHODS["overview"], **params)
        series, _ = await _call(client, METHODS["series"], **params, group_by="none", metric="spans", bucket="hour")
    assert (overview["totals"]["spans"], overview["totals"]["sessions"], overview["totals"]["traces"]) == (17, 17, 1)
    expected = {hour: sum(1 for start in starts if start // HOUR * HOUR == hour) for hour in {start // HOUR * HOUR for start in starts}}
    assert {point["start_ns"]: point["spans"] for point in overview["series"]} == expected
    assert {point["point"]["start_ns"]: point["point"]["spans"] for point in series["points"]} == expected


@row("redaction")
async def _redaction(root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Content sent in OTLP attributes, events, and status never reaches any read."""
    pytest.importorskip("opentelemetry.proto")
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )
    from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
    from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

    from zeta.telemetry.receiver.projector import project_request

    def attributes(values: dict[str, object]) -> list[KeyValue]:
        return [KeyValue(key=key, value=AnyValue(int_value=value) if isinstance(value, int) else AnyValue(string_value=str(value)))
                for key, value in values.items()]

    def span(number: int, name: str, values: dict[str, object], *, error: bool = False) -> Span:
        result = Span(trace_id=bytes.fromhex(TRACE_A), span_id=number.to_bytes(8, "big"), name=name, kind=Span.SPAN_KIND_INTERNAL,
                      start_time_unix_nano=RECENT + number, end_time_unix_nano=RECENT + number + 10,
                      status=Status(code=Status.STATUS_CODE_ERROR if error else Status.STATUS_CODE_OK, message=SECRET),
                      attributes=attributes({"gen_ai.conversation.id": ROOT, **values}),
                      events=[Span.Event(name="exception", time_unix_nano=RECENT, attributes=attributes({"exception.message": SECRET}))])
        return result

    version, home_id = importlib.metadata.version("zeta"), "f" * 32
    request = ExportTraceServiceRequest()
    resource = request.resource_spans.add()
    resource.resource.attributes.extend(attributes({"service.name": "zeta", "service.version": version, "zeta.home.id": home_id}))
    scope = resource.scope_spans.add()
    scope.scope.name, scope.scope.version = "zeta", version
    scope.spans.extend([
        span(1, "invoke_agent lead", {"gen_ai.operation.name": "invoke_agent", "zeta.agent.name": "lead", "gen_ai.system_instructions": SECRET}),
        span(2, "chat m", {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m", "gen_ai.input.messages": SECRET,
                           "gen_ai.output.messages": SECRET, "gen_ai.provider.name": "openai"}, error=True),
        span(3, "execute_tool bash", {"gen_ai.operation.name": "execute_tool", "zeta.tool.name": "bash", "zeta.tool.type": "function",
                                      "gen_ai.tool.call.arguments": SECRET, "gen_ai.tool.call.result": SECRET}, error=True),
    ])
    rows, _ = project_request(request.SerializeToString(), home_id)
    home = root / "redaction"
    await _write(home, rows)
    async with _serve(home) as client:
        replies = [await client.request_raw(key, method.name, method.params) for key, method in METHODS.items()]
        replies.append(await client.request_raw("status", "telemetry_status"))
    assert all('"result"' in reply for reply in replies)
    assert '"bash"' in replies[list(METHODS).index("tools")], "the projected rows did not reach the reads"
    assert not [reply for reply in replies if SECRET in reply], "content reached a telemetry read"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", list(ROWS))
async def test_telemetry_read_contract(name: str, contract_home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """One row of the read contract; shared rows read ``contract_home``, the others seed their own homes."""
    shared, check = ROWS[name]
    await check(contract_home if shared else tmp_path, monkeypatch)


@pytest.mark.asyncio
async def test_telemetry_port_gate_lists_every_v1_method(tmp_path: Path) -> None:
    """The full ``telemetry_read_v1`` list from design-api.md is the descriptor table's list, and all are advertised."""
    declared = [method.wire for method in MethodDescriptor if method.feature == "telemetry_read_v1"]
    async with _serve(tmp_path / "home") as client:
        advertised = set(client.hello["capabilities"]["requests"])  # type: ignore[attr-defined]
    assert sorted(declared) == sorted(TELEMETRY_READ_V1)
    assert set(TELEMETRY_READ_V1) <= advertised
