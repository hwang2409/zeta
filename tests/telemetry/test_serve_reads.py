"""``telemetry_read_v1`` through a real ``zeta serve``: negotiation, pages, cursors, and safety."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from pathlib import Path

import pytest

from tests.support.owned import await_or_fail
from tests.support.serve_close import close_serve
from tests.support.serve_contract import ServeContract
from tests.support.server_client import read, request, send
from tests.support.sockets import short_socket_path
from zeta.read_page import TooLarge
from zeta.server import ZetaServer
from zeta.telemetry.receiver import reader
from zeta.telemetry.receiver.projector import SPAN_COLUMNS
from zeta.telemetry.receiver.store import Writer, open_database

FEATURES = ["telemetry_read_v1"]
NOW = time.time_ns()
TRACE = "00112233445566778899aabbccddeeff"
SESSION_KEYS = [
    "session_id", "first_ns", "last_ns", "last_trace_id", "profile", "model", "span_count", "turn_count",
    "model_count", "tool_count", "error_count", "input_tokens", "output_tokens", "total_tokens",
]


def _span(trace: str, span: int, start: int, *, session: str = "0" * 32, operation: str = "turn", **values) -> dict:
    row = dict.fromkeys(SPAN_COLUMNS)
    row.update(
        trace_id=bytes.fromhex(trace), span_id=span.to_bytes(8, "big"), parent_span_id=None, operation=operation,
        kind=1, start_ns=start, end_ns=start + 10, duration_ns=10, status="ok", session_id=session,
        service_version="test", scope_version="test", events=[], **values,
    )
    return row


def _event(row: dict, index: int, **values) -> dict:
    event = dict.fromkeys(("retry_attempt", "retry_reason", "retry_delay", "retry_after", "decision", "scope", "duration_ns"))
    event.update(trace_id=row["trace_id"], span_id=row["span_id"], event_index=index, name="gen_ai.client.retry",
                 time_ns=row["start_ns"] + index, **values)
    return event


async def _write(home: Path, *batches: list[dict]) -> None:
    """Commit each batch through the receiver's real writer."""
    directory = home / "telemetry"
    directory.mkdir(parents=True, exist_ok=True)
    open_database(directory / "telemetry.sqlite3")
    writer = Writer(directory / "telemetry.sqlite3")
    writer.start()
    try:
        for batch in batches:
            await await_or_fail(writer.submit(batch), what="telemetry batch commit")
    finally:
        await asyncio.to_thread(writer.close)


async def _client(contract: ServeContract, role: str = "controller", features: list[str] = FEATURES):
    client = await contract.connect(role)  # type: ignore[arg-type]
    hello = await client.request("hello", "hello", {
        "protocol_version": "1.2", "features": features, **({"role": "observer"} if role == "observer" else {}),
    })
    return client, hello["result"]


async def test_telemetry_reads_require_the_negotiated_feature(tmp_path: Path) -> None:
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        plain, plain_hello = await _client(contract, features=[])
        observer, observer_hello = await _client(contract, "observer", ["serve_observers", *FEATURES])

        refused = await plain.request("status", "telemetry_status")
        allowed = await observer.request("status", "telemetry_status")

        assert "telemetry_status" not in plain_hello["capabilities"]["requests"]
        assert refused["error"]["code"] == -32601
        assert {"telemetry_status", "telemetry_sessions", "telemetry_trace"} <= set(observer_hello["capabilities"]["requests"])
        assert observer_hello["capabilities"]["features"] == ["serve_observers", "telemetry_read_v1"]
        assert allowed["result"]["enabled"] is False
    finally:
        await contract.close()


@pytest.mark.asyncio
async def test_missing_database_is_unavailable_and_is_never_created(tmp_path: Path) -> None:
    database = tmp_path / "home" / "telemetry" / "telemetry.sqlite3"
    database.parent.mkdir(parents=True)  # a receiver ran here once; only the database is missing
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        client, _ = await _client(contract)

        status = await client.request("status", "telemetry_status")
        sessions = await client.request("sessions", "telemetry_sessions")
        trace = await client.request("trace", "telemetry_trace", {"trace_id": TRACE})

        assert status["result"] == {
            "enabled": False, "receiver": "stopped", "endpoint": None, "last_export": None, "last_export_ns": None,
            "raw_days": 14, "rollup_days": 90, "max_bytes": 1 << 30, "used_bytes": 0, "oldest_raw_ns": None,
            "oldest_rollup_ns": None, "over_cap": False,
        }
        unavailable = {"code": -32005, "message": "state unavailable", "data": {"code": "telemetry_off", "retryable": True}}
        assert sessions["error"] == trace["error"] == unavailable
        assert not database.exists(), "a telemetry read created the database"
    finally:
        await contract.close()


@pytest.mark.asyncio
async def test_session_pages_follow_the_cursor_and_reject_other_filters(tmp_path: Path) -> None:
    home = tmp_path / "home"
    roots = ["a" * 32, "b" * 32, "c" * 32]
    await _write(home, [
        _span("01" * 16, 1, NOW - 3000, session=roots[0]),
        _span("02" * 16, 1, NOW - 1000, session=roots[1]),
        _span("03" * 16, 1, NOW - 1000, session=roots[2]),  # ties roots[1] on last_ns: session_id breaks it
        _span("04" * 16, 1, NOW - 2000, session=roots[0], operation="chat", request_model="m-a", input_tokens=3),
    ])
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        client, _ = await _client(contract)

        first = (await client.request("p1", "telemetry_sessions", {"limit": 2}))["result"]
        second = (await client.request("p2", "telemetry_sessions", {"limit": 2, "cursor": first["next_cursor"]}))["result"]
        by_model = (await client.request("m", "telemetry_sessions", {"limit": 1, "model": "m-a"}))["result"]
        mismatched = await client.request("x", "telemetry_sessions", {"limit": 1, "model": "m-b", "cursor": first["next_cursor"]})
        invalid = [
            (await client.request(f"bad{index}", "telemetry_sessions", params))["error"]
            for index, params in enumerate((
                {"cursor": "not-a-cursor"}, {"cursor": first["next_cursor"] + "="}, {"limit": 0}, {"limit": 201},
                {"limit": True}, {"status": "ok"}, {"start_ns": 5, "end_ns": 5}, {"unknown": 1},
            ))
        ]

        assert [item["session_id"] for item in first["items"]] == [roots[2], roots[1]]
        assert [item["session_id"] for item in second["items"]] == [roots[0]]
        assert second["next_cursor"] is None
        assert list(first["items"][0]) == SESSION_KEYS
        assert [item["session_id"] for item in by_model["items"]] == [roots[0]]
        assert by_model["items"][0]["input_tokens"] == 3
        assert mismatched.get("error") == {
            "code": -32602, "message": "invalid params", "data": {"code": "invalid_params"},
        }, "a cursor made for other filters was accepted"
        assert all(error["code"] == -32602 and error["data"] == {"code": "invalid_params"} for error in invalid)
    finally:
        await contract.close()


@pytest.mark.asyncio
async def test_trace_pages_spans_in_order_with_their_events(tmp_path: Path) -> None:
    home = tmp_path / "home"
    spans = [_span(TRACE, span, NOW - 100 + (span // 2)) for span in (5, 4, 3, 2, 1)]  # spans 2 and 3 tie on start
    spans[0]["stream"] = 1
    spans[0]["events"] = [_event(spans[0], 0, retry_attempt=1, retry_after=0)]
    big = [_span("ee" * 16, span, NOW) for span in range(1, 61)]
    for row in big:
        row["events"] = [_event(row, index, retry_reason="r" * 64) for index in range(128)]
    await _write(home, spans, big)
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        client, _ = await _client(contract)
        pages, cursor = [], None
        while True:
            params = {"trace_id": TRACE, "limit": 2, **({"cursor": cursor} if cursor else {})}
            page = (await client.request(f"t{len(pages)}", "telemetry_trace", params))["result"]
            pages.append(page)
            cursor = page["next_cursor"]
            if cursor is None:
                break
        other = await client.request("other", "telemetry_trace", {"trace_id": "ff" * 16, "cursor": pages[0]["next_cursor"]})
        missing = await client.request("missing", "telemetry_trace", {"trace_id": "ff" * 16})
        large, cursor = [], None
        while True:
            params = {"trace_id": "ee" * 16, **({"cursor": cursor} if cursor else {})}
            reply = await client.request(f"l{len(large)}", "telemetry_trace", params)
            assert "result" in reply, f"an oversized page was not cut to the frame budget: {reply.get('error')}"
            page = reply["result"]
            large.append(page)
            cursor = page["next_cursor"]
            if cursor is None:
                break

        assert [len(page["spans"]) for page in pages] == [2, 2, 1]
        assert all(page["trace"] == pages[0]["trace"] for page in pages)
        assert pages[0]["trace"]["span_count"] == 5
        ordered = [span["span_id"] for page in pages for span in page["spans"]]
        assert ordered == [(span).to_bytes(8, "big").hex() for span in (1, 2, 3, 4, 5)]
        last = pages[-1]["spans"][0]
        assert last["stream"] is True
        # the served Span is exactly the documented frame: same keys, same order
        document = (Path(__file__).parents[2] / "docs" / "serve-protocol.md").read_text()
        example = json.loads(next(line for line in document.splitlines() if line.startswith('{"trace":{')))["spans"][0]
        assert list(last) == list(example)
        assert last["events"] == [{"name": "gen_ai.client.retry", "time_ns": last["start_ns"], "retry_attempt": 1,
                                   "retry_reason": None, "retry_delay": None, "retry_after": False, "decision": None,
                                   "scope": None, "duration_ns": None}]
        assert other["error"]["data"] == {"code": "invalid_params"}
        assert missing["error"] == {"code": -32602, "message": "resource not found",
                                    "data": {"code": "trace_not_found", "id": "ff" * 16}}
        # 60 spans with 128 events each are ~1.8 MB: each page is the largest prefix that fits one frame
        assert len(large) > 1, "an oversized page was not cut to the frame budget"
        assert 0 < len(large[0]["spans"]) < 60
        assert [span["span_id"] for page in large for span in page["spans"]] == [n.to_bytes(8, "big").hex() for n in range(1, 61)]
        with pytest.raises(TooLarge):
            reader.trace(home, {"trace_id": "ee" * 16}, lambda result: not result["spans"])
    finally:
        await contract.close()


OFF = {"code": -32005, "message": "state unavailable", "data": {"code": "telemetry_off", "retryable": True}}
MISMATCH = {"code": -32000, "message": "telemetry schema is unsupported", "data": {"code": "telemetry_schema_mismatch"}}


@pytest.mark.asyncio
@pytest.mark.parametrize(("change", "status_error", "data_error"), [
    (None, None, OFF),  # no database file
    ("PRAGMA user_version = 0", MISMATCH, MISMATCH),
    ("PRAGMA user_version = 1", MISMATCH, MISMATCH),  # schema 1: the receiver has not migrated it yet
    ("PRAGMA user_version = 3", MISMATCH, MISMATCH),  # the slice-3 receiver left this version
    ("DROP TABLE viewer_meta", MISMATCH, MISMATCH),
    ("UPDATE viewer_meta SET schema_version = 1", MISMATCH, MISMATCH),
    ("UPDATE viewer_meta SET schema_version = 5", MISMATCH, MISMATCH),
])
async def test_only_a_missing_database_is_off_and_every_bad_schema_is_refused(
    tmp_path: Path, change: str | None, status_error: dict | None, data_error: dict,
) -> None:
    home = tmp_path / "home"
    if change is not None:
        await _write(home, [_span(TRACE, 1, NOW - 100, session="e" * 32)])
        connection = sqlite3.connect(home / "telemetry" / "telemetry.sqlite3")
        try:
            connection.execute(change)
            connection.commit()
        finally:
            connection.close()
    contract = await ServeContract.start(tmp_path, "websocket")
    try:
        client, _ = await _client(contract)
        replies = [
            await client.request("status", "telemetry_status"),
            await client.request("sessions", "telemetry_sessions"),
            await client.request("trace", "telemetry_trace", {"trace_id": TRACE}),
        ]

        assert [reply.get("error") for reply in replies] == [status_error, data_error, data_error], change
    finally:
        await contract.close()


@pytest.mark.asyncio
async def test_reads_stay_consistent_while_the_writer_commits(tmp_path: Path) -> None:
    home = tmp_path / "home"
    session = "d" * 32
    await _write(home, [_span("0a" * 16, 1, NOW - 10_000, session=session)])
    contract = await ServeContract.start(tmp_path, "websocket")
    writer = Writer(home / "telemetry" / "telemetry.sqlite3")
    writer.start()
    try:
        client, _ = await _client(contract)
        counts = []

        async def commit_batches() -> None:
            for batch in range(40):
                rows = [_span("0a" * 16, 2 + batch * 25 + index, NOW - 9_000 + batch, session=session) for index in range(25)]
                await await_or_fail(writer.submit(rows), what="concurrent batch commit")

        writing = asyncio.create_task(commit_batches())
        try:
            while not writing.done():
                reply = await client.request(f"r{len(counts)}", "telemetry_sessions")
                counts.append(reply["result"]["items"][0]["span_count"])
            await await_or_fail(writing, what="concurrent writer")
        finally:
            writing.cancel()
        final = await client.request("final", "telemetry_sessions")

        assert len(counts) > 1
        assert counts == sorted(counts)
        assert all((count - 1) % 25 == 0 for count in counts), "a read saw a partly committed batch"
        assert final["result"]["items"][0]["span_count"] == 1 + 40 * 25
    finally:
        await asyncio.to_thread(writer.close)
        await contract.close()


@pytest.mark.asyncio
async def test_a_slow_read_does_not_block_the_event_loop(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    release = threading.Event()
    released: list[bool] = []
    real_sessions = reader.sessions

    def slow_sessions(*arguments):
        # bounded so a read on the loop delays, rather than hangs, the test; it times out only then
        released.append(release.wait(5))
        return real_sessions(*arguments)

    monkeypatch.setattr(reader, "sessions", slow_sessions)
    server = ZetaServer(home=tmp_path / "home", socket_path=short_socket_path(tmp_path), provider="codex")
    await await_or_fail(server.start(), what="serve start")
    connect = asyncio.open_unix_connection
    reader_stream, writer = await await_or_fail(connect(str(server.socket_path)), what="controller connect")
    observer_stream, observer = await await_or_fail(connect(str(server.socket_path)), what="observer connect")
    try:
        await request(reader_stream, writer, "hello", "hello", {"protocol_version": "1.2", "features": FEATURES})
        await request(observer_stream, observer, "hello", "hello",
                      {"protocol_version": "1.2", "role": "observer", "features": ["serve_observers", "ping"]})
        await send(reader_stream, writer, "slow", "telemetry_sessions")
        await send(observer_stream, observer, "ping", "ping")  # a connection reads its own requests in order

        pong = await read(observer_stream)
        release.set()
        slow = await read(reader_stream)

        assert pong["result"] == {"pong": True}
        assert released == [True], "ping waited for the telemetry read: the read ran on the event loop"
        assert slow["id"] == "slow"
    finally:
        release.set()
        await close_serve(server, writer, observer)
