from __future__ import annotations

import fcntl
import functools
import http.client
import importlib.metadata
import json
import os
import select
import signal
import sqlite3
import stat
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.proto")
pytest.importorskip("starlette")

from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
    ExportTraceServiceRequest,
)
from opentelemetry.proto.common.v1.common_pb2 import AnyValue, KeyValue
from opentelemetry.proto.trace.v1.trace_pb2 import Span, Status

with warnings.catch_warnings():  # starlette warns that its test client prefers httpx2
    warnings.simplefilter("ignore")
    from starlette.testclient import TestClient

from zeta.telemetry.receiver import reader
from zeta.telemetry.receiver import store as store_module
from zeta.telemetry.receiver.app import create_app

VERSION = importlib.metadata.version("zeta")
ROOT = "0123456789abcdef0123456789abcdef"
TRACE = bytes.fromhex("00112233445566778899aabbccddeeff")
PROTOBUF = {"content-type": "application/x-protobuf"}
HOME_ID = "f" * 32


def _value(value: object) -> AnyValue:
    if isinstance(value, bool):
        return AnyValue(bool_value=value)
    if isinstance(value, int):
        return AnyValue(int_value=value)
    return AnyValue(string_value=str(value))


def _attributes(values: dict[str, object]) -> list[KeyValue]:
    return [KeyValue(key=key, value=_value(value)) for key, value in values.items()]


def _span(
    span_id: int,
    *,
    trace: bytes = TRACE,
    parent: int | None = None,
    name: str = "zeta.turn",
    start: int | None = None,
    end: int | None = None,
    status: int = Status.STATUS_CODE_OK,
    attributes: dict[str, object] | None = None,
    events: tuple[Span.Event, ...] = (),
) -> Span:
    if start is None:
        start = time.time_ns() - 100
    if end is None:
        end = start + 100
    return Span(
        trace_id=trace,
        span_id=span_id.to_bytes(8, "big"),
        parent_span_id=parent.to_bytes(8, "big") if parent is not None else b"",
        name=name,
        kind=Span.SPAN_KIND_INTERNAL,
        start_time_unix_nano=start,
        end_time_unix_nano=end,
        status=Status(code=status),
        attributes=_attributes({"gen_ai.conversation.id": ROOT, **(attributes or {})}),
        events=list(events),
    )


def _body(*spans: Span, resource: dict[str, object] | None = None, scope_name: str = "zeta") -> bytes:
    request = ExportTraceServiceRequest()
    resource_spans = request.resource_spans.add()
    resource_spans.resource.attributes.extend(_attributes(resource or {"service.name": "zeta", "service.version": VERSION, "zeta.home.id": HOME_ID}))
    scope_spans = resource_spans.scope_spans.add()
    scope_spans.scope.name = scope_name
    scope_spans.scope.version = VERSION
    scope_spans.spans.extend(spans)
    return request.SerializeToString()


@pytest.fixture
def receiver(tmp_path):
    directory = tmp_path / "telemetry"
    directory.mkdir()
    app = create_app(directory, HOME_ID)
    with TestClient(app, base_url="http://127.0.0.1:4323") as client:
        yield client, app, directory / "telemetry.sqlite3"


def _sessions(database: Path, **params) -> dict:
    return reader.sessions(database.parents[1], {"start_ns": 0, **params})


def _trace(database: Path, trace_id: str) -> dict:
    return reader.trace(database.parents[1], {"trace_id": trace_id}, lambda result: True)


def _rows(database: Path, table: str) -> list[tuple]:
    connection = sqlite3.connect(database)
    try:
        return connection.execute(f"SELECT * FROM {table} ORDER BY 1, 2").fetchall()
    finally:
        connection.close()


def _committed_sequence(database: Path) -> int:
    connection = sqlite3.connect(database)
    try:
        return connection.execute("SELECT committed_sequence FROM viewer_meta").fetchone()[0]
    finally:
        connection.close()


def test_receiver_retains_only_allowlisted_metadata(receiver):
    client, _, database = receiver
    secret = "SECRET-CONTENT-MARKER"
    span = _span(
        1,
        name="chat qwen3:4b",
        status=Status.STATUS_CODE_ERROR,
        attributes={
            "gen_ai.request.model": "qwen3:4b",
            "gen_ai.response.model": f"../{secret}",  # fails the model rule
            "gen_ai.provider.name": "ollama",
            "gen_ai.usage.input_tokens": 12,
            "error.type": "TimeoutError",
            "gen_ai.input.messages": secret,  # content key, not allowlisted
            "zeta.custom": secret,
        },
        events=(
            Span.Event(name="gen_ai.client.retry", time_unix_nano=150, attributes=_attributes({
                "gen_ai.retry.attempt": 1, "gen_ai.retry.reason": "overloaded_error", "prompt": secret,
            })),
            Span.Event(name="exception", time_unix_nano=160, attributes=_attributes({"exception.message": secret})),
        ),
    )
    span.status.message = secret
    span.trace_state = secret
    span.links.add(trace_id=TRACE, span_id=(9).to_bytes(8, "big"), attributes=_attributes({"link": secret}))
    body = _body(span, resource={"service.name": "zeta", "service.version": VERSION, "zeta.home.id": HOME_ID, "host.name": secret})

    assert client.post("/v1/traces", content=body, headers=PROTOBUF).status_code == 200

    stored = _trace(database, TRACE.hex())["spans"][0]
    assert stored["operation"] == "chat"
    assert stored["response_model"] == "_invalid"
    assert stored["request_model"] == "qwen3:4b"
    assert stored["error_type"] == "TimeoutError"
    assert [event["name"] for event in stored["events"]] == ["gen_ai.client.retry"]
    dump = "\n".join(sqlite3.connect(database).iterdump())
    assert secret not in dump
    assert "SECRET" not in dump


def test_schema_2_fields_are_closed(receiver):
    client, _, database = receiver
    secret = "SECRET-CONTENT-MARKER"
    chat = _span(1, name="chat qwen3:4b", status=Status.STATUS_CODE_ERROR, attributes={
        "gen_ai.request.model": "qwen3:4b",
        "zeta.error.code": f"rate_limit: {secret}",  # a message, not a closed code
        "gen_ai.usage.cache_read.input_tokens": secret,  # not an integer
        "gen_ai.usage.cache_creation.input_tokens": 4,
        "zeta.response.time_to_first_token": 1_500_000,
        "zeta.retry.count": 2,
    })
    agent = _span(2, name="invoke_agent worker", attributes={
        "zeta.agent.name": "worker", "zeta.agent.run_id": f"{ROOT}:1:{secret}",
        "zeta.agent.parent_run_id": f"{ROOT}:1", "zeta.outcome": secret,
    })
    turn = _span(3, attributes={"zeta.outcome": "turn_limit", "zeta.error.code": "overloaded"})

    assert client.post("/v1/traces", content=_body(chat, agent, turn), headers=PROTOBUF).status_code == 200

    stored = {span["span_id"]: span for span in _trace(database, TRACE.hex())["spans"]}
    chat_row, agent_row, turn_row = (stored[n.to_bytes(8, "big").hex()] for n in (1, 2, 3))
    assert (chat_row["error_code"], chat_row["cache_read_tokens"], chat_row["cache_write_tokens"]) == ("_invalid", 0, 4)
    assert (chat_row["ttft_ns"], chat_row["retry_count"]) == (1_500_000, 2)
    assert (agent_row["run_id"], agent_row["parent_run_id"], agent_row["outcome"]) == ("_invalid", f"{ROOT}:1", "_invalid")
    assert (turn_row["outcome"], turn_row["error_code"], turn_row["run_id"]) == ("turn_limit", "overloaded", None)
    assert secret not in "\n".join(sqlite3.connect(database).iterdump())


def _schema_1_database(path: Path) -> None:
    """A database exactly as the schema-1 receiver left it, with one span, event, trace, and session."""
    connection = sqlite3.connect(path)
    try:
        connection.executescript(store_module.SCHEMA)
        connection.execute("INSERT INTO viewer_meta (singleton, schema_version, committed_sequence) VALUES (1, 1, 1)")
        connection.execute(
            "INSERT INTO spans (trace_id, span_id, operation, kind, start_ns, end_ns, duration_ns, status, session_id, "
            "root_session_id, service_version, scope_version, request_model, input_tokens) "
            "VALUES (?, ?, 'chat', 1, 100, 200, 100, 'ok', ?, ?, 'v', 'v', 'qwen3:4b', 9)",
            (TRACE, (1).to_bytes(8, "big"), ROOT, ROOT),
        )
        connection.execute(
            "INSERT INTO span_events (trace_id, span_id, event_index, name, time_ns, retry_attempt) "
            "VALUES (?, ?, 0, 'gen_ai.client.retry', 150, 1)", (TRACE, (1).to_bytes(8, "big")),
        )
        connection.execute(
            "INSERT INTO traces (trace_id, first_ns, last_ns, root_span_id, root_session_id, status, span_count) "
            "VALUES (?, 100, 200, ?, ?, 'ok', 1)", (TRACE, (1).to_bytes(8, "big"), ROOT),
        )
        connection.execute(
            "INSERT INTO sessions (root_session_id, first_ns, last_ns, last_trace_id, last_span_id, span_count, model_count, "
            "input_tokens) VALUES (?, 100, 200, ?, ?, 1, 1, 9)", (ROOT, TRACE, (1).to_bytes(8, "big")),
        )
        connection.execute("PRAGMA user_version = 1")
        connection.commit()
    finally:
        connection.close()


def test_receiver_start_migrates_a_schema_1_database_and_keeps_its_rows(tmp_path):
    directory = tmp_path / "telemetry"
    directory.mkdir()
    database = directory / "telemetry.sqlite3"
    _schema_1_database(database)
    with pytest.raises(reader.SchemaMismatch):  # serve refuses it and never migrates it
        _trace(database, TRACE.hex())

    app = create_app(directory, "f" * 32)  # receiver start migrates
    app.state.writer._clock_ns = lambda: 0
    with TestClient(app, base_url="http://127.0.0.1:4323") as client:
        body = _body(_span(2, trace=TRACE, name="chat qwen3:4b", start=300, end=400,
                           attributes={"gen_ai.request.model": "qwen3:4b", "zeta.retry.count": 1}))
        assert client.post("/v1/traces", content=body, headers=PROTOBUF).status_code == 200

    old, new = _trace(database, TRACE.hex())["spans"]
    assert (old["request_model"], old["input_tokens"], old["retry_count"], old["ttft_ns"]) == ("qwen3:4b", 9, None, None)
    assert [event["retry_attempt"] for event in old["events"]] == [1]
    assert new["retry_count"] == 1
    assert _sessions(database)["items"][0]["span_count"] == 2
    connection = sqlite3.connect(database)
    try:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
        assert connection.execute("SELECT schema_version, committed_sequence FROM viewer_meta").fetchone() == (2, 2)
    finally:
        connection.close()


@pytest.mark.parametrize(
    "body",
    [
        pytest.param(b"\xff\xff\xff", id="malformed protobuf"),
        pytest.param(_body(_span(1), _span(2, name="chat_without_space")), id="unknown operation"),
        pytest.param(_body(_span(1), _span(2, name="chat m1", attributes={"gen_ai.request.model": "m2"})), id="name disagrees"),
        pytest.param(_body(_span(1), _span(2, name="execute_tool rm -rf")), id="invalid suffix"),
        pytest.param(_body(_span(1), _span(2, trace=bytes(16))), id="zero trace id"),
        pytest.param(_body(_span(1), _span(2, start=300, end=200)), id="end before start"),
        pytest.param(_body(_span(1), _span(2, attributes={"zeta.agent.name": "a" * 257})), id="string over 256 bytes"),
        pytest.param(_body(_span(1), scope_name="other"), id="foreign scope"),
    ],
)
def test_invalid_batch_is_rejected_whole(receiver, body):
    client, app, database = receiver

    response = client.post("/v1/traces", content=body, headers=PROTOBUF)

    assert response.status_code == 400
    assert _rows(database, "spans") == []
    assert _committed_sequence(database) == 0
    assert app.state.writer.committed_sequence == 0


def test_oversize_body_is_refused_while_streaming(receiver):
    client, _, database = receiver

    def chunks():
        for _ in range(9):
            yield b"\0" * (1024 * 1024)

    streamed = client.post("/v1/traces", content=chunks(), headers=PROTOBUF)
    declared = client.post("/v1/traces", content=b"", headers={**PROTOBUF, "content-length": str(8 * 1024 * 1024 + 1)})

    assert streamed.status_code == 413
    assert declared.status_code == 413
    assert _rows(database, "spans") == []


@pytest.mark.parametrize("host", ["localhost:4323", "127.0.0.1:4324", "127.0.0.1", "[::1]:4323", "attacker.example"])
def test_foreign_host_is_refused(receiver, host):
    client, _, database = receiver

    ingest = client.post("/v1/traces", content=_body(_span(1)), headers={**PROTOBUF, "host": host})
    health = client.get("/health", headers={"host": host})

    assert ingest.status_code == 400
    assert health.status_code == 400
    assert _rows(database, "spans") == []


def test_request_shape_rules(receiver):
    client, _, database = receiver
    body = _body(_span(1))

    assert client.post("/v1/traces", content=body, headers={"content-type": "application/json"}).status_code == 415
    assert client.post("/v1/traces", content=body, headers={**PROTOBUF, "content-encoding": "gzip"}).status_code == 415
    assert client.post("/v1/logs", content=body, headers=PROTOBUF).status_code == 404
    assert client.get("/v1/traces").status_code == 405
    assert client.post("/v1/traces", content=body, headers={**PROTOBUF, "origin": "http://evil.example"}).status_code == 403
    assert _rows(database, "spans") == []
    for path in ("/", "/api/sessions", f"/api/traces/{TRACE.hex()}", "/app.js"):
        assert client.get(path).status_code == 404  # the receiver serves no page and no read routes
    health = client.get("/health")
    assert health.status_code == 200
    assert health.headers["content-security-policy"] == "default-src 'none'; frame-ancestors 'none'"
    assert health.headers["x-content-type-options"] == "nosniff"


def test_spans_from_another_home_are_dropped_and_counted(receiver):
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceResponse,
    )

    client, _app, database = receiver
    other = {"service.name": "zeta", "service.version": VERSION, "zeta.home.id": "e" * 32}
    unmarked = {"service.name": "zeta", "service.version": VERSION}

    own = client.post("/v1/traces", content=_body(_span(1)), headers=PROTOBUF)
    foreign = client.post("/v1/traces", content=_body(_span(2), _span(3), resource=other), headers=PROTOBUF)
    legacy = client.post("/v1/traces", content=_body(_span(4), resource=unmarked), headers=PROTOBUF)
    health = client.get("/health").json()

    assert [response.status_code for response in (own, foreign, legacy)] == [200, 200, 200]
    assert [span["span_id"] for span in _trace(database, TRACE.hex())["spans"]] == [(1).to_bytes(8, "big").hex()], "another home's spans were stored"
    rejected = ExportTraceServiceResponse.FromString(foreign.content).partial_success
    assert (rejected.rejected_spans, rejected.error_message) == (2, "foreign_home")
    assert health["dropped"] == {"foreign_home": 3}


def test_duplicate_batch_and_span_count_once(receiver):
    client, app, database = receiver
    event = Span.Event(name="zeta.approval.requested", time_unix_nano=150, attributes=_attributes({"decision": "ask", "scope": "once"}))
    body = _body(_span(1, events=(event,)), _span(1, events=(event,)))

    first = client.post("/v1/traces", content=body, headers=PROTOBUF)
    sequence = app.state.writer.committed_sequence
    second = client.post("/v1/traces", content=body, headers=PROTOBUF)

    assert first.status_code == second.status_code == 200
    session = _sessions(database)["items"][0]
    assert session["span_count"] == 1
    assert session["turn_count"] == 1
    assert _trace(database, TRACE.hex())["trace"]["span_count"] == 1
    assert len(_rows(database, "span_events")) == 1
    assert sequence == 1
    assert app.state.writer.committed_sequence == _committed_sequence(database) == 1


def test_commit_failure_acknowledges_and_publishes_nothing(tmp_path, monkeypatch):
    crash = {"armed": False}

    class CrashingConnection(sqlite3.Connection):
        def commit(self) -> None:
            if crash["armed"]:
                raise sqlite3.OperationalError("disk I/O error")
            super().commit()

    monkeypatch.setattr(store_module.sqlite3, "connect", functools.partial(sqlite3.connect, factory=CrashingConnection))
    directory = tmp_path / "telemetry"
    directory.mkdir()
    app = create_app(directory, HOME_ID)
    database = directory / "telemetry.sqlite3"
    with TestClient(app, base_url="http://127.0.0.1:4323") as client:
        body = _body(_span(1), _span(2, parent=1, name="execute_tool read", attributes={"zeta.tool.name": "read"}))
        crash["armed"] = True
        failed = client.post("/v1/traces", content=body, headers=PROTOBUF)

        assert failed.status_code == 503
        assert app.state.writer.committed_sequence == 0
        assert _rows(database, "spans") == []
        assert _rows(database, "traces") == []
        assert _rows(database, "sessions") == []
        assert _committed_sequence(database) == 0

        crash["armed"] = False
        retried = client.post("/v1/traces", content=body, headers=PROTOBUF)

        assert retried.status_code == 200
        assert app.state.writer.committed_sequence == 1
        assert _sessions(database)["items"][0]["span_count"] == 2


def test_projection_is_identical_in_both_orders(tmp_path):
    other_trace = bytes.fromhex("ff112233445566778899aabbccddeeff")
    spans = [
        _span(1, start=100, end=900),  # parentless turn root
        _span(2, parent=1, name="invoke_agent worker", start=110, end=800, attributes={"zeta.agent.name": "worker"}),
        _span(3, parent=2, name="chat m-low", start=120, end=500, status=Status.STATUS_CODE_ERROR,
              attributes={"gen_ai.request.model": "m-low", "gen_ai.usage.input_tokens": 10, "gen_ai.usage.output_tokens": 2,
                          "gen_ai.usage.total_tokens": 12, "gen_ai.conversation.id": f"{ROOT}:1"}),
        _span(4, parent=2, name="chat m-high", start=130, end=500,
              attributes={"gen_ai.request.model": "m-high", "gen_ai.usage.input_tokens": 5, "gen_ai.usage.total_tokens": 5}),
        _span(5, parent=2, name="execute_tool read", start=140, end=150, attributes={"zeta.tool.name": "read"}),
        # same session, same end_ns and span_id as span 2, larger trace id: it wins every latest-value tie
        _span(2, trace=other_trace, name="invoke_agent reviewer", start=50, end=800, attributes={"zeta.agent.name": "reviewer"}),
        # ties span 1 on end_ns and span_id from the other trace: the larger trace id becomes the latest trace
        _span(1, trace=other_trace, parent=2, start=60, end=900),
    ]

    def ingest(order: list[Span], home: Path) -> tuple[list[tuple], list[tuple]]:
        home.mkdir()
        app = create_app(home, HOME_ID)
        app.state.writer._clock_ns = lambda: 0
        with TestClient(app, base_url="http://127.0.0.1:4323") as client:
            for span in order:
                assert client.post("/v1/traces", content=_body(span), headers=PROTOBUF).status_code == 200
        database = home / "telemetry.sqlite3"
        return _rows(database, "traces"), _rows(database, "sessions")

    forward = ingest(spans, tmp_path / "forward")
    backward = ingest(list(reversed(spans)), tmp_path / "backward")

    assert forward == backward
    traces, sessions = forward
    first = next(row for row in traces if row[0] == TRACE)
    assert first[1:] == (100, 900, (1).to_bytes(8, "big"), ROOT, "error", 5, 1)
    assert len(sessions) == 1
    session = dict(zip(
        ("root_session_id", "first_ns", "last_ns", "last_trace_id", "span_count", "turn_count", "model_count", "tool_count",
         "error_count", "input_tokens", "output_tokens", "total_tokens", "profile", "model", "last_span_id"),
        sessions[0],
    ))
    assert session["first_ns"] == 50
    assert session["last_ns"] == 900
    assert (session["span_count"], session["turn_count"], session["model_count"], session["tool_count"]) == (7, 2, 2, 1)
    assert (session["error_count"], session["input_tokens"], session["output_tokens"], session["total_tokens"]) == (1, 15, 2, 17)
    assert session["last_trace_id"] == other_trace
    assert session["profile"] == "reviewer"
    assert session["model"] == "m-high"


def _receiver_command(home: Path) -> tuple[list[str], dict[str, str]]:
    command = [sys.executable, "-c", "import sys; from zeta.cli.main import main; sys.exit(main())", "otel", "receiver"]
    return command, {**os.environ, "ZETA_HOME": str(home)}


def test_second_process_exits_and_bind_failure_releases_lock(tmp_path, builtin_port):
    owner_home, other_home = tmp_path / "owner", tmp_path / "other"
    command, environment = _receiver_command(owner_home)
    owner = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        ready, _, _ = select.select([owner.stdout], [], [], 20)
        assert ready, "receiver did not report ready"
        assert owner.stdout.readline().startswith(b"zeta otel receiver: http://127.0.0.1:4323/v1/traces")
        home_id = (owner_home / "telemetry" / "home-id").read_text().strip()
        # the ready line follows listen(), so this request waits in the backlog until uvicorn serves it
        connection = http.client.HTTPConnection("127.0.0.1", 4323, timeout=10)
        try:
            connection.request("GET", "/health")
            health = json.loads(connection.getresponse().read())
        finally:
            connection.close()
        assert health["home_id"] == home_id
        assert stat.S_IMODE((owner_home / "telemetry").stat().st_mode) == 0o700
        assert stat.S_IMODE((owner_home / "telemetry" / "home-id").stat().st_mode) == 0o600
        assert stat.S_IMODE((owner_home / "telemetry" / "telemetry.sqlite3").stat().st_mode) == 0o600

        same_home = subprocess.run(command, env=environment, capture_output=True, timeout=30, check=False)
        command_b, environment_b = _receiver_command(other_home)
        other = subprocess.run(command_b, env=environment_b, capture_output=True, timeout=30, check=False)

        assert same_home.returncode == 1
        assert same_home.stdout.decode().strip() == "already running at http://127.0.0.1:4323"
        assert other.returncode == 1
        assert b"cannot listen on 127.0.0.1:4323" in other.stderr
        assert not (other_home / "telemetry" / "telemetry.sqlite3").exists()
        descriptor = os.open(other_home / "telemetry" / "receiver.lock", os.O_RDWR)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)  # raises if bind failure kept the lock
        finally:
            os.close(descriptor)
    finally:
        owner.send_signal(signal.SIGTERM)
        try:
            owner.wait(timeout=10)
        except subprocess.TimeoutExpired:
            owner.kill()
            owner.wait(timeout=10)
        owner.stdout.close()
        owner.stderr.close()
    assert owner.returncode == -signal.SIGTERM  # uvicorn shuts down gracefully, then re-raises the signal


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_idle_receiver_exits_after_the_timeout(tmp_path):
    import asyncio
    import socket

    import uvicorn

    from zeta.telemetry.receiver.app import IdleExitServer, IdleTimer
    from zeta.telemetry.receiver.process import IDLE_EXIT_SECONDS

    clock = _Clock()
    idle = IdleTimer(IDLE_EXIT_SECONDS, clock)
    clock.now = IDLE_EXIT_SECONDS - 1
    assert not idle.expired()
    clock.now = IDLE_EXIT_SECONDS
    assert idle.expired()

    directory = tmp_path / "telemetry"
    directory.mkdir()
    app = create_app(directory, HOME_ID, idle)
    server = IdleExitServer(uvicorn.Config(app, log_level="warning", lifespan="on"), idle)
    listener = socket.create_server(("127.0.0.1", 0))
    try:
        asyncio.run(asyncio.wait_for(server.serve(sockets=[listener]), 10))  # returns only when the idle tick stops it
    finally:
        listener.close()
    assert server.started
    assert not app.state.writer._thread.is_alive()  # the lifespan closed the writer: a clean shutdown


@pytest.mark.parametrize("request_kind", ["health", "ingest"])
def test_health_or_ingest_resets_the_idle_timer(tmp_path, request_kind):
    from zeta.telemetry.receiver.app import IdleTimer
    from zeta.telemetry.receiver.process import IDLE_EXIT_SECONDS

    clock = _Clock()
    idle = IdleTimer(IDLE_EXIT_SECONDS, clock)
    directory = tmp_path / "telemetry"
    directory.mkdir()
    with TestClient(create_app(directory, HOME_ID, idle), base_url="http://127.0.0.1:4323") as client:
        clock.now = IDLE_EXIT_SECONDS - 1
        if request_kind == "health":
            assert client.get("/health").status_code == 200
        else:
            assert client.post("/v1/traces", content=_body(_span(1)), headers=PROTOBUF).status_code == 200
        clock.now = 2 * IDLE_EXIT_SECONDS - 2
        assert not idle.expired(), f"{request_kind} did not reset the idle timer"
        clock.now = 2 * IDLE_EXIT_SECONDS - 1
        assert idle.expired()


@pytest.mark.parametrize(("arguments", "exits_when_idle"), [([], False), (["--idle-exit"], True)], ids=["manual", "auto-started"])
def test_only_the_auto_started_receiver_idle_exits(tmp_path, monkeypatch, arguments, exits_when_idle):
    import socket

    from zeta.cli.main import main
    from zeta.telemetry.receiver import process
    from zeta.telemetry.receiver.app import IdleTimer

    served: list[float] = []
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    create_server = socket.create_server
    monkeypatch.setattr(process.socket, "create_server", lambda address: create_server(("127.0.0.1", 0)))  # not the fixed port
    # test-contract: seam _serve takes the idle timeout the CLI chose; a manual receiver can only be shown not to expire
    monkeypatch.setattr(process, "_serve", lambda listener, directory, home_id, idle_seconds: served.append(idle_seconds) or 0)

    assert main(["otel", "receiver", *arguments]) == 0
    clock = _Clock()
    idle = IdleTimer(served[0], clock)
    clock.now = 10.0**12
    assert idle.expired() is exits_when_idle
