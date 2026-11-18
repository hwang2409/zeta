from __future__ import annotations

import asyncio
import contextlib
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from tests.support.owned import await_or_fail
from zeta.telemetry.receiver import reader, store
from zeta.telemetry.receiver.projector import SPAN_COLUMNS
from zeta.telemetry.receiver.store import Writer, open_database

DAY = 86_400 * 10**9


def span(trace: int, start: int, *, operation: str = "chat", duration: int = 10, session: str | None = None) -> dict:
    row = dict.fromkeys(SPAN_COLUMNS)
    row.update(
        trace_id=trace.to_bytes(16, "big"), span_id=(trace + 1).to_bytes(8, "big"),
        parent_span_id=None, operation=operation, kind=1, start_ns=start,
        end_ns=start + duration, duration_ns=duration, status="ok", session_id=session,
        service_version="test", scope_version="test", events=[], provider="p",
        request_model="m", input_tokens=2, output_tokens=3, total_tokens=5,
    )
    return row


async def write(path: Path, rows: list[dict], clock: int) -> None:
    writer = Writer(path, clock_ns=lambda: clock)
    writer.start()
    try:
        await await_or_fail(writer.submit(rows), what="rollup commit")
    finally:
        await asyncio.to_thread(writer.close)


async def submit(writer: Writer, rows: list[dict], what: str) -> int:
    return await await_or_fail(writer.submit(rows), what=what)


def test_rollups_survive_raw_age_pruning_and_duplicates_do_not_change_them(tmp_path: Path) -> None:
    home = tmp_path / "home"
    directory = home / "telemetry"
    directory.mkdir(parents=True)
    path = directory / "telemetry.sqlite3"
    open_database(path)
    now = 200 * DAY
    old = span(1, now - 15 * DAY, duration=20)
    recent = span(2, now - DAY, duration=40)
    asyncio.run(write(path, [old, recent], now))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM spans").fetchone()[0] == 1
        before = db.execute("SELECT sum(spans), sum(duration_ns) FROM hourly_series WHERE group_by='operation'").fetchone()
        assert before == (2, 60)
        assert db.execute("SELECT spans, duration_ns FROM hourly_series WHERE group_by='operation' ORDER BY hour_ns").fetchall() == [(1, 20), (1, 40)]
    asyncio.run(write(path, [old], now))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT sum(spans), sum(duration_ns) FROM hourly_series WHERE group_by='operation'").fetchone() == before
    status = reader.status(home, None, {"receiver": None, "last_export": None, "last_export_ns": None})
    assert status["oldest_raw_ns"] == recent["start_ns"]
    assert status["oldest_rollup_ns"] is not None


def test_non_chat_model_metadata_does_not_create_model_rows(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    open_database(path)
    row = span(1, 1, operation="turn")
    writer = Writer(path, clock_ns=lambda: 1)
    writer.start()
    try:
        asyncio.run(submit(writer, [row], "family purity commit"))
        assert writer._connection.execute("SELECT count(*) FROM hourly_models").fetchone()[0] == 0
    finally:
        asyncio.run(asyncio.to_thread(writer.close))


def test_rollup_dimensions_and_histogram_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    open_database(path)
    asyncio.run(write(path, [span(i, i + 1, duration=i + 1) for i in range(1, 100)], 1_000_000))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM hourly_series WHERE group_by='operation'").fetchone()[0] == 1
        assert db.execute("SELECT count(*) FROM hourly_latency").fetchone()[0] > 0


NOW = 200 * DAY
A, B = "a" * 32, "b" * 32


def _measured_minus_one(path: Path) -> int:
    return store._database_bytes(path) - 1


@dataclass(frozen=True)
class Case:
    """One storage-bounds contract row: seed, start the receiver, run ``maintenance_step`` at ``NOW``, check."""

    id: str
    expect: dict[str, int]  # a table's row count, or a ``PRAGMA``/``SELECT`` scalar
    result: dict[str, Any] = field(default_factory=dict)  # a subset of the last step's result
    seed: tuple[dict, ...] = ()  # spans written before startup, at clock 0
    schema: int = 3  # below 3: an auto_vacuum=0 file of that schema holds the seed spans
    start_at_now: bool = False  # startup catch-up runs at NOW instead of clock 0
    chunk: int = store.PRUNE_CHUNK
    orphans: int = 0  # traces and sessions without spans, inserted after startup
    cap: Callable[[Path], int] | None = None
    steps: int = 0
    late: tuple[dict, ...] = ()  # spans submitted again after the steps


CASES = (
    Case("raw age expiry keeps younger spans", seed=(span(1, NOW - 15 * DAY), span(3, NOW - DAY)), steps=1,
         expect={"spans": 1, "span_fingerprints": 2, "traces": 1}, result={"age_family": "raw"}),
    Case("fingerprint age expiry after raw", seed=(span(1, NOW - 100 * DAY),), steps=2,
         expect={"spans": 0, "span_fingerprints": 0, "hourly_series": 3}, result={"age_family": "fingerprints"}),
    Case("rollup age expiry keeps rollups younger than 90 days", seed=(span(1, NOW - 100 * DAY), span(3, NOW - 30 * DAY)), steps=7,
         expect={"spans": 0, "span_fingerprints": 1, "hourly_series": 3, "hourly_distinct": 3, "hourly_models": 1, "hourly_latency": 1},
         result={"age_family": None, "caught_up": True}),
    Case("size chunk deletes raw spans first", seed=(span(1, NOW - DAY), span(3, NOW - DAY)), chunk=1, cap=_measured_minus_one, steps=1,
         expect={"spans": 1, "span_fingerprints": 2}, result={"size_family": "raw", "size_rows": 1}),
    Case("cap counts WAL and SHM bytes", seed=(span(1, NOW - DAY),), cap=lambda path: path.stat().st_size + 1, steps=1,
         expect={"span_fingerprints": 1}, result={"over_cap": True}),
    Case("stall at a 172,031-byte cap keeps the other families", seed=(span(1, NOW - DAY, session=A),), cap=lambda path: 172_031, steps=1,
         expect={"spans": 0, "traces": 0, "sessions": 0, "span_fingerprints": 1, "hourly_series": 3},
         result={"size_family": "raw", "over_cap": True, "stalled": True}),
    Case("no second size victim after a stall", seed=(span(1, NOW - DAY), span(3, NOW - DAY)), chunk=1, cap=_measured_minus_one, steps=2,
         expect={"spans": 1, "span_fingerprints": 2}, result={"size_rows": 0, "over_cap": True, "stalled": True}),
    Case("size chunk that frees pages does not stall", seed=tuple(span(2 * i + 1, NOW - DAY) for i in range(400)), chunk=200,
         cap=_measured_minus_one, steps=1, expect={"spans": 200}, result={"size_rows": 200, "stalled": False}),
    Case("orphan sweep is bounded by the chunk", orphans=3, chunk=1, steps=1, expect={"traces": 2, "sessions": 2}),
    Case("orphan sweep completes over steps", orphans=3, chunk=1, steps=3, expect={"traces": 0, "sessions": 0}),
    Case("expired span removes its session projection in the same step",
         seed=(span(1, NOW - 15 * DAY, session=A), span(3, NOW - DAY, session=B)), chunk=1, steps=1,
         expect={"spans": 1, "traces": 1, "sessions": 1}),
    Case("schema 1 migrates to incremental vacuum", schema=1, seed=(span(1, NOW - DAY),),
         expect={"PRAGMA auto_vacuum": 2, "PRAGMA user_version": 3, "spans": 1, "hourly_series": 3}),
    Case("schema 2 migrates to incremental vacuum", schema=2, seed=(span(1, NOW - DAY),),
         expect={"PRAGMA auto_vacuum": 2, "PRAGMA user_version": 3, "spans": 1, "hourly_series": 3}),
    Case("late duplicate after a migrated prune is not recounted", schema=2, seed=(span(1, NOW - 30 * DAY),), start_at_now=True,
         late=(span(1, NOW - 30 * DAY),),
         expect={"spans": 0, "span_fingerprints": 1, "SELECT sum(spans) FROM hourly_series WHERE group_by='operation'": 1}),
    Case("startup catches up over several steps", seed=tuple(span(2 * i + 1, NOW - 20 * DAY) for i in range(3)), chunk=1,
         start_at_now=True, expect={"spans": 0, "span_fingerprints": 3}),
)


def _old_database(path: Path, schema: int, rows: tuple[dict, ...]) -> None:
    """A schema-1 or schema-2 file as older receivers wrote it: no incremental auto-vacuum."""
    db = sqlite3.connect(path)
    try:
        db.executescript(store.SCHEMA)
        if schema == 2:
            for statement in store.MIGRATIONS_2:
                db.execute(statement)
        columns = [column[1] for column in db.execute("PRAGMA table_info(spans)")]
        db.executemany(f"INSERT INTO spans ({','.join(columns)}) VALUES ({','.join('?' * len(columns))})",
                       [[row.get(column) for column in columns] for row in rows])
        db.execute("INSERT INTO viewer_meta (singleton, schema_version) VALUES (1, ?)", (schema,))
        db.execute(f"PRAGMA user_version = {schema}")
        db.commit()
        assert db.execute("PRAGMA auto_vacuum").fetchone()[0] == 0
    finally:
        db.close()


async def _drive(case: Case, directory: Path, monkeypatch) -> tuple[dict | None, list[tuple[bool, bool]]]:
    from zeta.telemetry.receiver.app import create_app

    path = directory / "telemetry.sqlite3"
    loop, loop_thread = asyncio.get_running_loop(), threading.current_thread()
    calls: list[tuple[bool, bool]] = []  # per startup call: (off the loop thread, the loop ran a callback meanwhile)

    def recorded(function: Callable[..., Any]) -> Callable[..., Any]:
        def call(*args: Any) -> Any:
            off_loop, loop_ran = threading.current_thread() is not loop_thread, False
            if off_loop:
                with contextlib.suppress(TimeoutError):
                    loop_ran = asyncio.run_coroutine_threadsafe(asyncio.sleep(0, True), loop).result(timeout=10)
            calls.append((off_loop, loop_ran))
            return function(*args)
        return call

    monkeypatch.setattr(store, "open_database", recorded(store.open_database))
    monkeypatch.setattr(Writer, "maintenance_step", recorded(Writer.maintenance_step))
    app = create_app(directory, "f" * 32)
    writer = app.state.writer
    writer._clock_ns = lambda: NOW if case.start_at_now else 0
    result = None
    async with app.router.lifespan_context(app):
        startup_calls = list(calls)
        connection = writer._connection
        for index in range(case.orphans):
            trace = (1_000 + index).to_bytes(16, "big")
            connection.execute("INSERT INTO traces (trace_id, first_ns, last_ns, status) VALUES (?, 1, 1, 'ok')", (trace,))
            connection.execute("INSERT INTO sessions (root_session_id, first_ns, last_ns, last_trace_id, last_span_id) "
                               "VALUES (?, 1, 1, ?, ?)", (f"{index:032x}", trace, bytes(8)))
        connection.commit()
        if case.cap is not None:
            monkeypatch.setattr(store, "MAX_BYTES", case.cap(path))
        for _ in range(case.steps):
            result = writer.maintenance_step(NOW)
        if case.late:
            await submit(writer, list(case.late), "late duplicate commit")
    return result, startup_calls


@pytest.mark.parametrize("case", CASES, ids=[case.id for case in CASES])
def test_storage_bounds_contract(case: Case, tmp_path: Path, monkeypatch) -> None:
    pytest.importorskip("starlette")
    path = tmp_path / "telemetry.sqlite3"
    monkeypatch.setattr(store, "PRUNE_CHUNK", case.chunk)
    if case.schema < 3:
        _old_database(path, case.schema, case.seed)
    elif case.seed:
        open_database(path)
        asyncio.run(write(path, list(case.seed), 0))
    result, startup_calls = asyncio.run(_drive(case, tmp_path, monkeypatch))
    if case.result:
        assert result is not None and {key: result[key] for key in case.result} == case.result
    with contextlib.closing(sqlite3.connect(path)) as db:
        actual = {key: db.execute(key if key.startswith(("PRAGMA", "SELECT")) else f"SELECT count(*) FROM {key}").fetchone()[0]
                  for key in case.expect}
    assert actual == case.expect
    # migration and catch-up ran on the writer thread, never on the event loop, and the loop kept running meanwhile
    assert startup_calls and set(startup_calls) == {(True, True)}
