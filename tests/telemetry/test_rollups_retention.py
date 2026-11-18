from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path

from tests.support.owned import await_or_fail
from zeta.telemetry.receiver import reader, store
from zeta.telemetry.receiver.projector import SPAN_COLUMNS
from zeta.telemetry.receiver.store import Writer, open_database

DAY = 86_400 * 10**9


def span(trace: int, start: int, *, operation: str = "chat", duration: int = 10) -> dict:
    row = dict.fromkeys(SPAN_COLUMNS)
    row.update(
        trace_id=trace.to_bytes(16, "big"), span_id=(trace + 1).to_bytes(8, "big"),
        parent_span_id=None, operation=operation, kind=1, start_ns=start,
        end_ns=start + duration, duration_ns=duration, status="ok", session_id=None,
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


def test_schema_v2_migrates_to_v3_without_losing_rows(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    open_database(path)
    with sqlite3.connect(path) as db:
        db.execute("DROP TABLE hourly_rollups")
        db.execute("DROP TABLE span_fingerprints")
        db.execute("PRAGMA user_version = 2")
        db.execute("UPDATE viewer_meta SET schema_version = 2")
        db.commit()
    open_database(path)
    with sqlite3.connect(path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3
        assert db.execute("SELECT name FROM sqlite_master WHERE name = 'hourly_rollups'").fetchone()
        assert db.execute("SELECT schema_version FROM viewer_meta").fetchone()[0] == 3


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
        before = db.execute("SELECT sum(calls), sum(duration_ns) FROM hourly_rollups WHERE dimension='operation'").fetchone()
        assert before == (2, 60)
        assert db.execute("SELECT calls, duration_ns FROM hourly_rollups WHERE dimension='operation' ORDER BY hour_ns").fetchall() == [(1, 20), (1, 40)]
    asyncio.run(write(path, [old], now))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT sum(calls), sum(duration_ns) FROM hourly_rollups WHERE dimension='operation'").fetchone() == before
    status = reader.status(home, None, {"receiver": None, "last_export": None, "last_export_ns": None})
    assert status["oldest_raw_ns"] == recent["start_ns"]
    assert status["oldest_rollup_ns"] is not None


def test_pruning_deletes_the_oldest_chunk_first(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "telemetry.sqlite3"
    open_database(path)
    now = 200 * DAY
    monkeypatch.setattr(store, "PRUNE_CHUNK", 1)
    asyncio.run(write(path, [span(1, now - 20 * DAY), span(2, now - 19 * DAY)], now))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT trace_id FROM spans").fetchall() == [((2).to_bytes(16, "big"),)]


async def _cap_write(path: Path, now: int, cap: int) -> None:
    writer = Writer(path, clock_ns=lambda: now)
    writer.start()
    try:
        await await_or_fail(writer.submit([span(1, now - DAY)]), what="cap commit")
        path.with_name(path.name + "-wal").write_bytes(b"x" * 100)
        store.MAX_BYTES = cap
        writer._enforce_size(writer._connection)
        assert writer._connection.execute("SELECT count(*) FROM spans").fetchone()[0] == 0
    finally:
        await asyncio.to_thread(writer.close)


def test_size_cap_counts_wal_bytes(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "telemetry.sqlite3"
    open_database(path)
    cap = path.stat().st_size + 1
    monkeypatch.setattr(store, "MAX_BYTES", cap)
    asyncio.run(_cap_write(path, 200 * DAY, cap))


def test_rollup_dimensions_and_histogram_are_bounded(tmp_path: Path) -> None:
    path = tmp_path / "telemetry.sqlite3"
    open_database(path)
    asyncio.run(write(path, [span(i, i + 1, duration=i + 1) for i in range(1, 100)], 1_000_000))
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM hourly_rollups WHERE dimension='kind'").fetchone()[0] == 1
        histogram = db.execute("SELECT histogram FROM hourly_rollups LIMIT 1").fetchone()[0]
        assert len(__import__("json").loads(histogram)) == 64
