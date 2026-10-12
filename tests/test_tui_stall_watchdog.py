from __future__ import annotations

import asyncio
import json
import threading
import time
from datetime import UTC, datetime
from pathlib import Path

from zeta.diagnostics.tui_stalls import (
    TUIStallWatchdog,
    start_tui_stall_watchdog,
)


def _deliberately_block_loop(done: asyncio.Event) -> None:
    time.sleep(0.4)
    done.set()


async def test_blocked_loop_writes_one_record_with_blocking_stack(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="session-123",
        probe_interval=0.01,
        stall_threshold=0.05,
        sample_interval=0.02,
        utc_now=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
    )
    watchdog.start(asyncio.get_running_loop())
    done = asyncio.Event()
    asyncio.get_running_loop().call_soon(_deliberately_block_loop, done)

    await done.wait()
    await watchdog.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["start_time"] == "2026-01-02T03:04:05Z"
    assert records[0]["duration_ms"] >= 300
    assert records[0]["session_id"] == "session-123"
    assert any(
        frame["function"] == "_deliberately_block_loop"
        for stack in records[0]["stacks"]
        for frame in stack["frames"]
    )


async def test_responsive_loop_writes_no_record(tmp_path: Path) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="responsive",
        probe_interval=0.001,
        stall_threshold=10,
        sample_interval=1,
        clock=lambda: 10.0,
    )
    watchdog.start(asyncio.get_running_loop())

    for _ in range(5):
        await asyncio.sleep(0)
    await watchdog.close()

    assert not path.exists()


async def test_opt_out_starts_no_thread(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZETA_STALL_WATCHDOG", "0")
    before = {thread.ident for thread in threading.enumerate()}

    watchdog = start_tui_stall_watchdog(
        asyncio.get_running_loop(), zeta_home=tmp_path, session_id="disabled"
    )

    assert watchdog is None
    assert {thread.ident for thread in threading.enumerate()} == before


def test_log_files_stay_within_size_bound(tmp_path: Path) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="bounded",
        max_bytes=400,
    )
    record = {
        "start_time": "2026-01-02T03:04:05Z",
        "duration_ms": 400,
        "session_id": "bounded",
        "stacks": [],
    }

    for _ in range(20):
        watchdog._append_record(record)

    assert path.stat().st_size <= 400
    assert path.with_name("tui-stalls.jsonl.1").stat().st_size <= 400


async def test_close_stops_watchdog_thread(tmp_path: Path) -> None:
    watchdog = TUIStallWatchdog(
        tmp_path / "tui-stalls.jsonl",
        session_id="closing",
        probe_interval=0.001,
    )
    watchdog.start(asyncio.get_running_loop())
    thread = watchdog._thread
    assert thread is not None and thread.is_alive()

    await watchdog.close()

    assert not thread.is_alive()
