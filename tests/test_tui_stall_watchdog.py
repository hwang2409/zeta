from __future__ import annotations

import asyncio
import concurrent.futures
import json
import multiprocessing
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


def _spin_until(stopped: threading.Event) -> None:
    while not stopped.is_set():
        pass


def _append_records(path: str, attempts: int, errors) -> None:
    watchdog = TUIStallWatchdog(Path(path), session_id="writer", max_bytes=2_000)
    record = {
        "start_time": "2026-01-02T03:04:05Z",
        "duration_ms": 400,
        "session_id": "writer",
        "stacks": [],
        "other_threads": [],
    }
    for _ in range(attempts):
        try:
            watchdog._append_record(record)
        except Exception as exc:  # noqa: BLE001 - returned to the parent for assertion
            errors.put(repr(exc))


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


async def test_idle_executor_worker_has_no_busy_samples(tmp_path: Path) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="idle-executor",
        probe_interval=0.002,
        stall_threshold=0.01,
        sample_interval=0.002,
    )
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="idle-executor",
    ) as executor:
        executor.submit(lambda: None).result()
        watchdog.start(asyncio.get_running_loop())
        done = asyncio.Event()
        asyncio.get_running_loop().call_soon(_deliberately_block_loop, done)
        await done.wait()
        await watchdog.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    executor_threads = [
        thread
        for record in records
        for thread in record["other_threads"]
        if thread["name"].startswith("idle-executor")
    ]
    assert executor_threads
    assert all(thread["busy_samples"] == 0 for thread in executor_threads)


async def test_cpu_bound_worker_is_recorded_during_loop_starvation(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="cpu-worker",
        probe_interval=0.002,
        stall_threshold=0.01,
        sample_interval=0.002,
    )
    stopped = threading.Event()
    worker = threading.Thread(
        target=_spin_until,
        args=(stopped,),
        name="reconciliation-worker",
    )
    watchdog.start(asyncio.get_running_loop())
    worker.start()
    done = asyncio.Event()
    asyncio.get_running_loop().call_soon(_deliberately_block_loop, done)
    try:
        await done.wait()
    finally:
        stopped.set()
        worker.join()
        await watchdog.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    worker_records = [
        thread
        for record in records
        for thread in record["other_threads"]
        if thread["name"] == "reconciliation-worker"
    ]
    assert worker_records
    assert all(thread["busy_samples"] > 0 for thread in worker_records)
    # Other harness threads (e.g. pytest-xdist I/O) can also be busy, so assert
    # the ranking rule instead of a fixed first place: threads are ordered by
    # busy evidence, so the busy worker never ranks below an idler thread.
    for record in records:
        busy = [thread["busy_samples"] for thread in record["other_threads"]]
        assert busy == sorted(busy, reverse=True)
    assert any(
        frame["function"] == "_spin_until"
        for thread in worker_records
        for stack in thread["stacks"]
        for frame in stack["frames"]
    )


def test_many_threads_keep_bounded_busy_evidence(tmp_path: Path) -> None:
    watchdog = TUIStallWatchdog(
        tmp_path / "tui-stalls.jsonl",
        session_id="many-threads",
    )
    watchdog._loop_thread_id = threading.get_ident()
    stopped = threading.Event()
    idle_threads = [
        threading.Thread(
            target=stopped.wait,
            name=f"idle-worker-{index}",
        )
        for index in range(100)
    ]
    busy_threads = [
        threading.Thread(
            target=_spin_until,
            args=(stopped,),
            name=f"busy-worker-{index}",
        )
        for index in range(3)
    ]
    for thread in [*idle_threads, *busy_threads]:
        thread.start()
    other_threads = {}
    observed_thread_count = 0
    try:
        for _ in range(3):
            observed_thread_count = max(
                observed_thread_count,
                watchdog._sample_stacks({}, other_threads),
            )
    finally:
        stopped.set()
        for thread in [*idle_threads, *busy_threads]:
            thread.join()

    record = watchdog._record(
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        0.4,
        {},
        other_threads,
        observed_thread_count,
    )
    assert len(record["other_threads"]) <= 16
    assert record["threads_truncated"] >= 87
    assert {
        thread["name"]
        for thread in record["other_threads"][:3]
    } == {"busy-worker-0", "busy-worker-1", "busy-worker-2"}
    assert all(
        thread["busy_samples"] > 0
        for thread in record["other_threads"][:3]
    )


async def test_logging_failure_does_not_stop_watchdog(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="retry",
        probe_interval=0.005,
        stall_threshold=0.02,
        sample_interval=0.01,
    )
    append = watchdog._append_record
    attempts = 0

    def fail_once(record: dict[str, object]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected write failure")
        append(record)

    monkeypatch.setattr(watchdog, "_append_record", fail_once)
    watchdog.start(asyncio.get_running_loop())
    for _ in range(2):
        done = asyncio.Event()
        asyncio.get_running_loop().call_soon(_deliberately_block_loop, done)
        await done.wait()
        await asyncio.sleep(0.02)
    await watchdog.close()

    assert attempts >= 2
    assert path.exists()


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


def test_concurrent_processes_write_valid_bounded_logs(tmp_path: Path) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    context = multiprocessing.get_context("spawn")
    errors = context.Queue()
    processes = [
        context.Process(target=_append_records, args=(str(path), 100, errors))
        for _ in range(4)
    ]

    for process in processes:
        process.start()
    for process in processes:
        process.join(15)

    assert all(process.exitcode == 0 for process in processes)
    assert errors.empty()
    files = [path, path.with_name("tui-stalls.jsonl.1")]
    assert all(log.stat().st_size <= 2_000 for log in files)
    for log in files:
        for line in log.read_text().splitlines():
            assert json.loads(line)["session_id"] == "writer"


def test_oversized_record_is_reduced_and_written(tmp_path: Path) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="reduced",
        max_bytes=400,
    )
    frames = [
        {"file": f"/long/path/to/module-{index}.py", "line": index, "function": "work"}
        for index in range(20)
    ]
    record = {
        "start_time": "2026-01-02T03:04:05Z",
        "duration_ms": 400,
        "session_id": "reduced",
        "stacks": [{"count": 1, "frames": frames} for _ in range(8)],
        "other_threads": [],
        "threads_truncated": 0,
    }

    watchdog._append_record(record)

    assert path.stat().st_size <= 400
    written = json.loads(path.read_text())
    assert written["session_id"] == "reduced"
    assert sum(len(stack["frames"]) for stack in written["stacks"]) < 160


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
