from __future__ import annotations

import asyncio
import json
import multiprocessing
import queue
import threading
from datetime import UTC, datetime
from pathlib import Path

from zeta.diagnostics.tui_stalls import (
    TUIStallWatchdog,
    start_tui_stall_watchdog,
)


def _frame(function: str, *, file: str = "/app/work.py"):
    from types import SimpleNamespace

    return SimpleNamespace(
        f_code=SimpleNamespace(co_filename=file, co_name=function),
        f_lineno=10,
        f_back=None,
    )


def _empty_samples():
    return {}, {}


def _append_records(path: str, attempts: int, result) -> None:
    watchdog = TUIStallWatchdog(Path(path), session_id="writer", max_bytes=2_000)
    errors = []
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
            errors.append(repr(exc))
    result.put(errors)


async def test_blocked_loop_writes_one_record_with_blocking_stack(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    probe_ready = threading.Event()
    release_probe = threading.Event()
    loop_blocked = threading.Event()
    release_loop: queue.SimpleQueue[None] = queue.SimpleQueue()
    record_written = threading.Event()

    def before_probe() -> None:
        probe_ready.set()
        if not release_probe.wait(timeout=5):
            raise TimeoutError("probe release was not signaled")

    def after_sample() -> None:
        if loop_blocked.is_set():
            release_loop.put(None)

    def block_loop() -> None:
        # SimpleQueue.get waits in C, so block_loop stays the innermost Python
        # frame; stack trimming keeps only Zeta, stdlib and innermost frames.
        loop_blocked.set()
        try:
            release_loop.get(timeout=5)
        except queue.Empty:
            raise TimeoutError("loop release was not signaled") from None

    watchdog = TUIStallWatchdog(
        path,
        session_id="session-123",
        probe_interval=0.01,
        stall_threshold=0.05,
        sample_interval=0.02,
        utc_now=lambda: datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        before_probe=before_probe,
        after_sample=after_sample,
    )
    append = watchdog._append_record

    def append_and_signal(record: dict[str, object]) -> None:
        append(record)
        record_written.set()

    watchdog._append_record = append_and_signal  # type: ignore[method-assign]
    watchdog.start(asyncio.get_running_loop())
    try:
        assert await asyncio.to_thread(probe_ready.wait, 5)
        asyncio.get_running_loop().call_soon(block_loop)
        release_probe.set()
        assert await asyncio.to_thread(record_written.wait, 5)
    finally:
        release_probe.set()
        release_loop.put(None)
        await watchdog.close()

    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 1
    assert records[0]["start_time"] == "2026-01-02T03:04:05Z"
    assert records[0]["duration_ms"] >= 50
    assert records[0]["session_id"] == "session-123"
    assert any(
        frame["function"] == "block_loop"
        for stack in records[0]["stacks"]
        for frame in stack["frames"]
    )


def test_idle_executor_worker_has_no_busy_samples(tmp_path: Path) -> None:
    watchdog = TUIStallWatchdog(
        tmp_path / "tui-stalls.jsonl",
        session_id="idle-executor",
        frames=lambda: {
            1: _frame("block_loop"),
            2: _frame(
                "_worker",
                file="/usr/lib/python/concurrent/futures/thread.py",
            ),
        },
        thread_names=lambda: {1: "loop", 2: "idle-executor_0"},
    )
    watchdog._loop_thread_id = 1
    stacks, other_threads = _empty_samples()

    watchdog._sample_stacks(stacks, other_threads)
    record = watchdog._record(
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        0.4,
        stacks,
        other_threads,
        1,
    )

    assert record["other_threads"][0]["name"] == "idle-executor_0"
    assert record["other_threads"][0]["busy_samples"] == 0
    assert record["other_threads"][0]["stacks"][0]["idle"] is True


def test_cpu_bound_worker_is_recorded_during_loop_starvation(
    tmp_path: Path,
) -> None:
    watchdog = TUIStallWatchdog(
        tmp_path / "tui-stalls.jsonl",
        session_id="cpu-worker",
        frames=lambda: {
            1: _frame("block_loop"),
            2: _frame("reconcile"),
            3: _frame("wait"),
        },
        thread_names=lambda: {
            1: "loop",
            2: "reconciliation-worker",
            3: "idle-worker",
        },
    )
    watchdog._loop_thread_id = 1
    stacks, other_threads = _empty_samples()

    for _ in range(3):
        watchdog._sample_stacks(stacks, other_threads)
    record = watchdog._record(
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        0.4,
        stacks,
        other_threads,
        2,
    )

    assert [thread["name"] for thread in record["other_threads"]] == [
        "reconciliation-worker",
        "idle-worker",
    ]
    assert [thread["busy_samples"] for thread in record["other_threads"]] == [3, 0]
    assert (
        record["other_threads"][0]["stacks"][0]["frames"][0]["function"] == "reconcile"
    )


def test_many_threads_keep_bounded_busy_evidence(tmp_path: Path) -> None:
    frames = {index: _frame("wait") for index in range(2, 102)}
    frames.update({index: _frame("work") for index in range(102, 105)})
    names = {index: f"idle-worker-{index}" for index in range(2, 102)}
    names.update({index: f"busy-worker-{index}" for index in range(102, 105)})
    frames[1] = _frame("block_loop")
    names[1] = "loop"
    watchdog = TUIStallWatchdog(
        tmp_path / "tui-stalls.jsonl",
        session_id="many-threads",
        frames=lambda: frames,
        thread_names=lambda: names,
    )
    watchdog._loop_thread_id = 1
    stacks, other_threads = _empty_samples()
    observed_thread_count = 0

    for _ in range(3):
        observed_thread_count = max(
            observed_thread_count,
            watchdog._sample_stacks(stacks, other_threads),
        )
    record = watchdog._record(
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        0.4,
        stacks,
        other_threads,
        observed_thread_count,
    )

    assert len(record["other_threads"]) == 16
    assert record["threads_truncated"] == 87
    assert {thread["name"] for thread in record["other_threads"][:3]} == {
        "busy-worker-102",
        "busy-worker-103",
        "busy-worker-104",
    }
    assert all(thread["busy_samples"] == 3 for thread in record["other_threads"][:3])


def test_logging_failure_does_not_stop_watchdog(tmp_path: Path) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="retry",
        stall_threshold=0.25,
        clock=lambda: 1.0,
    )
    append = watchdog._append_record
    attempts = 0

    def fail_once(record: dict[str, object]) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise OSError("injected write failure")
        append(record)

    watchdog._append_record = fail_once  # type: ignore[method-assign]
    stacks, other_threads = _empty_samples()
    watchdog._finish_probe(
        0.5,
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        stacks,
        other_threads,
        0,
    )
    assert not watchdog._stop.is_set()
    watchdog._finish_probe(
        0.5,
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        stacks,
        other_threads,
        0,
    )

    assert attempts == 2
    assert path.exists()


def test_responsive_loop_writes_no_record(tmp_path: Path) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    watchdog = TUIStallWatchdog(
        path,
        session_id="responsive",
        stall_threshold=1.0,
        clock=lambda: 10.0,
    )
    stacks, other_threads = _empty_samples()

    watchdog._finish_probe(
        9.5,
        datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        stacks,
        other_threads,
        0,
    )

    assert not path.exists()


async def test_opt_out_starts_no_thread(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("ZETA_STALL_WATCHDOG", "0")

    started = []

    def record_start(thread) -> None:
        started.append(thread)

    monkeypatch.setattr(threading.Thread, "start", record_start)
    watchdog = start_tui_stall_watchdog(
        asyncio.get_running_loop(), zeta_home=tmp_path, session_id="disabled"
    )

    assert watchdog is None
    assert started == []


def test_concurrent_processes_write_valid_bounded_logs(tmp_path: Path) -> None:
    path = tmp_path / "tui-stalls.jsonl"
    context = multiprocessing.get_context("spawn")
    results = [context.Queue() for _ in range(4)]
    processes = [
        context.Process(target=_append_records, args=(str(path), 100, result))
        for result in results
    ]

    try:
        for process in processes:
            process.start()
        errors = [result.get(timeout=15) for result in results]
    finally:
        for process in processes:
            process.join(15)
            if process.is_alive():
                process.terminate()
                process.join(5)

    assert all(process.exitcode == 0 for process in processes)
    assert errors == [[], [], [], []]
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
    probe_ready = threading.Event()
    release_probe = threading.Event()

    def before_probe() -> None:
        probe_ready.set()
        if not release_probe.wait(timeout=5):
            raise TimeoutError("probe release was not signaled")

    watchdog = TUIStallWatchdog(
        tmp_path / "tui-stalls.jsonl",
        session_id="closing",
        probe_interval=0.001,
        before_probe=before_probe,
    )
    watchdog.start(asyncio.get_running_loop())
    assert await asyncio.to_thread(probe_ready.wait, 5)
    thread = watchdog._thread
    assert thread is not None and thread.is_alive()

    release_probe.set()
    await watchdog.close()

    assert not thread.is_alive()
