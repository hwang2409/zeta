from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

from zeta.cli.main import main
from zeta.stall_trace import StallWatchdog, read_stall_summary, render_stall_summary


def _deliberate_block() -> None:
    time.sleep(0.3)


async def test_watchdog_records_blocking_stack(tmp_path: Path) -> None:
    path = tmp_path / "stalls.jsonl"
    watchdog = StallWatchdog(path, threshold_seconds=0.05, max_bytes=100_000)
    watchdog.start(asyncio.get_running_loop())
    await asyncio.sleep(0.04)

    _deliberate_block()

    await asyncio.sleep(0.1)
    watchdog.close()
    records = [json.loads(line) for line in path.read_text().splitlines()]
    stall = next(record for record in records if record["kind"] == "loop_stall")
    assert stall["duration_ms"] >= 200
    assert any(frame["function"] == "_deliberate_block" for frame in stall["stack"])


def test_watchdog_is_noop_without_opt_in(
    tmp_path: Path, monkeypatch,
) -> None:
    monkeypatch.delenv("ZETA_STALL_TRACE", raising=False)
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))

    assert StallWatchdog.from_environment() is None
    assert not (tmp_path / "logs" / "stalls.jsonl").exists()


def test_stall_log_rotates_at_bound(tmp_path: Path) -> None:
    path = tmp_path / "stalls.jsonl"
    watchdog = StallWatchdog(path, threshold_seconds=0.05, max_bytes=500)
    for index in range(20):
        watchdog._write_record(
            {
                "kind": "loop_stall",
                "timestamp": index,
                "duration_ms": 100 + index,
                "stack": [{"file": "work.py", "line": index, "function": "busy"}],
            }
        )

    assert path.stat().st_size <= 500
    assert path.with_name("stalls.jsonl.1").stat().st_size <= 500


def test_stall_summary_aggregates_durations_and_stacks(tmp_path: Path) -> None:
    path = tmp_path / "stalls.jsonl"
    records = [
        {
            "kind": "loop_stall",
            "duration_ms": duration,
            "stack": [
                {"file": "app.py", "line": 10, "function": "paint"},
                {
                    "file": "/src/zeta/stall_trace.py",
                    "line": 164,
                    "function": "_gc_callback",
                },
            ],
        }
        for duration in (100, 200, 300, 400)
    ]
    records.append(
        {
            "kind": "loop_stall",
            "duration_ms": 500,
            "stack": [{"file": "store.py", "line": 20, "function": "load"}],
        }
    )
    records.append({"kind": "gc_pause", "duration_ms": 12, "generation": 2})
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    report = read_stall_summary(path)

    assert report["loop_stalls"] == 5
    assert report["duration_ms"] == {"p50": 300, "p95": 500, "max": 500}
    assert report["gc_pauses"] == 1
    assert report["top_stacks"][0] == {
        "location": "app.py:10 in paint",
        "count": 4,
        "total_ms": 1000,
    }
    rendered = render_stall_summary(report)
    assert "p50 300 ms · p95 500 ms · max 500 ms" in rendered
    assert "app.py:10 in paint" in rendered


def test_stalls_cli_summarizes_default_log(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    (log_dir / "stalls.jsonl").write_text(
        json.dumps(
            {
                "kind": "loop_stall",
                "duration_ms": 123,
                "stack": [{"file": "app.py", "line": 4, "function": "paint"}],
            }
        )
        + "\n"
    )
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))

    assert main(["stalls"]) == 0

    assert "p50 123 ms" in capsys.readouterr().out
