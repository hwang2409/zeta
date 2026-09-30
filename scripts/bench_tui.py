"""End-to-end and synthetic TUI transcript benchmarks.

Run synthetic cases with ``uv run python scripts/bench_tui.py`` and real resume
cases with one or more read-only fixture directories passed via ``--session-dir``.
Each real fixture is copied into a temporary ``ZETA_HOME`` before every run.
"""
from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import math
import os
import pstats
import shutil
import statistics
import tempfile
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from prompt_toolkit.application.current import set_app
from prompt_toolkit.input.defaults import create_pipe_input
from prompt_toolkit.output import DummyOutput
from rich.markdown import Markdown
from rich.text import Text

from zeta.cli.main import build_parser
from zeta.tui.app import create_app
from zeta.tui.key_bindings import FullScreenPromptSession
from zeta.tui.transcript import TranscriptWidget

TimerValues = dict[str, float]
REAL_PHASES = (
    "jsonl_read_parse",
    "store_replay_validation",
    "app_construction",
    "prompt_layout_construction",
    "message_to_unit",
    "rich_rendering",
    "ansi_line_assembly",
    "prompt_toolkit_layout",
    "location_map",
    "first_frame_total",
)


def synthetic_transcript(messages: int) -> TranscriptWidget:
    transcript = TranscriptWidget()
    for index in range(messages):
        if index % 3 == 0:
            value = Markdown(
                f"## Tool result {index}\n\n"
                + ("```python\nprint('long result')\n```\n" * 4)
                + ("tool output line with markdown and details. " * 8)
            )
        elif index % 3 == 1:
            value = Text("thinking block " + ("internal reasoning " * 18), style="dim")
        else:
            value = Markdown(
                f"assistant message {index}\n\n"
                + ("A paragraph with **markdown**, `code`, and a long explanation. " * 8)
            )
        transcript.append(value)
        transcript.append_blank()
    return transcript


def redraw(transcript: TranscriptWidget, width: int = 100, height: int = 30) -> None:
    transcript.create_content(width, height)


def measure_synthetic(messages: int) -> TimerValues:
    transcript = synthetic_transcript(messages)
    with create_pipe_input():
        started = time.perf_counter()
        redraw(transcript)
        startup = time.perf_counter() - started
        started = time.perf_counter()
        for _ in range(5):
            redraw(transcript)
        steady = (time.perf_counter() - started) / 5
        unit = transcript._units[-2]
        assert unit is not None
        started = time.perf_counter()
        for token in range(5):
            transcript.replace(unit, Text("streaming token " * (token + 1)))
            redraw(transcript)
        streaming = (time.perf_counter() - started) / 5
    return {"startup": startup, "steady": steady, "streaming": streaming}


def _percentiles(samples: list[float]) -> tuple[float, float]:
    ordered = sorted(samples)
    p90_index = max(0, math.ceil(len(ordered) * 0.9) - 1)
    return statistics.median(ordered), ordered[p90_index]


def _print_synthetic(sizes: list[int], repeats: int) -> None:
    print("synthetic_messages,phase,median_s,p90_s")
    for size in sizes:
        samples = [measure_synthetic(size) for _ in range(repeats)]
        for phase in ("startup", "steady", "streaming"):
            median, p90 = _percentiles([sample[phase] for sample in samples])
            print(f"{size},{phase},{median:.6f},{p90:.6f}")


@contextmanager
def _instrument_store(timers: TimerValues) -> Iterator[None]:
    from zeta.core.store import _store

    original_load = _store.ConversationStore._load
    original_read = _store.read_session_file
    original_json = _store.load_session_json
    active = 0

    def timed_read(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_read(*args, **kwargs)
        finally:
            if active:
                timers["jsonl_read_parse"] += time.perf_counter() - started

    def timed_json(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_json(*args, **kwargs)
        finally:
            if active:
                timers["jsonl_read_parse"] += time.perf_counter() - started

    def timed_load(store: Any) -> None:
        nonlocal active
        active += 1
        started = time.perf_counter()
        try:
            original_load(store)
        finally:
            elapsed = time.perf_counter() - started
            active -= 1
            timers["store_replay_validation"] += elapsed

    _store.read_session_file = timed_read
    _store.load_session_json = timed_json
    _store.ConversationStore._load = timed_load
    try:
        yield
    finally:
        _store.read_session_file = original_read
        _store.load_session_json = original_json
        _store.ConversationStore._load = original_load
        timers["store_replay_validation"] = max(
            0.0,
            timers["store_replay_validation"] - timers["jsonl_read_parse"],
        )


@contextmanager
def _instrument_paint(transcript: TranscriptWidget, timers: TimerValues) -> Iterator[None]:
    original_render = transcript._render_unit
    original_lines = transcript._unit_parsed_lines

    def timed_render(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_render(*args, **kwargs)
        finally:
            timers["rich_rendering"] += time.perf_counter() - started

    def timed_lines(*args: Any, **kwargs: Any) -> Any:
        rich_before = timers["rich_rendering"]
        started = time.perf_counter()
        try:
            return original_lines(*args, **kwargs)
        finally:
            total = time.perf_counter() - started
            rich_inside = timers["rich_rendering"] - rich_before
            timers["ansi_line_assembly"] += max(0.0, total - rich_inside)

    transcript._render_unit = timed_render  # type: ignore[method-assign]
    transcript._unit_parsed_lines = timed_lines  # type: ignore[method-assign]
    try:
        yield
    finally:
        transcript._render_unit = original_render  # type: ignore[method-assign]
        transcript._unit_parsed_lines = original_lines  # type: ignore[method-assign]


def _copy_fixture(fixture: Path, root: Path) -> str:
    session_id = fixture.name
    sessions = root / "sessions"
    sessions.mkdir()
    shutil.copytree(fixture, sessions / session_id)
    return session_id


async def _measure_real_async(fixture: Path, profiler: cProfile.Profile | None) -> TimerValues:
    timers = {phase: 0.0 for phase in REAL_PHASES}
    with tempfile.TemporaryDirectory(prefix="zeta-tui-bench-") as temporary:
        home = Path(temporary)
        session_id = _copy_fixture(fixture, home)
        old_home = os.environ.get("ZETA_HOME")
        os.environ["ZETA_HOME"] = str(home)
        try:
            args = build_parser().parse_args(["--resume", session_id])
            if profiler is not None:
                profiler.enable()
            total_started = time.perf_counter()
            with _instrument_store(timers):
                started = time.perf_counter()
                app = create_app(args)
                timers["app_construction"] = time.perf_counter() - started
            with create_pipe_input() as pipe:
                started = time.perf_counter()
                session = FullScreenPromptSession(
                    input=pipe,
                    output=DummyOutput(),
                    multiline=True,
                )
                app._active_session = session
                app._install_full_screen_layout(session)
                timers["prompt_layout_construction"] = time.perf_counter() - started

                started = time.perf_counter()
                app._rebuild_transcript()
                timers["message_to_unit"] = time.perf_counter() - started

                with set_app(session.app), _instrument_paint(app._transcript, timers):
                    started = time.perf_counter()
                    session.app.renderer.render(session.app, session.app.layout)
                    render_total = time.perf_counter() - started
                    timers["prompt_toolkit_layout"] = max(
                        0.0,
                        render_total
                        - timers["rich_rendering"]
                        - timers["ansi_line_assembly"],
                    )
                    timers["first_frame_total"] = time.perf_counter() - total_started
                    first_frame_rich = timers["rich_rendering"]
                    first_frame_ansi = timers["ansi_line_assembly"]

                    started = time.perf_counter()
                    app._transcript._locations(app._transcript._content_width)
                    timers["location_map"] = time.perf_counter() - started
                    timers["rich_rendering"] = first_frame_rich
                    timers["ansi_line_assembly"] = first_frame_ansi
            app.loop.store.close()
            if profiler is not None:
                profiler.disable()
        finally:
            if old_home is None:
                os.environ.pop("ZETA_HOME", None)
            else:
                os.environ["ZETA_HOME"] = old_home
    return timers


def measure_real(fixture: Path, profiler: cProfile.Profile | None = None) -> TimerValues:
    return asyncio.run(_measure_real_async(fixture, profiler))


def _print_real(fixtures: list[Path], repeats: int, profile: bool) -> None:
    print("real_session,phase,median_s,p90_s")
    for fixture in fixtures:
        samples = [measure_real(fixture) for _ in range(repeats)]
        for phase in REAL_PHASES:
            median, p90 = _percentiles([sample[phase] for sample in samples])
            print(f"{fixture.name},{phase},{median:.6f},{p90:.6f}")
        if profile:
            profiler = cProfile.Profile()
            measure_real(fixture, profiler)
            output = io.StringIO()
            pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(20)
            print(f"profile_session={fixture.name}\n{output.getvalue()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 2000])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--session-dir", action="append", type=Path, default=[])
    parser.add_argument("--real-only", action="store_true")
    args = parser.parse_args()
    if args.repeats < 5:
        parser.error("--repeats must be at least 5")
    if not args.real_only:
        _print_synthetic(args.sizes, args.repeats)
    if args.session_dir:
        _print_real(args.session_dir, args.repeats, args.profile)


if __name__ == "__main__":
    main()
