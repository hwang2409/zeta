"""Synthetic TUI transcript benchmark.

Run with ``uv run python scripts/bench_tui.py``. ``--session-dir`` records
metadata for a copied real session without reading or storing its contents.
"""
from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import time
from pathlib import Path

from prompt_toolkit.input.defaults import create_pipe_input
from rich.markdown import Markdown
from rich.text import Text

from zeta.tui.transcript import TranscriptWidget


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


def measure(messages: int) -> dict[str, float]:
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


def profile(messages: int) -> None:
    transcript = synthetic_transcript(messages)
    profiler = cProfile.Profile()
    profiler.enable()
    redraw(transcript)
    profiler.disable()
    output = io.StringIO()
    pstats.Stats(profiler, stream=output).sort_stats("cumulative").print_stats(12)
    print(f"\nprofile n={messages}\n{output.getvalue()}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[100, 500, 2000])
    parser.add_argument("--profile", type=int, default=0)
    parser.add_argument("--session-dir", type=Path)
    args = parser.parse_args()
    if args.session_dir:
        path = args.session_dir / "conversation.jsonl"
        print(f"session_dir={args.session_dir} bytes={path.stat().st_size if path.exists() else 0}")
    print("messages,startup_s,steady_redraw_s,streaming_event_s")
    for size in args.sizes:
        result = measure(size)
        print(size, *(f"{result[key]:.6f}" for key in ("startup", "steady", "streaming")), sep=",")
    if args.profile:
        profile(args.profile)


if __name__ == "__main__":
    main()
