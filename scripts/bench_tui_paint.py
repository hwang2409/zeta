"""Measure TUI paint and event-loop cost with deterministic local fixtures."""

from __future__ import annotations

import asyncio
import inspect
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from rich.text import Text

from zeta.core.store import ConversationStore
from zeta.protocol.types import (
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
)
from zeta.tui.render import render_event
from zeta.tui.transcript import TranscriptWidget
from zeta.tui.transcript.streaming_text import StreamingText


def percentile(xs, p):
    xs = sorted(xs)
    return xs[max(0, int(len(xs) * p + 0.999) - 1)]


def base_transcript(n=2000):
    t = TranscriptWidget()
    for i in range(n):
        t.append(Text(f"unit {i} " + "history words " * 4))
    return t


async def measured(work, rounds, interval=0.01):
    gaps = []
    paints = []
    stop = False

    async def ticker():
        expected = time.perf_counter() + 0.01
        while not stop:
            await asyncio.sleep(max(0, expected - time.perf_counter()))
            now = time.perf_counter()
            gaps.append(max(0, now - expected))
            expected = now + 0.01

    task = asyncio.create_task(ticker())
    await asyncio.sleep(0.02)
    wall = time.perf_counter()
    cpu = time.process_time()
    for i in range(rounds):
        s = time.perf_counter()
        result = work(i)
        if inspect.isawaitable(result):
            await result
        paints.append(time.perf_counter() - s)
        await asyncio.sleep(interval)
    cpu = time.process_time() - cpu
    wall = time.perf_counter() - wall
    stop = True
    await task
    return {
        "gap_max_ms": max(gaps) * 1000,
        "gap_p95_ms": percentile(gaps, 0.95) * 1000,
        "paint_p95_ms": percentile(paints, 0.95) * 1000,
        "paint_mean_ms": statistics.mean(paints) * 1000,
        "cpu_percent": cpu / wall * 100,
        "frames": rounds,
    }


async def main():
    t = base_transcript()
    s = StreamingText("", palette_role="body")
    u = t.append(s)
    t.create_content(100, 30)
    s.append("x" * 48000)
    t.touch(u)
    t.create_content(100, 30)
    chunks = ["x" * 100 for _ in range(20)]
    print(
        "streaming",
        await measured(
            lambda i: (s.append(chunks[i]), t.touch(u), t.create_content(100, 30)),
            20,
            0.016,
        ),
    )

    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        template = ConversationStore(root / "template-parent", session_id="0")
        row_count = 22_000
        template.append_many(
            (
                "message",
                {
                    "message": Message(
                        MessageRole.ASSISTANT,
                        [TextContent("child output line")],
                    ).to_dict()
                },
            )
            for _ in range(row_count)
        )
        template.close()
        print(
            "agent_fixture_mb",
            template.session_dir.joinpath("conversation.jsonl").stat().st_size
            / 1_000_000,
        )
        t = base_transcript()
        for i in range(8):
            child_path = root / f"agent-{i}" / "0"
            shutil.copytree(template.session_dir, child_path)
            call = ToolCall(
                f"a{i}", "agent", {"prompt": "inspect", "description": f"agent {i}"}
            )
            ev = StreamEvent(
                StreamEventType.TOOL_EXECUTION_START,
                tool_call=call,
                data={"child_session_path": str(child_path)},
            )
            t.start_tool(call.id, call, render_event(ev), ev)
            t.update_tool(
                call.id,
                Text("turn 1"),
                StreamEvent(
                    StreamEventType.TOOL_EXECUTION_UPDATE,
                    tool_call=call,
                    data={"child_session_path": str(child_path)},
                ),
            )
        t.create_content(100, 30)

        async def refresh_agents(_index):
            await t.refresh_agent_transcripts()
            t.create_content(100, 30)

        print("agents", await measured(refresh_agents, 6, 0.2))

    t = base_transcript(20000)
    t.create_content(100, 30)
    for _ in range(20):
        t.page_up()
        t.create_content(100, 30)
    print("scrolled", await measured(lambda i: t.create_content(100, 30), 100, 0.01))


if __name__ == "__main__":
    asyncio.run(main())
