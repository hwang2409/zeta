#!/usr/bin/env python3
"""Exploratory production benchmark for copied zeta sessions.

Not a CI benchmark. It records operation scale and a load reading beside every
timed sample. All copies live in one per-run sandbox (see sandbox.py) that is
removed on exit unless --keep is used. Every copied session is pinned to the
local provider stub and a network guard refuses non-local hosts. Every zeta
process and the in-process probes import the tree named by --tree.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import fcntl
import io
import json
import os
import pstats
import re
import selectors
import shutil
import signal
import statistics
import struct
import subprocess
import sys
import termios
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parents[2]
PROFILE_DIR = Path.home() / ".zeta/scratch/perf/profiles"
SOURCE_HOME = Path.home() / ".zeta"
DEFAULTS = {
    "phoebe": SOURCE_HOME / "sessions/7897c1aed2e5444caba907f462306b10",
    "orchestrator": SOURCE_HOME / "sessions/86e3d775a40b42ceaa76ca571e27ff1b",
}
PROMPT = " \u203a ".encode()  # the composer prompt the TUI tests wait for
ANSI = re.compile(
    rb"\x1b\[[0-?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)|\x1b[@-_]"
)
KEY_CHARS = b"zxqjkvwy"
ROWS, COLS = 50, 200
# Appended to the sandbox guard in each child's sitecustomize. SIGUSR1 starts a
# cProfile, SIGUSR2 stops it and writes "<prefix>.<n>.prof". With
# ZETA_BENCH_PROFILE_AT_START=1 the first profile starts before zeta is imported.
SITECUSTOMIZE = """
import cProfile, os, signal
_prefix = os.environ.get("ZETA_BENCH_PROFILE")
if _prefix:
    _state = {"p": None, "n": 0}
    def _on(*_):
        _state["p"] = cProfile.Profile(); _state["p"].enable()
    def _off(*_):
        if _state["p"] is not None:
            _state["p"].disable(); _state["p"].dump_stats(f"{_prefix}.{_state['n']}.prof")
            _state["p"] = None; _state["n"] += 1
    signal.signal(signal.SIGUSR1, _on); signal.signal(signal.SIGUSR2, _off)
    if os.environ.get("ZETA_BENCH_PROFILE_AT_START") == "1":
        _on()
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", action="append", choices=[*DEFAULTS, "fresh"])
    parser.add_argument(
        "--probe",
        action="append",
        choices=["store", "scaling", "context", "serve", "tui"],
    )
    parser.add_argument(
        "--tree", type=Path, default=HERE, help="zeta checkout whose src/ is measured"
    )
    parser.add_argument("--idle-seconds", type=int, default=60)
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--serve-traffic-only",
        action="store_true",
        help="serve probe: one lifetime per session and no timed samples",
    )
    return parser.parse_args()


ARGS = parse_args()
TREE_SRC = ARGS.tree.resolve() / "src"
if os.environ.get("ZETA_BENCH_TREE") != str(TREE_SRC):
    # Re-exec so the in-process probes import the measured tree, not an editable install.
    env = {**os.environ, "ZETA_BENCH_TREE": str(TREE_SRC), "PYTHONPATH": str(TREE_SRC)}
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

from sandbox import (
    CP_CLONE,
    STUB_MODEL,
    STUB_PROVIDER,
    Sandbox,
    StubOllama,
    load1,
    pin_stub_provider,
    summary,
    timed,
)

import zeta
from zeta.core.context import ContextAssembler
from zeta.core.session import SessionManager
from zeta.server import ZetaServer
from zeta.storage_format import advance

assert Path(zeta.__file__).resolve().is_relative_to(TREE_SRC), zeta.__file__


def load() -> tuple[float, float, float]:
    return os.getloadavg()


def top_cumulative(path: Path, count: int = 15) -> list[str]:
    text = io.StringIO()
    pstats.Stats(str(path), stream=text).sort_stats("cumulative").print_stats(count)
    lines = text.getvalue().splitlines()
    start = next((i for i, line in enumerate(lines) if "ncalls" in line), len(lines))
    return [line.rstrip() for line in lines[start + 1 :] if line.strip()]


def profile_call(name: str, fn: Callable[[], Any]) -> tuple[str, list[str]]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"{name}.prof"
    profiler = cProfile.Profile()
    profiler.runcall(fn)
    profiler.dump_stats(path)
    return str(path), top_cumulative(path)


def conversation_stats(session: Path) -> tuple[int, int]:
    path = session / "conversation.jsonl"
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    return rows, path.stat().st_size


def clone(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    subprocess.run([*CP_CLONE, str(source), str(target)], check=True)
    return target


def make_home(path: Path, sessions: list[Path]) -> Path:
    """Build a sandbox home with the same identity, skills, and project memory."""
    (path / "sessions").mkdir(parents=True)
    advance(path, expected_epoch=0)
    for name in ("AGENTS.md", "skills", "agents"):
        if (SOURCE_HOME / name).exists():
            clone(SOURCE_HOME / name, path)
    # No mcp.json: MCP servers would be real external processes.
    (path / "settings.toml").write_text(
        'theme = "gruvbox-dark"\ntoken_budget = 400000\n\n[agents]\nexplicit_finish = true\n'
    )
    for source in sessions:
        # Metadata comes from the copy, never the live source.
        project = pin_stub_provider(clone(source, path / "sessions")).get("project_id")
        if project and not (path / "projects" / project).exists():
            clone(SOURCE_HOME / "projects" / project, path / "projects")
    return path


def truncate(source: Path, destination: Path, fraction: float) -> Path:
    target = clone(source, destination)
    conversation = target / "conversation.jsonl"
    lines = conversation.read_bytes().splitlines(keepends=True)
    conversation.write_bytes(b"".join(lines[: max(1, int(len(lines) * fraction))]))
    # Derived indexes and reconciliation ledgers must not describe removed rows.
    for path in target.rglob("*.sqlite*"):
        path.unlink(missing_ok=True)
    return target


def store_probe(home: Path, sid: str, name: str, repeats: int = 10) -> dict[str, Any]:
    open_times: list[tuple[float, float]] = []
    for _ in range(repeats):
        opened, sample = timed(lambda: SessionManager(home).open(sid, _read_only=True))
        opened.store.close()
        open_times.append(sample)
    opened = SessionManager(home).open(sid, _read_only=True)
    store = opened.store
    try:
        replay_times = [timed(store.replay)[1] for _ in range(repeats)]
        message_times = [timed(store.messages)[1] for _ in range(repeats)]
        profile, top = profile_call(f"replay-{name}", store.replay)
        return {
            "open_resume": summary(open_times),
            "replay": summary(replay_times),
            "messages": summary(message_times),
            "rows_replayed": len(store.replay()),
            "messages_count": len(store.messages()),
            "profile": profile,
            "profile_top": top,
        }
    finally:
        store.close()


def context_probe(home: Path, sid: str) -> dict[str, Any]:
    opened = SessionManager(home).open(sid, _read_only=True)
    store = opened.store
    try:

        def assemble() -> Any:
            meta = opened.metadata
            assembler = ContextAssembler(
                store,
                token_budget=meta.compaction_budget,
                retained_tail=meta.retained_tail,
                system_prompt=meta.system_prompt,
            )
            return asyncio.run(assembler.assemble())

        return {"context_assembly": summary([timed(assemble)[1] for _ in range(10)])}
    finally:
        store.close()


async def serve_one(
    home: Path, sid: str, stub: StubOllama, repeats: int
) -> dict[str, Any]:
    """One `zeta serve` process lifetime: hello, one resume, then history/status.

    The copied session is pinned to the stub, so resume's model lookup and any
    wake turn reach only the local stub. `stub_lookups` counts the lookups;
    `stub_requests` counts turn requests that arrived before the last reply.
    Each step prints a progress line to stderr, so a run that hangs names its step.
    """
    started = time.perf_counter()

    def step(name: str) -> None:
        elapsed = time.perf_counter() - started
        print(f"serve {sid[:8]}: {name} +{elapsed:.2f}s", file=sys.stderr, flush=True)

    socket = home / "bench.sock"
    server = await asyncio.to_thread(
        lambda: ZetaServer(
            home=home,
            socket_path=socket,
            provider="ollama",
            model="bench-stub",
            auto_memory=False,
            approval_reconnect_grace=0,
        )
    )
    await server.start()
    step("server started")
    try:
        return await _serve_session(socket, sid, stub, repeats, step)
    finally:
        try:
            step("close start")
            await server.close()
            step("close done")
        finally:
            socket.unlink(missing_ok=True)


async def _serve_session(
    socket: Path,
    sid: str,
    stub: StubOllama,
    repeats: int,
    step: Callable[[str], None],
) -> dict[str, Any]:
    # Frames (resume, notifications) exceed StreamReader's 64 KiB default line limit.
    reader, writer = await asyncio.open_unix_connection(str(socket), limit=1 << 26)
    ident = 0

    async def request(
        method: str, params: dict[str, Any]
    ) -> tuple[tuple[float, float], dict[str, Any]]:
        nonlocal ident
        ident += 1
        line = {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
        started = time.perf_counter_ns()
        writer.write((json.dumps(line) + "\n").encode())
        await writer.drain()
        while True:
            raw = await asyncio.wait_for(reader.readline(), 120)
            if not raw:
                raise RuntimeError(f"{method}: serve closed the connection")
            frame = json.loads(raw)
            if frame.get("id") == ident:
                if "error" in frame:
                    raise RuntimeError(f"{method}: {frame['error']}")
                step(f"{method} (request {ident})")
                return ((time.perf_counter_ns() - started) / 1e6, load1()), frame

    seen, lookups = len(stub.requests), stub.lookups
    try:
        await request(
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["session_history_fidelity", "session_host_commands"],
            },
        )
        resume, _ = await request("resume", {"session_id": sid})
        history = [
            (
                await request(
                    "session_history", {"session_id": sid, "limit": 100, "offset": 0}
                )
            )[0]
            for _ in range(repeats)
        ]
        status = [(await request("status", {}))[0] for _ in range(repeats)]
        return {
            "resume": resume,
            "history": history,
            "status": status,
            "stub_requests": len(stub.requests) - seen,
            "stub_lookups": stub.lookups - lookups,
        }
    finally:
        writer.close()
        await writer.wait_closed()
        step("client closed")


def serve_probe(
    home: Path, sid: str, stub: StubOllama, traffic_only: bool = False
) -> dict[str, Any]:
    # A resume can start a wake turn, so a second resume on one server is refused;
    # each resume sample gets its own server. Traffic-only runs one lifetime with
    # no history/status samples: enough to see where provider traffic goes.
    repeats = [0] if traffic_only else [10, 0, 0, 0, 0]
    runs = [asyncio.run(serve_one(home, sid, stub, n)) for n in repeats]
    return {
        "resume": summary([r["resume"] for r in runs]),
        "history": summary(runs[0]["history"]),
        "status": summary(runs[0]["status"]),
        "stub_requests_per_server": [r["stub_requests"] for r in runs],
        "stub_lookups_per_server": [r["stub_lookups"] for r in runs],
    }


class Tui:
    """One real zeta CLI process on a pty that is its controlling tty."""

    def __init__(
        self,
        box: Sandbox,
        home: Path,
        sid: str | None,
        cwd: Path,
        profile: str | None = None,
        profile_at_start: bool = False,
    ) -> None:
        self.box = box
        self.output = bytearray()
        self.chunks: list[tuple[int, int]] = []
        self.exit_code: int | None = None
        env = {
            k: v
            for k, v in box.env(
                home,
                TERM="xterm-256color",
                COLORTERM="truecolor",
                PYTHONPATH=f"{box.site}{os.pathsep}{TREE_SRC}",
            ).items()
            if k
            not in {
                "NO_COLOR",
                "FORCE_COLOR",
                "CLICOLOR",
                "CLICOLOR_FORCE",
                "ZETA_BENCH_TREE",
            }
        }
        if profile:
            env["ZETA_BENCH_PROFILE"] = profile
            if profile_at_start:
                env["ZETA_BENCH_PROFILE_AT_START"] = "1"
        # zeta has no __main__, so run the console-script entry point. spawn's setsid plus
        # TIOCSCTTY makes the pty the controlling tty; macOS does not do that on open.
        launcher = "import fcntl, sys, termios; fcntl.ioctl(0, termios.TIOCSCTTY, 0); from zeta.cli.main import main; sys.exit(main())"
        argv = [
            sys.executable,
            "-c",
            launcher,
            "--no-auto-memory",
            "--provider",
            "ollama",
            "--model",
            "bench-stub",
        ]
        if sid:
            argv += ["--resume", sid, "--force-provider"]
        self.master, slave = os.openpty()
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        self.started = time.perf_counter_ns()
        self.process = box.spawn(
            argv, stdin=slave, stdout=slave, stderr=slave, env=env, cwd=cwd
        )
        os.close(slave)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.master, selectors.EVENT_READ)
        self.last_byte = self.started

    def ms(self, ns: int) -> float:
        return (ns - self.started) / 1e6

    def read(self, timeout: float) -> int:
        """Read what is available within timeout; return bytes read (-1 when the process is gone)."""
        if not self.selector.select(timeout):
            return 0
        try:
            chunk = os.read(self.master, 65536)
        except OSError:
            chunk = b""
        if not chunk:
            self.exit_code = self.process.wait(timeout=5)
            return -1
        self.output.extend(chunk)
        self.last_byte = time.perf_counter_ns()
        self.chunks.append((self.last_byte, len(chunk)))
        return len(chunk)

    def wait_for(self, predicate: Callable[[], bool], timeout: float) -> int | None:
        deadline = time.monotonic() + timeout
        while not predicate():
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.read(remaining) < 0:
                return None
        return time.perf_counter_ns()

    def quiet(self, window: float, timeout: float = 30) -> int | None:
        """Wait until no byte arrives for `window` seconds; return the last byte time."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            got = self.read(window)
            if got < 0:
                return None
            if got == 0:
                return self.last_byte
        return None

    def startup(self, stub: StubOllama, timeout: float = 120) -> dict[str, Any]:
        """Time to the first prompt, then to the first request of a turn typed at that prompt.

        A resumed TUI paints the prompt first and rebuilds the transcript afterwards;
        a turn submitted meanwhile waits for the rebuild, so the request time is when
        the session is really usable.
        """
        prompt_at = self.wait_for(lambda: PROMPT in self.output, timeout)
        result: dict[str, Any] = {
            "prompt_ms": self.ms(prompt_at) if prompt_at else None,
            "ready_ms": None,
        }
        if prompt_at:
            mark = len(self.chunks)
            seen = len(stub.requests)
            stub.finished.clear()
            os.write(self.master, b"bench ready\r")
            if self.wait_for(lambda: len(stub.requests) > seen, timeout):
                result["ready_ms"] = self.ms(stub.requests[seen])
                paints = [
                    (t, n) for t, n in self.chunks[mark:] if t <= stub.requests[seen]
                ]
                if paints:
                    result["paint_after_prompt_ms"] = (
                        self.ms(paints[0][0]),
                        self.ms(paints[-1][0]),
                    )
                    result["paint_after_prompt_bytes"] = sum(n for _, n in paints)
                self.wait_for(stub.finished.is_set, 60)
                self.quiet(0.5)
        return {
            **result,
            "bytes": len(self.output),
            "load": load1(),
            **self.failure(),
        }

    def failure(self) -> dict[str, Any]:
        if self.exit_code is None and self.process.poll() is None:
            return {}
        self.exit_code = self.process.poll()
        return {
            "exit_code": self.exit_code,
            "tail": ANSI.sub(b"", bytes(self.output[-2048:])).decode(
                "utf-8", "replace"
            ),
        }

    def key(self, timeout: float = 5) -> dict[str, Any] | None:
        """Type one char not visible in recent output; time to its first byte and to its render."""
        recent = ANSI.sub(b"", bytes(self.output[-8192:]))
        char = next(
            (bytes((c,)) for c in KEY_CHARS if bytes((c,)) not in recent), KEY_CHARS[:1]
        )
        offset, start = len(self.output), len(self.chunks)
        sent = time.perf_counter_ns()
        os.write(self.master, char)
        # One bounded loop; the first post-key chunk carries its own timestamp.
        shown = self.wait_for(
            lambda: char in ANSI.sub(b"", bytes(self.output[offset:])), timeout
        )
        if shown is None or len(self.chunks) == start:
            return None  # a failed sample, counted against `attempted`
        return {
            "first_byte_ms": (self.chunks[start][0] - sent) / 1e6,
            "render_ms": (shown - sent) / 1e6,
            "bytes": len(self.output) - offset,
            "load": load1(),
        }

    def keys(self, count: int, gap: float = 0.05) -> dict[str, Any]:
        samples = []
        for _ in range(count):
            sample = self.key()
            if sample is not None:
                samples.append(sample)
            self.quiet(gap, timeout=gap * 20)
        return {
            "render": summary([(s["render_ms"], s["load"]) for s in samples]),
            "first_byte": summary([(s["first_byte_ms"], s["load"]) for s in samples]),
            "bytes_per_key_p50": statistics.median([s["bytes"] for s in samples])
            if samples
            else None,
            "attempted": count,
            **self.failure(),
        }

    def clear_composer(self) -> None:
        os.write(self.master, b"\x15")  # ctrl-u
        self.quiet(0.2, timeout=5)

    def signal(self, sig: int) -> None:
        os.kill(self.process.pid, sig)

    def close(self) -> None:
        try:
            self.selector.close()
            self.box.stop(self.process)
        finally:
            os.close(self.master)


def ps_sample(pid: int) -> tuple[float, int, float]:
    out = subprocess.run(
        ["ps", "-o", "%cpu=,rss=,time=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    minutes, seconds = out[2].split(":")
    return float(out[0]), int(out[1]), int(minutes) * 60 + float(seconds)


def idle_probe(tui: Tui, seconds: int, profile_prefix: str) -> dict[str, Any]:
    start_bytes = len(tui.output)
    before = ps_sample(tui.process.pid)[2]
    samples = []
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        tui.read(1.0)
        samples.append((*ps_sample(tui.process.pid), load1()))
    after = samples[-1][2]
    # A separate 10 s profiled window counts loop iterations and the callbacks that fire.
    tui.signal(signal.SIGUSR1)
    end = time.monotonic() + 10
    while time.monotonic() < end:
        tui.read(0.5)
    tui.signal(signal.SIGUSR2)
    time.sleep(1)
    path = Path(f"{profile_prefix}.0.prof")
    counts = {}
    if path.exists():
        stats = pstats.Stats(str(path)).stats  # type: ignore[attr-defined]
        for (file, line, func), (_, ncalls, *_rest) in stats.items():
            if ncalls >= 5 and (
                "zeta" in file or func in {"_run_once", "select", "control"}
            ):
                counts[f"{file}:{line}({func})"] = ncalls
    return {
        "seconds": seconds,
        "cpu_seconds": round(after - before, 3),
        "cpu_percent_p50": statistics.median(s[0] for s in samples),
        "rss_kb_first_last_max": (
            samples[0][1],
            samples[-1][1],
            max(s[1] for s in samples),
        ),
        "output_bytes": len(tui.output) - start_bytes,
        "profile_10s": str(path),
        "calls_10s": dict(sorted(counts.items(), key=lambda kv: -kv[1])[:25]),
        "load_range": [min(s[3] for s in samples), max(s[3] for s in samples)],
        **tui.failure(),
    }


def submit(
    tui: Tui,
    stub: StubOllama,
    text: bytes,
    stream_seconds: float,
    before_enter: Callable[[], None] | None = None,
) -> tuple[float, float] | None:
    """Type text, then time Enter until the stub provider receives the request."""
    stub.stream_seconds.append(stream_seconds)
    stub.finished.clear()
    os.write(tui.master, text)
    tui.quiet(0.3, timeout=10)
    seen = len(stub.requests)
    if before_enter is not None:
        before_enter()
    sent = time.perf_counter_ns()
    os.write(tui.master, b"\r")
    if (
        tui.wait_for(lambda: len(stub.requests) > seen, 60) is None
        and len(stub.requests) == seen
    ):
        return None
    return (stub.requests[seen] - sent) / 1e6, load1()


def finish_turn(tui: Tui, stub: StubOllama) -> None:
    tui.wait_for(stub.finished.is_set, 60)
    tui.quiet(0.5)


def _startup_phase(
    box: Sandbox, home: Path, sid: str | None, cwd: Path
) -> dict[str, Any]:
    startups = []
    for _ in range(5):
        tui = Tui(box, home, sid, cwd)
        try:
            startups.append(tui.startup(box.stub))
        finally:
            tui.close()
    ok = [s for s in startups if s["ready_ms"] is not None]
    return {
        "startups": startups,
        "startup_ok": len(ok),
        "startup_prompt": summary([(s["prompt_ms"], s["load"]) for s in ok]),
        "startup_ready": summary([(s["ready_ms"], s["load"]) for s in ok]),
    }


def _scroll_page_up(tui: Tui) -> dict[str, Any]:
    scroll = []
    for _ in range(10):
        start = len(tui.output)
        sent = time.perf_counter_ns()
        os.write(tui.master, b"\x1b[5~")  # PageUp
        first = tui.wait_for(lambda start=start: len(tui.output) > start, 5)
        last = tui.quiet(0.15, timeout=10)
        if first and last:
            scroll.append(
                {
                    "first_byte_ms": (first - sent) / 1e6,
                    "last_byte_ms": (last - sent) / 1e6,
                    "bytes": len(tui.output) - start,
                    "load": load1(),
                }
            )
    os.write(tui.master, b"\x1b[F")  # End: back to the live tail
    tui.quiet(0.3)
    return {
        "first_byte": summary([(s["first_byte_ms"], s["load"]) for s in scroll]),
        "last_byte": summary([(s["last_byte_ms"], s["load"]) for s in scroll]),
        "first_sample": scroll[0] if scroll else None,
        "bytes_p50": statistics.median(s["bytes"] for s in scroll) if scroll else None,
    }


def _turns(tui: Tui, stub: StubOllama) -> dict[str, Any]:
    result: dict[str, Any] = {}
    turn_start = []
    for index in range(10):
        sample = submit(tui, stub, b"bench turn %d" % index, 0)
        if sample is None:
            result["turn_error"] = tui.failure() or {
                "tail": ANSI.sub(b"", bytes(tui.output[-4096:])).decode(
                    "utf-8", "replace"
                )
            }
            break
        turn_start.append(sample)
        finish_turn(tui, stub)
    result["turn_start_first_ms"] = turn_start[0][0] if turn_start else None
    result["turn_start"] = summary(turn_start)
    if turn_start and submit(tui, stub, b"bench stream", 10) is not None:
        result["key_streaming"] = tui.keys(30, gap=0.1)
        finish_turn(tui, stub)
    return result


def _interaction_phase(
    box: Sandbox, home: Path, sid: str | None, cwd: Path, prefix: str, idle_seconds: int
) -> dict[str, Any]:
    tui = Tui(box, home, sid, cwd, profile=f"{prefix}-idle")
    try:
        tui.startup(box.stub)
        result = {"idle": idle_probe(tui, idle_seconds, f"{prefix}-idle")}
        result["key_idle"] = tui.keys(30)
        tui.clear_composer()
        result["scroll_page_up"] = _scroll_page_up(tui)
        result.update(_turns(tui, box.stub))
        result.update(tui.failure())
        return result
    finally:
        tui.close()


def _profile_phase(
    box: Sandbox, home: Path, sid: str | None, cwd: Path, prefix: str
) -> dict[str, Any]:
    """Profiles run in their own process so profiler overhead never enters a timing."""
    stub = box.stub
    tui = Tui(box, home, sid, cwd, profile=f"{prefix}-span", profile_at_start=True)
    try:
        # profile 0: process start to the first request; profile 1: one later turn's Enter to its request.
        stub.on_request = lambda: tui.signal(signal.SIGUSR2)
        if tui.startup(stub)["ready_ms"] is not None:
            submit(
                tui,
                stub,
                b"bench profiled turn",
                0,
                before_enter=lambda: tui.signal(signal.SIGUSR1),
            )
            finish_turn(tui, stub)
        stub.on_request = None
        time.sleep(1)
        result = {}
        for key, index in (("startup_profile", 0), ("turn_profile", 1)):
            path = Path(f"{prefix}-span.{index}.prof")
            result[key] = (
                {"path": str(path), "top": top_cumulative(path)}
                if path.exists()
                else None
            )
        return result
    finally:
        stub.on_request = None
        tui.close()


def tui_probe(
    box: Sandbox, home: Path, sid: str | None, name: str, cwd: Path, idle_seconds: int
) -> dict[str, Any]:
    result = _startup_phase(box, home, sid, cwd)
    if not result["startup_ok"]:
        return result
    prefix = str(PROFILE_DIR / f"tui-{name}")
    result.update(_interaction_phase(box, home, sid, cwd, prefix, idle_seconds))
    result.update(_profile_phase(box, home, sid, cwd, prefix))
    return result


def scaling_probe(box: Sandbox, home: Path, sid: str, name: str) -> dict[str, Any]:
    curves = {}
    for fraction in (0.1, 0.5, 1.0):
        scaled_home = box.root / f"scaled-{name}-{fraction}"
        (scaled_home / "sessions").mkdir(parents=True)
        try:
            advance(scaled_home, expected_epoch=0)
            # Cut the sandbox copy, never the live source.
            scaled = truncate(
                home / "sessions" / sid, scaled_home / "sessions", fraction
            )
            scaled_rows, scaled_bytes = conversation_stats(scaled)
            store = store_probe(scaled_home, sid, f"{name}-{fraction}")
        finally:
            shutil.rmtree(scaled_home, ignore_errors=True)
        # Report both per-MB and per-row cost: bytes per row differ between sessions.
        curves[str(fraction)] = {
            "rows_in_file": scaled_rows,
            "bytes": scaled_bytes,
            "rows_replayed": store["rows_replayed"],
            "replay_ms_per_mb": round(
                store["replay"]["p50_ms"] / (scaled_bytes / 1e6), 3
            ),
            "replay_us_per_row": round(
                store["replay"]["p50_ms"] * 1000 / scaled_rows, 2
            ),
            "store": store,
        }
    return curves


# Run order and handler per --probe value: (box, home, sid, name, cwd) -> result.
PROBES: dict[str, Callable[..., dict[str, Any]]] = {
    "store": lambda box, home, sid, name, cwd: store_probe(home, sid, name),
    "context": lambda box, home, sid, name, cwd: context_probe(home, sid),
    "serve": lambda box, home, sid, name, cwd: serve_probe(
        home, sid, box.stub, ARGS.serve_traffic_only
    ),
    "tui": lambda box, home, sid, name, cwd: tui_probe(
        box, home, sid, name, cwd, ARGS.idle_seconds
    ),
    "scaling": lambda box, home, sid, name, cwd: scaling_probe(box, home, sid, name),
}


def sandbox_sessions(box: Sandbox, names: list[str]) -> tuple[Path, dict[str, Any]]:
    """Copy the named sessions (and create `fresh`) in the sandbox; return sid and cwd."""
    home = make_home(box.root / "home", [DEFAULTS[n] for n in names if n != "fresh"])
    sessions: dict[str, tuple[str, Path]] = {}
    for name in names:
        if name == "fresh":
            opened = SessionManager(home).create(
                provider=STUB_PROVIDER, model=STUB_MODEL, cwd=HERE, auto_project=False
            )
            sessions[name] = (opened.metadata.session_id, HERE)
            opened.store.close()
        else:
            sid = DEFAULTS[name].name
            meta = json.loads((home / "sessions" / sid / "meta.json").read_text())
            sessions[name] = (sid, Path(meta["cwd"]))
    return home, sessions


def main() -> None:
    names = ARGS.session or ["fresh", "phoebe", "orchestrator"]
    selected = set(ARGS.probe or PROBES)
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    with Sandbox("sandbox-bench", keep=ARGS.keep) as box:
        box.write_sitecustomize(SITECUSTOMIZE)
        child_zeta = subprocess.run(
            [sys.executable, "-c", "import zeta; print(zeta.__file__)"],
            env=box.env(PYTHONPATH=f"{box.site}{os.pathsep}{TREE_SRC}"),
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        results: dict[str, Any] = {
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "load_average": load(),
            "profiles": str(PROFILE_DIR),
            "zeta_file": zeta.__file__,
            "child_zeta_file": child_zeta,
        }
        print(f"zeta: {zeta.__file__}; child: {child_zeta}", file=sys.stderr)
        home, sessions = sandbox_sessions(box, names)
        box.use_home(home)
        box.check_stubbed(home)
        for name, (sid, cwd) in sessions.items():
            rows, size = conversation_stats(home / "sessions" / sid)
            agents = home / "sessions" / sid / "agents"
            children = len(list(agents.iterdir())) if agents.is_dir() else 0
            result: dict[str, Any] = {
                "scale": {"rows": rows, "bytes": size, "children": children},
                "load_average": load(),
            }
            for probe, run in PROBES.items():
                if probe in selected and not (probe == "scaling" and name == "fresh"):
                    result[probe] = run(box, home, sid, name, cwd)
            result["load_average_after"] = load()
            results[name] = result
            print(json.dumps({name: result}, indent=2), flush=True)
        meta = {k: results[k] for k in ("generated_at", "zeta_file", "child_zeta_file")}
        print(json.dumps({"meta": meta}), flush=True)


if __name__ == "__main__":
    main()
