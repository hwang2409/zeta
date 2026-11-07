#!/usr/bin/env python3
"""Exploratory production benchmark for copied zeta sessions.

Not a CI benchmark. It records operation scale and load averages beside noisy
timings. Copies are removed unless --keep is used. Every zeta process and the
in-process probes import the tree named by --tree (default: this checkout).
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import fcntl
import http.server
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
import tempfile
import termios
import threading
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
# Loaded through PYTHONPATH as sitecustomize in each TUI process. SIGUSR1 starts a
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
    return parser.parse_args()


ARGS = parse_args()
TREE_SRC = ARGS.tree.resolve() / "src"
if os.environ.get("ZETA_BENCH_TREE") != str(TREE_SRC):
    # Re-exec so the in-process probes import the measured tree, not an editable install.
    env = {**os.environ, "ZETA_BENCH_TREE": str(TREE_SRC), "PYTHONPATH": str(TREE_SRC)}
    os.execve(sys.executable, [sys.executable, *sys.argv], env)

import zeta
from zeta.core.context import ContextAssembler
from zeta.core.session import SessionManager
from zeta.server import ZetaServer
from zeta.storage_format import advance

assert Path(zeta.__file__).resolve().is_relative_to(TREE_SRC), zeta.__file__


def load() -> tuple[float, float, float]:
    return os.getloadavg()


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    started = time.perf_counter_ns()
    value = fn()
    return value, (time.perf_counter_ns() - started) / 1e6


def summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"p50_ms": None, "p95_ms": None, "n": 0, "load_average": load()}
    ordered = sorted(values)
    return {
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[max(0, -(-len(ordered) * 95 // 100) - 1)], 3),
        "max_ms": round(ordered[-1], 3),
        "n": len(ordered),
        "load_average": load(),
    }


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
    subprocess.run(["cp", "-cR", str(source), str(target)], check=True)
    return target


def make_home(path: Path, sessions: list[Path]) -> Path:
    """Build a throwaway home with the same identity, skills, and project memory."""
    (path / "sessions").mkdir(parents=True)
    advance(path, expected_epoch=0)
    for name in ("AGENTS.md", "skills", "agents"):
        if (SOURCE_HOME / name).exists():
            subprocess.run(
                ["cp", "-cR", str(SOURCE_HOME / name), str(path / name)], check=True
            )
    # No mcp.json: MCP servers would be real external processes.
    (path / "settings.toml").write_text(
        'theme = "gruvbox-dark"\ntoken_budget = 400000\n\n[agents]\nexplicit_finish = true\n'
    )
    for source in sessions:
        meta_path = clone(source, path / "sessions") / "meta.json"
        # The throwaway home has no accounts; the copy must not pin the user's.
        meta = json.loads(meta_path.read_text())
        meta["selected_accounts"] = {}
        meta_path.write_text(json.dumps(meta))
        project = json.loads((source / "meta.json").read_text()).get("project_id")
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
    open_times: list[float] = []
    for _ in range(repeats):
        opened, elapsed = timed(lambda: SessionManager(home).open(sid, _read_only=True))
        opened.store.close()
        open_times.append(elapsed)
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
    home: Path, sid: str, stub_url: str, repeats: int
) -> dict[str, Any]:
    """One `zeta serve` process lifetime: hello, one resume, then history/status."""
    os.environ["ZETA_OLLAMA_BASE_URL"] = stub_url
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
    # Frames (resume, notifications) exceed StreamReader's 64 KiB default line limit.
    reader, writer = await asyncio.open_unix_connection(str(socket), limit=1 << 26)
    ident = 0

    async def request(
        method: str, params: dict[str, Any]
    ) -> tuple[float, dict[str, Any]]:
        nonlocal ident
        ident += 1
        line = {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
        started = time.perf_counter_ns()
        writer.write((json.dumps(line) + "\n").encode())
        await writer.drain()
        while True:
            frame = json.loads(await asyncio.wait_for(reader.readline(), 120))
            if frame.get("id") == ident:
                if "error" in frame:
                    raise RuntimeError(f"{method}: {frame['error']}")
                return (time.perf_counter_ns() - started) / 1e6, frame

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
        return {"resume": resume, "history": history, "status": status}
    finally:
        writer.close()
        await writer.wait_closed()
        await server.close()


def serve_probe(home: Path, sid: str, stub_url: str) -> dict[str, Any]:
    # A resume can start a wake turn, so a second resume on one server is refused;
    # each resume sample gets its own server.
    runs = [
        asyncio.run(serve_one(home, sid, stub_url, 10 if i == 0 else 0))
        for i in range(5)
    ]
    return {
        "resume": summary([r["resume"] for r in runs]),
        "history": summary(runs[0]["history"]),
        "status": summary(runs[0]["status"]),
    }


class StubOllama:
    """A local Ollama-compatible server, so the production client streams over HTTP."""

    def __init__(self) -> None:
        self.requests: list[int] = []  # perf_counter_ns at request-line arrival
        self.stream_seconds: list[float] = []  # per-request stream length; default 0
        self.finished = threading.Event()
        self.on_request: Callable[[], None] | None = None
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def do_POST(self) -> None:
                arrived = time.perf_counter_ns()
                body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/api/show":
                    reply = json.dumps(
                        {"model_info": {"stub.context_length": 400000}}
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(reply)))
                    self.end_headers()
                    self.wfile.write(reply)
                    return
                del body
                stub.requests.append(arrived)
                if stub.on_request is not None:
                    stub.on_request()
                seconds = stub.stream_seconds.pop(0) if stub.stream_seconds else 0.0
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                deadline = time.monotonic() + seconds
                while True:
                    # Dots only, so typed letters stay distinguishable in the render.
                    self.wfile.write(
                        json.dumps(
                            {
                                "message": {"role": "assistant", "content": ". "},
                                "done": False,
                            }
                        ).encode()
                        + b"\n"
                    )
                    self.wfile.flush()
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                self.wfile.write(
                    json.dumps(
                        {
                            "message": {"role": "assistant", "content": ""},
                            "done": True,
                            "done_reason": "stop",
                            "prompt_eval_count": 1,
                            "eval_count": 1,
                        }
                    ).encode()
                    + b"\n"
                )
                self.wfile.flush()
                stub.finished.set()

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Tui:
    """One real zeta CLI process on a pty that is its controlling tty."""

    def __init__(
        self,
        home: Path,
        sid: str | None,
        stub: StubOllama,
        cwd: Path,
        profile: str | None = None,
        profile_at_start: bool = False,
    ) -> None:
        self.output = bytearray()
        self.chunks: list[tuple[int, int]] = []
        self.exit_code: int | None = None
        site_dir = home.parent / "bench-site"
        site_dir.mkdir(exist_ok=True)
        (site_dir / "sitecustomize.py").write_text(SITECUSTOMIZE)
        env = {
            k: v
            for k, v in os.environ.items()
            if k
            not in {
                "COLORTERM",
                "NO_COLOR",
                "FORCE_COLOR",
                "CLICOLOR",
                "CLICOLOR_FORCE",
                "ZETA_BENCH_TREE",
            }
        }
        env.update(
            ZETA_HOME=str(home),
            ZETA_ACCOUNT_HOME=str(home),
            TERM="xterm-256color",
            COLORTERM="truecolor",
            PYTHONPATH=f"{site_dir}{os.pathsep}{TREE_SRC}",
            ZETA_OLLAMA_BASE_URL=stub.url,
        )
        if profile:
            env["ZETA_BENCH_PROFILE"] = profile
            if profile_at_start:
                env["ZETA_BENCH_PROFILE_AT_START"] = "1"
        # zeta has no __main__, so run the console-script entry point. Popen's setsid plus
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
        self.process = subprocess.Popen(
            argv,
            stdin=slave,
            stdout=slave,
            stderr=slave,
            env=env,
            cwd=cwd,
            start_new_session=True,
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
            "load_average": load(),
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
        start = len(self.output)
        sent = time.perf_counter_ns()
        os.write(self.master, char)
        first = self.wait_for(lambda: len(self.output) > start, timeout)
        shown = self.wait_for(
            lambda: char in ANSI.sub(b"", bytes(self.output[start:])), timeout
        )
        if shown is None:
            return None
        return {
            "first_byte_ms": (first - sent) / 1e6,
            "render_ms": (shown - sent) / 1e6,
            "bytes": len(self.output) - start,
        }

    def keys(self, count: int, gap: float = 0.05) -> dict[str, Any]:
        samples = []
        for _ in range(count):
            sample = self.key()
            if sample is not None:
                samples.append(sample)
            self.quiet(gap, timeout=gap * 20)
        return {
            "render": summary([s["render_ms"] for s in samples]),
            "first_byte": summary([s["first_byte_ms"] for s in samples]),
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
        self.selector.close()
        try:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=5)
        except ProcessLookupError:
            pass
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGKILL)
            self.process.wait()
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
        samples.append(ps_sample(tui.process.pid))
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
        "load_average": load(),
        **tui.failure(),
    }


def submit(
    tui: Tui,
    stub: StubOllama,
    text: bytes,
    stream_seconds: float,
    before_enter: Callable[[], None] | None = None,
) -> float | None:
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
    return (stub.requests[seen] - sent) / 1e6


def finish_turn(tui: Tui, stub: StubOllama) -> None:
    tui.wait_for(stub.finished.is_set, 60)
    tui.quiet(0.5)


def tui_probe(
    home: Path,
    sid: str | None,
    name: str,
    cwd: Path,
    stub: StubOllama,
    idle_seconds: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {"startups": []}
    for _ in range(5):
        tui = Tui(home, sid, stub, cwd)
        try:
            result["startups"].append(tui.startup(stub))
        finally:
            tui.close()
    ok = [s for s in result["startups"] if s["ready_ms"] is not None]
    result["startup_prompt"] = summary([s["prompt_ms"] for s in ok])
    result["startup_ready"] = summary([s["ready_ms"] for s in ok])
    if not ok:
        return result

    prefix = str(PROFILE_DIR / f"tui-{name}")
    tui = Tui(home, sid, stub, cwd, profile=f"{prefix}-idle")
    try:
        tui.startup(stub)
        result["idle"] = idle_probe(tui, idle_seconds, f"{prefix}-idle")
        result["key_idle"] = tui.keys(30)
        tui.clear_composer()
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
                    }
                )
        os.write(tui.master, b"\x1b[F")  # End: back to the live tail
        tui.quiet(0.3)
        result["scroll_page_up"] = {
            "first_byte": summary([s["first_byte_ms"] for s in scroll]),
            "last_byte": summary([s["last_byte_ms"] for s in scroll]),
            "first_sample": scroll[0] if scroll else None,
            "bytes_p50": statistics.median(s["bytes"] for s in scroll)
            if scroll
            else None,
        }
        turn_start = []
        for index in range(10):
            elapsed = submit(tui, stub, b"bench turn %d" % index, 0)
            if elapsed is None:
                result["turn_error"] = tui.failure() or {
                    "tail": ANSI.sub(b"", bytes(tui.output[-4096:])).decode(
                        "utf-8", "replace"
                    )
                }
                break
            turn_start.append(elapsed)
            finish_turn(tui, stub)
        result["turn_start_first_ms"] = turn_start[0] if turn_start else None
        result["turn_start"] = summary(turn_start)
        if turn_start and submit(tui, stub, b"bench stream", 10) is not None:
            result["key_streaming"] = tui.keys(30, gap=0.1)
            finish_turn(tui, stub)
        result.update(tui.failure())
    finally:
        tui.close()

    # Profiles run in their own process so profiler overhead never enters a timing above.
    tui = Tui(home, sid, stub, cwd, profile=f"{prefix}-span", profile_at_start=True)
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
        for key, index in (("startup_profile", 0), ("turn_profile", 1)):
            path = Path(f"{prefix}-span.{index}.prof")
            result[key] = (
                {"path": str(path), "top": top_cumulative(path)}
                if path.exists()
                else None
            )
    finally:
        stub.on_request = None
        tui.close()
    return result


def main() -> None:
    names = ARGS.session or ["fresh", "phoebe", "orchestrator"]
    probes = set(ARGS.probe or ["store", "scaling", "context", "serve", "tui"])
    child_zeta = subprocess.run(
        [sys.executable, "-c", "import zeta; print(zeta.__file__)"],
        env={**os.environ, "PYTHONPATH": str(TREE_SRC)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    base = Path(
        tempfile.mkdtemp(prefix="zeta-perf-", dir=Path.home() / ".zeta/scratch/perf")
    )
    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load_average": load(),
        "profiles": str(PROFILE_DIR),
        "zeta_file": zeta.__file__,
        "child_zeta_file": child_zeta,
    }
    print(f"zeta: {zeta.__file__}; child: {child_zeta}", file=sys.stderr)
    stub = StubOllama()
    try:
        home = make_home(base / "home", [DEFAULTS[n] for n in names if n != "fresh"])
        for name in names:
            if name == "fresh":
                opened = SessionManager(home).create(
                    provider="ollama", model="bench-stub", cwd=HERE, auto_project=False
                )
                sid = opened.metadata.session_id
                opened.store.close()
                cwd = HERE
            else:
                sid = DEFAULTS[name].name
                cwd = Path(
                    json.loads((DEFAULTS[name] / "meta.json").read_text())["cwd"]
                )
            source = home / "sessions" / sid
            rows, size = conversation_stats(source)
            children = (
                len(list((source / "agents").iterdir()))
                if (source / "agents").is_dir()
                else 0
            )
            result: dict[str, Any] = {
                "scale": {"rows": rows, "bytes": size, "children": children},
                "load_average": load(),
            }
            if "store" in probes:
                result["store"] = store_probe(home, sid, name)
            if "context" in probes:
                result["context"] = context_probe(home, sid)
            if "serve" in probes:
                result["serve"] = serve_probe(home, sid, stub.url)
            if "tui" in probes:
                result["tui"] = tui_probe(home, sid, name, cwd, stub, ARGS.idle_seconds)
            if "scaling" in probes and name != "fresh":
                curves = {}
                for fraction in (0.1, 0.5, 1.0):
                    scaled_home = base / f"scaled-{name}-{fraction}"
                    (scaled_home / "sessions").mkdir(parents=True)
                    advance(scaled_home, expected_epoch=0)
                    scaled = truncate(
                        DEFAULTS[name], scaled_home / "sessions", fraction
                    )
                    scaled_rows, scaled_bytes = conversation_stats(scaled)
                    store = store_probe(scaled_home, sid, f"{name}-{fraction}")
                    # Replay time is linear in bytes; ms/MB makes sizes comparable.
                    curves[str(fraction)] = {
                        "rows_in_file": scaled_rows,
                        "bytes": scaled_bytes,
                        "rows_replayed": store["rows_replayed"],
                        "replay_ms_per_mb": round(
                            store["replay"]["p50_ms"] / (scaled_bytes / 1e6), 3
                        ),
                        "store": store,
                    }
                    shutil.rmtree(scaled_home, ignore_errors=True)
                result["scaling"] = curves
            result["load_average_after"] = load()
            results[name] = result
            print(json.dumps({name: result}, indent=2), flush=True)
    finally:
        stub.close()
        if not ARGS.keep:
            shutil.rmtree(base, ignore_errors=True)
        print(
            json.dumps(
                {
                    "meta": {
                        k: results[k]
                        for k in ("generated_at", "zeta_file", "child_zeta_file")
                    }
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
