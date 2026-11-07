#!/usr/bin/env python3
"""Exploratory production benchmark for copied zeta sessions.

The script is deliberately not a CI benchmark. It records operation scale and
load averages beside noisy timings. Copies are removed unless --keep is used.
"""

from __future__ import annotations

import argparse
import asyncio
import cProfile
import io
import json
import os
import pstats
import pty
import shutil
import signal
import statistics
import subprocess
import tempfile
import termios
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from zeta.core.context import ContextAssembler
from zeta.core.session import SessionManager
from zeta.server import ZetaServer
from zeta.storage_format import advance

ROOT = Path(__file__).resolve().parents[3]
DEFAULTS = {
    "phoebe": Path.home() / ".zeta/sessions/7897c1aed2e5444caba907f462306b10",
    "orchestrator": Path.home() / ".zeta/sessions/86e3d775a40b42ceaa76ca571e27ff1b",
}
PROFILE_DIR = Path.home() / ".zeta/scratch/perf/profiles"


def load() -> tuple[float, float, float]:
    try:
        return os.getloadavg()
    except OSError:
        return (float("nan"),) * 3


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
        "p95_ms": round(ordered[max(0, int(len(ordered) * 0.95) - 1)], 3),
        "n": len(ordered),
        "load_average": load(),
    }


def profile_call(name: str, fn: Callable[[], Any]) -> tuple[Any, str, str]:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    path = PROFILE_DIR / f"{name}.prof"
    profiler = cProfile.Profile()
    value = profiler.runcall(fn)
    profiler.dump_stats(path)
    text = io.StringIO()
    pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(15)
    hot = next(
        (line.strip() for line in text.getvalue().splitlines() if ".py:" in line),
        "profile unavailable",
    )
    return value, str(path), hot


def conversation_stats(session: Path) -> tuple[int, int]:
    path = session / "conversation.jsonl"
    with path.open() as handle:
        rows = sum(1 for _ in handle)
    return rows, path.stat().st_size


def clone(source: Path, destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    target = destination / source.name
    subprocess.run(["cp", "-cR", str(source), str(target)], check=True)
    return target


def truncate(source: Path, destination: Path, fraction: float) -> Path:
    target = clone(source, destination)
    conversation = target / "conversation.jsonl"
    lines = conversation.read_bytes().splitlines(keepends=True)
    keep = max(1, int(len(lines) * fraction))
    conversation.write_bytes(b"".join(lines[:keep]))
    # Derived indexes and reconciliation ledgers must not describe removed rows.
    for path in target.rglob("*.sqlite*"):
        path.unlink(missing_ok=True)
    return target


def store_probe(home: Path, sid: str, repeats: int = 10) -> dict[str, Any]:
    manager = SessionManager(home)
    opened, open_ms = timed(lambda: manager.open(sid, _read_only=True))
    store = opened.store
    replay_times: list[float] = []
    message_times: list[float] = []
    profile_path = ""
    hot = ""
    try:
        for _ in range(repeats):

            def replay() -> list[Any]:
                nonlocal profile_path, hot
                entries, profile_path, hot = profile_call("replay", store.replay)
                return entries

            entries, elapsed = timed(replay)
            replay_times.append(elapsed)
            _, message_elapsed = timed(store.messages)
            message_times.append(message_elapsed)
        return {
            "open_resume": summary([open_ms] * repeats),
            "replay": summary(replay_times),
            "messages": summary(message_times),
            "rows_replayed": len(entries),
            "messages_count": len(store.messages()),
            "profile": profile_path,
            "profile_hot_path": hot,
        }
    finally:
        store.close()


def pty_once(home: Path, sid: str) -> tuple[float | None, float, int]:
    env = os.environ.copy()
    env.update(ZETA_HOME=str(home), ZETA_ACCOUNT_HOME=str(home), TERM="xterm-256color")
    master, slave = pty.openpty()
    started = time.perf_counter_ns()
    process = subprocess.Popen(
        ["zeta", "--resume", sid, "--no-auto-memory", "--provider", "codex"],
        stdin=slave,
        stdout=slave,
        stderr=slave,
        env=env,
        cwd=ROOT,
        start_new_session=True,
    )
    os.close(slave)
    try:
        os.set_blocking(master, False)
        data = b""
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            try:
                data += os.read(master, 65536)
            except BlockingIOError:
                time.sleep(0.01)
            if time.monotonic() - (deadline - 15) > 0.5 and data:
                # startup is ready when output has been quiet for 500 ms
                quiet_until = time.monotonic() + 0.5
                while time.monotonic() < quiet_until:
                    try:
                        chunk = os.read(master, 65536)
                        if chunk:
                            data += chunk
                            quiet_until = time.monotonic() + 0.5
                    except BlockingIOError:
                        time.sleep(0.01)
                break
        startup_ms = (time.perf_counter_ns() - started) / 1e6
        # Disable kernel echo only after prompt_toolkit has configured the pty.
        # The marker must come from the real render, not line-discipline echo.
        attrs = termios.tcgetattr(master)
        attrs[3] &= ~termios.ECHO
        termios.tcsetattr(master, termios.TCSANOW, attrs)
        marker = f"qzx{time.time_ns()}".encode()
        key_started = time.perf_counter_ns()
        try:
            for byte in marker:
                os.write(master, bytes((byte,)))
                time.sleep(0.01)
        except OSError:
            return None, startup_ms, len(data)
        deadline = time.monotonic() + 4
        while time.monotonic() < deadline and marker not in data:
            try:
                data += os.read(master, 65536)
            except BlockingIOError:
                time.sleep(0.005)
        key_ms = (
            (time.perf_counter_ns() - key_started) / 1e6 if marker in data else None
        )
        return key_ms, startup_ms, len(data)
    finally:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            process.wait()
        os.close(master)


def pty_probe(home: Path, sid: str) -> dict[str, Any]:
    startup: list[float] = []
    keys: list[float] = []
    bytes_seen = 0
    for _ in range(5):
        key, start, count = pty_once(home, sid)
        startup.append(start)
        bytes_seen += count
        if key is not None:
            keys.append(key)
    # A fresh process is required for each sample to avoid buffered terminal data.
    for _ in range(30):
        key, _, count = pty_once(home, sid)
        bytes_seen += count
        if key is not None:
            keys.append(key)
    return {
        "startup": summary(startup),
        "key_to_render": summary(keys),
        "bytes": bytes_seen,
    }


def context_probe(home: Path, sid: str) -> dict[str, Any]:
    manager = SessionManager(home)
    opened = manager.open(sid, _read_only=True)
    store = opened.store
    try:
        timings: list[float] = []
        for _ in range(3):

            def assemble() -> Any:
                assembler = ContextAssembler(
                    store,
                    token_budget=opened.metadata.compaction_budget,
                    retained_tail=opened.metadata.retained_tail,
                    system_prompt=opened.metadata.system_prompt,
                )
                return asyncio.run(assembler.assemble())

            _, elapsed = timed(assemble)
            timings.append(elapsed)
        return {
            "context_assembly": summary(timings),
            "provider_request": "not sent; provider boundary stubbed by omission",
        }
    finally:
        store.close()


async def serve_probe(home: Path, sid: str) -> dict[str, Any]:
    socket = home / "bench.sock"
    server = await asyncio.to_thread(
        lambda: ZetaServer(
            home=home,
            socket_path=socket,
            provider="codex",
            auto_memory=False,
            approval_reconnect_grace=0,
        )
    )
    await server.start()

    async def request(
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        ident: int,
        method: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        writer.write(
            (
                json.dumps(
                    {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
                )
                + "\n"
            ).encode()
        )
        await writer.drain()
        while True:
            frame = json.loads(await reader.readline())
            if frame.get("id") == ident:
                return frame

    try:
        reader, writer = await asyncio.open_unix_connection(str(socket))
        await request(
            reader,
            writer,
            1,
            "hello",
            {
                "protocol_version": "1.2",
                "features": ["session_history_fidelity", "session_host_commands"],
            },
        )
        timings: dict[str, list[float]] = {"resume": [], "history": [], "status": []}
        for i in range(3):
            started = time.perf_counter_ns()
            await request(reader, writer, 10 + i, "resume", {"session_id": sid})
            timings["resume"].append((time.perf_counter_ns() - started) / 1e6)
            started = time.perf_counter_ns()
            await request(
                reader,
                writer,
                20 + i,
                "session_history",
                {"session_id": sid, "limit": 100, "offset": 0},
            )
            timings["history"].append((time.perf_counter_ns() - started) / 1e6)
            started = time.perf_counter_ns()
            await request(reader, writer, 30 + i, "status", {})
            timings["status"].append((time.perf_counter_ns() - started) / 1e6)
        writer.close()
        await writer.wait_closed()
        return {name: summary(values) for name, values in timings.items()}
    finally:
        await server.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", action="append", choices=[*DEFAULTS, "fresh"])
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    names = args.session or ["fresh", "phoebe", "orchestrator"]
    base = Path(
        tempfile.mkdtemp(prefix="zeta-perf-", dir=Path.home() / ".zeta/scratch/perf")
    )
    home = base / "home"
    (home / "sessions").mkdir(parents=True)
    advance(home, expected_epoch=0)
    results: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "load_average": load(),
        "profiles": str(PROFILE_DIR),
    }
    try:
        for name in names:
            if name == "fresh":
                manager = SessionManager(home)
                opened = manager.create(
                    provider="codex", model="offline", cwd=ROOT, auto_project=False
                )
                sid = opened.metadata.session_id
                opened.store.close()
                source = home / "sessions" / sid
            else:
                source = clone(DEFAULTS[name], home / "sessions")
                sid = source.name
            rows, size = conversation_stats(source)
            result: dict[str, Any] = {
                "scale": {
                    "rows": rows,
                    "bytes": size,
                    "children": len(list((source / "agents").iterdir()))
                    if (source / "agents").is_dir()
                    else 0,
                },
                "load_average": load(),
                "store": store_probe(home, sid),
            }
            if name != "fresh":
                result["pty"] = pty_probe(home, sid)
                result["context"] = context_probe(home, sid)
                result["serve"] = asyncio.run(serve_probe(home, sid))
                curves = {}
                for fraction in (0.1, 0.5, 1.0):
                    scaled_home = (
                        base / f"scaled-{name}-{fraction} /".replace(" ", "") / "home"
                    )
                    scaled_home.mkdir(parents=True)
                    advance(scaled_home, expected_epoch=0)
                    scaled = truncate(
                        DEFAULTS[name], scaled_home / "sessions", fraction
                    )
                    curves[str(fraction)] = {
                        "scale": conversation_stats(scaled),
                        "store": store_probe(scaled_home, sid, repeats=3),
                    }
                result["scaling"] = curves
                result["load_average_after"] = load()
            results[name] = result
        print(json.dumps(results, indent=2))
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    main()
