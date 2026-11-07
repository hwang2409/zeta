"""Isolation shared by the benchmark entry points.

One `Sandbox` owns a per-run throwaway root: copied homes, the local provider
stub, scrubbed environments, the network guard, and every spawned process.
Leaving the `with` block (normally, on an exception, or on SIGTERM) stops the
owned processes and removes the root unless `keep` is set. Imports no zeta code,
so `bench.py` can load it before it re-executes on the measured tree.
"""

from __future__ import annotations

import contextlib
import http.server
import json
import math
import os
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self
from urllib.parse import urlsplit

PERF = Path.home() / ".zeta/scratch/perf"
STUB_PROVIDER = "ollama"
STUB_MODEL = "bench-stub"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}
CREDENTIAL_MARKERS = ("ANTHROPIC", "OPENAI", "CLAUDE", "CODEX")
CREDENTIAL_SUFFIXES = ("API_KEY", "_TOKEN", "_SECRET", "PASSWORD")
# Copy-on-write clone on APFS; GNU cp reads -c as --preserve=context and fails.
CP_CLONE = ["cp", "-cR"] if sys.platform == "darwin" else ["cp", "-R"]

# Runs in-process (exec) and in every child (as sitecustomize). Any connect or name
# lookup to a non-local host, or a keychain read, is logged and refused, so no
# provider credential or endpoint can be reached even through a fallback path.
GUARD_SOURCE = """
import os as _os, sys as _sys
_guard_log = _os.environ.get("ZETA_BENCH_GUARD_LOG")
if _guard_log:
    _guard_local = {"127.0.0.1", "localhost", "::1"}
    def _guard_deny(what):
        with open(_guard_log, "a") as handle:
            handle.write(f"pid {_os.getpid()}: {what}\\n")
        raise PermissionError(f"zeta bench network guard refused {what}")
    def _guard_hook(event, args):
        if event == "socket.connect":
            address = args[1]
            if isinstance(address, tuple) and address[0] not in _guard_local:
                _guard_deny(f"connect {address!r}")
        elif event == "socket.getaddrinfo":
            host = args[0]
            if isinstance(host, bytes):
                host = host.decode(errors="replace")
            if host and host not in _guard_local:
                _guard_deny(f"resolve {host!r}")
        elif event == "subprocess.Popen":
            argv = args[1]
            first = args[0] or (argv[0] if isinstance(argv, (list, tuple)) and argv else argv)
            if first and _os.path.basename(_os.fsdecode(first)) == "security":
                _guard_deny("keychain read")
    _sys.addaudithook(_guard_hook)
"""


def is_credential_name(name: str) -> bool:
    upper = name.upper()
    return (
        any(marker in upper for marker in CREDENTIAL_MARKERS)
        or upper.endswith(CREDENTIAL_SUFFIXES)
        or upper == "ZETA_ALLOW_API_KEY"
    )


def load1() -> float:
    return round(os.getloadavg()[0], 2)


def timed(fn: Callable[[], Any]) -> tuple[Any, tuple[float, float]]:
    """Run fn once; return its value and one (elapsed ms, 1-minute load) sample."""
    started = time.perf_counter_ns()
    value = fn()
    return value, ((time.perf_counter_ns() - started) / 1e6, load1())


def summary(samples: list[tuple[float, float]]) -> dict[str, Any]:
    """p50/p95 from the elapsed values; every sample keeps its own load reading."""
    if not samples:
        return {"n": 0}
    ordered = sorted(ms for ms, _ in samples)
    loads = [load for _, load in samples]
    return {
        "n": len(ordered),
        "p50_ms": round(statistics.median(ordered), 3),
        "p95_ms": round(ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)], 3),
        "min_ms": round(ordered[0], 3),
        "max_ms": round(ordered[-1], 3),
        "load_range": [min(loads), max(loads)],
        "samples": [[round(ms, 3), load] for ms, load in samples],
    }


def pin_stub_provider(session: Path) -> dict[str, Any]:
    """Point a copied session at the local stub; return the copy's metadata."""
    meta_path = session / "meta.json"
    meta = json.loads(meta_path.read_text())
    # Resume uses the provider stored here, not the CLI or constructor provider.
    meta.update(provider=STUB_PROVIDER, model=STUB_MODEL, selected_accounts={})
    meta_path.write_text(json.dumps(meta))
    return meta


class StubOllama:
    """A local Ollama-compatible server, so the production client streams over HTTP."""

    def __init__(self) -> None:
        self.requests: list[int] = []  # perf_counter_ns at request-line arrival
        self.lookups = 0  # model lookups (/api/show), e.g. one per resume
        self.stream_seconds: list[float] = []  # per-request stream length; default 0
        self.finished = threading.Event()
        self.on_request: Callable[[], None] | None = None
        stub = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args: Any) -> None:
                pass

            def _line(self, content: str, done: bool) -> None:
                frame: dict[str, Any] = {
                    "message": {"role": "assistant", "content": content},
                    "done": done,
                }
                if done:
                    frame.update(done_reason="stop", prompt_eval_count=1, eval_count=1)
                self.wfile.write(json.dumps(frame).encode() + b"\n")
                self.wfile.flush()

            def do_POST(self) -> None:
                arrived = time.perf_counter_ns()
                self.rfile.read(int(self.headers.get("Content-Length", 0)))
                if self.path == "/api/show":
                    stub.lookups += 1
                    reply = json.dumps(
                        {"model_info": {"stub.context_length": 400000}}
                    ).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(reply)))
                    self.end_headers()
                    self.wfile.write(reply)
                    return
                stub.requests.append(arrived)
                if stub.on_request is not None:
                    stub.on_request()
                seconds = stub.stream_seconds.pop(0) if stub.stream_seconds else 0.0
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                deadline = time.monotonic() + seconds
                while True:
                    self._line(". ", False)  # dots keep typed letters distinguishable
                    if time.monotonic() >= deadline:
                        break
                    time.sleep(0.05)
                self._line("", True)
                stub.finished.set()

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_address[1]}"
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


class Sandbox:
    def __init__(self, prefix: str, *, keep: bool = False) -> None:
        self.prefix = prefix
        self.keep = keep
        self.processes: list[subprocess.Popen[Any]] = []
        self.stub: StubOllama | None = None

    def __enter__(self) -> Self:
        # The root exists only inside the with block: __exit__ does not run when
        # __enter__ raises, so setup failures clean up here.
        PERF.mkdir(parents=True, exist_ok=True)
        self.root = Path(tempfile.mkdtemp(prefix=f"{self.prefix}-", dir=PERF))
        self.guard_log = self.root / "network-guard.log"
        self.site = self.root / "site"
        try:
            self.site.mkdir()
            # External timeouts (lockf, kill) send SIGTERM; make it run the cleanup.
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
            self.stub = StubOllama()
            env = self.env()
            os.environ.clear()
            os.environ.update(env)
            exec(GUARD_SOURCE, {})  # noqa: S102 - the guard is this module's own code
            self.write_sitecustomize("")
        except BaseException:
            self._cleanup()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        violations = self._cleanup()
        if violations:
            raise SystemExit(f"network guard refused non-local access:\n{violations}")

    def _cleanup(self) -> str:
        """Stop owned processes and the stub, remove the root; return guard refusals."""
        try:
            for process in self.processes:
                self.stop(process)
            if self.stub is not None:
                self.stub.close()
        finally:
            violations = self.guard_log.read_text() if self.guard_log.exists() else ""
            if self.keep:
                print(f"kept {self.root}", file=sys.stderr)
            else:
                shutil.rmtree(self.root, ignore_errors=True)
        return violations

    def env(self, home: Path | None = None, **extra: str) -> dict[str, str]:
        """The process environment without credential sources, pinned to the stub."""
        env = {k: v for k, v in os.environ.items() if not is_credential_name(k)}
        env.update(
            # Nonexistent paths: Codex/Claude bootstrap finds no file and, with
            # CLAUDE_CONFIG_DIR set, never falls back to the keychain.
            CODEX_HOME=str(self.root / "no-codex"),
            CLAUDE_CONFIG_DIR=str(self.root / "no-claude"),
            ZETA_OLLAMA_BASE_URL=self.stub_url,
            ZETA_BENCH_GUARD_LOG=str(self.guard_log),
        )
        if home is not None:
            env.update(ZETA_HOME=str(home), ZETA_ACCOUNT_HOME=str(home))
        env.update(extra)
        return env

    @property
    def stub_url(self) -> str:
        assert self.stub is not None
        return self.stub.url

    def use_home(self, home: Path) -> None:
        """Point in-process zeta code at a copied home."""
        os.environ.update(ZETA_HOME=str(home), ZETA_ACCOUNT_HOME=str(home))

    def write_sitecustomize(self, extra: str) -> None:
        """Every child gets the guard; `extra` adds code after it (e.g. a profiler)."""
        (self.site / "sitecustomize.py").write_text(GUARD_SOURCE + extra)

    def check_stubbed(self, home: Path) -> None:
        """Fail before any probe if a copied session could pick a real provider."""
        for meta_path in (home / "sessions").glob("*/meta.json"):
            provider = json.loads(meta_path.read_text()).get("provider")
            if provider != STUB_PROVIDER:
                raise SystemExit(f"{meta_path}: provider {provider!r} is not the stub")
        host = urlsplit(os.environ.get("ZETA_OLLAMA_BASE_URL", "")).hostname
        if host != "127.0.0.1":
            raise SystemExit(f"ollama base url host {host!r} is not 127.0.0.1")

    def spawn(self, argv: list[str], **kwargs: Any) -> subprocess.Popen[Any]:
        """Start an owned child in its own process group; the sandbox stops it."""
        process = subprocess.Popen(argv, start_new_session=True, **kwargs)
        self.processes.append(process)
        return process

    @staticmethod
    def stop(process: subprocess.Popen[Any]) -> None:
        """SIGTERM the group, wait, then SIGKILL what is left of it and wait again."""
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        finally:
            # Also reaches descendants that outlived the group leader.
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10)
