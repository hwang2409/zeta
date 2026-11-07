"""The benchmark sandbox keeps the serve probe off the real provider.

The probe runs in a child process because the sandbox installs a permanent
audit hook and replaces os.environ; neither may leak into the pytest process.
"""

from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_sandbox():
    spec = importlib.util.spec_from_file_location(
        "bench_sandbox", ROOT / "scripts/bench/sandbox.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# The serve probe path from scripts/bench/bench.py (sandbox, pinned clone,
# in-process ZetaServer, hello, resume) on a session whose stored provider is
# claude, then one send so a turn reaches whatever provider resume selected.
# The keychain fallback stays reachable, as in the 2026-11-07 runs: ZETA_HOME
# and CLAUDE_CONFIG_DIR are not exported, and the keychain read is stubbed to
# return fake Claude tokens. A recorder hook, added after the guard, sees every
# connect and name lookup the guard lets through and refuses non-local ones, so
# the test never reaches the network.
PROBE = r"""
import asyncio, json, os, shutil, sys, tempfile, time
from pathlib import Path

sys.path.insert(0, sys.argv[1])
out = Path(sys.argv[2])
from sandbox import Sandbox, pin_stub_provider

from zeta.core.session import SessionManager
from zeta.oauth import OAuthTokens
from zeta.providers import anthropic, retry_policy
from zeta.server import ZetaServer

LOCAL = {"127.0.0.1", "localhost", "::1"}
escaped, keychain_reads = [], []


def recorder(event, args):
    host = None
    if event == "socket.getaddrinfo":
        host = args[0].decode() if isinstance(args[0], bytes) else args[0]
    elif event == "socket.connect" and isinstance(args[1], tuple):
        host = args[1][0]
    if host and host not in LOCAL:
        escaped.append(f"{event} {host}")
        raise ConnectionRefusedError(f"test recorder refused {host}")


def fake_keychain():
    keychain_reads.append(1)
    return OAuthTokens("fake-access", "fake-refresh", time.time() + 3600)


anthropic._keychain_claude_tokens = fake_keychain
# Each refused attempt is retried; skip the backoff so the test stays fast.
retry_policy.retry_wait_seconds = lambda *args: 0.0


async def probe(home, sid):
    # A short path: AF_UNIX paths are limited to about 100 bytes.
    sock = Path(tempfile.mkdtemp(prefix="zb-", dir="/tmp")) / "s.sock"
    server = await asyncio.to_thread(
        lambda: ZetaServer(
            home=home,
            socket_path=sock,
            provider="ollama",
            model="bench-stub",
            auto_memory=False,
            approval_reconnect_grace=0,
        )
    )
    await server.start()
    reader, writer = await asyncio.open_unix_connection(str(sock), limit=1 << 26)
    ident = 0

    async def call(method, params):
        nonlocal ident
        ident += 1
        line = {"jsonrpc": "2.0", "id": ident, "method": method, "params": params}
        writer.write((json.dumps(line) + "\n").encode())
        await writer.drain()

    async def frame():
        return json.loads(await asyncio.wait_for(reader.readline(), 60))

    try:
        await call("hello", {"protocol_version": "1.2"})
        await call("resume", {"session_id": sid})
        await call("send", {"text": "hi"})
        while (event := (await frame()).get("params", {}).get("event")) not in {
            "agent_end",
            "error",
        }:
            pass
        return event
    finally:
        writer.close()
        await server.close()
        sock.unlink(missing_ok=True)
        shutil.rmtree(sock.parent, ignore_errors=True)


result = {}
try:
    with Sandbox("sandbox-test") as box:
        home = box.root / "home"
        sessions = {}
        for name in ("pinned", "unpinned"):
            opened = SessionManager(home).create(
                provider="claude",
                model="claude-opus-5-5",
                cwd=box.root,
                auto_project=False,
            )
            sessions[name] = opened.metadata.session_id
            opened.store.close()
        pin_stub_provider(home / "sessions" / sessions["pinned"])
        box.use_home(home)
        for name in ("ZETA_HOME", "ZETA_ACCOUNT_HOME", "CLAUDE_CONFIG_DIR"):
            os.environ.pop(name, None)
        sys.addaudithook(recorder)
        for name, sid in sessions.items():
            seen = len(box.stub.requests)
            reads = len(keychain_reads)
            end = asyncio.run(probe(home, sid))
            result[name] = {
                "end": end,
                "stub_requests": len(box.stub.requests) - seen,
                "keychain_reads": len(keychain_reads) - reads,
            }
except SystemExit as exc:
    result["guard"] = str(exc)
result["escaped"] = escaped
out.write_text(json.dumps(result))
"""


def test_serve_probe_reaches_only_the_local_stub(tmp_path: Path) -> None:
    out = tmp_path / "result.json"
    env = {
        key: value
        for key, value in os.environ.items()
        # The conftest exports ZETA_HOME and a scripted provider; the 2026-11-07
        # runs had neither, so resume picked the stored provider.
        if key not in {"ZETA_HOME", "ZETA_TEST_SCRIPTED_PROVIDER"}
    }
    # A throwaway HOME: no ~/.claude, ~/.codex, or account home is readable.
    env.update(
        HOME=str(tmp_path),
        PYTHONPATH=os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")]),
    )
    run = subprocess.run(
        [sys.executable, "-c", PROBE, str(ROOT / "scripts/bench"), str(out)],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert run.returncode == 0, run.stderr
    result = json.loads(out.read_text())

    # Nothing passed the guard to a non-local host.
    assert result["escaped"] == []
    # The pinned clone (what the harness runs) streams from the local stub only.
    assert result["pinned"] == {
        "end": "agent_end",
        "stub_requests": 1,
        "keychain_reads": 0,
    }
    # A clone left on claude bootstraps from the (fake) keychain, and the guard
    # refuses its provider host before any connect; the sandbox fails the run.
    assert result["unpinned"]["stub_requests"] == 0
    assert result["unpinned"]["keychain_reads"] >= 1
    assert "api.anthropic.com" in result["guard"]


@pytest.mark.parametrize("failing", ["stub", "env"])
def test_sandbox_setup_failure_removes_root_and_closes_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failing: str
) -> None:
    sandbox = _load_sandbox()
    perf = tmp_path / "perf"
    monkeypatch.setattr(sandbox, "PERF", perf)
    stubs = []

    class RecordingStub(sandbox.StubOllama):
        def __init__(self) -> None:
            if failing == "stub":
                raise OSError("stub bind failed")
            super().__init__()
            self.closed = False
            stubs.append(self)

        def close(self) -> None:
            super().close()
            self.closed = True

    def failing_env(self, *args, **kwargs):
        raise RuntimeError("env failed")

    monkeypatch.setattr(sandbox, "StubOllama", RecordingStub)
    if failing == "env":
        # Fails after the stub starts and before the environment or guard change.
        monkeypatch.setattr(sandbox.Sandbox, "env", failing_env)
    sigterm = signal.getsignal(signal.SIGTERM)
    try:
        with pytest.raises((OSError, RuntimeError)), sandbox.Sandbox("setup-failure"):
            pytest.fail("the with body must not run")
    finally:
        signal.signal(signal.SIGTERM, sigterm)
        for stub in stubs:
            if not stub.closed:
                stub.close()

    assert list(perf.iterdir()) == []
    assert [stub.closed for stub in stubs] == ([] if failing == "stub" else [True])
