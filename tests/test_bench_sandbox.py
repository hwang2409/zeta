"""The benchmark sandbox keeps the serve probe off the real provider.

The serve probe runs as bench.py itself in a child process: the harness's own
home copy, provider pinning, and serve probe, on two fixture source sessions.
The sandbox installs a permanent audit hook and replaces os.environ; neither
may leak into the pytest process.
"""

from __future__ import annotations

import importlib.util
import json
import os
import pwd
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import uuid
from collections.abc import Iterator
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


# Installed as sitecustomize in the bench.py process, so it applies to the
# unmodified harness. It records the provider traffic of the run: each TCP
# connect to a loopback host (the local stub), each read of the Claude keychain
# fallback (stubbed: fake tokens, the real keychain is never read), each
# connect or name lookup to a non-local host, and each file open, listing,
# change, copy, or spawned command line that names a path under the real
# ~/.zeta (BENCH_TEST_FORBIDDEN). The last two end the process at once, so
# nothing reaches the network or the real sessions. Once the harness adds its
# own audit hook (the sandbox network guard), the recorder moves behind it and
# sees only what the guard let through, and it makes one lookup of a .invalid
# host that the guard must refuse and log.
SITECUSTOMIZE = r"""
import os, sys, time

_LOCAL = {"127.0.0.1", "localhost", "::1"}
_PROBE_HOST = "bench-guard-probe.invalid"
_moved = []


def _note(line):
    with open(os.environ["BENCH_TEST_RECORD"], "a") as handle:
        handle.write(line + "\n")


_FORBIDDEN = os.path.realpath(os.environ["BENCH_TEST_FORBIDDEN"])
# Events that name a path (first, and for two-path events second, argument).
_PATH_EVENTS = {
    "open", "os.mkdir", "os.rmdir", "os.remove", "os.listdir", "os.scandir",
    "os.chmod", "os.chown", "os.chflags", "os.utime", "os.truncate",
    "os.rename", "os.link", "os.symlink", "shutil.copyfile", "shutil.copymode",
    "shutil.copystat", "shutil.copytree", "shutil.move", "shutil.rmtree",
    "shutil.chown",
}


def _forbidden(path):
    if path is None or isinstance(path, int):
        return False
    try:
        path = os.path.realpath(os.fsdecode(path))
    except (TypeError, ValueError):
        return False
    return path == _FORBIDDEN or path.startswith(_FORBIDDEN + os.sep)


def _paths(event, args):
    if event in _PATH_EVENTS:
        return [args[0] if args[0] is not None else "."] + (
            [args[1]] if event.startswith(("os.rename", "os.link", "os.symlink",
                                           "shutil.copy", "shutil.move"))
            else []
        )
    if event == "subprocess.Popen":  # (executable, args, cwd, env): cp, rm, ...
        argv = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
        return [args[0], *argv, args[2]]
    if event in ("os.posix_spawn", "os.exec"):  # (path, argv, env)
        return [args[0], *args[1]]
    return []


def _record(event, args):
    for path in _paths(event, args):
        if _forbidden(path):
            _note(f"forbidden {event} {os.fsdecode(path)}")
            os._exit(98)
    host = None
    if event == "socket.getaddrinfo":
        host = args[0].decode() if isinstance(args[0], bytes) else args[0]
    elif event == "socket.connect" and isinstance(args[1], tuple):
        host = args[1][0]
    if host in _LOCAL and event == "socket.connect":
        _note("loopback connect")
    elif host and host not in _LOCAL:
        _note(f"{event} {host}")
        os._exit(97)


def _early(event, args):
    if not _moved:
        _record(event, args)


_add = sys.addaudithook


def _probe_guard():
    # One controlled non-local lookup right after the harness guard is in
    # place: the guard must refuse it (PermissionError) and log the refusal.
    # Without the guard, the recorder sees the lookup and ends the run (97).
    import socket

    try:
        socket.getaddrinfo(_PROBE_HOST, 443)
    except PermissionError:
        _note("guard refused probe")
    log = os.environ.get("ZETA_BENCH_GUARD_LOG")
    if log and os.path.exists(log):
        with open(log) as handle:
            lines = handle.read().splitlines()
        for line in lines:
            if _PROBE_HOST in line:
                _note(f"guard log: {line}")
        # Drop only the probe's own line, so the harness still fails the run
        # on any other refusal.
        with open(log, "w") as handle:
            handle.writelines(f"{x}\n" for x in lines if _PROBE_HOST not in x)


def _add_then_record(hook):
    _add(hook)
    if not _moved:
        _moved.append(hook)
        _add(_record)
        _probe_guard()


sys.addaudithook = _add_then_record
_add(_early)

from zeta.oauth import OAuthTokens
from zeta.providers import anthropic


def _fake_keychain():
    _note("keychain read")
    return OAuthTokens("fake-access", "fake-refresh", time.time() + 3600)


anthropic._keychain_claude_tokens = _fake_keychain
"""

# bench.py's default sources, resolved under the throwaway HOME.
SOURCES = {
    "phoebe": "7897c1aed2e5444caba907f462306b10",
    "orchestrator": "86e3d775a40b42ceaa76ca571e27ff1b",
}


def _source_sessions(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Two sessions stored with provider claude, each with an undelivered child
    notification, so a serve resume starts a turn on the stored provider."""
    from zeta.core import session
    from zeta.core.session import SessionManager

    real_uuid4 = uuid.uuid4
    for sid in SOURCES.values():
        ids = iter([uuid.UUID(sid)])
        monkeypatch.setattr(
            session.uuid, "uuid4", lambda ids=ids: next(ids, None) or real_uuid4()
        )
        opened = SessionManager(home / ".zeta").create(
            provider="claude",
            model="claude-opus-5-5",
            cwd=home,
            auto_project=False,
        )
        try:
            assert opened.metadata.session_id == sid
            opened.store.append_agent_notification(
                f"{sid}:1",
                child_session_path=str(home / "child"),
                description="child",
                status="completed",
                text="done",
            )
        finally:
            opened.store.close()
    monkeypatch.setattr(session.uuid, "uuid4", real_uuid4)


@pytest.fixture
def short_home() -> Iterator[Path]:
    # bench.py binds its serve socket inside the home; AF_UNIX paths are
    # limited to about 100 bytes, which pytest's tmp_path can exceed.
    home = Path(tempfile.mkdtemp(prefix="zb-", dir="/tmp"))
    try:
        yield home
    finally:
        shutil.rmtree(home, ignore_errors=True)


def test_bench_serve_probe_reaches_only_the_local_stub(
    tmp_path: Path, short_home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real home, resolved from the account database before anything
    # changes HOME. The child's audit hook ends the run on any path under it.
    real_zeta = Path(pwd.getpwuid(os.getuid()).pw_dir) / ".zeta"
    # Any zeta code in this process resolves its home to a temp dir.
    monkeypatch.setenv("ZETA_HOME", str(tmp_path / "pytest-zeta-home"))
    home = short_home
    _source_sessions(home, monkeypatch)
    site = tmp_path / "site"
    site.mkdir()
    (site / "sitecustomize.py").write_text(SITECUSTOMIZE)
    record = tmp_path / "record.log"
    env = {
        key: value
        for key, value in os.environ.items()
        # As in the 2026-11-07 runs: no zeta home, account home, Claude config
        # dir, scripted provider, or credential variable is exported, so a
        # session left on claude falls back to the (stubbed) keychain. The child
        # cannot get ZETA_HOME: Claude bootstrap skips the keychain whenever it
        # is set (providers/anthropic.py, AnthropicCredentialDriver.bootstrap),
        # and that fallback must stay reachable for this test to catch it. The
        # real ~/.zeta is protected instead by the throwaway HOME, the audit
        # hook (BENCH_TEST_FORBIDDEN).
        if not key.startswith("ZETA_")
        and not any(
            m in key.upper() for m in ("ANTHROPIC", "CLAUDE", "OPENAI", "CODEX")
        )
    }
    src = str(ROOT / "src")
    env.update(
        HOME=str(home),  # throwaway: bench.py reads its sources under ~/.zeta
        PYTHONPATH=os.pathsep.join([str(site), src]),
        ZETA_BENCH_TREE=src,  # already on the tree, so bench.py keeps PYTHONPATH
        BENCH_TEST_RECORD=str(record),
        BENCH_TEST_FORBIDDEN=str(real_zeta),
    )
    try:
        run = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/bench/bench.py"),
                "--tree",
                str(ROOT),
                *("--session", "phoebe", "--session", "orchestrator"),
                *("--probe", "serve"),
            ],
            env=env,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        returncode, stderr = run.returncode, run.stderr
    except subprocess.TimeoutExpired as exc:
        # The run is killed; the traffic it recorded is still checked first.
        # The last progress lines name the step that was still running.
        err = exc.stderr or b""
        tail = (err.decode() if isinstance(err, bytes) else err).splitlines()[-20:]
        run, returncode, stderr = None, "timed out", "\n".join(tail)

    # Provider traffic went to the local stub only: no keychain read and no
    # connect or lookup to a non-local host got past the harness. The harness
    # guard refused the controlled probe lookup and logged it. No path under
    # the real ~/.zeta was opened, listed, changed, or passed to a command.
    lines = record.read_text().splitlines() if record.exists() else []
    forbidden = [line for line in lines if line.startswith("forbidden ")]
    probe = [line for line in lines if line.startswith("guard ")]
    assert {
        "forbidden": forbidden,
        "guard_probe": [re.sub(r"pid \d+:", "pid N:", line) for line in probe],
        "outside": [
            line
            for line in lines
            if line != "loopback connect" and line not in forbidden + probe
        ],
        "reached_loopback": "loopback connect" in lines,
    } == {
        "forbidden": [],
        "guard_probe": [
            "guard refused probe",
            "guard log: pid N: resolve 'bench-guard-probe.invalid'",
        ],
        "outside": [],
        "reached_loopback": True,
    }, stderr
    assert returncode == 0, stderr
    assert run is not None
    decoder, text, results = json.JSONDecoder(), run.stdout, {}
    while text := text.lstrip():
        value, end = decoder.raw_decode(text)
        results.update(value)
        text = text[end:]
    # Every resume of each copied session looked up its model at the local stub.
    for name in SOURCES:
        assert results[name]["serve"]["stub_lookups_per_server"] == [1] * 5


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
