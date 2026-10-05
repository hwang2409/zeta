"""Opt-in integration tests against the dedicated Lima VM.

Run with ``ZETA_COMPUTER_DOCKER=1`` after ``zeta computer setup``. The suite
fakes ``$HOME``, so these tests point Lima at the real ``~/.lima`` unless
``LIMA_HOME`` is already set.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from queue import Empty, Queue

import pytest

from zeta.computer.backend import DesktopOptions
from zeta.computer.docker import DockerClient
from zeta.computer.lima import SandboxVM, host_home
from zeta.computer.local import (
    LocalDockerBackend,
    check_container_policy,
    remove_session_desktops,
    session_label,
)
from zeta.computer.server import ComputerServer

pytestmark = pytest.mark.skipif(
    os.environ.get("ZETA_COMPUTER_DOCKER") != "1",
    reason="set ZETA_COMPUTER_DOCKER=1 to run against the zeta-sandbox VM",
)


@pytest.fixture(autouse=True)
def real_lima_home(monkeypatch: pytest.MonkeyPatch) -> None:
    if "LIMA_HOME" not in os.environ:
        monkeypatch.setenv("LIMA_HOME", str(host_home() / ".lima"))


@pytest.fixture
def backend():
    desktop = LocalDockerBackend(DesktopOptions(f"it-{uuid.uuid4().hex[:10]}", 600))
    try:
        yield desktop
    finally:
        desktop.close()


def _docker() -> DockerClient:
    return DockerClient(SandboxVM().require_running().docker_host)


def test_vm_verifies_isolation_before_use() -> None:
    with _docker() as docker:
        report = SandboxVM().verify_isolation(docker)
    assert any("absent in the guest" in check for check in report.checks)
    assert any("Docker engine is lima-zeta-sandbox" in check for check in report.checks)


def test_desktop_is_hardened_and_writes_a_note(backend: LocalDockerBackend) -> None:
    server = ComputerServer(backend)
    first = server.call("screenshot", {})
    assert first["isError"] is False
    observation = json.loads(first["content"][1]["text"].removeprefix("Observation: "))
    assert "Mousepad" in observation["active_window"]["title"]
    with _docker() as docker:
        details = json.loads(docker.output("inspect", backend.name))[0]
    check_container_policy(details)
    assert details["Config"]["Labels"][session_label(backend.options.session_id).split("=")[0]]

    result = server.call(
        "batch",
        {
            "actions": [
                {"type": "click", "x": 500, "y": 300},
                {"type": "type", "text": "integration line\nsecond"},
                {"type": "key", "keys": "ctrl+s"},
            ]
        },
    )
    assert result["isError"] is False, result["content"][0]["text"]
    dialog = server.call("screenshot", {})
    assert "Save" in dialog["content"][1]["text"]
    server.call(
        "batch",
        {
            "actions": [
                {"type": "key", "keys": "ctrl+a"},
                {"type": "type", "text": "/home/zeta/notes/it.txt"},
                {"type": "wait", "seconds": 0.5},
                {"type": "key", "keys": "Return"},
                {"type": "wait", "seconds": 1},
            ]
        },
    )
    assert backend._exec(("cat", "/home/zeta/notes/it.txt")) == b"integration line\nsecond"
    network = backend._exec(("cat", "/proc/net/dev")).decode()
    assert [line.split(":")[0].strip() for line in network.splitlines()[2:]] == ["lo"]
    assert backend._exec(("sh", "-c", "test ! -e /Users && id -u")).strip() == b"65532"


def test_session_removal_deletes_every_desktop(backend: LocalDockerBackend) -> None:
    backend.start()
    label = session_label(backend.options.session_id)
    with _docker() as docker:
        assert len(docker.running(label)) == 1
        remove_session_desktops(backend.options.session_id)
        assert docker.running(label) == ()


def test_watch_live_serves_vnc_through_docker_exec(
    backend: LocalDockerBackend, tmp_path: Path
) -> None:
    backend.start()
    home = tmp_path / "zeta-home"
    recording = home / "sessions" / backend.options.session_id / "computer"
    recording.mkdir(parents=True)
    (recording / "metadata.json").write_text(json.dumps({"active": True}))
    environment = {**os.environ, "ZETA_HOME": str(home)}
    process = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "from zeta.cli.main import main; raise SystemExit(main())",
            "computer",
            "watch",
            "--live",
            backend.options.session_id[:12],
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    try:
        lines: list[str] = []
        queue: Queue[str] = Queue()
        threading.Thread(
            target=lambda: [queue.put(line) for line in process.stdout], daemon=True
        ).start()
        deadline = time.monotonic() + 60
        while not any(line.startswith("password:") for line in lines):
            try:
                lines.append(queue.get(timeout=max(0.1, deadline - time.monotonic())).strip())
            except Empty:
                break
        output = "\n".join(lines)
        assert "spectator: http://127.0.0.1:" in output, output
        vnc = next(line for line in lines if line.startswith("live view: vnc://127.0.0.1:"))
        port = int(vnc.rsplit(":", 1)[1])
        with socket.create_connection(("127.0.0.1", port), timeout=10) as connection:
            assert connection.recv(12).startswith(b"RFB 003.")
    finally:
        process.send_signal(2)
        process.wait(timeout=30)
    assert process.returncode == 0
    assert backend._exec(("sh", "-c", "ls /tmp/zeta-vnc-* 2>/dev/null || true")) == b""
