"""Builds the native macOS helper with swiftc and checks a read-only hello round-trip.

Skipped where swiftc is unavailable (so Ubuntu CI skips and macOS CI builds). Only ``hello`` is
sent: it reads permission state via CGPreflightScreenCaptureAccess / AXIsProcessTrusted without
prompting and performs no capture or input, so it is safe on a headless CI runner.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from zeta.tools.computer.helper_client import HelperClient

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build-computer-helper.sh"
HELPER_SOURCE = REPO_ROOT / "helpers" / "macos-computer" / "main.swift"

pytestmark = [
    pytest.mark.skipif(shutil.which("swiftc") is None, reason="swiftc not available"),
    pytest.mark.skipif(sys.platform != "darwin", reason="helper is macOS-only"),
]


def test_cursor_position_uses_nearest_integer_rule() -> None:
    source = HELPER_SOURCE.read_text()
    assert 'Int(loc.x.rounded())' in source
    assert 'Int(loc.y.rounded())' in source


def test_indicator_lifecycle_and_interruptible_steps_are_present() -> None:
    source = HELPER_SOURCE.read_text()
    assert 'NSStatusBar.system.statusItem' in source
    assert 'window.ignoresMouseEvents = true' in source
    assert 'Thread {' in source
    assert 'app.run()' in source
    assert 'for point in path.dropFirst()' in source
    assert 'try checkKilled()' in source
    assert 'for ch in text' in source


def test_helper_builds_and_answers_hello() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["sh", str(BUILD_SCRIPT), tmp], check=True, capture_output=True, timeout=300
        )
        binary = Path(tmp) / "zeta-computer-helper"
        assert binary.exists() and binary.stat().st_mode & 0o111

        with HelperClient([str(binary)], timeout=15.0) as client:
            reply = client.request("hello")
    assert reply["version"] == "0.1.0"
    assert isinstance(reply["screen_recording"], bool)
    assert isinstance(reply["accessibility"], bool)


def test_helper_exits_when_stdin_reaches_eof() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["sh", str(BUILD_SCRIPT), tmp], check=True, capture_output=True, timeout=300
        )
        binary = Path(tmp) / "zeta-computer-helper"
        process = subprocess.Popen([str(binary)], stdin=subprocess.PIPE)
        assert process.stdin is not None
        process.stdin.close()
        assert process.wait(timeout=15) == 0
