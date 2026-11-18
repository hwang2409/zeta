"""Builds the native macOS helper with swiftc and checks a read-only hello round-trip.

Skipped where swiftc is unavailable (so Ubuntu CI skips and macOS CI builds). Only ``hello`` is
sent: it reads permission state via CGPreflightScreenCaptureAccess / AXIsProcessTrusted without
prompting and performs no capture or input, so it is safe on a headless CI runner.
"""

from __future__ import annotations

import json
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


def test_native_scroll_keeps_quartz_line_wheel_signs() -> None:
    from zeta.tools.computer import _SCROLL_UNIT

    assert _SCROLL_UNIT == {"up": (0, 1), "down": (0, -1), "left": (1, 0), "right": (-1, 0)}
    source = HELPER_SOURCE.read_text()
    assert "units: .line" in source
    assert "wheel1: Int32(dy)" in source
    assert "wheel2: Int32(dx)" in source


def test_indicator_lifecycle_and_interruptible_steps_are_present() -> None:
    source = HELPER_SOURCE.read_text()
    assert 'NSStatusBar.system.statusItem' in source
    assert 'window.ignoresMouseEvents = true' in source
    assert 'Thread {' in source
    assert 'app.run()' in source
    assert 'runInterruptibleSteps(path.dropFirst()' in source
    assert 'try checkKilled(currentKill())' in source
    assert 'final class EventGate' in source
    assert 'commitReply()' in source
    assert 'runInterruptibleSteps(Array(text)' in source


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


def test_selftest_scroll_reads_unposted_axis_values() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["sh", str(BUILD_SCRIPT), tmp], check=True, capture_output=True, timeout=300
        )
        output = subprocess.check_output(
            [str(Path(tmp) / "zeta-computer-helper"), "--selftest-scroll"],
            text=True,
            timeout=15,
        )
    assert output.strip() == "scroll self-test ok"


def test_selftest_lifecycle_refuses_raced_capture_and_orders_shutdown() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["sh", str(BUILD_SCRIPT), tmp], check=True, capture_output=True, timeout=300
        )
        output = subprocess.check_output(
            [str(Path(tmp) / "zeta-computer-helper"), "--selftest-lifecycle"],
            text=True,
            timeout=15,
        )
    result = json.loads(output)
    assert result["capture"] == "refused"
    assert result["safety_calls"] == 2
    assert result["events"][-2:] == ["mouse_up", "key_up"]
    assert result["events"].count("mouse_up") == 1
    assert result["events"].count("key_up") == 1


def test_selftest_kill_interrupts_steps_and_releases_drag() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(
            ["sh", str(BUILD_SCRIPT), tmp], check=True, capture_output=True, timeout=300
        )
        output = subprocess.check_output(
            [str(Path(tmp) / "zeta-computer-helper"), "--selftest-kill"], text=True, timeout=15
        )
    results = {item["name"]: item for item in map(json.loads, output.splitlines())}
    assert results["drag"] == {
        "name": "drag", "events": ["mouse_down", "drag_1", "mouse_up"],
        "reply": "killed", "later_reply": "killed"
    }
    assert results["type"] == {
        "name": "type", "events": ["key_down", "key_up"], "reply": "killed", "later_reply": "killed"
    }
    assert results["mouse_down"]["events"] == ["mouse_down", "mouse_up"]
    assert results["mouse_down"]["later_reply"] == "killed"
    assert results["kill_after_last_step"]["reply"] == "killed"
    assert results["kill_after_last_step"]["later_reply"] == "killed"
    assert results["kill_before_commit"]["reply"] == "killed"
    assert results["kill_before_commit"]["later_reply"] == "killed"
    assert results["kill_after_commit"]["reply"] == "ok"
    assert results["kill_after_commit"]["later_reply"] == "killed"


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
