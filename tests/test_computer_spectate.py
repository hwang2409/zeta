"""Tests for computer-use recording and local spectating."""

import importlib.util
import json
import socket
import sys
import threading
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import urlopen

import pytest

ROOT = Path(__file__).resolve().parents[1] / "prototypes/computer-mcp"


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend = sys.modules.get("backend") or _module("backend", "backend.py")
recording = _module("computer_recording", "recording.py")
spectate = _module("computer_spectate", "spectate.py")


def test_recording_format_has_frames_coordinates_and_session_state(tmp_path) -> None:
    clock_values = iter([100.0, 101.25])
    recorder = recording.SessionRecorder(tmp_path / "run", clock=lambda: next(clock_values))
    frame = recorder.record_frame(b"jpeg")
    recorder.record_tool(
        "computer_click",
        {"x": 512, "y": 320},
        {"content": [{"type": "text", "text": "WARNING: verify me"}], "isError": False},
        checklist=[{"index": 0, "step": "Save", "done": False, "note": None}],
        frame=frame,
    )
    recorder.close()

    metadata = json.loads((tmp_path / "run/metadata.json").read_text())
    event = json.loads((tmp_path / "run/events.jsonl").read_text())
    assert metadata["active"] is False
    assert metadata["model_frame"] == {"width": 1024, "height": 640}
    assert (tmp_path / "run/frames/000001.jpg").read_bytes() == b"jpeg"
    assert event["args"]["_coordinates"] == {
        "point": {
            "model": {"x": 512, "y": 320},
            "physical": {"x": 640, "y": 400},
        }
    }
    assert event["frame"] == "frames/000001.jpg"
    assert event["checklist"][0]["step"] == "Save"
    assert event["verify_warnings"] == ["WARNING: verify me"]

    recording.append_usage_event(tmp_path / "run", {"input_tokens": 12}, 1.5)
    usage = json.loads((tmp_path / "run/events.jsonl").read_text().splitlines()[1])
    assert usage["tool"] == "session_usage"
    assert usage["token_usage"] == {"input_tokens": 12}
    assert usage["elapsed"] == 1.5


def test_overlay_math_scales_model_frame() -> None:
    assert spectate.overlay_position(512, 320, 800, 500) == (400, 250)
    assert spectate.overlay_position(1024, 640, 2048, 1280) == (2048, 1280)


def test_hostile_text_is_not_emitted_as_html() -> None:
    hostile = '</script><img src=x onerror="alert(1)">&'
    encoded = spectate.safe_json({"value": hostile})
    assert b"</script>" not in encoded
    assert b"<img" not in encoded
    assert b"\\u003c" in encoded and b"\\u0026" in encoded
    assert "innerHTML" not in spectate.VIEWER
    assert "textContent" in spectate.VIEWER


def test_web_token_and_loopback_binding(tmp_path) -> None:
    (tmp_path / "frames").mkdir()
    (tmp_path / "metadata.json").write_text('{"active":true,"model_frame":{"width":1024,"height":640}}')
    server, token = spectate.make_web_server(tmp_path, token="known-token")
    assert server.server_address[0] == "127.0.0.1"
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with pytest.raises(HTTPError) as missing:
            urlopen(f"http://127.0.0.1:{port}/", timeout=2)
        assert missing.value.code == 403
        with pytest.raises(HTTPError) as wrong:
            urlopen(f"http://127.0.0.1:{port}/?token=wrong", timeout=2)
        assert wrong.value.code == 403
        with urlopen(f"http://127.0.0.1:{port}/?token={token}", timeout=2) as response:
            assert response.status == 200
            assert b"Zeta computer spectator" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
    assert not thread.is_alive()


def test_live_state_follows_recent_events_not_only_metadata(tmp_path) -> None:
    metadata = {"active": True, "model_frame": {"width": 1024, "height": 640}}
    (tmp_path / "events.jsonl").write_text("{}\n")
    assert spectate.is_live(tmp_path, metadata, time.time()) is True
    stale = time.time() + spectate.LIVE_WINDOW_SECONDS + 1
    assert spectate.is_live(tmp_path, metadata, stale) is False
    assert spectate.is_live(tmp_path, {"active": False}, time.time()) is False
    assert spectate.is_live(tmp_path / "missing", metadata, time.time()) is False


def test_tunnel_command_is_an_argument_list_and_rejects_injection(tmp_path) -> None:
    command = spectate.tunnel_command(tmp_path, "unix:///safe.sock", "abc123")
    assert command == [
        "docker",
        "--host",
        "unix:///safe.sock",
        "--config",
        str(tmp_path),
        "exec",
        "-i",
        "abc123",
        "socat",
        "-",
        "TCP:127.0.0.1:5900",
    ]
    with pytest.raises(ValueError, match="invalid container"):
        spectate.tunnel_command(tmp_path, "unix:///safe.sock", "abc; touch /tmp/pwn")
    assert "shell=True" not in (ROOT / "spectate.py").read_text()


def test_tunnel_cleanup_closes_listener_and_processes() -> None:
    listener = socket.socket()

    class Process:
        terminated = False
        waited = False

        def terminate(self) -> None:
            self.terminated = True

        def wait(self, timeout: int) -> None:
            assert timeout == 2
            self.waited = True

    process = Process()
    spectate.close_tunnel(listener, [process])
    assert process.terminated and process.waited
    assert listener.fileno() == -1


def test_recording_defaults_under_zeta_home(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("ZETA_HOME", str(tmp_path))
    monkeypatch.setenv("ZETA_COMPUTER_RUN_ID", "run-1")
    monkeypatch.delenv("ZETA_COMPUTER_RECORDING_DIR", raising=False)
    monkeypatch.delenv("ZETA_COMPUTER_RECORDING", raising=False)
    assert recording.default_recording_dir() == tmp_path / "recordings/run-1"
