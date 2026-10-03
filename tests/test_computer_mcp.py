"""Tests for the standalone computer-use MCP prototype."""

import importlib.util
import io
import json
import os
import pwd
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1] / "prototypes/computer-mcp"


def _module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    os.sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


backend = _module("backend", "backend.py")
server = _module("computer_server", "server.py")


class FakeBackend:
    def __init__(self) -> None:
        self.started = False
        self.destroyed = False
        self.actions = []

    def start(self) -> None:
        self.started = True

    def reset(self) -> None:
        self.actions.clear()

    def destroy(self) -> None:
        self.destroyed = True

    def screenshot(self):
        return backend.Screenshot(b"\x89PNG\r\n\x1a\n", "image/png")

    def input(self, action, arguments) -> None:
        self.actions.append((action, arguments))


def test_coordinate_scaling_clamps_rounding_and_rejects_out_of_range() -> None:
    assert backend.model_coordinate(0, axis="x") == 0
    assert backend.model_coordinate(512, axis="x") == 640
    assert backend.model_coordinate(1023.999, axis="x") == 1279
    assert backend.model_coordinate(320, axis="y") == 400
    assert backend.model_coordinate(639.999, axis="y") == 799
    for value, axis in (
        (-1, "x"),
        (1024, "x"),
        (640, "y"),
        (True, "x"),
        ("4", "y"),
    ):
        with pytest.raises(ValueError, match="model frame|must be a number"):
            backend.model_coordinate(value, axis=axis)
    with pytest.raises(ValueError, match="axis"):
        backend.model_coordinate(1, axis="z")


def test_mcp_protocol_round_trip_and_image_block_shape() -> None:
    fake = FakeBackend()
    requests = [
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "computer_click",
                "arguments": {"x": 10, "y": 20},
            },
        },
    ]
    source = io.StringIO("".join(json.dumps(item) + "\n" for item in requests))
    sink = io.StringIO()
    server.serve(fake, source, sink)
    replies = [json.loads(line) for line in sink.getvalue().splitlines()]

    assert fake.started and fake.destroyed
    assert fake.actions == [("click", {"x": 10, "y": 20})]
    assert len(replies[1]["result"]["tools"]) == 8
    content = replies[2]["result"]["content"]
    assert content[0]["type"] == "text" and "8 bytes" in content[0]["text"]
    assert content[1] == {
        "type": "image",
        "data": "iVBORw0KGgo=",
        "mimeType": "image/png",
    }


def test_action_error_is_a_clear_tool_result() -> None:
    result = server.ComputerServer(FakeBackend()).call(
        "computer_click", {"x": 1024, "y": 0}
    )
    assert result["isError"] is True
    assert "outside model frame" in result["content"][0]["text"]


def test_container_policy_has_no_network_mounts_or_ports() -> None:
    args = backend.container_args("test", "image", "run")
    assert args[args.index("--network") + 1] == "none"
    assert args[args.index("--user") + 1] == "65532:65532"
    assert args[args.index("--memory") + 1] == "1g"
    assert args[args.index("--pids-limit") + 1] == "256"
    assert "--read-only" in args and "--init" in args
    assert "ALL" in args and "no-new-privileges" in args
    assert not {"-v", "--volume", "--mount", "-p", "--publish"}.intersection(args)


def test_docker_command_uses_only_the_configured_socket(monkeypatch) -> None:
    captured = {}

    class Result:
        returncode = 0
        stdout = b"ok"
        stderr = b""

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["env"] = kwargs["env"]
        return Result()

    monkeypatch.setattr(backend.subprocess, "run", fake_run)
    desktop = backend.DockerDesktopBackend(docker_host="unix:///isolated/docker.sock")
    assert desktop._docker("version") == b"ok"
    assert captured["command"] == [
        "docker",
        "--host",
        "unix:///isolated/docker.sock",
        "--config",
        captured["env"]["DOCKER_CONFIG"],
        "version",
    ]
    assert captured["env"]["DOCKER_HOST"] == "unix:///isolated/docker.sock"
    assert captured["env"]["DOCKER_CONFIG"].startswith(
        "/tmp/zeta-computer-docker-config-"
    )
    assert "DOCKER_CONTEXT" not in captured["env"]


def test_backend_does_not_overwrite_a_docker_config(tmp_path, monkeypatch) -> None:
    config = tmp_path / "docker"
    config.mkdir()
    config_file = config / "config.json"
    config_file.write_text('{"credsStore": "do-not-touch"}\n')
    monkeypatch.setenv("ZETA_COMPUTER_DOCKER_CONFIG", str(config))

    with pytest.raises(RuntimeError, match="empty isolated config"):
        backend.DockerDesktopBackend()
    assert json.loads(config_file.read_text()) == {"credsStore": "do-not-touch"}


@pytest.mark.skipif(
    os.environ.get("ZETA_COMPUTER_DOCKER") != "1",
    reason="set ZETA_COMPUTER_DOCKER=1",
)
def test_docker_desktop_screenshot_and_cleanup() -> None:
    script = f"""
import sys
sys.path.insert(0, {str(ROOT)!r})
import backend

desktop = backend.DockerDesktopBackend(ttl_seconds=60)
try:
    desktop.start()
    shot = desktop.screenshot()
    assert shot.media_type == "image/jpeg"
    assert shot.data.startswith(b"\\xff\\xd8")
    desktop.input("key", {{"keys": "ctrl+s"}})
finally:
    desktop.destroy()
"""
    child_env = {
        "HOME": pwd.getpwuid(os.getuid()).pw_dir,
        "PATH": os.environ["PATH"],
        "ZETA_COMPUTER_DOCKER_HOST": (
            f"unix://{pwd.getpwuid(os.getuid()).pw_dir}"
            "/.lima/zeta-sandbox/sock/docker.sock"
        ),
    }
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        env=child_env,
        text=True,
        timeout=120,
        check=False,
    )
    assert result.returncode == 0, result.stderr
