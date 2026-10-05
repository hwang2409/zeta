"""Unit tests for zeta.computer: geometry, actions, observation, server, backend policy."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import subprocess
import threading
import urllib.error
import urllib.request
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import pytest

from zeta.computer import server as server_module
from zeta.computer.actions import (
    MODEL_HEIGHT,
    MODEL_WIDTH,
    model_coordinate,
    physical_bounds_to_model,
    validate_action,
    validate_batch,
)
from zeta.computer.backend import DesktopOptions, Screenshot
from zeta.computer.docker import DockerClient
from zeta.computer.lima import IsolationError, SandboxVM, mounted_host_paths
from zeta.computer.local import (
    LocalDockerBackend,
    check_container_policy,
    container_args,
    image_tag,
    session_label,
)
from zeta.computer.observe import format_observation, frame_difference, wait_for_stable
from zeta.computer.recording import SessionRecorder, mark_finished
from zeta.computer.server import ComputerServer, handle_request, serve
from zeta.computer.settings import ComputerSettingsError, load_computer_settings
from zeta.computer.spectate import (
    Spectator,
    load_recording,
    make_web_server,
    safe_json,
    vnc_command,
)
from zeta.computer.tools import QUALIFIED_TOOL_NAMES, TOOL_NAMES
from zeta.computer.x11 import input_command, model_observation

JPEG = b"\xff\xd8fake-jpeg"


class FakeBackend:
    def __init__(self, *, fail_on: str | None = None) -> None:
        self.calls: list[tuple[str, object]] = []
        self.fail_on = fail_on
        self.closed = False

    def start(self) -> None:
        self.calls.append(("start", None))

    def destroy(self) -> None:
        self.calls.append(("destroy", None))

    def close(self) -> None:
        self.closed = True

    def screenshot(self) -> Screenshot:
        self.calls.append(("screenshot", None))
        return Screenshot(JPEG)

    def input(self, action: str, arguments: Mapping[str, object]) -> None:
        if action == self.fail_on:
            raise RuntimeError(f"{action} failed")
        self.calls.append((action, dict(arguments)))

    def observe(self) -> dict[str, object]:
        return {"active_window": {"title": "Mousepad", "bounds": None}, "windows": []}

    def settle(self) -> float:
        self.calls.append(("settle", None))
        return 0.25

    @property
    def inputs(self) -> list[str]:
        return [name for name, _ in self.calls if name not in {"start", "screenshot", "settle"}]


# Coordinate scaling


def test_model_coordinates_scale_to_the_physical_display() -> None:
    assert model_coordinate(0, axis="x") == 0
    assert model_coordinate(512, axis="x") == 640
    assert model_coordinate(320, axis="y") == 400
    assert model_coordinate(MODEL_WIDTH - 0.01, axis="x") == 1279
    assert model_coordinate(MODEL_HEIGHT - 0.01, axis="y") == 799
    assert model_coordinate(10.4, axis="x") == 13


@pytest.mark.parametrize("value", [-1, MODEL_WIDTH, "5", True, None, float("nan")])
def test_model_coordinates_reject_values_outside_the_frame(value: object) -> None:
    with pytest.raises(ValueError):
        model_coordinate(value, axis="x")


def test_physical_bounds_convert_and_clamp_to_the_model_frame() -> None:
    assert physical_bounds_to_model({"x": 640, "y": 400, "width": 1280, "height": 800}) == {
        "x": 512,
        "y": 320,
        "w": 512,
        "h": 320,
    }
    assert physical_bounds_to_model({"x": -50, "y": -10, "width": 10, "height": 10})["x"] == 0


# Action and batch validation


def test_action_validation_checks_every_field() -> None:
    assert validate_action("click", {"x": 1, "y": 2, "button": "right"}) == {
        "x": 1,
        "y": 2,
        "button": "right",
    }
    with pytest.raises(ValueError, match="missing"):
        validate_action("click", {"x": 1})
    with pytest.raises(ValueError, match="unknown fields"):
        validate_action("type", {"text": "a", "x": 1})
    with pytest.raises(ValueError, match="button"):
        validate_action("click", {"x": 1, "y": 2, "button": "side"})
    with pytest.raises(ValueError, match="seconds"):
        validate_action("wait", {"seconds": 6})
    with pytest.raises(ValueError, match="keys"):
        validate_action("key", {"keys": "x" * 101})
    with pytest.raises(ValueError, match="dx and dy"):
        validate_action("scroll", {"x": 1, "y": 1, "dx": 0, "dy": 10_001})


def test_batch_validation_rejects_bad_shapes_with_the_item_index() -> None:
    with pytest.raises(ValueError, match="1..10"):
        validate_batch({"actions": []})
    with pytest.raises(ValueError, match="1..10"):
        validate_batch({"actions": [{"type": "wait", "seconds": 0}] * 11})
    with pytest.raises(ValueError, match=r"actions\[1\]: unsupported action: zoom"):
        validate_batch({"actions": [{"type": "wait", "seconds": 0}, {"type": "zoom"}]})
    with pytest.raises(ValueError, match="screenshot must be a boolean"):
        validate_batch({"actions": [{"type": "wait", "seconds": 0}], "screenshot": "no"})
    with pytest.raises(ValueError, match="only actions and screenshot"):
        validate_batch({"actions": [{"type": "wait", "seconds": 0}], "extra": 1})


def test_invalid_batch_item_prevents_every_action_from_running() -> None:
    backend = FakeBackend()
    server = ComputerServer(backend)
    result = server.call(
        "batch",
        {"actions": [{"type": "click", "x": 1, "y": 1}, {"type": "click", "x": 5000, "y": 1}]},
    )
    assert result["isError"] is True
    assert "actions[1]" in result["content"][0]["text"]
    assert backend.calls == []


def test_batch_stops_at_the_first_runtime_error_and_reports_statuses() -> None:
    backend = FakeBackend(fail_on="key")
    result = ComputerServer(backend).call(
        "batch",
        {
            "actions": [
                {"type": "type", "text": "hello"},
                {"type": "key", "keys": "ctrl+s"},
                {"type": "click", "x": 1, "y": 1},
            ]
        },
    )
    assert backend.inputs == ["type"]
    statuses = json.loads(result["content"][1]["text"].removeprefix("Batch results: "))
    assert [item["status"] for item in statuses] == ["ok", "error"]
    assert result["isError"] is True
    assert sum(1 for item in result["content"] if item["type"] == "image") == 1
    assert backend.calls.count(("settle", None)) == 1


def test_batch_without_screenshot_returns_text_only() -> None:
    backend = FakeBackend()
    result = ComputerServer(backend).call(
        "batch", {"actions": [{"type": "wait", "seconds": 0}], "screenshot": False}
    )
    assert all(item["type"] == "text" for item in result["content"])
    assert ("screenshot", None) not in backend.calls


def test_one_tool_call_has_one_verification_scope() -> None:
    class ScopedBackend(FakeBackend):
        def __init__(self) -> None:
            super().__init__()
            self.verifications = 0

        @contextmanager
        def operation(self):
            self.verifications += 1
            yield

    backend = ScopedBackend()
    server = ComputerServer(backend)
    server.call("click", {"x": 10, "y": 20})
    server.call("screenshot", {})
    assert backend.verifications == 2


class CountingVerifier:
    def __init__(self, *, barrier: threading.Barrier | None = None) -> None:
        self.count = 0
        self.barrier = barrier
        self._lock = threading.Lock()

    def require_running(self) -> object:
        return type("VMInfo", (), {"docker_host": "unix:///fake/docker.sock"})()

    def verify_isolation(self, _docker: DockerClient) -> None:
        with self._lock:
            self.count += 1
        if self.barrier is not None:
            self.barrier.wait(timeout=5)


def _scoped_backend(verifier: CountingVerifier, runner: Recorder | None = None) -> LocalDockerBackend:
    backend = LocalDockerBackend(
        DesktopOptions("scope-test", 60),
        vm=verifier,  # type: ignore[arg-type]
        docker_factory=lambda host: DockerClient(host, runner=runner or Recorder()),
        clock=lambda: 0.0,
    )
    backend._docker = DockerClient("unix:///fake/docker.sock", runner=runner or Recorder())
    backend.name = "fake-desktop"
    backend._started_at = 0.0
    return backend


def _many_execs() -> dict[str, object]:
    return {
        "actions": [
            {"type": "wait", "seconds": 0},
            {"type": "wait", "seconds": 0},
            {"type": "wait", "seconds": 0},
        ],
        "screenshot": False,
    }


def test_one_server_call_verifies_once_across_many_execs() -> None:
    verifier = CountingVerifier()
    backend = _scoped_backend(verifier)
    result = ComputerServer(backend).call("batch", _many_execs())
    assert result["isError"] is False
    assert verifier.count == 1


def test_second_server_call_verifies_again() -> None:
    verifier = CountingVerifier()
    backend = _scoped_backend(verifier)
    server = ComputerServer(backend)
    server.call("batch", _many_execs())
    server.call("batch", _many_execs())
    assert verifier.count == 2


def test_failed_call_still_forces_verification_on_next_call() -> None:
    class FailingRunner(Recorder):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
            if self.fail:
                self.fail = False
                raise RuntimeError("fake exec failed")
            return super().__call__(argv, **kwargs)

    verifier = CountingVerifier()
    runner = FailingRunner()
    server = ComputerServer(_scoped_backend(verifier, runner))
    first = server.call("batch", _many_execs())
    second = server.call("batch", _many_execs())
    assert first["isError"] is True
    assert second["isError"] is False
    assert verifier.count == 2


def test_concurrent_calls_are_serialized_and_each_verifies() -> None:
    verifier = CountingVerifier()
    backend = _scoped_backend(verifier)
    entered = threading.Event()
    release = threading.Event()
    events: list[str] = []
    event_lock = threading.Lock()

    original_operation = backend.operation

    @contextmanager
    def operation():
        with original_operation():
            with event_lock:
                events.append("enter")
                first = events.count("enter") == 1
            if first:
                entered.set()
                assert release.wait(timeout=5)
            yield
            with event_lock:
                events.append("exit")

    backend.operation = operation  # type: ignore[method-assign]
    server = ComputerServer(backend)
    results: list[dict[str, object]] = []
    first = threading.Thread(target=lambda: results.append(server.call("batch", _many_execs())))
    second = threading.Thread(target=lambda: results.append(server.call("batch", _many_execs())))
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    assert not any(event == "enter" for event in events[2:])
    release.set()
    first.join(timeout=10)
    second.join(timeout=10)
    assert not first.is_alive() and not second.is_alive()
    assert all(result["isError"] is False for result in results)
    assert events == ["enter", "exit", "enter", "exit"]
    assert verifier.count == 2


@pytest.mark.asyncio
async def test_cancel_while_waiting_releases_cleanly() -> None:
    verifier = CountingVerifier()
    backend = _scoped_backend(verifier)
    entered = threading.Event()
    release = threading.Event()
    original_operation = backend.operation

    @contextmanager
    def operation():
        with original_operation():
            if not entered.is_set():
                entered.set()
                assert release.wait(timeout=5)
            yield

    backend.operation = operation  # type: ignore[method-assign]
    server = ComputerServer(backend)
    first = asyncio.create_task(asyncio.to_thread(server.call, "batch", _many_execs()))
    await asyncio.to_thread(entered.wait, 5)
    second = asyncio.create_task(asyncio.to_thread(server.call, "batch", _many_execs()))
    await asyncio.sleep(0.01)
    second.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second
    release.set()
    await first
    for _ in range(100):
        if verifier.count == 2:
            break
        await asyncio.sleep(0.01)
    third = await asyncio.to_thread(server.call, "batch", _many_execs())
    assert third["isError"] is False
    assert verifier.count == 3


def test_single_action_settles_then_returns_screenshot_and_observation() -> None:
    backend = FakeBackend()
    result = ComputerServer(backend).call("click", {"x": 10, "y": 20})
    assert [name for name, _ in backend.calls] == ["start", "click", "settle", "screenshot"]
    texts = [item["text"] for item in result["content"] if item["type"] == "text"]
    assert texts[1] == "Screen settled in 0.250 seconds."
    assert texts[2].startswith("Observation: ")
    image = result["content"][-1]
    assert image["mimeType"] == "image/jpeg"
    assert base64.b64decode(image["data"]) == JPEG


def test_backend_timeouts_become_tool_errors() -> None:
    class SlowBackend(FakeBackend):
        def screenshot(self) -> Screenshot:
            raise subprocess.TimeoutExpired(["docker"], 120)

    result = ComputerServer(SlowBackend()).call("screenshot", {})
    assert result["isError"] is True
    assert "timed out" in result["content"][0]["text"]


def test_unknown_or_removed_prototype_tools_are_errors() -> None:
    for name in ("zoom", "plan", "check", "computer_click"):
        result = ComputerServer(FakeBackend()).call(name, {})
        assert result["isError"] is True


def test_tool_list_is_the_benchmarked_default_set() -> None:
    assert TOOL_NAMES == (
        "screenshot", "click", "double_click", "drag", "type", "key", "scroll", "wait", "batch",
    )
    assert QUALIFIED_TOOL_NAMES[0] == "computer__screenshot"
    listed = handle_request(ComputerServer(FakeBackend()), {"method": "tools/list"})
    assert [tool["name"] for tool in listed["tools"]] == list(TOOL_NAMES)
    assert all("observation" in tool["description"] for tool in listed["tools"])


def test_stdio_server_answers_requests_and_reports_errors() -> None:
    source = io.StringIO(
        "\n".join(
            [
                json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize"}),
                json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}),
                "not json",
                json.dumps({"jsonrpc": "2.0", "id": 2, "method": "nope"}),
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": 3,
                        "method": "tools/call",
                        "params": {"name": "screenshot", "arguments": {}},
                    }
                ),
            ]
        )
        + "\n"
    )
    sink = io.StringIO()
    serve(ComputerServer(FakeBackend()), source, sink)
    replies = [json.loads(line) for line in sink.getvalue().splitlines()]
    assert [reply["id"] for reply in replies] == [1, 2, 3]
    assert replies[0]["result"]["serverInfo"]["name"] == "zeta-computer"
    assert "unsupported method" in replies[1]["error"]["message"]
    assert replies[2]["result"]["isError"] is False


def test_server_main_closes_backend_and_recording_on_eof(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend = FakeBackend()
    monkeypatch.setattr(server_module, "create_backend", lambda name, options: backend)
    monkeypatch.setenv("ZETA_COMPUTER_SESSION", "s1")
    monkeypatch.setenv("ZETA_COMPUTER_RECORDING_DIR", str(tmp_path / "rec"))
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.setattr(server_module.signal, "signal", lambda *args: None)
    assert server_module.main() == 0
    assert backend.closed is True
    metadata = json.loads((tmp_path / "rec" / "metadata.json").read_text())
    assert metadata["active"] is False


# Observation and settle


def test_observation_formatting_tracks_screen_changes() -> None:
    raw = {"active_window": {"title": "A"}, "windows": [], "mouse": {"x": 1, "y": 2}}
    first, first_hash = format_observation(raw, b"one", None)
    second, second_hash = format_observation(raw, b"two", first_hash)
    third, _ = format_observation(raw, b"two", second_hash)
    assert json.loads(first)["screen_change"]["changed"] is None
    assert json.loads(second)["screen_change"] == {
        "changed": True,
        "hash": second_hash,
        "previous_hash": first_hash,
    }
    assert json.loads(third)["screen_change"]["changed"] is False
    assert list(json.loads(first)) == sorted(json.loads(first))


def test_guest_observation_is_converted_to_model_frame() -> None:
    raw = {
        "active_id": "7",
        "active_title": "notes.txt - Mousepad",
        "windows": [
            {"id": "7", "title": "notes.txt - Mousepad", "bounds": {"x": 0, "y": 0, "width": 1280, "height": 800}},
            {"id": "8", "title": "", "bounds": {"x": 0, "y": 0, "width": 10, "height": 10}},
        ],
        "focused_widget": {"role": "text", "value": "hi"},
        "mouse": {"x": 640, "y": 400},
    }
    observation = model_observation(raw)
    assert observation["active_window"] == {
        "title": "notes.txt - Mousepad",
        "bounds": {"x": 0, "y": 0, "w": 1024, "h": 640},
    }
    assert len(observation["windows"]) == 1
    assert observation["mouse"] == {"x": 512, "y": 320}


def test_settle_returns_when_two_samples_match() -> None:
    now = [0.0]
    samples = iter([b"\x00" * 4, b"\xff" * 4, b"\xff" * 4])

    def sleep(seconds: float) -> None:
        now[0] += seconds

    elapsed = wait_for_stable(lambda: next(samples), monotonic=lambda: now[0], sleep=sleep)
    assert elapsed == pytest.approx(0.2)


def test_settle_is_capped_for_an_animated_screen() -> None:
    now = [0.0]
    counter = iter(range(1000))

    def sleep(seconds: float) -> None:
        now[0] += seconds

    elapsed = wait_for_stable(
        lambda: bytes([next(counter) % 2 * 255] * 4), monotonic=lambda: now[0], sleep=sleep
    )
    assert elapsed == pytest.approx(2.0, abs=0.11)
    assert frame_difference(b"\x00\x00", b"\x00\x01") < 0.002
    assert frame_difference(b"", b"") == 1.0


# Guest commands


def test_input_commands_scale_coordinates_and_pass_text_as_one_argument() -> None:
    assert input_command("click", {"x": 512, "y": 320}) == (
        "xdotool", "mousemove", "640", "400", "click", "--repeat", "1", "--delay", "100", "1",
    )
    typed = input_command("type", {"text": "a; rm -rf /\nb"})
    assert typed[0:2] == ("bash", "-c") and typed[-1] == "a; rm -rf /\nb"
    assert input_command("scroll", {"x": 0, "y": 0, "dx": 0, "dy": -250})[-3:] == ("--repeat", "2", "4")
    assert input_command("double_click", {"x": 0, "y": 0})[6] == "2"


# Container and VM policy


def test_container_args_are_the_hardened_launch_policy() -> None:
    args = container_args("zeta-computer-x", "zeta-computer:abc", "session1", 3660)
    joined = " ".join(args)
    for flag in (
        "--network none", "--read-only", "--cap-drop ALL", "--security-opt no-new-privileges",
        "--memory 1g", "--cpus 1", "--pids-limit 256", "--user 65532:65532", "--init",
        "--label zeta.computer.session=session1",
    ):
        assert flag in joined
    assert "-v" not in args and "--volume" not in args and "--mount" not in args
    assert "-p" not in args and "--publish" not in args and "--privileged" not in args
    assert args[-2:] == ("zeta-computer:abc", "3660")


def _inspect(**overrides: object) -> dict[str, object]:
    host = {
        "NetworkMode": "none", "ReadonlyRootfs": True, "CapDrop": ["ALL"],
        "SecurityOpt": ["no-new-privileges"], "PidsLimit": 256, "Memory": 1024**3,
        "PortBindings": {}, "Privileged": False,
    }
    host.update(overrides)
    return {"Mounts": [], "HostConfig": host, "Config": {"User": "65532:65532"}}


def test_container_policy_check_rejects_any_missing_control() -> None:
    check_container_policy(_inspect())
    with pytest.raises(IsolationError, match="network"):
        check_container_policy(_inspect(NetworkMode="bridge"))
    with pytest.raises(IsolationError, match="published ports"):
        check_container_policy(_inspect(PortBindings={"5900/tcp": [{}]}))
    details = _inspect()
    details["Mounts"] = [{"Source": "/Users"}]
    with pytest.raises(IsolationError, match="mounts"):
        check_container_policy(details)


def test_session_labels_reject_unsafe_ids() -> None:
    assert session_label("abc-1") == "zeta.computer.session=abc-1"
    with pytest.raises(ValueError):
        session_label("a b")


def test_image_tag_is_content_addressed() -> None:
    assert image_tag().startswith("zeta-computer:")
    assert image_tag() == image_tag()


class Recorder:
    def __init__(self, replies: dict[str, subprocess.CompletedProcess[bytes]] | None = None) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.replies = replies or {}

    def __call__(self, argv: list[str], **kwargs: object) -> subprocess.CompletedProcess[bytes]:
        self.calls.append((list(argv), kwargs))
        for key, reply in self.replies.items():
            if key in " ".join(argv):
                return reply
        return subprocess.CompletedProcess(argv, 0, b"", b"")


def test_docker_client_always_selects_the_socket_and_an_empty_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DOCKER_CONTEXT", "colima")
    runner = Recorder()
    with DockerClient("unix:///vm/sock/docker.sock", runner=runner) as docker:
        docker.run("ps")
        config = docker.config_dir
        assert json.loads((config / "config.json").read_text()) == {}
        assert config.stat().st_mode & 0o077 == 0
    argv, kwargs = runner.calls[0]
    assert argv[:5] == ["docker", "--host", "unix:///vm/sock/docker.sock", "--config", str(config)]
    env = kwargs["env"]
    assert env["DOCKER_HOST"] == "unix:///vm/sock/docker.sock"
    assert "DOCKER_CONTEXT" not in env
    assert set(env) == {"PATH", "HOME", "DOCKER_HOST", "DOCKER_CONFIG"}
    assert not config.exists()
    with pytest.raises(ValueError):
        DockerClient("tcp://127.0.0.1:2375", runner=runner)


def _lima_list(tmp_path: Path, **config: object) -> bytes:
    return json.dumps(
        {
            "name": "zeta-sandbox", "status": "Running", "dir": str(tmp_path),
            "cpus": 4, "memory": 6 * 1024**3, "disk": 50 * 1024**3,
            "config": {
                "mounts": None,
                "portForwards": [
                    {"guestSocket": "/run/user/501/docker.sock",
                     "hostSocket": str(tmp_path / "sock" / "docker.sock")}
                ],
                **config,
            },
        }
    ).encode()


def _vm_runner(
    tmp_path: Path,
    *,
    guest_mounts: str = "",
    users: int = 0,
    engine: bytes = b"lima-zeta-sandbox\n",
    config: dict[str, object] | None = None,
) -> Recorder:
    ok = subprocess.CompletedProcess
    return Recorder(
        {
            "limactl list": ok([], 0, _lima_list(tmp_path, **(config or {})), b""),
            "cat /proc/mounts": ok([], 0, ("/dev/vda1 / ext4 rw 0 0\n" + guest_mounts).encode(), b""),
            "test ! -e /Users": ok([], users, b"", b""),
            "docker --host": ok([], 0, engine, b""),
        }
    )


def test_isolation_verification_passes_for_a_mountless_vm(tmp_path: Path) -> None:
    runner = _vm_runner(tmp_path)
    vm = SandboxVM(runner=runner)
    with DockerClient(f"unix://{tmp_path}/sock/docker.sock", runner=runner) as docker:
        report = vm.verify_isolation(docker)
    assert len(report.checks) == 6
    assert not any("colima" in " ".join(argv) for argv, _ in runner.calls)


@pytest.mark.parametrize(
    ("runner_options", "message"),
    [
        ({"guest_mounts": "host /mnt/host virtiofs rw 0 0\n"}, "host mounts"),
        ({"guest_mounts": "h /Users/someone 9p rw 0 0\n"}, "host mounts"),
        ({"users": 1}, "exists"),
        ({"engine": b"colima\n"}, "not the VM"),
        ({"config": {"mounts": [{"location": "~"}]}}, "configures host mounts"),
        (
            {"config": {"portForwards": [{"guestSocket": "/var/run/x.sock", "hostSocket": "/x"}]}},
            "unexpected socket",
        ),
        ({"config": {"portForwards": [{"reverse": True}]}}, "reverse"),
    ],
)
def test_isolation_verification_refuses_any_host_exposure(
    tmp_path: Path, runner_options: dict[str, object], message: str
) -> None:
    runner = _vm_runner(tmp_path, **runner_options)
    with (
        DockerClient(f"unix://{tmp_path}/sock/docker.sock", runner=runner) as docker,
        pytest.raises(IsolationError, match=message),
    ):
        SandboxVM(runner=runner).verify_isolation(docker)


def test_mounted_host_paths_ignores_guest_filesystems() -> None:
    mounts = "tmpfs /tmp tmpfs rw 0 0\n/dev/vda1 / ext4 rw 0 0\nnone /home/me sshfs rw 0 0\n"
    assert mounted_host_paths(mounts, Path("/Users/me")) == ["/home/me (sshfs)"]


def test_backend_refuses_to_start_a_desktop_when_isolation_fails(tmp_path: Path) -> None:
    runner = _vm_runner(tmp_path, guest_mounts="h /mnt virtiofs rw 0 0\n")
    backend = LocalDockerBackend(
        DesktopOptions("s1", 60),
        vm=SandboxVM(runner=runner),
        docker_factory=lambda host: DockerClient(host, runner=runner),
    )
    with pytest.raises(IsolationError):
        backend.start()
    assert not any(" run " in f" {' '.join(argv)} " for argv, _ in runner.calls if argv[0] == "docker")
    backend.close()


def test_start_fast_path_reverifies_isolation(tmp_path: Path) -> None:
    runner = _vm_runner(tmp_path, guest_mounts="h /mnt virtiofs rw 0 0\n")
    backend = LocalDockerBackend(
        DesktopOptions("s1", 60),
        vm=SandboxVM(runner=runner),
        docker_factory=lambda host: DockerClient(host, runner=runner),
        clock=lambda: 1.0,
    )
    backend._started_at = 0.0
    with pytest.raises(IsolationError):
        backend.start()
    assert not any(" run " in f" {' '.join(argv)} " for argv, _ in runner.calls if argv[0] == "docker")
    backend.close()


# Recording and spectating


def test_recording_writes_frames_events_and_coordinates(tmp_path: Path) -> None:
    recorder = SessionRecorder(tmp_path, clock=lambda: 100.0)
    recorder.record_frame(JPEG)
    recorder.record_tool("click", {"x": 512, "y": 320}, {"content": [{"type": "text", "text": "ok"}]})
    recorder.record_tool("wait", {"seconds": 1}, {"content": [], "isError": True})
    events = [json.loads(line) for line in (tmp_path / "events.jsonl").read_text().splitlines()]
    assert events[0]["frame"] == "frames/000001.jpg"
    assert events[0]["args"]["_coordinates"]["point"]["physical"] == {"x": 640, "y": 400}
    assert events[1]["frame"] is None and events[1]["result"]["error"] is True
    assert (tmp_path / "frames" / "000001.jpg").read_bytes() == JPEG
    mark_finished(tmp_path)
    assert json.loads((tmp_path / "metadata.json").read_text())["active"] is False


def _get(url: str) -> tuple[int, bytes, dict[str, str]]:
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as exc:
        return exc.code, b"", dict(exc.headers)


def test_spectator_requires_its_token_and_binds_loopback(tmp_path: Path) -> None:
    recorder = SessionRecorder(tmp_path)
    recorder.record_frame(JPEG)
    recorder.record_tool("type", {"text": "</script><img src=x onerror=alert(1)>"}, {"content": []})
    spectator = Spectator(tmp_path)
    try:
        assert spectator.url.startswith("http://127.0.0.1:")
        base, token = spectator.url.split("/?token=")
        assert _get(f"{base}/")[0] == 403
        assert _get(f"{base}/?token=wrong")[0] == 403
        assert _get(f"{base}/api/state?token={token}&token={token}")[0] == 403
        status, page, headers = _get(spectator.url)
        assert status == 200 and b"<canvas" in page
        assert "default-src 'none'" in headers["Content-Security-Policy"]
        assert headers["X-Content-Type-Options"] == "nosniff"
        status, state, _ = _get(f"{base}/api/state?token={token}")
        assert status == 200
        assert b"<" not in state and b">" not in state
        assert json.loads(state)["events"][0]["args"]["text"].startswith("</script>")
        assert _get(f"{base}/frames/000001.jpg?token={token}")[1] == JPEG
        assert _get(f"{base}/frames/../metadata.json?token={token}")[0] == 404
        assert _get(f"{base}/frames/%2e%2e%2fmetadata.json?token={token}")[0] == 404
    finally:
        spectator.stop()


def test_web_server_listens_only_on_ipv4_loopback(tmp_path: Path) -> None:
    server, token = make_web_server(tmp_path)
    try:
        assert server.server_address[0] == "127.0.0.1"
        assert len(token) >= 32
    finally:
        server.server_close()


def test_safe_json_escapes_markup_and_recording_state_without_files(tmp_path: Path) -> None:
    assert safe_json({"a": "<&>"}) == b'{"a":"\\u003c\\u0026\\u003e"}'
    assert load_recording(tmp_path)["events"] == []
    assert load_recording(tmp_path)["metadata"]["active"] is False


# Settings


def test_computer_settings_defaults_and_validation(tmp_path: Path) -> None:
    assert load_computer_settings(tmp_path).backend == "local"
    (tmp_path / "settings.toml").write_text("[computer]\ncpus = 2\nrecording = false\n")
    settings = load_computer_settings(tmp_path)
    assert (settings.cpus, settings.recording) == (2, False)
    for body in ("cpus = 0", "backend = 'e2b'", "recording = 1", "unknown = 1"):
        (tmp_path / "settings.toml").write_text(f"[computer]\n{body}\n")
        with pytest.raises(ComputerSettingsError):
            load_computer_settings(tmp_path)


def test_live_vnc_command_is_always_view_only() -> None:
    assert "-viewonly" in vnc_command("desktop-1", "/tmp/password")


def test_active_tool_and_cleanup_refuse_after_isolation_failure() -> None:
    class FailingVM:
        def verify_isolation(self, _docker: object) -> None:
            raise IsolationError("engine changed")

    class Docker:
        def __init__(self) -> None:
            self.calls: list[tuple[str, ...]] = []

        def output(self, *args: str, **kwargs: object) -> bytes:
            self.calls.append(args)
            return b"unexpected"

        def run(self, *args: str, **kwargs: object) -> object:
            self.calls.append(args)
            return object()

        def close(self) -> None:
            return

    backend = LocalDockerBackend(DesktopOptions("s1", 60), vm=FailingVM())
    docker = Docker()
    backend.name = "desktop"
    backend._docker = docker
    with pytest.raises(IsolationError, match="engine changed"):
        backend._exec(("true",))
    backend.destroy()
    assert docker.calls == []
