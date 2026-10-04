"""Desktop runtime boundary and hardened Docker implementation."""

from __future__ import annotations

import json
import os
import pwd
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

PHYSICAL_WIDTH = 1280
PHYSICAL_HEIGHT = 800
MODEL_WIDTH = 1024
MODEL_HEIGHT = 640
DEFAULT_IMAGE = "zeta-computer-mcp:local"
DEFAULT_DOCKER_HOST = f"unix://{Path.home()}/.lima/zeta-sandbox/sock/docker.sock"


@dataclass(frozen=True, slots=True)
class Screenshot:
    data: bytes
    media_type: str = "image/jpeg"


class DesktopBackend(Protocol):
    """Runtime seam for a disposable graphical desktop."""

    def start(self) -> None: ...

    def reset(self) -> None: ...

    def destroy(self) -> None: ...

    def screenshot(self) -> Screenshot: ...

    def input(self, action: str, arguments: dict[str, object]) -> None: ...


def model_coordinate(value: object, *, axis: str) -> int:
    """Validate a model-frame coordinate and scale it to the physical frame."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be x or y")
    limit = MODEL_WIDTH if axis == "x" else MODEL_HEIGHT
    physical = PHYSICAL_WIDTH if axis == "x" else PHYSICAL_HEIGHT
    if type(value) not in {int, float}:
        raise ValueError(f"{axis} must be a number in model frame 0..{limit - 1}")
    number = float(value)
    if not 0 <= number < limit:
        raise ValueError(f"{axis}={value} is outside model frame 0..{limit - 1}")
    return min(physical - 1, max(0, round(number * physical / limit)))


def container_args(
    name: str, image: str, run_id: str, *, enable_vnc: bool = False
) -> tuple[str, ...]:
    """Return the complete no-mount Docker launch policy."""
    return (
        "run",
        "-d",
        "--rm",
        "--name",
        name,
        "--label",
        "zeta.computer-mcp=true",
        "--label",
        f"zeta.computer-mcp.run={run_id}",
        "--network",
        "none",
        "--read-only",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--memory",
        "1g",
        "--cpus",
        "1",
        "--pids-limit",
        "256",
        "--user",
        "65532:65532",
        "--init",
        "--shm-size",
        "256m",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        "--tmpfs",
        "/home/zeta:rw,nosuid,nodev,size=128m,mode=1777",
        *(("--env", "ZETA_COMPUTER_VNC=1") if enable_vnc else ()),
        image,
    )


class DockerDesktopBackend:
    """A networkless Xvfb desktop controlled only with Docker exec."""

    def __init__(
        self,
        *,
        ttl_seconds: int = 900,
        docker_host: str | None = None,
        image: str | None = None,
    ) -> None:
        self.name = f"zeta-computer-{uuid.uuid4().hex[:12]}"
        self.run_id = os.environ.get("ZETA_COMPUTER_RUN_ID", uuid.uuid4().hex)
        self.ttl_seconds = ttl_seconds
        self.docker_host = docker_host or os.environ.get(
            "ZETA_COMPUTER_DOCKER_HOST", DEFAULT_DOCKER_HOST
        )
        self.image = image or os.environ.get("ZETA_COMPUTER_IMAGE", DEFAULT_IMAGE)
        self.docker_config = Path(
            os.environ.get(
                "ZETA_COMPUTER_DOCKER_CONFIG",
                f"/tmp/zeta-computer-docker-config-{os.getpid()}",
            )
        )
        self.docker_config.mkdir(mode=0o700, parents=True, exist_ok=True)
        docker_config_file = self.docker_config / "config.json"
        if docker_config_file.exists():
            if json.loads(docker_config_file.read_text()) != {}:
                raise RuntimeError("Docker CLI config must be an empty isolated config")
        else:
            docker_config_file.write_text("{}\n")
        self.enable_vnc = os.environ.get("ZETA_COMPUTER_VNC") == "1"
        self.created = False
        self.started_at: float | None = None
        self.metrics_path = os.environ.get("ZETA_COMPUTER_METRICS")
        self.artifact_dir = os.environ.get("ZETA_COMPUTER_ARTIFACT_DIR")

    def _docker(self, *args: str, timeout: int = 120, check: bool = True) -> bytes:
        env = {
            "DOCKER_CONFIG": str(self.docker_config),
            "DOCKER_HOST": self.docker_host,
            "HOME": pwd.getpwuid(os.getuid()).pw_dir,
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        }
        result = subprocess.run(
            [
                "docker",
                "--host",
                self.docker_host,
                "--config",
                str(self.docker_config),
                *args,
            ],
            capture_output=True,
            timeout=timeout,
            check=False,
            env=env,
        )
        if check and result.returncode:
            detail = result.stderr.decode(errors="replace")[-4000:]
            raise RuntimeError(f"docker {args[0]} failed ({result.returncode}): {detail}")
        return result.stdout

    def _alive(self) -> None:
        if self.started_at is None:
            self.start()
        elif time.monotonic() - self.started_at > self.ttl_seconds:
            self.destroy()
            raise RuntimeError(f"desktop TTL of {self.ttl_seconds}s expired")

    def start(self) -> None:
        if self.started_at is not None:
            return
        self._docker(
            *container_args(
                self.name, self.image, self.run_id, enable_vnc=self.enable_vnc
            )
        )
        self.created = True
        try:
            details = json.loads(self._docker("inspect", self.name))[0]
            host = details["HostConfig"]
            if (
                details["Mounts"]
                or host["NetworkMode"] != "none"
                or host["ReadonlyRootfs"] is not True
                or details["Config"]["User"] != "65532:65532"
                or host["CapDrop"] != ["ALL"]
                or "no-new-privileges" not in (host.get("SecurityOpt") or [])
                or host["PidsLimit"] != 256
                or host["Memory"] != 1024**3
                or host["PortBindings"]
            ):
                raise RuntimeError("desktop container failed isolation check")
            for _ in range(150):
                result = self._docker(
                    "exec", self.name, "xdotool", "getactivewindow", check=False
                )
                if result.strip():
                    self.started_at = time.monotonic()
                    self._metric("lifecycle_start", self.started_at)
                    return
                time.sleep(0.1)
            raise RuntimeError("desktop did not become ready")
        except BaseException:
            self.destroy()
            raise

    def reset(self) -> None:
        self.destroy()
        self.name = f"zeta-computer-{uuid.uuid4().hex[:12]}"
        self.start()

    def _save_grader_artifacts(self) -> None:
        if not self.artifact_dir or self.started_at is None:
            return
        target = Path(self.artifact_dir)
        target.mkdir(parents=True, exist_ok=True)
        note = self._docker(
            "exec", self.name, "cat", "/home/zeta/notes/demo.txt", check=False
        )
        (target / "demo.txt").write_bytes(note)
        try:
            (target / "final.jpg").write_bytes(
                self._capture_screenshot(metric=False, ensure_alive=False).data
            )
        except RuntimeError as exc:
            (target / "screenshot-error.txt").write_text(str(exc))

    def destroy(self) -> None:
        if not self.created:
            return
        try:
            self._save_grader_artifacts()
        finally:
            self._docker("stop", "--timeout", "1", self.name, timeout=20, check=False)
            self._metric("lifecycle_destroy", time.monotonic())
            self.created = False
            self.started_at = None

    def screenshot(self) -> Screenshot:
        return self._capture_screenshot(metric=True, ensure_alive=True)

    def _capture_screenshot(self, *, metric: bool, ensure_alive: bool) -> Screenshot:
        if ensure_alive:
            self._alive()
        started = time.monotonic()
        data = self._docker(
            "exec",
            "-e",
            "DISPLAY=:99",
            self.name,
            "import",
            "-window",
            "root",
            "-resize",
            f"{MODEL_WIDTH}x{MODEL_HEIGHT}!",
            "-quality",
            "75",
            "jpeg:-",
        )
        if metric:
            self._metric("screenshot", started, len(data))
        if not data.startswith(b"\xff\xd8"):
            raise RuntimeError("desktop returned an invalid JPEG screenshot")
        return Screenshot(data)

    def input(self, action: str, arguments: dict[str, object]) -> None:
        self._alive()
        started = time.monotonic()
        command = self._input_command(action, arguments)
        self._docker("exec", "-e", "DISPLAY=:99", self.name, *command)
        self._metric(action, started)

    def _input_command(
        self, action: str, arguments: dict[str, object]
    ) -> tuple[str, ...]:
        if action in {"click", "double_click"}:
            x = model_coordinate(arguments["x"], axis="x")
            y = model_coordinate(arguments["y"], axis="y")
            button = {"left": "1", "middle": "2", "right": "3"}.get(
                str(arguments.get("button", "left"))
            )
            if button is None:
                raise ValueError("button must be left, middle, or right")
            repeat = "2" if action == "double_click" else "1"
            return (
                "xdotool",
                "mousemove",
                str(x),
                str(y),
                "click",
                "--repeat",
                repeat,
                "--delay",
                "100",
                button,
            )
        if action == "drag":
            x1 = model_coordinate(arguments["x1"], axis="x")
            y1 = model_coordinate(arguments["y1"], axis="y")
            x2 = model_coordinate(arguments["x2"], axis="x")
            y2 = model_coordinate(arguments["y2"], axis="y")
            return (
                "xdotool",
                "mousemove",
                str(x1),
                str(y1),
                "mousedown",
                "1",
                "mousemove",
                "--sync",
                str(x2),
                str(y2),
                "mouseup",
                "1",
            )
        if action == "type":
            text = arguments.get("text")
            if type(text) is not str:
                raise ValueError("text must be a string")
            script = (
                "text=$1; while case $text in *$'\\n'*) true;; *) false;; esac; do "
                "head=${text%%$'\\n'*}; xdotool type --clearmodifiers --delay 1 -- "
                '"$head"; xdotool key Return; text=${text#*$\'\\n\'}; done; '
                'xdotool type --clearmodifiers --delay 1 -- "$text"'
            )
            return ("bash", "-c", script, "zeta-type", text)
        if action == "key":
            keys = arguments.get("keys")
            if type(keys) is not str or not keys or len(keys) > 100:
                raise ValueError(
                    "keys must be a nonempty xdotool key string of at most 100 characters"
                )
            return ("xdotool", "key", "--clearmodifiers", keys)
        if action == "wait":
            seconds = arguments.get("seconds")
            if type(seconds) not in {int, float} or not 0 <= float(seconds) <= 5:
                raise ValueError("seconds must be a number between 0 and 5")
            return ("sleep", str(float(seconds)))
        if action == "scroll":
            x = model_coordinate(arguments["x"], axis="x")
            y = model_coordinate(arguments["y"], axis="y")
            dx, dy = arguments.get("dx"), arguments.get("dy")
            if type(dx) not in {int, float} or type(dy) not in {int, float}:
                raise ValueError("dx and dy must be numbers")
            if abs(float(dx)) > 10_000 or abs(float(dy)) > 10_000:
                raise ValueError("dx and dy must be between -10000 and 10000")
            command = ["xdotool", "mousemove", str(x), str(y)]
            for delta, negative, positive in (
                (float(dy), "4", "5"),
                (float(dx), "6", "7"),
            ):
                if delta:
                    clicks = min(100, max(1, round(abs(delta) / 100)))
                    command += [
                        "click",
                        "--repeat",
                        str(clicks),
                        positive if delta > 0 else negative,
                    ]
            return tuple(command)
        raise ValueError(f"unsupported input action: {action}")

    def _metric(
        self, action: str, started: float, screenshot_bytes: int | None = None
    ) -> None:
        if not self.metrics_path:
            return
        record: dict[str, object] = {
            "action": action,
            "latency_seconds": max(0.0, time.monotonic() - started),
        }
        if screenshot_bytes is not None:
            record["screenshot_bytes"] = screenshot_bytes
        with Path(self.metrics_path).open("a") as output:
            output.write(json.dumps(record) + "\n")
