"""The local backend: a hardened Docker desktop inside the dedicated Lima VM."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from importlib import resources
from pathlib import Path

from .backend import DesktopOptions
from .docker import DockerClient, DockerError
from .lima import IsolationError, SandboxError, SandboxVM
from .x11 import DISPLAY, X11Desktop

CONTAINER_LABEL = "zeta.computer"
SESSION_LABEL = "zeta.computer.session"
CONTAINER_USER = "65532:65532"
MEMORY_LIMIT_BYTES = 1024**3
PIDS_LIMIT = 256
IMAGE_ASSETS = ("Dockerfile", "entrypoint.sh")
IMAGE_SIZE_NOTE = "about 1.2 GiB with Chromium"
_SAFE_LABEL = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def _asset(name: str) -> bytes:
    return (resources.files("zeta.computer") / "assets" / name).read_bytes()


def image_tag() -> str:
    """Return a content-addressed tag, so a changed Dockerfile forces a rebuild."""

    digest = hashlib.sha256()
    for name in IMAGE_ASSETS:
        digest.update(name.encode() + b"\0" + _asset(name) + b"\0")
    return f"zeta-computer:{digest.hexdigest()[:12]}"


def image_present(docker: DockerClient, tag: str) -> bool:
    return docker.run("image", "inspect", tag, timeout=30, check=False).returncode == 0


def build_image(docker: DockerClient) -> str:
    """Build the packaged desktop image in the VM engine and return its tag."""

    tag = image_tag()
    with tempfile.TemporaryDirectory(prefix="zeta-computer-image-") as context:
        for name in IMAGE_ASSETS:
            (Path(context) / name).write_bytes(_asset(name))
        docker.run("build", "--pull", "-t", tag, context, timeout=3600)
    return tag


def session_label(session_id: str) -> str:
    if not _SAFE_LABEL.fullmatch(session_id):
        raise ValueError("computer session id must be a simple identifier")
    return f"{SESSION_LABEL}={session_id}"


def container_args(name: str, image: str, session_id: str, lifetime_seconds: int) -> tuple[str, ...]:
    """Return the complete Docker launch policy. No mounts, ports, or network."""

    return (
        "run", "-d", "--rm",
        "--name", name,
        "--label", f"{CONTAINER_LABEL}=true",
        "--label", session_label(session_id),
        "--network", "none",
        "--read-only",
        "--cap-drop", "ALL",
        "--security-opt", "no-new-privileges",
        "--memory", "1g",
        "--cpus", "1",
        "--pids-limit", str(PIDS_LIMIT),
        "--user", CONTAINER_USER,
        "--init",
        "--shm-size", "256m",
        "--tmpfs", "/tmp:rw,nosuid,nodev,size=256m,mode=1777",
        "--tmpfs", "/home/zeta:rw,nosuid,nodev,size=128m,mode=1777",
        image,
        str(lifetime_seconds),
    )


def check_container_policy(details: dict[str, object]) -> None:
    """Raise ``IsolationError`` unless ``docker inspect`` shows every control."""

    host = details.get("HostConfig")
    config = details.get("Config")
    if not isinstance(host, dict) or not isinstance(config, dict):
        raise IsolationError("desktop container inspect output is incomplete")
    failures = [
        label
        for label, ok in (
            ("mounts", not details.get("Mounts")),
            ("network", host.get("NetworkMode") == "none"),
            ("read-only root", host.get("ReadonlyRootfs") is True),
            ("user", config.get("User") == CONTAINER_USER),
            ("capabilities", host.get("CapDrop") == ["ALL"]),
            ("no-new-privileges", "no-new-privileges" in (host.get("SecurityOpt") or [])),
            ("pids limit", host.get("PidsLimit") == PIDS_LIMIT),
            ("memory limit", host.get("Memory") == MEMORY_LIMIT_BYTES),
            ("published ports", not host.get("PortBindings")),
            ("privileged", host.get("Privileged") is False),
        )
        if not ok
    ]
    if failures:
        raise IsolationError("desktop container failed isolation check: " + ", ".join(failures))


def verified_docker(
    vm: SandboxVM,
    factory: Callable[[str], DockerClient] = DockerClient,
) -> DockerClient:
    """Acquire a Docker client only after proving the VM and engine identity."""

    info = vm.require_running()
    docker = factory(info.docker_host)
    try:
        vm.verify_isolation(docker)
    except BaseException:
        docker.close()
        raise
    return docker


def remove_session_desktops(session_id: str, *, vm: SandboxVM | None = None) -> None:
    """Remove desktops only through a currently verified sandbox engine."""

    try:
        sandbox = vm or SandboxVM()
        with verified_docker(sandbox) as docker:
            docker.remove_labeled(session_label(session_id))
    except (OSError, ValueError, subprocess.SubprocessError, SandboxError, DockerError) as exc:
        print(f"zeta computer: cleanup skipped; isolation verification failed: {exc}", file=sys.stderr)
        return


class LocalDockerBackend(X11Desktop):
    """A networkless Xvfb desktop in the Lima VM, controlled with docker exec."""

    def __init__(
        self,
        options: DesktopOptions,
        *,
        vm: SandboxVM | None = None,
        docker_factory: Callable[[str], DockerClient] = DockerClient,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        session_label(options.session_id)
        self.options = options
        self._vm = vm or SandboxVM()
        self._docker_factory = docker_factory
        self._docker: DockerClient | None = None
        self._clock = clock
        self._sleep = sleep
        self.name: str | None = None
        self._started_at: float | None = None
        self._verified_scope = False
        self._operation_lock = threading.Lock()

    @contextmanager
    def operation(self) -> Iterator[None]:
        """Serialize calls, verify once, and reuse that verified client."""

        with self._operation_lock:
            if self._verified_scope:
                raise RuntimeError("nested computer operation")
            self._client()
            self._verified_scope = True
            try:
                yield
            finally:
                self._verified_scope = False

    def _client(self) -> DockerClient:
        if self._docker is None:
            self._docker = verified_docker(self._vm, self._docker_factory)
        elif not self._verified_scope:
            self._vm.verify_isolation(self._docker)
        return self._docker

    def start(self) -> None:
        docker = self._client()
        if self._started_at is not None:
            if self._clock() - self._started_at <= self.options.ttl_seconds:
                return
            self.destroy()
            raise RuntimeError(
                f"desktop lifetime of {self.options.ttl_seconds}s expired; "
                "the next action starts a fresh desktop"
            )
        image = image_tag()
        if not image_present(docker, image):
            raise SandboxError(f"image {image} is missing; run `zeta computer setup`")
        # One server owns one session's desktop: remove leftovers of a crashed
        # predecessor before starting a fresh one.
        docker.remove_labeled(session_label(self.options.session_id))
        self.name = f"zeta-computer-{uuid.uuid4().hex[:12]}"
        lifetime = self.options.ttl_seconds + 60
        docker.run(*container_args(self.name, image, self.options.session_id, lifetime))
        try:
            details = json.loads(docker.output("inspect", self.name, timeout=30))[0]
            check_container_policy(details)
            for _ in range(150):
                ready = docker.run(
                    "exec", "-e", f"DISPLAY={DISPLAY}", self.name,
                    "xdotool", "getactivewindow", timeout=30, check=False,
                )
                if ready.returncode == 0 and ready.stdout.strip():
                    self._started_at = self._clock()
                    return
                self._sleep(0.1)
            raise RuntimeError("desktop did not become ready")
        except BaseException:
            self.destroy()
            raise

    def destroy(self) -> None:
        name, self.name, self._started_at = self.name, None, None
        if name is not None and self._docker is not None:
            try:
                if not self._verified_scope:
                    self._vm.verify_isolation(self._docker)
            except (OSError, SandboxError, DockerError) as exc:
                print(
                    f"zeta computer: cleanup skipped; isolation verification failed: {exc}",
                    file=sys.stderr,
                )
                return
            self._docker.run("rm", "-f", name, timeout=60, check=False)

    def close(self) -> None:
        """Destroy the desktop and release the isolated Docker config."""

        try:
            self.destroy()
        finally:
            if self._docker is not None:
                self._docker.close()
                self._docker = None

    def _exec(self, command: Sequence[str], *, stdin: bytes | None = None) -> bytes:
        if self.name is None or self._docker is None:
            raise RuntimeError("desktop is not running")
        if not self._verified_scope:
            self._vm.verify_isolation(self._docker)
        interactive = ("-i",) if stdin is not None else ()
        return self._docker.output(
            "exec", *interactive, "-e", f"DISPLAY={DISPLAY}", self.name, *command,
            stdin=stdin,
        )


__all__ = [
    "CONTAINER_LABEL",
    "IMAGE_SIZE_NOTE",
    "SESSION_LABEL",
    "LocalDockerBackend",
    "build_image",
    "check_container_policy",
    "container_args",
    "image_present",
    "image_tag",
    "remove_session_desktops",
    "session_label",
    "verified_docker",
]
