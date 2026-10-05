"""The dedicated Lima VM that hosts computer-use desktops.

Zeta uses one VM, ``zeta-sandbox``, created with no host mounts. Zeta reaches
its Docker engine only through the VM's forwarded socket. ``verify_isolation``
proves the boundary before any use: if one check fails, the caller must refuse
to run a desktop. This module never calls Colima and never reads the default
Docker context.
"""

from __future__ import annotations

import json
import os
import pwd
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from .docker import DockerClient, Runner

VM_NAME = "zeta-sandbox"
HOST_MOUNT_FILESYSTEMS = frozenset({"virtiofs", "9p", "sshfs", "fuse.sshfs"})
GUEST_DOCKER_SOCKET_SUFFIX = "/docker.sock"


class SandboxError(RuntimeError):
    """The sandbox VM is absent, stopped, or failed a command."""


class IsolationError(SandboxError):
    """The sandbox VM failed an isolation check. Do not use it."""


@dataclass(frozen=True, slots=True)
class VMInfo:
    name: str
    status: str
    directory: Path
    cpus: int
    memory_bytes: int
    disk_bytes: int
    config: dict[str, object] = field(default_factory=dict)

    @property
    def docker_host(self) -> str:
        return f"unix://{self.directory / 'sock' / 'docker.sock'}"


@dataclass(frozen=True, slots=True)
class IsolationReport:
    """The checks that passed, in order, for display."""

    checks: tuple[str, ...]


def host_home() -> Path:
    """Return the real host home from the user database, not from ``$HOME``."""

    return Path(pwd.getpwuid(os.getuid()).pw_dir)


def mounted_host_paths(proc_mounts: str, home: Path) -> list[str]:
    """Return guest mounts that could expose host files."""

    findings: list[str] = []
    for line in proc_mounts.splitlines():
        fields = line.split()
        if len(fields) < 3:
            continue
        mount_point, filesystem = fields[1], fields[2]
        if (
            filesystem in HOST_MOUNT_FILESYSTEMS
            or mount_point in {"/Users", str(home)}
            or mount_point.startswith(("/Users/", f"{home}/"))
        ):
            findings.append(f"{mount_point} ({filesystem})")
    return findings


class SandboxVM:
    """Create, start, stop, delete, and verify the dedicated Lima VM."""

    def __init__(self, name: str = VM_NAME, *, runner: Runner = subprocess.run) -> None:
        self.name = name
        self._runner = runner

    def _limactl(
        self, *args: str, timeout: float = 600, check: bool = True
    ) -> subprocess.CompletedProcess[bytes]:
        try:
            result = self._runner(
                ["limactl", *args], capture_output=True, timeout=timeout, check=False
            )
        except FileNotFoundError as exc:
            raise SandboxError("limactl is not installed; install Lima 2.x first") from exc
        if check and result.returncode:
            detail = (result.stderr or b"").decode(errors="replace")[-2000:].strip()
            raise SandboxError(f"limactl {args[0]} failed ({result.returncode}): {detail}")
        return result

    def info(self) -> VMInfo | None:
        """Return the VM description, or None when the VM does not exist."""

        result = self._limactl("list", "--json", self.name, timeout=60, check=False)
        if result.returncode:
            return None
        lines = [line for line in result.stdout.decode().splitlines() if line.strip()]
        if len(lines) != 1:
            return None
        data = json.loads(lines[0])
        return VMInfo(
            name=str(data["name"]),
            status=str(data["status"]),
            directory=Path(str(data["dir"])),
            cpus=int(data.get("cpus") or 0),
            memory_bytes=int(data.get("memory") or 0),
            disk_bytes=int(data.get("disk") or 0),
            config=dict(data.get("config") or {}),
        )

    def require_running(self) -> VMInfo:
        info = self.info()
        if info is None:
            raise SandboxError(f"VM {self.name} does not exist; run `zeta computer setup`")
        if info.status != "Running":
            raise SandboxError(
                f"VM {self.name} is {info.status}; run `zeta computer setup` to start it"
            )
        return info

    def create(self, *, cpus: int, memory_gib: int, disk_gib: int) -> None:
        self._limactl(
            "create",
            f"--name={self.name}",
            "--mount-none",
            f"--cpus={cpus}",
            f"--memory={memory_gib}",
            f"--disk={disk_gib}",
            "--tty=false",
            "template:docker",
            timeout=1800,
        )

    def start(self) -> None:
        self._limactl("start", "--tty=false", self.name, timeout=900)

    def stop(self) -> None:
        self._limactl("stop", self.name, timeout=300)

    def delete(self) -> None:
        self._limactl("delete", "--force", self.name, timeout=300)

    def shell(self, *command: str, check: bool = True) -> subprocess.CompletedProcess[bytes]:
        return self._limactl(
            "shell", "--workdir", "/", self.name, "--", *command, timeout=60, check=check
        )

    def verify_isolation(self, docker: DockerClient | None = None) -> IsolationReport:
        """Prove the VM exposes no host files; raise ``IsolationError`` if not.

        With ``docker``, also prove that the client's socket reaches this VM's
        engine and no other engine.
        """

        info = self.require_running()
        checks: list[str] = [f"VM {self.name} is Running ({info.directory})"]
        mounts = info.config.get("mounts")
        if mounts:
            raise IsolationError(f"VM {self.name} configures host mounts: {mounts}")
        checks.append("Lima config has no mounts")
        expected_socket = str(info.directory / "sock" / "docker.sock")
        for forward in info.config.get("portForwards") or []:
            if not isinstance(forward, dict):
                raise IsolationError("VM has an unreadable port forward")
            if forward.get("reverse"):
                raise IsolationError("VM has a reverse port forward into the guest")
            guest_socket = forward.get("guestSocket")
            if guest_socket and (
                not str(guest_socket).endswith(GUEST_DOCKER_SOCKET_SUFFIX)
                or forward.get("hostSocket") != expected_socket
            ):
                raise IsolationError(f"VM forwards an unexpected socket: {guest_socket}")
        checks.append("only the Docker socket is forwarded")
        home = host_home()
        proc_mounts = self.shell("cat", "/proc/mounts").stdout.decode(errors="replace")
        findings = mounted_host_paths(proc_mounts, home)
        if findings:
            raise IsolationError("VM has host mounts: " + ", ".join(findings))
        checks.append("guest has no virtiofs, 9p, or sshfs mounts")
        absent = self.shell(
            "sh", "-c", 'test ! -e /Users && test ! -e "$1"', "sh", str(home), check=False
        )
        if absent.returncode:
            raise IsolationError(f"/Users or {home} exists inside VM {self.name}")
        checks.append(f"/Users and {home} are absent in the guest")
        if docker is not None:
            if docker.host != info.docker_host:
                raise IsolationError(f"Docker host {docker.host} is not the VM socket")
            engine = docker.output("info", "--format", "{{.Name}}", timeout=30).decode().strip()
            if engine != f"lima-{self.name}":
                raise IsolationError(f"Docker socket reaches engine {engine!r}, not the VM")
            checks.append(f"Docker engine is {engine} via {docker.host}")
        return IsolationReport(tuple(checks))


__all__ = [
    "VM_NAME",
    "IsolationError",
    "IsolationReport",
    "SandboxError",
    "SandboxVM",
    "VMInfo",
    "host_home",
    "mounted_host_paths",
]
