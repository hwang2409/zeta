"""A Docker CLI bound to one explicit daemon socket.

Every command passes ``--host`` and ``--config`` explicitly and runs with a
minimal environment, so the default Docker context, ``DOCKER_CONTEXT``, a
credential helper, or another engine (for example Colima or Docker Desktop)
can never receive a command.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any, Self

Runner = Callable[..., subprocess.CompletedProcess[bytes]]


class DockerError(RuntimeError):
    """A Docker command failed."""


class DockerClient:
    """Run Docker CLI commands against exactly one ``unix://`` socket."""

    def __init__(self, host: str, *, runner: Runner = subprocess.run) -> None:
        if not host.startswith("unix:///"):
            raise ValueError("computer Docker host must be an absolute unix:// socket")
        self.host = host
        self._runner = runner
        # mkdtemp creates the directory with mode 0700.
        self._config_dir = Path(tempfile.mkdtemp(prefix="zeta-computer-docker-"))
        (self._config_dir / "config.json").write_text("{}\n", encoding="utf-8")

    @property
    def config_dir(self) -> Path:
        return self._config_dir

    def command(self, *args: str) -> list[str]:
        """Return the full argv for one Docker command."""

        return ["docker", "--host", self.host, "--config", str(self._config_dir), *args]

    @property
    def env(self) -> dict[str, str]:
        """Return the only environment a Docker command receives."""

        return {
            "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
            "HOME": str(Path.home()),
            "DOCKER_HOST": self.host,
            "DOCKER_CONFIG": str(self._config_dir),
        }

    def run(
        self,
        *args: str,
        stdin: bytes | None = None,
        timeout: float = 120,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        result = self._runner(
            self.command(*args),
            input=stdin,
            capture_output=True,
            timeout=timeout,
            check=False,
            env=self.env,
        )
        if check and result.returncode:
            detail = (result.stderr or b"").decode(errors="replace")[-4000:].strip()
            raise DockerError(f"docker {args[0]} failed ({result.returncode}): {detail}")
        return result

    def output(self, *args: str, **kwargs: Any) -> bytes:
        return self.run(*args, **kwargs).stdout

    def remove_labeled(self, label: str) -> tuple[str, ...]:
        """Force-remove every container with ``label`` and return their IDs."""

        ids = tuple(
            self.output("ps", "-aq", "--filter", f"label={label}", timeout=30).decode().split()
        )
        if ids:
            self.run("rm", "-f", *ids, timeout=60, check=False)
        return ids

    def running(self, label: str) -> tuple[str, ...]:
        return tuple(
            self.output("ps", "-q", "--filter", f"label={label}", timeout=30).decode().split()
        )

    def close(self) -> None:
        shutil.rmtree(self._config_dir, ignore_errors=True)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


__all__ = ["DockerClient", "DockerError", "Runner"]
