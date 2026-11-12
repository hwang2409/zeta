"""Detect source updates without blocking the TUI loop."""

from __future__ import annotations

import asyncio
import hashlib
import subprocess
from collections.abc import Awaitable, Callable
from pathlib import Path

CHECK_INTERVAL_SECONDS = 180.0


def source_revision(package_file: str | Path) -> str | None:
    """Return the checkout revision containing package_file, or disable checking."""
    try:
        result = subprocess.run(
            ["git", "-C", str(Path(package_file).resolve().parent), "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    if result.returncode == 0 and revision:
        return revision
    try:
        package_dir = Path(package_file).resolve().parent
        digest = hashlib.sha256()
        for path in sorted(package_dir.rglob("*.py")):
            stat = path.stat()
            digest.update(str(path.relative_to(package_dir)).encode())
            digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
        return digest.hexdigest()
    except OSError:
        return None


class UpdateChecker:
    """Poll the imported package revision and report the first change once."""

    def __init__(
        self,
        package_file: str | Path,
        on_update: Callable[[str, str], Awaitable[None]],
        *,
        interval: float = CHECK_INTERVAL_SECONDS,
        read_revision: Callable[[str | Path], str | None] = source_revision,
    ) -> None:
        self._package_file = package_file
        self._on_update: Callable[[str, str], Awaitable[None]] | None = on_update
        self._interval = interval
        self._read_revision = read_revision
        self._initial = read_revision(package_file)
        self._task: asyncio.Task[None] | None = None
        self._reported = False

    def start(self) -> None:
        if self._task is None and self._initial is not None:
            self._task = asyncio.create_task(self._run())

    async def close(self) -> None:
        if self._task is None:
            self._on_update = None
            return
        self._task.cancel()
        await asyncio.gather(self._task, return_exceptions=True)
        self._task = None
        self._on_update = None

    async def tick(self) -> bool:
        """Check once; this seam lets tests trigger a check without sleeping."""
        if self._reported or self._initial is None:
            return False
        current = await asyncio.to_thread(self._read_revision, self._package_file)
        if current is None or current == self._initial:
            return False
        self._reported = True
        on_update = self._on_update
        if on_update is not None:
            await on_update(self._initial, current)
        return True

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:  # noqa: BLE001, S112 - checks are best effort
                # A transient checkout/read failure must not affect the TUI.
                continue
