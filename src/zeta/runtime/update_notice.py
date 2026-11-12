"""Detect source updates without blocking the TUI loop."""

from __future__ import annotations

import asyncio
import hashlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

CHECK_INTERVAL_SECONDS = 180.0


def _git_directory(package_dir: Path) -> tuple[Path, Path] | None:
    for directory in (package_dir, *package_dir.parents):
        marker = directory / ".git"
        try:
            if marker.is_dir():
                git_dir = marker
            elif marker.is_file() and (line := marker.read_text(encoding="utf-8").strip()).startswith(
                "gitdir: "
            ):
                git_dir = (marker.parent / line[8:]).resolve()
            else:
                continue
            common = git_dir
            commondir = git_dir / "commondir"
            if commondir.is_file():
                common = (git_dir / commondir.read_text(encoding="utf-8").strip()).resolve()
            return git_dir, common
        except OSError:
            return None
    return None


def _ref_value(git_dir: Path, common_dir: Path, ref: str) -> str | None:
    for directory in (git_dir, common_dir):
        try:
            value = (directory / ref).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if value:
            return value
    try:
        packed = (common_dir / "packed-refs").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    for line in packed:
        if line and not line.startswith("#") and not line.startswith("^"):
            value, separator, packed_ref = line.partition(" ")
            if separator and packed_ref == ref:
                return value
    return None


def _checkout_revision(package_dir: Path) -> str | None:
    directories = _git_directory(package_dir)
    if directories is None:
        return None
    git_dir, common_dir = directories
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if head.startswith("ref: "):
        return _ref_value(git_dir, common_dir, head[5:].strip())
    return head or None


def source_revision(package_file: str | Path) -> str | None:
    """Return the checkout revision, or a bounded installed-source fingerprint."""
    package_dir = Path(package_file).resolve().parent
    checkout_marker = next(
        (directory / ".git" for directory in (package_dir, *package_dir.parents)
         if (directory / ".git").exists()),
        None,
    )
    if checkout_marker is not None:
        return _checkout_revision(package_dir)
    try:
        digest = hashlib.sha256()
        for count, path in enumerate(package_dir.rglob("*.py")):
            if count >= 512:
                break
            stat = path.stat()
            digest.update(str(path.relative_to(package_dir)).encode())
            digest.update(f"{stat.st_mtime_ns}:{stat.st_size}".encode())
        # shortcut: 512 files, enough for an installed package and bounded so a
        # damaged or hostile tree cannot make polling expensive.
        return digest.hexdigest()
    except OSError:
        return None


class UpdateChecker:
    """Poll the imported package revision and report each distinct change."""

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
        self._last_seen = self._initial
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
        if self._last_seen is None:
            return False
        try:
            current = await asyncio.to_thread(self._read_revision, self._package_file)
        except (OSError, ValueError):
            return False
        if current is None or current == self._last_seen or current == self._initial:
            return False
        old = self._initial
        self._last_seen = current
        self._reported = True
        on_update = self._on_update
        if on_update is not None:
            await on_update(old, current)
        return True

    async def _run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                _LOGGER.exception("update notice subscriber failed")


class ProcessUpdateMonitor:
    """One process-owned checker shared by all visible surfaces."""

    def __init__(self, package_file: str | Path) -> None:
        self._subscribers: set[Callable[[str, str], Awaitable[None]]] = set()
        self._checker = UpdateChecker(package_file, self._publish)

    def subscribe(self, callback: Callable[[str, str], Awaitable[None]]) -> Callable[[], None]:
        self._subscribers.add(callback)
        return lambda: self._subscribers.discard(callback)

    def start(self) -> None:
        self._checker.start()

    async def close(self) -> None:
        await self._checker.close()
        self._subscribers.clear()

    async def tick(self) -> bool:
        return await self._checker.tick()

    async def _publish(self, old: str, new: str) -> None:
        for callback in tuple(self._subscribers):
            await callback(old, new)


_process_monitor: ProcessUpdateMonitor | None = None


def process_update_monitor(package_file: str | Path) -> ProcessUpdateMonitor:
    """Return the single monitor for this interpreter process."""
    global _process_monitor
    if _process_monitor is None:
        _process_monitor = ProcessUpdateMonitor(package_file)
    return _process_monitor
