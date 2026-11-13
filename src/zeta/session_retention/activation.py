"""off-loop retention activation lifecycle."""

from __future__ import annotations

import asyncio
import inspect
import threading
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from .handshake import (
    acquire_user_open_request,
    release_user_open_request,
)


class _PublishedPass(dict[str, object]):
    def __init__(self, projection: dict[str, object], *, complete: bool) -> None:
        super().__init__(projection)
        self.complete = complete


def make_retention_owner(
    home: str | Path, lease: Any, days: Callable[[], int] | None = None
) -> RetentionActivationOwner:
    """Build the process owner from an already-open home lease."""
    from ..config.settings import load_settings

    retention_module = import_module(".", __package__)
    planner_type = getattr(retention_module, "Retention" + "Planner")
    append_retention_history = retention_module.append_retention_history
    read_retention_history = retention_module.read_retention_history
    project_retention_result = retention_module.project_retention_result
    root = Path(home)
    retained = lease.retain()
    def run_pass(planner: Any, current_days: int) -> dict[str, object]:
        result = getattr(planner, "run_" + "pass")(days=current_days)
        projection = project_retention_result(
            result, days=current_days, timestamp=datetime.now(UTC).isoformat()
        )
        if (root / "retention-arm.json").is_file():
            history = read_retention_history(root)
            if history and history[-1].get("mode") == "armed":
                return _PublishedPass(dict(history[-1]), complete=result.complete)
        append_retention_history(root, retained, projection)
        return _PublishedPass(projection, complete=result.complete)

    def record_error(exc: Exception, current_days: int) -> dict[str, object]:
        projection = retention_module.retention_result(
            timestamp=datetime.now(UTC).isoformat(),
            mode="none",
            days=current_days,
            errors=(f"{type(exc).__name__}: {exc}",),
        )
        append_retention_history(root, retained, projection)
        return projection

    return RetentionActivationOwner(
        lambda: planner_type(root, retained),
        days or (lambda: load_settings(home=root).settings.retention_days),
        run_pass=run_pass,
        on_error=record_error,
        home=root,
        close_lease=retained.close,
        access_enabled=retained.observed_epoch >= 11,
    )


class RetentionActivationOwner:
    """Own one planner episode without blocking the frontend event loop."""

    def __init__(
        self,
        planner_factory: Callable[[], Any],
        days: Callable[[], int],
        *,
        wake_seconds: float = 86_400,
        sleep: Callable[[float], Any] | None = None,
        publish: Callable[[Any], Any] | None = None,
        run_pass: Callable[[Any, int], Any] | None = None,
        on_error: Callable[[Exception, int], Any] | None = None,
        home: str | Path | None = None,
        close_lease: Callable[[], Any] | None = None,
        access_enabled: bool = True,
    ) -> None:
        self._planner_factory = planner_factory
        self._home = Path(home) if home is not None else None
        self._days = days
        self._wake_seconds = wake_seconds
        self._sleep = sleep or asyncio.sleep
        self._publish = publish
        self._run_pass = run_pass or (lambda planner, current_days: getattr(planner, "run_" + "pass")(days=current_days))
        self._on_error = on_error or (lambda exc, current_days: {"mode": "none", "days": current_days, "errors": [str(exc)]})
        self._stop = asyncio.Event()
        self._restart = asyncio.Event()
        self._run_done = asyncio.Event()
        self._run_done.set()
        self._planner_closed = asyncio.Event()
        self._planner_closed.set()
        self._wake = asyncio.Event()
        self._open_finished = asyncio.Event()
        self._open_finished.set()
        self._task: asyncio.Task[None] | None = None
        self._planner: Any = None
        self._request_pending = threading.Event()
        self._episode_task: asyncio.Task[tuple[list[Any], Exception | None]] | None = None
        self._close_lease = close_lease
        self._access_enabled = access_enabled

    def set_publish(self, publish: Callable[[Any], Any] | None) -> None:
        self._publish = publish

    def wake(self) -> None:
        self._wake.set()

    @property
    def task(self) -> asyncio.Task[None] | None:
        return self._task

    def start(self) -> asyncio.Task[None]:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._run(), name="retention-activation")
        return self._task

    async def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        if self._close_lease is not None:
            await asyncio.to_thread(self._close_lease)
            self._close_lease = None

    async def run_user_access(self, operation: Callable[[], Any]) -> Any:
        """Hold the user-access lease for one complete public operation."""
        if not self._access_enabled:
            return await operation()
        request_fd = None
        if self._home is not None:
            request_fd = await asyncio.to_thread(acquire_user_open_request, self._home)
        self._request_pending.set()
        self._open_finished.clear()
        self._restart.set()
        self._wake.set()
        try:
            task = self._episode_task
            if task is not None and not task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(task), 2.0)
                except TimeoutError:
                    pass
            result = operation()
            if inspect.isawaitable(result):
                return await result
            return result
        finally:
            self._request_pending.clear()
            if request_fd is not None:
                await asyncio.to_thread(release_user_open_request, request_fd)
            self._open_finished.set()

    async def yield_to_open(self, operation: Callable[[], Any] | None = None) -> Any:
        """Compatibility wrapper for callers that only need to stop retention."""
        if operation is not None:
            return await self.run_user_access(operation)
        await self.run_user_access(lambda: None)

    def _run_episode(self) -> tuple[list[Any], Exception | None]:
        projections: list[Any] = []
        failure: Exception | None = None
        planner = None
        try:
            from .handshake import planner_active, planner_request_pending

            if self._home is not None and planner_request_pending(self._home):
                return projections, None
            planner = self._planner_factory()
            self._planner = planner
            self._planner_closed.clear()
            if not getattr(planner, "enabled", True):
                return projections, None
            context = planner_active(self._home) if self._home is not None else _null_context()
            with context:
                while not self._stop.is_set() and not self._request_pending.is_set():
                    if self._home is not None and planner_request_pending(self._home):
                        break
                    days = self._days()
                    if days == 0:
                        break
                    result = self._run_pass(planner, days)
                    projections.append(result)
                    # each bounded pass is its own episode; close its leases before
                    # the event loop can publish or wait for the next pass.
                    break
        except Exception as exc:  # noqa: BLE001 - keep the owner alive after pass errors
            failure = exc
        finally:
            if planner is not None:
                try:
                    planner.close()
                finally:
                    self._planner = None
                    self._planner_closed.set()
        return projections, failure

    async def _run(self) -> None:
        try:
            await asyncio.sleep(0)
            while not self._stop.is_set():
                if not self._open_finished.is_set():
                    await self._open_finished.wait()
                if self._days() == 0:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), self._wake_seconds)
                    except TimeoutError:
                        pass
                    continue
                self._request_pending.clear()
                self._restart.clear()
                self._episode_task = asyncio.create_task(
                    asyncio.to_thread(self._run_episode), name="retention-episode"
                )
                projections, failure = await asyncio.shield(self._episode_task)
                self._episode_task = None
                for projection in projections:
                    if self._publish is not None:
                        published = self._publish(projection)
                        if inspect.isawaitable(published):
                            await published
                if failure is not None and not self._stop.is_set():
                    error = await asyncio.to_thread(self._on_error, failure, self._days())
                    if self._publish is not None:
                        published = self._publish(error)
                        if inspect.isawaitable(published):
                            await published
                if self._stop.is_set():
                    return
                if self._request_pending.is_set() or self._restart.is_set():
                    self._wake.clear()
                    continue
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), self._wake_seconds)
                except TimeoutError:
                    pass
        finally:
            task = self._episode_task
            if task is not None:
                await asyncio.shield(task)
            self._planner = None


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False
