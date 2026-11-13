"""off-loop retention activation lifecycle."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any

from .handshake import (
    acquire_user_open_request,
    open_with_retention_yield,
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

    async def yield_to_open(self, operation: Callable[[], Any] | None = None) -> Any:
        """Abandon the current preview before an interactive open proceeds."""
        request_fd = None
        if self._home is not None:
            request_fd = await asyncio.to_thread(acquire_user_open_request, self._home)
        self._open_finished.clear()
        self._restart.set()
        self._wake.set()
        task = self._task
        timed_out = False
        if task is not None and not task.done() and self._planner is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._planner_closed.wait()), 2.0)
            except TimeoutError:
                timed_out = True
        try:
            if operation is None:
                if timed_out and self._home is not None:
                    await asyncio.to_thread(release_user_open_request, request_fd)
                    request_fd = None
                    return await asyncio.to_thread(
                        open_with_retention_yield, self._home, lambda: None
                    )
                return None
            if self._home is None:
                return await asyncio.to_thread(operation)
            return await asyncio.to_thread(open_with_retention_yield, self._home, operation)
        finally:
            if request_fd is not None:
                await asyncio.to_thread(release_user_open_request, request_fd)
            self._open_finished.set()

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
                failed = False
                planner = None
                error: Any = None
                failure: Exception | None = None
                try:
                    planner = self._planner_factory()
                    self._planner = planner
                    self._planner_closed.clear()
                    while not self._stop.is_set() and not self._restart.is_set():
                        days = self._days()
                        if days == 0:
                            break
                        self._run_done.clear()
                        try:
                            result = await asyncio.to_thread(self._run_pass, planner, days)
                        finally:
                            self._run_done.set()
                        if self._publish is not None:
                            await self._publish(result)
                        if not result.complete:
                            continue
                        if self._restart.is_set():
                            break
                        self._wake.clear()
                        try:
                            await asyncio.wait_for(self._wake.wait(), self._wake_seconds)
                        except TimeoutError:
                            continue
                except Exception as exc:  # noqa: BLE001 - keep the owner alive after any pass failure
                    failed = True
                    failure = exc
                finally:
                    self._planner = None
                    if planner is not None:
                        await asyncio.to_thread(planner.close)
                    self._planner_closed.set()
                if failure is not None and not self._stop.is_set():
                    error = await asyncio.to_thread(self._on_error, failure, self._days())
                    if self._publish is not None:
                        await self._publish(error)
                if self._stop.is_set():
                    return
                self._restart.clear()
                if failed:
                    self._wake.clear()
                    try:
                        await asyncio.wait_for(self._wake.wait(), self._wake_seconds)
                    except TimeoutError:
                        pass
        finally:
            self._planner = None
