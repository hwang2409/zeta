"""off-loop retention activation lifecycle."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ..locking import RETENTION_HOME_ID, domain_authority
from ..storage_format import StorageFeature, feature_enabled
from . import (
    _COORDINATOR_DOMAIN,
    RetentionPlanner,
    Yielded,
    append_retention_history,
    project_retention_result,
    read_retention_history,
    retention_authority,
    retention_result,
)
from .handshake import ensure_handshake_files, planner_active, planner_request_pending


class Published:
    def __init__(self, projection: dict[str, object], *, complete: bool, progressed: bool = False) -> None:
        self.projection = projection
        self.complete = complete
        self.progressed = progressed

    def __getitem__(self, key: str) -> object:
        return self.projection[key]

    def get(self, key: str, default: object = None) -> object:
        return self.projection.get(key, default)


def _progress_marker(planner: Any, projection: dict[str, object] | None = None) -> str:
    progress = projection or {}
    return repr(
        (
            getattr(planner, "progress_marker", lambda: ())(),
            progress.get("progress_counts"), progress.get("skip_counts"),
            progress.get("mode"), progress.get("root_ids"),
        )
    )


def _same_projection(left: dict[str, object], right: dict[str, object]) -> bool:
    return {key: value for key, value in left.items() if key != "timestamp"} == {
        key: value for key, value in right.items() if key != "timestamp"
    }


def make_retention_owner(
    home: str | Path, lease: Any, days: Callable[[], int] | None = None,
    *, bounds: Any | None = None, now_ns: Callable[[], int] | None = None,
    session_available: bool = True,
) -> RetentionActivationOwner:
    """Build the process owner from an already-open home lease."""
    from ..config.settings import load_settings

    root = Path(home)
    retained = lease.retain()
    activation_enabled = feature_enabled(
        retained.observed_epoch, StorageFeature.SESSION_RETENTION_ACTIVATION
    )
    if activation_enabled:
        ensure_handshake_files(root)
    last_projection: dict[str, object] | None = None

    def run_pass(
        planner: Any, current_days: int, yield_check: Callable[[], bool] | None = None
    ) -> Published | Yielded:
        before = _progress_marker(planner)
        result = planner.run_pass(
            days=current_days, commit_state=False, yield_check=yield_check
        )
        if isinstance(result, Yielded):
            planner.close()
            return result
        projection = project_retention_result(
            result, days=current_days, timestamp=datetime.now(UTC).isoformat()
        )
        history = read_retention_history(root)
        if (
            history
            and history[-1].get("mode") == "armed"
            and history[-1].get("days") == current_days
            and result.mode is None
        ):
            projection = dict(history[-1])
        nonlocal last_projection
        with domain_authority(root, RETENTION_HOME_ID, _COORDINATOR_DOMAIN, exclusive=True):
            committed = planner.commit_state()
            if not committed:
                planner.close()
                history = read_retention_history(root)
                if result.mode is None and history:
                    projection = dict(history[-1])
                elif result.mode is not None and (
                    not history or not _same_projection(history[-1], projection)
                ):
                    append_retention_history(
                        root, retained, projection, authority_held=True
                    )
                return Published(projection, complete=True)
            if last_projection is None or not _same_projection(last_projection, projection):
                append_retention_history(root, retained, projection, authority_held=True)
                last_projection = projection
        return Published(
            projection, complete=result.complete,
            progressed=before != _progress_marker(planner, projection),
        )

    def record_error(exc: Exception, current_days: int) -> dict[str, object]:
        projection = retention_result(
            timestamp=datetime.now(UTC).isoformat(),
            mode="none",
            days=current_days,
            errors=(f"{type(exc).__name__}: {exc}",),
        )
        append_retention_history(root, retained, projection)
        return projection

    return RetentionActivationOwner(
        lambda: RetentionPlanner(
            root, retained,
            **{
                key: value for key, value in (("bounds", bounds), ("now_ns", now_ns))
                if value is not None
            },
        ),
        days or (lambda: load_settings(home=root).settings.retention_days),
        run_pass=run_pass,
        on_error=record_error,
        home=root,
        close_lease=retained.close,
        activation_enabled=activation_enabled,
        session_available=session_available,
        disable=lambda: retention_authority(root, retained, days=0).projection,
    )


class RetentionActivationOwner:
    """Own one planner episode without blocking the frontend event loop."""

    def __init__(
        self,
        planner_factory: Callable[[], Any],
        days: Callable[[], int],
        *,
        wake_seconds: float = 86_400,
        publish: Callable[[Any], Any] | None = None,
        run_pass: Callable[[Any, int], Published | Yielded] | None = None,
        on_error: Callable[[Exception, int], Any] | None = None,
        home: str | Path | None = None,
        close_lease: Callable[[], Any] | None = None,
        activation_enabled: bool = True,
        session_available: bool = True,
        disable: Callable[[], Any] | None = None,
    ) -> None:
        self._planner_factory = planner_factory
        self._home = Path(home) if home is not None else None
        self._days = days
        self._wake_seconds = wake_seconds
        self._publish = publish
        self._run_pass = run_pass or (
            lambda planner, current_days, yield_check=None: planner.run_pass(
                days=current_days, yield_check=yield_check
            )
        )
        self._on_error = on_error or (lambda exc, current_days: {"mode": "none", "days": current_days, "errors": [str(exc)]})
        self._stop = asyncio.Event()
        self._wake = asyncio.Event()
        self._task: asyncio.Task[None] | None = None
        self._episode_task: asyncio.Task[tuple[list[Any], Exception | None]] | None = None
        self._close_lease = close_lease
        self._activation_enabled = activation_enabled
        self._session_available = session_available
        self._disable = disable

    def set_publish(self, publish: Callable[[Any], Any] | None) -> None:
        self._publish = publish

    def set_session_available(self) -> None:
        self._session_available = True
        self.wake()

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

    def _run_episode(self) -> tuple[list[Published], Yielded | Exception | None]:
        projections: list[Published] = []
        failure: Exception | None = None
        planner = None
        try:
            if not self._activation_enabled or not self._session_available or (self._home is not None and planner_request_pending(self._home)):
                return projections, None
            days = self._days()
            if days == 0:
                if self._disable is not None and self._home is not None and any(
                    (self._home / name).is_file()
                    for name in ("retention-arm.json", "retention-state.json")
                ):
                    projections.append(Published(self._disable(), complete=True))
                return projections, None
            planner = self._planner_factory()
            if not getattr(planner, "enabled", True):
                return projections, None
            context = planner_active(self._home) if self._home is not None else _null_context()
            with context:
                days = self._days()
                if days == 0:
                    if self._disable is not None:
                        projections.append(Published(self._disable(), complete=True))
                else:
                    while not self._stop.is_set():
                        if self._home is not None and planner_request_pending(self._home):
                            return projections, Yielded("user")
                        projection = self._run_pass(
                            planner, days,
                            lambda: "stop" if self._stop.is_set() else (
                                "user" if self._home is not None and planner_request_pending(self._home) else None
                            ),
                        )
                        if isinstance(projection, Yielded):
                            return projections, projection
                        projections.append(projection)
                        if projection.complete or not projection.progressed:
                            break
                planner.close()
                planner = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - keep the owner alive after pass errors
            failure = exc
        finally:
            if planner is not None:
                planner.close()
        return projections, failure

    async def _run(self) -> None:
        try:
            await asyncio.sleep(0)
            while not self._stop.is_set():
                self._episode_task = asyncio.create_task(
                    asyncio.to_thread(self._run_episode), name="retention-episode"
                )
                projections, outcome = await asyncio.shield(self._episode_task)
                self._episode_task = None
                try:
                    published_projection: dict[str, object] | None = None
                    for projection in projections:
                        if (
                            published_projection is not None
                            and _same_projection(published_projection, projection.projection)
                        ):
                            continue
                        if self._publish is not None:
                            published = self._publish(projection.projection)
                            if inspect.isawaitable(published):
                                await published
                        published_projection = projection.projection
                    if isinstance(outcome, Exception) and not self._stop.is_set():
                        error = await asyncio.to_thread(self._on_error, outcome, self._days())
                        if self._publish is not None:
                            published = self._publish(error)
                            if inspect.isawaitable(published):
                                await published
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001,S110 - publication must not kill the owner
                    pass
                if self._stop.is_set():
                    return
                if isinstance(outcome, Yielded):
                    if outcome.reason == "stop":
                        return
                    await asyncio.sleep(0)
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


class _null_context:
    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False
