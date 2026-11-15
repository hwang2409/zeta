"""off-loop retention activation lifecycle."""

from __future__ import annotations

import asyncio
import inspect
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from ..storage_format import (
    StorageFeature,
    StorageFormatError,
    feature_enabled,
    validate_retention_arm,
)
from . import (
    _ARM_FILENAME,
    RetentionPlanner,
    Yielded,
    _coordinator_authority,
    _load_state,
    _read_json,
    append_retention_history,
    disarm_retention,
    project_retention_result,
    read_retention_history,
    retention_result,
    same_retention_result,
)
from .handshake import ensure_handshake_files, planner_active, planner_request_pending


class Published:
    def __init__(
        self, projection: dict[str, object], *, complete: bool,
        mode: str | None = None, stepped: bool = False,
        changed: bool = False, moved: bool = False,
    ) -> None:
        self.projection = projection
        self.complete = complete
        # the facts next_step decides from: the transaction mode (None for a sweep pass),
        # an armed root's terminal outcome, a durable change, and a moved progress marker
        self.mode = mode
        self.stepped = stepped
        self.changed = changed
        self.moved = moved

    def __getitem__(self, key: str) -> object:
        return self.projection[key]

    def get(self, key: str, default: object = None) -> object:
        return self.projection.get(key, default)


Step = Literal["pass", "episode", "wait", "yield", "stop"]
# a user open is short: retry this soon after yielding to one, not at the next daily wake
_YIELD_RETRY_SECONDS = 1.0


def next_step(outcome: Published | Yielded | Exception | None) -> Step:
    """The one owner rule: from one pass outcome, continue now, end and wait, or yield.

    pass: next pass in this episode. episode: publish this card, then the next root at once.
    wait: publish and wait for the next wake. yield: release, retry after a user open.
    """
    if isinstance(outcome, Yielded):
        return "stop" if outcome.reason == "stop" else "yield"
    if not isinstance(outcome, Published):
        # no pass ran, or it failed: its error is published once and the next wake retries
        return "wait"
    if outcome.stepped:
        return "episode"
    if outcome.complete or outcome.mode is not None:
        # a transaction outcome that is not a step (repeated, or a failure before any root
        # outcome) touched disk but moved nothing: continuing would repeat it forever
        return "wait"
    return "pass" if outcome.changed or outcome.moved else "wait"


_UNSEEDED = object()


def _progress_marker(planner: Any) -> tuple[object, ...]:
    return getattr(planner, "progress_marker", lambda: ())()


def _stale_projection(
    root: Path, current_days: int,
    history: tuple[dict[str, object], ...],
) -> Published:
    """Re-decide stale no-side-effect work from bounded durable state."""
    if history and (
        history[-1].get("days") == current_days
        or history[-1].get("mode") in {"disarmed", "disabled"}
    ):
        return Published(dict(history[-1]), complete=True)

    state: dict[str, object] | None = None
    arm: dict[str, object] | None = None
    home_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        state = _load_state(home_fd)
        value = _read_json(home_fd, _ARM_FILENAME)
        if isinstance(value, dict):
            try:
                validate_retention_arm(value)
            except StorageFormatError:
                arm = None
            else:
                arm = value
    finally:
        os.close(home_fd)

    preview = state.get("preview") if state is not None else None
    if current_days == 0:
        projection = retention_result(
            timestamp=datetime.now(UTC).isoformat(), mode="disabled", days=0,
        )
        complete = True
    elif isinstance(preview, dict) and preview.get("days") == current_days:
        mode = "armed" if arm is not None and arm.get("preview_id") == preview.get("preview_id") else "preview"
        projection = retention_result(
            timestamp=datetime.now(UTC).isoformat(), mode=mode, days=current_days,
            root_ids=tuple(preview.get("root_ids", ())), count=preview.get("count"),
            bytes=preview.get("bytes", 0), preview_id=preview.get("preview_id"),
            cutoff=preview.get("cutoff"),
            armed_by_zeta_version=arm.get("armed_by_zeta_version") if mode == "armed" and arm else None,
        )
        complete = True
    else:
        projection = retention_result(
            timestamp=datetime.now(UTC).isoformat(), mode="preview", days=current_days,
            progress_counts={"roots": 0},
        )
        complete = False
    return Published(projection, complete=complete)


def make_retention_owner(
    home: str | Path, lease: Any, days: Callable[[], int] | None = None,
    *, bounds: Any | None = None, now_ns: Callable[[], int] | None = None,
    session_available: bool = True,
) -> RetentionActivationOwner:
    """Build the process owner from an already-open home lease."""
    from ..config.settings import load_settings

    root = Path(home)
    days = days or (lambda: load_settings(home=root).settings.retention_days)
    retained = lease.retain()
    activation_enabled = feature_enabled(
        retained.observed_epoch, StorageFeature.SESSION_RETENTION_ACTIVATION
    )
    if activation_enabled:
        ensure_handshake_files(root)

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
        with _coordinator_authority(root) as token:
            # the one read of the setting at publication; every decision below uses current.days
            current = token.generation(days())
            history = read_retention_history(root)
            # a transaction outcome (deleted, rejected, or blocked) is a fact: always publish it
            effectful = result.mode is not None or result.changed
            if (
                history
                and history[-1].get("mode") == "armed"
                and history[-1].get("days") == current.days
                and not effectful
            ):
                planner.close()
                projection = dict(history[-1])
                return Published(projection, complete=True)
            status = append_retention_history(
                token, root, retained, projection,
                expected=result.expected_generation,
                current=current,
                before_append=None if effectful else lambda: planner.commit_state(token),
                publish_if_stale=effectful,
            )
            if status == "stale":
                planner.close()
                redecided = _stale_projection(root, current.days, history)
                if not redecided.complete:
                    append_retention_history(token, root, retained, redecided.projection)
                return redecided
            if status == "not_committed":
                planner.close()
                return _stale_projection(root, current.days, history)
        return Published(
            projection,
            complete=result.complete,
            mode=result.mode,
            stepped=result.stepped,
            changed=result.changed,
            moved=before != _progress_marker(planner),
        )

    def record_error(exc: Exception, current_days: int) -> dict[str, object]:
        projection = retention_result(
            timestamp=datetime.now(UTC).isoformat(),
            mode="none",
            days=current_days,
            errors=(f"{type(exc).__name__}: {exc}",),
        )
        with _coordinator_authority(root) as token:
            append_retention_history(token, root, retained, projection)
        return projection

    return RetentionActivationOwner(
        lambda: RetentionPlanner(
            root, retained,
            **{
                key: value for key, value in (("bounds", bounds), ("now_ns", now_ns))
                if value is not None
            },
        ),
        days,
        run_pass=run_pass,
        on_error=record_error,
        home=root,
        close_lease=retained.close,
        activation_enabled=activation_enabled,
        session_available=session_available,
        disable=lambda: disarm_retention(root, retained).projection,
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
        # one planner per process keeps the armed preview and its attempted roots across episodes
        self._planner: Any = None
        self._last_published: object = _UNSEEDED

    def set_publish(self, publish: Callable[[Any], Any] | None) -> None:
        self._publish = publish

    def set_session_available(self) -> None:
        self._session_available = True
        self.wake()

    def wake(self, shown: dict[str, object] | None = None) -> None:
        """Start an episode now; ``shown`` is a card the caller already published."""
        if shown is not None:
            self._last_published = shown
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

    def _should_publish(self, projection: dict[str, object]) -> bool:
        """Publish only what changed since the last card, across episodes and restarts."""
        if (
            projection.get("mode") == "preview"
            and projection.get("preview_id") is None
            and not projection.get("errors")
        ):
            return False
        last = self._last_published
        return not (isinstance(last, dict) and same_retention_result(last, projection))

    def _run_episode(self) -> tuple[list[Published], Yielded | Exception | None]:
        """Run passes while next_step says pass; return the cards and a yield or failure."""
        projections: list[Published] = []
        failure: Exception | None = None
        planner = None
        try:
            if not self._activation_enabled or not self._session_available:
                return projections, None
            days = self._days()
            # an idle home (days 0, nothing to disarm) is left untouched: no probe, no lock
            disarm = days == 0 and self._disable is not None and self._home is not None and any(
                (self._home / name).is_file()
                for name in ("retention-arm.json", "retention-state.json")
            )
            if days == 0 and not disarm:
                return projections, None
            if self._last_published is _UNSEEDED and self._home is not None:
                try:
                    history = read_retention_history(self._home)
                except Exception:  # noqa: BLE001 - a corrupt history only means no seed
                    history = ()
                self._last_published = history[-1] if history else None
            if self._home is not None and planner_request_pending(self._home):
                return projections, Yielded("user")
            if disarm:
                projections.append(Published(self._disable(), complete=True))
                return projections, None
            if self._planner is None:
                self._planner = self._planner_factory()
            planner = self._planner
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
                        if next_step(projection) != "pass":
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
                # cleared before the episode, so a wake (an arm) during the episode is not lost
                self._wake.clear()
                self._episode_task = asyncio.create_task(
                    asyncio.to_thread(self._run_episode), name="retention-episode"
                )
                projections, outcome = await asyncio.shield(self._episode_task)
                self._episode_task = None
                try:
                    outputs = [projection.projection for projection in projections]
                    if isinstance(outcome, Exception) and not self._stop.is_set():
                        outputs.append(await asyncio.to_thread(lambda failed=outcome: self._on_error(failed, self._days())))
                    for output in outputs:
                        if not self._should_publish(output):
                            continue
                        self._last_published = output
                        if self._publish is not None:
                            published = self._publish(output)
                            if inspect.isawaitable(published):
                                await published
                except asyncio.CancelledError:
                    raise
                except Exception:  # noqa: BLE001,S110 - publication must not kill the owner
                    pass
                step = next_step(outcome if outcome is not None else (projections[-1] if projections else None))
                if self._stop.is_set() or step == "stop":
                    return
                if step in {"pass", "episode"}:
                    # continuation: the armed set is not done until no unattempted root is left
                    await asyncio.sleep(0)
                    continue
                try:
                    await asyncio.wait_for(
                        self._wake.wait(),
                        _YIELD_RETRY_SECONDS if step == "yield" else self._wake_seconds,
                    )
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
