"""Async in-process extension hooks on one private per-session loop."""

from __future__ import annotations

import asyncio
import contextvars
import copy
import inspect
import threading
import weakref
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from functools import partial
from typing import Any

HOOK_OUTPUT_LIMIT = 2048
HOOK_TRUNCATION_MARKER = "...[truncated]"
DEFAULT_HOOK_TIMEOUT_SECONDS = 5.0
RUNNER_CLOSE_GRACE_SECONDS = 0.25
_THREAD_HAS_CONTEXT = "context" in inspect.signature(threading.Thread).parameters


@dataclass(frozen=True, slots=True)
class HookOutcome:
    kind: str
    value: Any = None


@dataclass(frozen=True, slots=True)
class HookRunnerCloseResult:
    resistant: bool


@dataclass(slots=True)
class ExtensionHook:
    extension: str
    event: str
    handler: Callable[..., Awaitable[Any]]
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS
    disabled: bool = False


class _ResidualTasksError(RuntimeError):
    pass


class _RunnerBridge:
    """Weak handoff from the daemon to one session-loop waiter."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loop: weakref.ReferenceType[asyncio.AbstractEventLoop] | None = None
        self._waiter: weakref.ReferenceType[asyncio.Future[HookOutcome]] | None = None
        self._stopped: weakref.ReferenceType[asyncio.Event] | None = None
        self._generation = 0
        self._cleared = False

    def bind(
        self,
        generation: int,
        loop: asyncio.AbstractEventLoop,
        waiter: asyncio.Future[HookOutcome],
        stopped: asyncio.Event,
    ) -> None:
        with self._lock:
            if self._cleared:
                raise RuntimeError("extension hook runner is closed")
            self._generation = generation
            self._loop = weakref.ref(loop)
            self._waiter = weakref.ref(waiter)
            self._stopped = weakref.ref(stopped)

    def claim_close(self, generation: int) -> asyncio.Future[HookOutcome] | None:
        with self._lock:
            if self._cleared or self._generation != generation:
                return None
            waiter = self._waiter() if self._waiter is not None else None
            self._waiter = None
            return waiter

    def abandon(self, generation: int) -> None:
        with self._lock:
            if self._generation == generation:
                self._waiter = None

    def snapshot_result(
        self, generation: int
    ) -> tuple[asyncio.AbstractEventLoop, asyncio.Future[HookOutcome]] | None:
        with self._lock:
            if self._cleared or generation != self._generation:
                return None
            loop = self._loop() if self._loop is not None else None
            waiter = self._waiter() if self._waiter is not None else None
        if loop is None or waiter is None:
            return None
        return loop, waiter

    def snapshot_stopped(self) -> tuple[asyncio.AbstractEventLoop, asyncio.Event] | None:
        with self._lock:
            if self._cleared:
                return None
            loop = self._loop() if self._loop is not None else None
            stopped = self._stopped() if self._stopped is not None else None
        if loop is None or stopped is None:
            return None
        return loop, stopped

    def clear(self) -> None:
        with self._lock:
            self._cleared = True
            self._loop = None
            self._waiter = None
            self._stopped = None


class _RunnerCapsule:
    """State retained by the daemon; it never owns session objects."""

    def __init__(self, bridge: _RunnerBridge) -> None:
        self.lock = threading.Lock()
        self.bridge = weakref.ref(bridge)
        self.loop: asyncio.AbstractEventLoop | None = None
        self.task: asyncio.Task[Any] | None = None
        self.generation = 0
        self.state = "starting"
        self.handler: Callable[..., Awaitable[Any]] | None = None
        self.payload: tuple[Any, ...] | None = None


def _post(
    loop: asyncio.AbstractEventLoop, callback: Callable[..., object], *args: object
) -> bool:
    """Own every cross-thread post without carrying the caller's context."""
    try:
        loop.call_soon_threadsafe(
            callback, *args, context=contextvars.Context()
        )
    except RuntimeError:
        return False
    return True


def _create_callback_task(
    loop: asyncio.AbstractEventLoop, coroutine: Awaitable[Any]
) -> asyncio.Task[Any]:
    """Create every callback-owned task in a distinct empty context."""
    return loop.create_task(coroutine, context=contextvars.Context())


def _settle_waiter(
    waiter_ref: weakref.ReferenceType[asyncio.Future[HookOutcome]], outcome: HookOutcome
) -> None:
    waiter = waiter_ref()
    if waiter is not None and not waiter.done():
        waiter.set_result(outcome)


def _settle_stopped(stopped_ref: weakref.ReferenceType[asyncio.Event]) -> None:
    stopped = stopped_ref()
    if stopped is not None:
        stopped.set()


def _publish_outcome(capsule: _RunnerCapsule, generation: int, outcome: HookOutcome) -> None:
    bridge = capsule.bridge()
    if bridge is None:
        return
    target = bridge.snapshot_result(generation)
    if target is not None:
        loop, waiter = target
        _post(loop, _settle_waiter, weakref.ref(waiter), outcome)


def _finish_private_task(task: asyncio.Task[Any], capsule: _RunnerCapsule, generation: int) -> None:
    try:
        value = task.result()
    except asyncio.CancelledError:
        outcome = HookOutcome("cancelled")
    except BaseException as exc:  # noqa: BLE001 - contain extension code
        outcome = HookOutcome("error", exc)
    else:
        outcome = HookOutcome("value", value)

    with capsule.lock:
        current = generation == capsule.generation
        closing = capsule.state == "closing"
        if current:
            capsule.task = None
            capsule.handler = None
            capsule.payload = None
            if closing:
                capsule.state = "closing"
            elif outcome.kind == "error" and isinstance(
                outcome.value, _ResidualTasksError
            ):
                capsule.state = "occupied"
            else:
                capsule.state = "idle"
    if current:
        _publish_outcome(capsule, generation, outcome)
    if closing and capsule.loop is not None:
        capsule.loop.stop()


async def _invoke_handler(
    handler: Callable[..., Awaitable[Any]], payload: tuple[Any, ...]
) -> Any:
    current = asyncio.current_task()
    baseline = asyncio.all_tasks()
    error: BaseException | None = None
    value: Any = None
    try:
        coroutine = contextvars.Context().run(handler, *payload)
        if not inspect.isawaitable(coroutine):
            raise TypeError("extension hook did not return an awaitable")
        value = await coroutine
    except BaseException as exc:  # noqa: BLE001 - contain extension code
        error = exc
    residual = asyncio.all_tasks() - baseline - {current}
    for task in residual:
        task.cancel()
    if residual:
        _, pending = await asyncio.wait(residual, timeout=RUNNER_CLOSE_GRACE_SECONDS)
        if pending:
            raise _ResidualTasksError(
                f"{len(pending)} extension hook task(s) resisted cancellation"
            )
    if error is not None:
        raise error
    return value


def _start_private_task(capsule: _RunnerCapsule, generation: int) -> None:
    with capsule.lock:
        if generation != capsule.generation or capsule.state not in {"starting", "active"}:
            return
        handler = capsule.handler
        payload = capsule.payload
        loop = capsule.loop
        capsule.state = "active"
    assert handler is not None and payload is not None and loop is not None
    try:
        task = _create_callback_task(loop, _invoke_handler(handler, payload))
    except BaseException as exc:  # noqa: BLE001 - contain extension code
        with capsule.lock:
            capsule.handler = None
            capsule.payload = None
            capsule.state = "idle"
        _publish_outcome(capsule, generation, HookOutcome("error", exc))
        return
    with capsule.lock:
        capsule.task = task
    task.add_done_callback(
        partial(_finish_private_task, capsule=capsule, generation=generation),
        context=contextvars.Context(),
    )


def _cancel_private_task(capsule: _RunnerCapsule, generation: int, closing: bool) -> None:
    with capsule.lock:
        if generation != capsule.generation:
            return
        if closing:
            capsule.state = "closing"
        elif capsule.state != "idle":
            capsule.state = "occupied"
        task = capsule.task
        loop = capsule.loop
    if task is not None:
        task.cancel()
    elif closing and loop is not None:
        loop.stop()


def _bootstrap(capsule: _RunnerCapsule) -> None:
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    with capsule.lock:
        capsule.loop = loop
        generation = capsule.generation
    loop.call_soon(_start_private_task, capsule, generation, context=contextvars.Context())
    loop.run_forever()
    pending = asyncio.all_tasks(loop)
    if pending:
        # A cooperative stop has no resistant task. Never gather a resistant one.
        with capsule.lock:
            resistant = any(not task.done() for task in pending)
        if resistant:
            return
    loop.close()
    bridge = capsule.bridge()
    if bridge is not None:
        target = bridge.snapshot_stopped()
        if target is not None:
            main_loop, stopped = target
            _post(main_loop, _settle_stopped, weakref.ref(stopped))


def _thread_target(capsule: _RunnerCapsule) -> None:
    contextvars.Context().run(_bootstrap, capsule)


def _new_daemon(capsule: _RunnerCapsule) -> threading.Thread:
    kwargs: dict[str, Any] = {
        "target": _thread_target,
        "args": (capsule,),
        "name": "zeta-extension-hooks",
        "daemon": True,
    }
    if _THREAD_HAS_CONTEXT:
        kwargs["context"] = contextvars.Context()
    return threading.Thread(**kwargs)


class HookRunner:
    """Run one extension hook at a time on a lazy private daemon loop."""

    def __init__(self, *, close_grace: float = RUNNER_CLOSE_GRACE_SECONDS) -> None:
        self._lock = threading.Lock()
        self._bridge = _RunnerBridge()
        self._capsule: _RunnerCapsule | None = None
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._state = "open"
        self._stopped: asyncio.Event | None = None
        self._close_grace = close_grace
        self._finalizer: asyncio.Task[HookRunnerCloseResult] | None = None

    async def invoke(
        self,
        handler: Callable[..., Awaitable[Any]],
        payload: tuple[Any, ...],
        deadline: float,
    ) -> HookOutcome:
        copied = copy.deepcopy(payload)
        loop = asyncio.get_running_loop()
        waiter: asyncio.Future[HookOutcome] = loop.create_future()
        stopped = self._stopped or asyncio.Event()
        self._stopped = stopped
        with self._lock:
            if self._state != "open":
                return HookOutcome("closing")
            capsule = self._capsule
            if capsule is not None:
                with capsule.lock:
                    if capsule.state != "idle":
                        return HookOutcome("busy")
            self._generation += 1
            generation = self._generation
            self._bridge.bind(generation, loop, waiter, stopped)
            if capsule is None:
                capsule = _RunnerCapsule(self._bridge)
                self._capsule = capsule
                with capsule.lock:
                    capsule.generation = generation
                    capsule.handler = handler
                    capsule.payload = copied
                thread = _new_daemon(capsule)
                self._thread = thread
                thread.start()
            else:
                with capsule.lock:
                    capsule.generation = generation
                    capsule.handler = handler
                    capsule.payload = copied
                    capsule.state = "active"
                    private_loop = capsule.loop
                assert private_loop is not None
                if not _post(private_loop, _start_private_task, capsule, generation):
                    return HookOutcome("closing")
        try:
            return await asyncio.wait_for(asyncio.shield(waiter), deadline)
        except TimeoutError:
            self._abandon(capsule, generation, closing=False)
            return HookOutcome("timeout")
        except asyncio.CancelledError:
            self._abandon(capsule, generation, closing=False)
            raise

    def _abandon(self, capsule: _RunnerCapsule, generation: int, *, closing: bool) -> None:
        self._bridge.abandon(generation)
        with capsule.lock:
            private_loop = capsule.loop
        if private_loop is not None:
            _post(private_loop, _cancel_private_task, capsule, generation, closing)

    def close_unused(self) -> bool:
        """Close synchronously only before the lazy daemon was created."""
        with self._lock:
            if self._capsule is not None:
                return False
            self._state = "closed"
        self._bridge.clear()
        return True

    async def aclose(self) -> HookRunnerCloseResult:
        if self._finalizer is None:
            self._finalizer = asyncio.create_task(self._finalize_close())
        try:
            return await asyncio.shield(self._finalizer)
        except asyncio.CancelledError:
            await asyncio.shield(self._finalizer)
            raise

    async def _finalize_close(self) -> HookRunnerCloseResult:
        with self._lock:
            if self._state == "closed":
                return HookRunnerCloseResult(False)
            self._state = "closing"
            capsule = self._capsule
            generation = self._generation
        if capsule is None:
            self._state = "closed"
            self._bridge.clear()
            return HookRunnerCloseResult(False)
        waiter = self._bridge.claim_close(generation)
        if waiter is not None and not waiter.done():
            waiter.set_result(HookOutcome("closing"))
        self._abandon(capsule, generation, closing=True)
        stopped = self._stopped
        resistant = True
        if stopped is not None:
            try:
                await asyncio.wait_for(stopped.wait(), self._close_grace)
                resistant = False
            except TimeoutError:
                pass
        self._bridge.clear()
        with self._lock:
            self._capsule = None
            self._thread = None
            self._state = "closed"
        return HookRunnerCloseResult(resistant)


def _normalize_pre_tool_result(result: Any) -> tuple[str, dict[str, Any] | None]:
    if result is None:
        return "continue", None
    if isinstance(result, str):
        return result, None
    if isinstance(result, Mapping):
        replacement = result.get("arguments")
        if type(replacement) is dict:
            return "continue", replacement
    return "invalid", None


class CompositeToolHook:
    """Run configured hooks and in-process extension pre-tool hooks."""

    def __init__(
        self,
        base: Callable[[str, dict[str, Any]], Awaitable[bool | str]] | None,
        hooks: list[ExtensionHook],
        notice: Callable[[str, str], Awaitable[None]],
        runner: HookRunner,
    ) -> None:
        self._base = base
        self._hooks = hooks
        self._notice = notice
        self._runner = runner

    async def _run_hook(
        self, hook: ExtensionHook, tool: str, args: dict[str, Any]
    ) -> tuple[str, dict[str, Any] | None]:
        if hook.disabled:
            await self._failure(hook, "disabled")
            return "continue", None
        outcome = await self._runner.invoke(
            hook.handler, (tool, copy.deepcopy(args)), hook.timeout_seconds
        )
        if outcome.kind == "value":
            normalized, replacement = _normalize_pre_tool_result(outcome.value)
            if normalized == "invalid":
                hook.disabled = True
                await self._failure(hook, "returned invalid tool arguments")
            return normalized, replacement
        if outcome.kind == "cancelled":
            raise asyncio.CancelledError
        hook.disabled = True
        detail = {
            "timeout": "timed out",
            "busy": "runner busy",
            "closing": "runner closing",
        }.get(outcome.kind, f"failed: {_short(outcome.value)}")
        await self._failure(hook, detail)
        return "continue", None

    async def __call__(self, tool: str, args: dict[str, Any]) -> bool | str:
        if self._base is not None:
            result = await self._base(tool, args)
            if result is not True:
                return result
        for hook in self._hooks:
            if hook.event != "pre_tool":
                continue
            outcome, replacement = await self._run_hook(hook, tool, args)
            if replacement is not None:
                await self._notice(
                    hook.extension, "pre_tool returned replacement arguments; ignored"
                )
            if outcome != "continue":
                return outcome
        return True

    async def _failure(self, hook: ExtensionHook, detail: str) -> None:
        await self._notice(
            hook.extension, f"pre_tool {detail}; call allowed unchecked"
        )


async def run_extension_hooks(
    hooks: list[ExtensionHook],
    event: str,
    payload: tuple[Any, ...],
    notice: Callable[[str, str], Awaitable[None]],
    runner: HookRunner,
) -> str | None:
    """Run lifecycle hooks and return bounded injected text, if any."""
    injected: str | None = None
    for hook in hooks:
        if hook.event != event or hook.disabled:
            continue
        outcome = await runner.invoke(hook.handler, payload, hook.timeout_seconds)
        if outcome.kind == "cancelled":
            raise asyncio.CancelledError
        if outcome.kind != "value":
            hook.disabled = True
            detail = "timed out" if outcome.kind == "timeout" else f"failed: {_short(outcome.value)}"
            await notice(hook.extension, f"{event} {detail}")
            continue
        result = outcome.value
        if isinstance(result, str) and event == "user_prompt_submit":
            injected = result[:HOOK_OUTPUT_LIMIT]
            if len(result) > HOOK_OUTPUT_LIMIT:
                injected += HOOK_TRUNCATION_MARKER
    return injected


def _short(exc: object) -> str:
    if isinstance(exc, BaseException):
        text = str(exc) or exc.__class__.__name__
    else:
        text = str(exc)
    return text.replace("\n", " ")[:200]


__all__ = [
    "CompositeToolHook",
    "ExtensionHook",
    "HookOutcome",
    "HookRunner",
    "HookRunnerCloseResult",
    "run_extension_hooks",
]
