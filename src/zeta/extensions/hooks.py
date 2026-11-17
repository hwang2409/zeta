"""Async in-process extension event hooks."""

from __future__ import annotations

import asyncio
import copy
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

HOOK_OUTPUT_LIMIT = 2048
HOOK_TRUNCATION_MARKER = "...[truncated]"
DEFAULT_HOOK_TIMEOUT_SECONDS = 5.0


class TaskOwner(Protocol):
    def __call__(self, coroutine: Awaitable[Any]) -> asyncio.Task[Any]: ...


def _missing_task_owner(coroutine: Awaitable[Any]) -> asyncio.Task[Any]:
    coroutine.close()
    raise RuntimeError("extension hooks require a session task owner")


@dataclass(slots=True)
class ExtensionHook:
    extension: str
    event: str
    handler: Callable[..., Awaitable[Any]]
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS
    disabled: bool = False


def _consume_task_result(task: asyncio.Task[Any]) -> None:
    if not task.cancelled():
        task.exception()


async def _invoke_with_deadline(
    handler: Callable[..., Awaitable[Any]],
    timeout_seconds: float,
    task_owner: TaskOwner,
    *payload: Any,
) -> Any:
    task = task_owner(handler(*payload))
    try:
        done, _ = await asyncio.wait(
            (task,), timeout=timeout_seconds, return_when=asyncio.FIRST_COMPLETED
        )
    except asyncio.CancelledError:
        task.cancel()
        task.add_done_callback(_consume_task_result)
        raise
    if task not in done:
        task.cancel()
        task.add_done_callback(_consume_task_result)
        raise TimeoutError
    return task.result()


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
    return "invalid", None


class CompositeToolHook:
    """Run the configured hook manager and in-process extension pre-tool hooks."""

    def __init__(
        self,
        base: Callable[[str, dict[str, Any]], Awaitable[bool | str]] | None,
        hooks: list[ExtensionHook],
        notice: Callable[[str, str], None],
        task_owner: TaskOwner,
    ) -> None:
        self._base = base
        self._hooks = hooks
        self._notice = notice
        self._task_owner = task_owner

    async def _run_hook(
        self, hook: ExtensionHook, tool: str, args: dict[str, Any]
    ) -> tuple[str, dict[str, Any] | None]:
        if hook.disabled:
            self._failure(hook, "disabled")
            return "continue", None
        try:
            result = await _invoke_with_deadline(
                hook.handler,
                hook.timeout_seconds,
                self._task_owner,
                tool,
                copy.deepcopy(args),
            )
        except asyncio.CancelledError:
            raise
        except TimeoutError:
            hook.disabled = True
            self._failure(hook, "timed out")
            return "continue", None
        except Exception as exc:  # noqa: BLE001 - extension code must fail open
            hook.disabled = True
            self._failure(hook, f"failed: {_short(exc)}")
            return "continue", None
        outcome, replacement = _normalize_pre_tool_result(result)
        if outcome == "invalid":
            hook.disabled = True
            self._failure(hook, "returned invalid tool arguments")
        return outcome, replacement

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
                self._notice(
                    hook.extension,
                    "pre_tool returned replacement arguments; ignored",
                )
            if outcome != "continue":
                return outcome
        return True

    def _failure(self, hook: ExtensionHook, detail: str) -> None:
        # This is intentionally emitted on every failure, not only when disabling.
        self._notice(
            hook.extension,
            f"pre_tool {detail}; call allowed unchecked",
        )


async def run_extension_hooks(
    hooks: list[ExtensionHook],
    event: str,
    payload: tuple[Any, ...],
    notice: Callable[[str, str], None],
    task_owner: TaskOwner,
) -> str | None:
    """Run non-blocking hooks and return bounded injected text, if any."""

    injected: str | None = None
    for hook in hooks:
        if hook.event != event or hook.disabled:
            continue
        try:
            result = await _invoke_with_deadline(
                hook.handler, hook.timeout_seconds, task_owner, *payload
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - extension code must fail open
            hook.disabled = True
            notice(hook.extension, f"{event} failed: {_short(exc)}")
            continue
        if isinstance(result, str) and event == "user_prompt_submit":
            injected = result[:HOOK_OUTPUT_LIMIT]
            if len(result) > HOOK_OUTPUT_LIMIT:
                injected += HOOK_TRUNCATION_MARKER
    return injected


def _short(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    return text.replace("\n", " ")[:200]


__all__ = ["CompositeToolHook", "ExtensionHook", "run_extension_hooks"]
