"""Async in-process extension event hooks."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

HOOK_OUTPUT_LIMIT = 2048
HOOK_TRUNCATION_MARKER = "...[truncated]"
DEFAULT_HOOK_TIMEOUT_SECONDS = 5.0


@dataclass(slots=True)
class ExtensionHook:
    extension: str
    event: str
    handler: Callable[..., Awaitable[Any]]
    timeout_seconds: float = DEFAULT_HOOK_TIMEOUT_SECONDS
    disabled: bool = False


class CompositeToolHook:
    """Run the configured hook manager and in-process extension pre-tool hooks."""

    def __init__(
        self,
        base: Callable[[str, dict[str, Any]], Awaitable[bool | str]] | None,
        hooks: list[ExtensionHook],
        notice: Callable[[str], None],
    ) -> None:
        self._base = base
        self._hooks = hooks
        self._notice = notice

    async def __call__(self, tool: str, args: dict[str, Any]) -> bool | str:
        if self._base is not None:
            result = await self._base(tool, args)
            if result is not True:
                return result
        for hook in self._hooks:
            if hook.event != "pre_tool" or hook.disabled:
                continue
            try:
                result = await asyncio.wait_for(
                    hook.handler(tool, args), timeout=hook.timeout_seconds
                )
            except asyncio.CancelledError:
                raise
            except TimeoutError:
                hook.disabled = True
                self._failure(hook, "timed out")
                continue
            except Exception as exc:  # noqa: BLE001 - extension code must fail open
                hook.disabled = True
                self._failure(hook, f"failed: {_short(exc)}")
                continue
            if result is None:
                continue
            if isinstance(result, str):
                return result
            if isinstance(result, Mapping):
                replacement = result.get("arguments")
                if type(replacement) is not dict:
                    hook.disabled = True
                    self._failure(hook, "returned invalid tool arguments")
                    continue
                args.clear()
                args.update(replacement)
                continue
            hook.disabled = True
            self._failure(hook, "returned an invalid result")
        return True

    def _failure(self, hook: ExtensionHook, detail: str) -> None:
        # This is intentionally emitted on every failure, not only when disabling.
        self._notice(
            f"extension hook warning: {hook.extension} pre_tool {detail}; call allowed unchecked"
        )


async def run_extension_hooks(
    hooks: list[ExtensionHook],
    event: str,
    payload: tuple[Any, ...],
    notice: Callable[[str], None],
) -> str | None:
    """Run non-blocking hooks and return bounded injected text, if any."""

    injected: str | None = None
    for hook in hooks:
        if hook.event != event or hook.disabled:
            continue
        try:
            result = await asyncio.wait_for(
                hook.handler(*payload), timeout=hook.timeout_seconds
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - extension code must fail open
            hook.disabled = True
            notice(f"extension hook warning: {hook.extension} {event} failed: {_short(exc)}")
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
