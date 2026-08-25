"""User-configured lifecycle hooks."""

from __future__ import annotations

import asyncio
import json
import math
import os
import signal
import tomllib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


HookEvent = Literal[
    "session_start",
    "user_prompt_submit",
    "pre_tool",
    "post_tool",
    "stop",
]
HOOK_EVENTS = frozenset(
    {"session_start", "user_prompt_submit", "pre_tool", "post_tool", "stop"}
)
DEFAULT_TIMEOUT_SECONDS = 10.0
MAX_TIMEOUT_SECONDS = 60.0


class HookConfigError(ValueError):
    """Raised when hooks.toml does not match the supported schema."""


@dataclass(frozen=True, slots=True)
class Hook:
    event: HookEvent
    command: str
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    tools: tuple[str, ...] = ()

    def matches(self, event: str, tool: str | None = None) -> bool:
        if self.event != event:
            return False
        return tool is None or not self.tools or tool in self.tools

    def status_entry(self) -> str:
        return f"{self.event}: {self.command}"


@dataclass(frozen=True, slots=True)
class _CommandResult:
    returncode: int | None
    stderr: str
    timed_out: bool = False


def load_hooks(home: str | Path, *, enabled: bool = True) -> HookManager:
    """Load the flat hooks.toml file below one zeta home."""

    if not enabled:
        return HookManager((), session_id="")
    path = Path(home) / "hooks.toml"
    if not path.exists():
        return HookManager((), session_id="")
    try:
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise HookConfigError(f"invalid hook config {path}: {exc}") from exc
    except OSError as exc:
        raise HookConfigError(f"could not read hook config {path}: {exc}") from exc
    return HookManager(_parse_hooks(document, path), session_id="")


def load_hooks_for_provider(home: str | Path, provider: str) -> HookManager:
    """Keep fake-provider smoke runs isolated from the real hook home."""

    return load_hooks(
        home,
        enabled=provider != "fake" or "ZETA_HOME" in os.environ,
    )


def _parse_hooks(document: object, path: Path) -> tuple[Hook, ...]:
    if type(document) is not dict:
        raise HookConfigError(f"invalid hook config {path}: expected a table")
    unknown_document_keys = set(document) - {"hook"}
    if unknown_document_keys:
        raise HookConfigError(
            f"invalid hook config {path}: unknown keys {sorted(unknown_document_keys)}"
        )
    entries = document.get("hook", [])
    if type(entries) is not list:
        raise HookConfigError(f"invalid hook config {path}: hook must be an array")
    hooks: list[Hook] = []
    for index, entry in enumerate(entries):
        if type(entry) is not dict:
            raise HookConfigError(f"invalid hook config {path}: hook {index} must be a table")
        unknown_keys = set(entry) - {"event", "command", "timeout_seconds", "tools"}
        if unknown_keys:
            raise HookConfigError(
                f"invalid hook config {path}: hook {index} has unknown keys "
                f"{sorted(unknown_keys)}"
            )
        event = entry.get("event")
        command = entry.get("command")
        if type(event) is not str or event not in HOOK_EVENTS:
            raise HookConfigError(f"invalid hook config {path}: hook {index} has an unknown event")
        if type(command) is not str or not command.strip():
            raise HookConfigError(f"invalid hook config {path}: hook {index} needs a command")
        timeout = entry.get("timeout_seconds", DEFAULT_TIMEOUT_SECONDS)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            raise HookConfigError(f"invalid hook config {path}: hook {index} timeout is invalid")
        raw_tools = entry.get("tools", [])
        if type(raw_tools) is not list or any(type(tool) is not str or not tool for tool in raw_tools):
            raise HookConfigError(f"invalid hook config {path}: hook {index} tools is invalid")
        if raw_tools and event not in {"pre_tool", "post_tool"}:
            raise HookConfigError(f"invalid hook config {path}: hook {index} tools only applies to tool events")
        hooks.append(
            Hook(
                event=event,
                command=command,
                timeout_seconds=min(float(timeout), MAX_TIMEOUT_SECONDS),
                tools=tuple(raw_tools),
            )
        )
    return tuple(hooks)


class HookManager:
    """Run configured hooks and keep their output outside model context."""

    def __init__(
        self,
        hooks: tuple[Hook, ...],
        *,
        session_id: str,
        notice_sink: Callable[[str], None] | None = None,
    ) -> None:
        self.hooks = hooks
        self.session_id = session_id
        self.notice_sink = notice_sink
        self._tasks: set[asyncio.Task[None]] = set()
        self._stop_emitted = False

    @property
    def status_entries(self) -> tuple[str, ...]:
        return tuple(hook.status_entry() for hook in self.hooks)

    def bind_session(self, session_id: str) -> None:
        self.session_id = session_id

    async def pre_tool(self, tool: str, args: dict[str, Any]) -> bool | str:
        """Run blocking pre-tool hooks through the existing ToolRegistry seam."""

        for hook in self._matching("pre_tool", tool):
            result = await self._run(hook, {"tool": tool, "args": args})
            if result.returncode == 0:
                continue
            if result.returncode == 2:
                reason = result.stderr.strip() or "hook denied tool"
                self._notice(f"hook denied pre_tool: {reason}")
                return reason
            self._notice(self._failure_message("pre_tool", result))
        return True

    def user_prompt_submit(self, prompt: str) -> None:
        self.emit("user_prompt_submit", {"prompt": prompt})

    def session_start(self) -> None:
        self.emit("session_start", {})

    def post_tool(self, tool: str, result_summary: str) -> None:
        self.emit("post_tool", {"tool": tool, "result_summary": result_summary})

    def stop(self) -> None:
        if self._stop_emitted:
            return
        self._stop_emitted = True
        self.emit("stop", {})

    def emit(self, event: HookEvent, payload: Mapping[str, Any]) -> None:
        for hook in self._matching(event, payload.get("tool")):
            task = asyncio.create_task(self._run_nonblocking(hook, payload))
            self._tasks.add(task)
            task.add_done_callback(self._tasks.discard)

    async def close(self) -> None:
        while self._tasks:
            await asyncio.gather(*tuple(self._tasks), return_exceptions=True)

    def _matching(self, event: str, tool: object = None) -> tuple[Hook, ...]:
        tool_name = tool if isinstance(tool, str) else None
        return tuple(hook for hook in self.hooks if hook.matches(event, tool_name))

    async def _run_nonblocking(
        self, hook: Hook, payload: Mapping[str, Any]
    ) -> None:
        result = await self._run(hook, payload)
        if result.returncode != 0:
            self._notice(self._failure_message(hook.event, result))

    async def _run(self, hook: Hook, payload: Mapping[str, Any]) -> _CommandResult:
        event = {"event": hook.event, "session_id": self.session_id, **payload}
        environment = os.environ.copy()
        environment["ZETA_SESSION_ID"] = self.session_id
        process: asyncio.subprocess.Process | None = None
        try:
            process = await asyncio.create_subprocess_exec(
                "sh",
                "-c",
                hook.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=environment,
                start_new_session=True,
            )
            output = json.dumps(event, separators=(",", ":")).encode() + b"\n"
            try:
                _, stderr = await asyncio.wait_for(
                    process.communicate(output), timeout=hook.timeout_seconds
                )
            except asyncio.TimeoutError:
                _kill_process_group(process.pid)
                await process.communicate()
                return _CommandResult(None, "", timed_out=True)
            return _CommandResult(process.returncode, stderr.decode(errors="replace"))
        except OSError as exc:
            return _CommandResult(None, str(exc))

    def _notice(self, message: str) -> None:
        if self.notice_sink is not None:
            self.notice_sink(" ".join(message.splitlines()) or "hook failed")

    @staticmethod
    def _failure_message(event: str, result: _CommandResult) -> str:
        if result.timed_out:
            return f"hook {event} timed out"
        if result.returncode is None:
            return f"hook {event} failed: {result.stderr}"
        return f"hook {event} failed with exit {result.returncode}"


def _kill_process_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
