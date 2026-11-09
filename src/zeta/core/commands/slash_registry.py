"""Typed declarations for commands shared by slash-command surfaces."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, fields, replace
from enum import StrEnum
from functools import partial
from pathlib import PurePath
from types import FunctionType, MappingProxyType
from typing import TYPE_CHECKING, Any

from ...mcp.client import MCPPrompt
from ...mcp.prompt_commands import MCPPromptCommands, SlashPromptError, dispatch_prompt
from ...model_input import ModelInputEnvelope, skill_model_input
from ...protocol.types import MessageOrigin
from ...skills import SkillCatalog, SkillMeta, is_slash_safe_name, load_skill_prompt
from ...skills.catalog import LiveSkillProjection, prepare_slash_projection
from ...skills.invocation import dollar_skill_mentions
from ..checkpoints import ConversationEntry
from .custom_commands import (
    CustomCommand,
    InlineShellRunner,
    needs_inline_shell_resolution,
    render_custom_input,
    resolve_custom_input,
)
from .slash_errors import unknown_slash_command

if TYPE_CHECKING:
    from ..slash import SlashSession

type SlashResult = (
    str
    | ModelInputEnvelope
    | SlashPromptError
    | Awaitable[str | ModelInputEnvelope | SlashPromptError]
)
type SlashHandler = Callable[[SlashSession, str], SlashResult]
type _Outcome = str | ModelInputEnvelope | SlashPromptError | None
type _Route = tuple[str | None, Callable[[SlashLock], Awaitable[_Outcome]]]


async def _resolved(value: _Outcome) -> _Outcome:
    return value


def _never(_args: str) -> bool:
    return False


def _always(_args: str) -> bool:
    return True


@dataclass(frozen=True, slots=True)
class ServeSlash:
    """Serve-specific policy for one shared slash command."""

    mutates: Callable[[str], bool] = _always
    runs_while_busy: Callable[[str], bool] = _never

    def can_run_while_busy(self, args: str) -> bool:
        return self.runs_while_busy(args) and not self.mutates(args)


class SlashSurface(StrEnum):
    TUI = "tui"
    SERVE = "serve"


def waiting_message(holder: str) -> str:
    return f"waiting for {holder} to finish"


def _ignore(_value: object) -> None:
    return None


class SlashLock:
    """One per live session: serializes slash dispatch.

    Lock order: runtime session-execution lease (process lifetime) -> this
    asyncio lock -> operation-local identity/project/shape file locks. It is
    awaited only in ``hold``, from ``SlashCommandRegistry.dispatch_async`` on an
    event-loop task; those file locks are taken in synchronous code, which
    cannot await this lock while it holds one.
    """

    def __init__(
        self,
        *,
        contended: Callable[[str], None] = _ignore,
        waiting: Callable[[str], None] = _ignore,
        # Both receive the holder label, such as "/status".
        late_output: Callable[[object], None] = _ignore,
        offload: Callable[[Callable[[], Any]], Awaitable[Any]] = asyncio.to_thread,
    ) -> None:
        self._lock = asyncio.Lock()
        self._contended = contended
        self._waiting = waiting
        self.late_output = late_output
        # Runs worker steps off the loop; serve uses its shutdown-aware reader.
        self.offload = offload
        self._owner: asyncio.Task[Any] | None = None
        self._holder: str | None = None
        self._state = "running"

    @property
    def holder(self) -> str | None:
        return self._holder

    def status(self) -> dict[str, str] | None:
        """Project the current holder for status surfaces."""

        if self._holder is None:
            return None
        return {"name": self._holder, "state": self._state}

    @asynccontextmanager
    async def hold(self, label: str) -> AsyncIterator[None]:
        task = asyncio.current_task()
        if task is not None and self._owner is task:
            raise RuntimeError(
                f"{self._holder} cannot dispatch {label} while holding the slash lock"
            )
        if self._lock.locked() and self._holder is not None:
            self._contended(self._holder)
        async with self._lock:
            self._owner, self._holder, self._state = task, label, "running"
            try:
                yield
            finally:
                self._owner = self._holder = None

    def defer_cancellation(self) -> None:
        """Show that a cancelled holder is finishing its started worker."""

        self._state = "waiting"
        if self._holder is not None:
            self._waiting(self._holder)


_SCALARS = (type(None), bool, int, float, str, bytes, PurePath)


def _is_frozen_leaf(value: object) -> bool:
    if isinstance(value, _SCALARS):
        return True
    if isinstance(value, ConversationEntry):
        return True  # resident append-only transcript rows, shared with workers on main
    # A plain module-level function: no bound state, closure, or lambda.
    return (
        isinstance(value, FunctionType)
        and value.__closure__ is None
        and "<" not in value.__qualname__
    )


def _frozen_children(value: object) -> Iterable[object] | None:
    """Return the items of an immutable container, or None for anything else."""
    if type(value) is tuple:
        return value
    if type(value) is MappingProxyType:
        return (item for pair in value.items() for item in pair)
    params = getattr(type(value), "__dataclass_params__", None)
    if params is not None and params.frozen:
        return (getattr(value, field.name) for field in fields(value))  # type: ignore[arg-type]
    return None


def _check_frozen(value: object, name: str) -> None:
    if _is_frozen_leaf(value):
        return
    children = _frozen_children(value)
    if children is None:
        raise TypeError(
            f"/{name} prepare returned unsupported/live state {type(value).__name__}; "
            "copy immutable inputs instead"
        )
    for child in children:
        _check_frozen(child, name)


def _standard_prepare(_session: SlashSession, args: str) -> str:
    return args


def _standard_publish(_session: SlashSession, output: object) -> SlashResult:
    if not isinstance(output, (str, ModelInputEnvelope, SlashPromptError)):
        raise TypeError("slash worker returned an unsupported result")
    return output


@dataclass(frozen=True, slots=True)
class _Worker:
    prepare: Callable[[SlashSession, str], object]
    run: Callable[[Any], object]
    publish: Callable[[SlashSession, Any], SlashResult]


@dataclass(frozen=True, slots=True)
class _LoopSafe:
    handler: SlashHandler
    reason: str


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """One named command: worker (prepare, run, publish) or reviewed loop-safe."""

    name: str
    policy: _Worker | _LoopSafe
    description: str = ""
    serve: ServeSlash | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.policy, (_Worker, _LoopSafe)):
            raise TypeError("use SlashCommand.worker or SlashCommand.loop_safe")

    @classmethod
    def worker(
        cls,
        name: str,
        prepare: Callable[[SlashSession, str], object] | None,
        run: Callable[[Any], object],
        publish: Callable[[SlashSession, Any], SlashResult] | None,
        description: str = "",
        *,
        serve: ServeSlash | None = None,
    ) -> SlashCommand:
        """Run ``prepare`` on the loop, ``run`` on a thread, ``publish`` on the loop."""

        for step in (prepare, run, publish):
            if inspect.iscoroutinefunction(step):
                raise TypeError(f"/{name} worker steps must be synchronous")
        policy = _Worker(
            prepare or _standard_prepare, run, publish or _standard_publish
        )
        return cls(name, policy, description, serve)

    @classmethod
    def loop_safe(
        cls,
        name: str,
        handler: SlashHandler,
        description: str = "",
        *,
        serve: ServeSlash | None = None,
        reason: str,
    ) -> SlashCommand:
        """Run ``handler`` on the event loop; ``reason`` records the review."""

        if not reason.strip():
            raise ValueError(f"/{name} loop_safe needs a reason")
        return cls(name, _LoopSafe(handler, reason), description, serve)

    async def run(
        self, session: SlashSession, args: str, lock: SlashLock
    ) -> str | ModelInputEnvelope | SlashPromptError:
        policy = self.policy
        if isinstance(policy, _LoopSafe):
            result = policy.handler(session, args)
            if inspect.isawaitable(result):
                if not inspect.iscoroutinefunction(policy.handler):
                    if inspect.iscoroutine(result):
                        result.close()
                    raise TypeError(
                        f"/{self.name} returned an awaitable from a synchronous "
                        "handler; make it async def"
                    )
                return await result
            return result
        return await _run_worker(self.name, policy, session, args, lock)


async def _run_worker(
    name: str, policy: _Worker, session: SlashSession, args: str, lock: SlashLock
) -> str | ModelInputEnvelope | SlashPromptError:
    inputs = policy.prepare(session, args)
    _check_frozen(inputs, name)
    worker = asyncio.ensure_future(lock.offload(partial(policy.run, inputs)))
    output, cancelled_after_start = await _await_worker(worker, lock)
    result = policy.publish(session, output)
    if cancelled_after_start:
        lock.late_output(result)
        raise asyncio.CancelledError
    return result


async def _await_worker(
    worker: asyncio.Future[object], lock: SlashLock
) -> tuple[object, bool]:
    """Wait for a started worker; return its output and whether the caller cancelled.

    A caller cancel waits for the worker so its output is still published. If
    the worker itself is cancelled (its reader shut down), this re-raises.
    """
    cancelled = False
    while True:
        try:
            return await asyncio.shield(worker), cancelled
        except asyncio.CancelledError:
            if worker.cancelled():
                raise
            if not cancelled:
                cancelled = True
                lock.defer_cancellation()


class SlashCommandRegistry:
    """Map registered command names to their handlers for one surface."""

    def __init__(self, surface: SlashSurface = SlashSurface.TUI) -> None:
        self.surface = surface
        self._commands: dict[str, SlashCommand] = {}
        self._custom_commands: dict[str, CustomCommand] = {}
        self._skills: dict[str, SkillMeta] = {}
        self._skill_notices: tuple[str, ...] = ()
        self._skill_warning_notices: frozenset[str] = frozenset()
        self._notices: list[str] = []
        self._warning_notices: set[str] = set()
        self._mcp_prompts = MCPPromptCommands(self._notices, self._warning_notices)

    def register(self, command: SlashCommand) -> None:
        if not is_slash_safe_name(command.name):
            raise ValueError("slash command name must be one nonempty word")
        if command.name in self._commands:
            raise ValueError(f"slash command already registered: {command.name}")
        self._commands[command.name] = command

    @property
    def custom_commands(self) -> tuple[CustomCommand, ...]:
        """Return loaded custom commands in stable name order."""

        return tuple(
            self._custom_commands[name] for name in sorted(self._custom_commands)
        )

    @property
    def builtin_commands(self) -> tuple[SlashCommand, ...]:
        """Return built-in commands in registration order."""

        return tuple(self._commands.values())

    @property
    def skill_entries(self) -> tuple[SkillMeta, ...]:
        """Return registered skills in stable insertion order."""

        return tuple(self._skills.values())

    @property
    def mcp_prompt_entries(self) -> tuple[tuple[str, str, str], ...]:
        """Return live MCP prompt commands as ``(name, description, server)``."""

        return self._mcp_prompts.completion_entries()

    @property
    def notices(self) -> tuple[str, ...]:
        """Return non-fatal load notices for the session-start display."""

        return (*self._notices, *self._skill_notices)

    @property
    def warning_notices(self) -> frozenset[str]:
        """Return notices that need prominent display."""

        return self._skill_warning_notices.union(self._warning_notices)

    def command_for(self, name: str) -> SlashCommand | None:
        """Return the shared builtin named ``name``, if it exists."""

        return self._commands.get(name)

    def available(self, command: SlashCommand) -> bool:
        """Return whether a builtin is available on this registry's surface."""

        return self.surface is SlashSurface.TUI or command.serve is not None

    @property
    def completion_entries(self) -> tuple[tuple[str, str, str], ...]:
        """Return command names, descriptions, and custom source badges."""

        builtins = tuple(
            (command.name, command.description, "")
            for command in self._commands.values()
            if self.available(command)
        )
        custom = tuple(
            (command.name, command.description, command.source)
            for command in self.custom_commands
        )
        skills = tuple(
            (skill.name, skill.description, skill.source)
            for skill in self._skills.values()
        )
        prompts = self._mcp_prompts.completion_entries()
        return builtins + custom + skills + prompts

    def set_mcp_prompts(self, entries: Sequence[tuple[str, str, MCPPrompt]]) -> None:
        """Replace live MCP prompt commands from one mount snapshot."""
        self._mcp_prompts.replace(tuple(entries))

    def register_custom(self, command: CustomCommand) -> None:
        """Register a custom command unless a built-in owns its name."""

        if command.name in self._commands:
            notice = (
                f"ignored custom command {command.path}: "
                f"shadows built-in /{command.name}"
            )
            self._notices.append(notice)
            self._warning_notices.add(notice)
            return
        if ":" in command.name:
            notice = (
                f"ignored custom command {command.path}: "
                "colon names are reserved for MCP prompts"
            )
            self._notices.append(notice)
            self._warning_notices.add(notice)
            return
        if command.name in self._custom_commands and command.source == "home":
            return
        self._custom_commands[command.name] = command

    def prepare_live_config(
        self, catalog: SkillCatalog, _agents: object
    ) -> LiveSkillProjection:
        return prepare_slash_projection(
            catalog, self._commands, self._custom_commands
        )

    def commit_live_config(self, prepared: object) -> None:
        """Commit one worker-prepared slash projection by reference."""

        if not isinstance(prepared, LiveSkillProjection):
            raise TypeError("expected a live skill projection")
        self._skills = prepared.skills
        self._skill_notices = prepared.notices
        self._skill_warning_notices = prepared.warning_notices

    def register_skills(self, catalog: SkillCatalog) -> None:
        """Replace skills after built-ins and custom commands."""

        self.commit_live_config(self.prepare_live_config(catalog, None))

    def _route(self, session: SlashSession, value: str) -> _Route | None:
        """Resolve one input to its lock label and its deferred execution."""

        first_line = value.split("\n", 1)[0]
        if not first_line.startswith("/") or first_line.startswith("//"):
            mentions = dollar_skill_mentions(value, self._skills)
            if not mentions:
                return None

            def expand() -> ModelInputEnvelope:
                prompts = [
                    load_skill_prompt(self._skills[item.name]) for item in mentions
                ]
                if mentions[0].start == 0 and len(mentions) == 1:
                    request = value[mentions[0].end :]
                    return skill_model_input(prompts[0], request, value)
                return ModelInputEnvelope(
                    "\n\n".join((*prompts, f"User request:\n{value}")),
                    display_text=value,
                    origin=MessageOrigin.SKILL_EXPANSION,
                )

            return f"${mentions[0].name}", lambda _lock: _resolved(expand())
        parts = first_line[1:].split(maxsplit=1)
        if not parts:
            return None
        name = parts[0]
        if not is_slash_safe_name(name):
            return None
        args = parts[1] if len(parts) == 2 else ""
        label = " ".join((f"/{name}", *args.split(maxsplit=1)[:1]))
        command = self._commands.get(name)
        if command is not None and self.available(command):
            return label, lambda lock: command.run(session, args, lock)
        skill = self._skills.get(name)
        if skill is not None:
            request = value[1 + len(name) :]
            return label, lambda _lock: _resolved(
                skill_model_input(load_skill_prompt(skill), request, value)
            )
        prompt = self._mcp_prompts.get(name)
        if prompt is not None:
            return label, lambda _lock: dispatch_prompt(
                session, name, prompt[1], args, display_text=value
            )
        custom = self._custom_commands.get(name)
        if custom is not None:
            if custom.kind != "exec":
                return None
            return label, lambda _lock: session.slash_exec_macro(custom, args)
        return None, lambda _lock: unknown_slash_command(
            value, name, args, (entry[0] for entry in self.completion_entries)
        )

    def routes(self, session: SlashSession, value: str) -> bool:
        """Return whether ``dispatch_async`` handles this input as a command."""

        return self._route(session, value) is not None

    def dispatch(self, session: SlashSession, value: str) -> SlashResult | None:
        """Run ``dispatch_async`` to completion when no event loop is running."""

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.dispatch_async(session, value))
        return self.dispatch_async(session, value)

    async def dispatch_async(
        self, session: SlashSession, value: str
    ) -> str | ModelInputEnvelope | SlashPromptError | None:
        """Run one command while holding the session's slash lock."""

        route = self._route(session, value)
        if route is None:
            return None
        label, execute = route
        lock = session.slash_lock
        if label is None:
            return await execute(lock)
        async with lock.hold(label):
            result = await execute(lock)
        if isinstance(result, ModelInputEnvelope):
            return replace(result, display_text=value)
        return result

    def input_for_model(self, value: str) -> ModelInputEnvelope:
        """Expand a custom command or turn an escape into a literal slash."""
        return render_custom_input(value, self._custom_commands)

    async def resolve_for_model(
        self,
        value: str,
        inline_shell: InlineShellRunner,
    ) -> ModelInputEnvelope | None:
        """Expand a prompt macro and resolve its inline shell spans."""

        return await resolve_custom_input(value, self._custom_commands, inline_shell)

    def needs_inline_shell_resolution(self, value: str) -> bool:
        """Return whether model resolution can wait for shell approval."""

        return needs_inline_shell_resolution(value, self._custom_commands)

    def custom_command_for(self, name: str) -> CustomCommand | None:
        """Return the loaded custom command named ``name``, or None."""

        return self._custom_commands.get(name)

    def exec_command_for(self, value: str) -> CustomCommand | None:
        """Return the custom execution command named by one input value."""

        first_line = value.split("\n", 1)[0]
        if not first_line.startswith("/") or first_line.startswith("//"):
            return None
        parts = first_line[1:].split(maxsplit=1)
        if not parts:
            return None
        name = parts[0]
        if not is_slash_safe_name(name):
            return None
        command = self._custom_commands.get(name)
        return command if command is not None and command.kind == "exec" else None

    def help_text(self) -> str:
        """Format the built-in and loaded custom commands for /help."""

        lines = ["built-in commands:"]
        for command in self._commands.values():
            if not self.available(command):
                continue
            description = f" — {command.description}" if command.description else ""
            lines.append(f"  /{command.name}{description}")
        lines.append("custom commands:")
        if not self._custom_commands:
            lines.append("  none")
        for command in self.custom_commands:
            description = f" — {command.description}" if command.description else ""
            kind = " [exec]" if command.kind == "exec" else ""
            lines.append(
                f"  /{command.name}{kind}{description} (source: {command.path})"
            )
        lines.append("skills (use /name or $name; $name also works inline):")
        if not self._skills:
            lines.append("  none")
        for skill in self._skills.values():
            description = f" — {skill.description}" if skill.description else ""
            lines.append(f"  /{skill.name}{description} (source: {skill.path})")
        return "\n".join(lines)
