"""Small input-time slash-command dispatcher."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class SlashStatus:
    """Read-only session values rendered by ``/status``."""

    session_id: str
    provider: str
    model: str
    retained_tail: int
    tokens_used_this_session: int
    tokens_in_current_context: int | None
    compaction_marker_count: int
    pending_approvals: tuple[str, ...]
    checkpoint_count: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    uncached_input_tokens: int = 0
    output_tokens_this_session: int = 0
    context_files: tuple[str, ...] = ()
    vim_mode: bool = True
    hooks: tuple[str, ...] = ()
    todo_counts: tuple[int, int, int] | None = None


class SlashSession(Protocol):
    def slash_status(self) -> SlashStatus: ...

    def slash_model(self, args: str) -> str: ...

    def slash_vim(self, args: str) -> str: ...

    def slash_paste(self, args: str) -> str: ...

    async def slash_compact(self) -> str: ...

    def slash_checkpoint(self, args: str) -> str: ...

    def slash_fork(self, args: str) -> str: ...


SlashResult = str | Awaitable[str]
SlashHandler = Callable[[SlashSession, str], SlashResult]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """One named command in the input-time registry."""

    name: str
    handler: SlashHandler

    def run(self, session: SlashSession, args: str) -> SlashResult:
        return self.handler(session, args)


class SlashCommandRegistry:
    """Map registered command names to their handlers."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}

    def register(self, command: SlashCommand) -> None:
        if not command.name or any(character.isspace() for character in command.name):
            raise ValueError("slash command name must be one nonempty word")
        if command.name in self._commands:
            raise ValueError(f"slash command already registered: {command.name}")
        self._commands[command.name] = command

    def _dispatch(self, session: SlashSession, value: str) -> SlashResult | None:
        """Run a known command from the first line, or pass the input through."""

        first_line = value.split("\n", 1)[0]
        if not first_line.startswith("/") or first_line.startswith("//"):
            return None
        parts = first_line[1:].split(maxsplit=1)
        if not parts:
            return None
        command = self._commands.get(parts[0])
        if command is None:
            return None
        return command.run(session, parts[1] if len(parts) == 2 else "")

    def dispatch(self, session: SlashSession, value: str) -> SlashResult | None:
        """Run a known command, returning an awaitable for async commands."""

        return self._dispatch(session, value)

    async def dispatch_async(self, session: SlashSession, value: str) -> str | None:
        """Run a known command, awaiting it when the command is asynchronous."""

        result = self._dispatch(session, value)
        if result is None:
            return None
        if isinstance(result, str):
            return result
        return await result

    @staticmethod
    def input_for_model(value: str) -> str:
        """Turn the double-slash escape into one literal leading slash."""

        return value[1:] if value.startswith("//") else value


def _format_status(status: SlashStatus) -> str:
    context_tokens = (
        str(status.tokens_in_current_context)
        if status.tokens_in_current_context is not None
        else "unknown"
    )
    pending = ", ".join(status.pending_approvals) or "none"
    cache_total = (
        status.cache_read_input_tokens
        + status.cache_creation_input_tokens
        + status.uncached_input_tokens
    )
    cache_hit_rate = (
        "n/a"
        if cache_total == 0
        else f"{status.cache_read_input_tokens / cache_total * 100:.1f}%"
    )
    lines = [
            f"session_id: {status.session_id}",
            f"provider: {status.provider}",
            f"model: {status.model}",
            f"vim_mode: {'on' if status.vim_mode else 'off'}",
            f"retained_tail: {status.retained_tail}",
            f"tokens_used_this_session: {status.tokens_used_this_session}",
            f"tokens_in_current_context: {context_tokens}",
            f"compaction_marker_count: {status.compaction_marker_count}",
            f"checkpoint_count: {status.checkpoint_count}",
            f"live_pending_approvals: {len(status.pending_approvals)} ({pending})",
            f"prompt_cache_read: {status.cache_read_input_tokens}",
            f"prompt_cache_write: {status.cache_creation_input_tokens}",
            f"prompt_cache_uncached_input: {status.uncached_input_tokens}",
            f"prompt_cache_hit_rate: {cache_hit_rate}",
            f"output_tokens_this_session: {status.output_tokens_this_session}",
            "context_files: " + (", ".join(status.context_files) or "none"),
            "hooks: " + (", ".join(status.hooks) or "none"),
        ]
    if status.todo_counts is not None:
        pending, in_progress, completed = status.todo_counts
        lines.append(
            "todo: "
            f"pending={pending}, in_progress={in_progress}, completed={completed}"
        )
    return "\n".join(lines)


def _run_status(session: SlashSession, args: str) -> str:
    del args
    return _format_status(session.slash_status())


def _run_model(session: SlashSession, args: str) -> str:
    return session.slash_model(args.strip())


def _run_vim(session: SlashSession, args: str) -> str:
    return session.slash_vim(args.strip())


async def _run_compact(session: SlashSession, args: str) -> str:
    del args
    return await session.slash_compact()


def _run_paste(session: SlashSession, args: str) -> str:
    return session.slash_paste(args)


def _run_checkpoint(session: SlashSession, args: str) -> str:
    return session.slash_checkpoint(args)


def _run_fork(session: SlashSession, args: str) -> str:
    return session.slash_fork(args)


def create_slash_registry() -> SlashCommandRegistry:
    """Create the built-in registry."""

    registry = SlashCommandRegistry()
    registry.register(SlashCommand("status", _run_status))
    registry.register(SlashCommand("model", _run_model))
    registry.register(SlashCommand("vim", _run_vim))
    registry.register(SlashCommand("paste", _run_paste))
    registry.register(SlashCommand("compact", _run_compact))
    registry.register(SlashCommand("checkpoint", _run_checkpoint))
    registry.register(SlashCommand("fork", _run_fork))
    return registry
