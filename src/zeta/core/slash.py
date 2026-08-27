"""Small input-time slash-command dispatcher."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from .store import ConversationEntry
from ..types import Message, MessageRole, TextContent


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """One provider completion's token usage."""

    turn: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def cache_total(self) -> int:
        return (
            self.input_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )

    @property
    def cache_hit_rate(self) -> float | None:
        if self.cache_total == 0:
            return None
        return self.cache_read_input_tokens / self.cache_total * 100


@dataclass(frozen=True, slots=True)
class CompactionSummary:
    """Display data for one durable compaction marker."""

    turn: int
    entries_folded: int
    tokens_saved: int


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """USD price per million tokens for one model."""

    input: float
    output: float
    cache_read: float
    cache_write: float


# Keep this table small and explicit. Unknown models remain valid and show no
# cost estimate. Prices are standard USD per million tokens.
MODEL_PRICES: dict[str, dict[str, ModelPricing | None]] = {
    "claude": {
        "claude-opus-4-6": ModelPricing(5.0, 25.0, 0.5, 6.25),
        "claude-sonnet-4-6": ModelPricing(3.0, 15.0, 0.3, 3.75),
        "claude-haiku-4-5": ModelPricing(1.0, 5.0, 0.1, 1.25),
    },
    "codex": {
        "gpt-5.6-sol": ModelPricing(4.0, 20.0, 0.4, 5.0),
        "gpt-5.6-terra": ModelPricing(2.0, 12.0, 0.2, 2.5),
        "gpt-5.6-luna": ModelPricing(0.2, 1.2, 0.02, 0.25),
        "gpt-5.4": ModelPricing(2.5, 15.0, 0.25, 3.125),
    },
}

MODEL_CONTEXT_WINDOWS: dict[str, dict[str, int | None]] = {
    "claude": {
        "claude-opus-4-6": 1_000_000,
        "claude-sonnet-4-6": 1_000_000,
        "claude-haiku-4-5": 200_000,
    },
    "codex": {
        "gpt-5.6-sol": 1_050_000,
        "gpt-5.6-terra": 1_050_000,
        "gpt-5.6-luna": 1_050_000,
        "gpt-5.4": 1_050_000,
    },
}


def usage_snapshot(usage: dict[str, object], turn: int) -> UsageSnapshot:
    """Convert one normalized usage mapping into safe display values."""

    def integer(*keys: str) -> int:
        for key in keys:
            value = usage.get(key)
            if type(value) is int and value >= 0:
                return value
        return 0

    return UsageSnapshot(
        turn=turn,
        input_tokens=integer("input_tokens", "prompt_tokens"),
        output_tokens=integer("output_tokens", "completion_tokens"),
        cache_read_input_tokens=integer("cache_read_input_tokens"),
        cache_creation_input_tokens=integer("cache_creation_input_tokens"),
    )


def context_fill_percent(
    token_count: int | None,
    model_window: int | None,
) -> int | None:
    """Return a bounded, whole-number context fill percentage."""

    if token_count is None or model_window is None or model_window <= 0:
        return None
    return max(0, min(100, round(token_count * 100 / model_window)))


def render_context_gauge(
    token_count: int | None,
    model_window: int | None,
    *,
    width: int = 20,
) -> str:
    """Render a bounded text gauge, including its percentage."""

    width = max(1, width)
    percent = context_fill_percent(token_count, model_window)
    if percent is None:
        return f"[{'?' * width}] unknown"
    filled = width if percent == 100 else round(width * percent / 100)
    return f"[{'#' * filled}{'-' * (width - filled)}] {percent}%"


def compaction_summary(
    marker: ConversationEntry,
    entries: Sequence[ConversationEntry],
    token_counter: Callable[[Message], int],
    turn: int,
) -> CompactionSummary:
    """Build display data from one durable compaction marker."""

    start = marker.data["source_seq_start"]
    end = marker.data["source_seq_end"]
    source_tokens = sum(
        token_counter(Message.from_dict(entry.data["message"]))
        for entry in entries
        if entry.type == "message" and start <= entry.seq <= end
    )
    summary = Message(
        MessageRole.ASSISTANT,
        [TextContent(marker.data["summary"])],
    )
    return CompactionSummary(
        turn=turn,
        entries_folded=max(0, end - start + 1),
        tokens_saved=max(0, source_tokens - token_counter(summary)),
    )


def usage_history(snapshots: Sequence[dict[str, object]]) -> tuple[UsageSnapshot, ...]:
    """Convert ordered usage snapshots into turn-labelled display data."""

    return tuple(
        usage_snapshot(snapshot, turn)
        for turn, snapshot in enumerate(snapshots, 1)
    )


def compaction_history(
    entries: Sequence[ConversationEntry],
    token_counter: Callable[[Message], int],
) -> tuple[CompactionSummary, ...]:
    """Convert durable compaction markers into ordered display data."""

    return tuple(
        compaction_summary(
            marker,
            entries,
            token_counter,
            max(
                1,
                sum(
                    entry.type == "message"
                    and Message.from_dict(entry.data["message"]).role
                    is MessageRole.USER
                    for entry in entries
                    if entry.seq <= marker.seq
                ),
            ),
        )
        for turn, marker in enumerate(
            (entry for entry in entries if entry.type == "compaction"), 1
        )
    )


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
    usage_history: tuple[UsageSnapshot, ...] = ()
    compaction_history: tuple[CompactionSummary, ...] = ()
    model_window: int | None = None


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
    pricing = MODEL_PRICES.get(status.provider, {}).get(status.model)
    if pricing is None:
        cost_text = f"unavailable (unknown model: {status.model})"
    else:
        cost = sum(
            (
                status.uncached_input_tokens * pricing.input,
                status.output_tokens_this_session * pricing.output,
                status.cache_read_input_tokens * pricing.cache_read,
                status.cache_creation_input_tokens * pricing.cache_write,
            )
        ) / 1_000_000
        cost_text = f"${cost:.6f}"
    trend = status.usage_history[-8:]
    trend_text = " ".join(
        f"{snapshot.turn}:{snapshot.cache_hit_rate:.0f}%"
        if snapshot.cache_hit_rate is not None
        else f"{snapshot.turn}:n/a"
        for snapshot in trend
    ) or "none"
    context_gauge = render_context_gauge(
        status.tokens_in_current_context,
        status.model_window,
    )
    window_text = (
        f"{status.tokens_in_current_context} / {status.model_window}"
        if status.tokens_in_current_context is not None and status.model_window is not None
        else "unknown"
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
        "usage:",
        f"  input_tokens: {status.uncached_input_tokens}",
        f"  output_tokens: {status.output_tokens_this_session}",
        f"  cache_read_tokens: {status.cache_read_input_tokens}",
        f"  cache_write_tokens: {status.cache_creation_input_tokens}",
        f"  cache_hit_rate: {cache_hit_rate}",
        f"  cache_hit_trend: {trend_text}",
        f"  estimated_cost_usd: {cost_text}",
        f"prompt_cache_read: {status.cache_read_input_tokens}",
        f"prompt_cache_write: {status.cache_creation_input_tokens}",
        f"prompt_cache_uncached_input: {status.uncached_input_tokens}",
        f"prompt_cache_hit_rate: {cache_hit_rate}",
        f"output_tokens_this_session: {status.output_tokens_this_session}",
        "context:",
        f"  window: {window_text}",
        f"  fill: {context_gauge}",
        "compaction_history:",
    ]
    lines.extend(
        f"  turn {item.turn}: {item.entries_folded} entries, "
        f"{item.tokens_saved} tokens saved"
        for item in status.compaction_history
    )
    if not status.compaction_history:
        lines.append("  none")
    lines.extend(
        [
            "context_files: " + (", ".join(status.context_files) or "none"),
            "hooks: " + (", ".join(status.hooks) or "none"),
        ]
    )
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
