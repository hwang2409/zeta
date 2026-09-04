"""Small input-time slash-command dispatcher."""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

import yaml

from ..mcp.client import MCPPrompt
from ..types import Message, MessageRole, StreamEventType, TextContent
from ..mcp.prompt_commands import (
    MCPPromptCommands,
    SlashModelInput,
    SlashPromptError,
    dispatch_prompt,
)
from .commands.custom_commands import (
    COMMAND_FILE_SIZE_LIMIT,  # noqa: F401 - public compatibility export
    CustomCommand,
    InlineShellRunner,
    load_custom_commands,
    needs_inline_shell_resolution,
    render_custom_input,
    resolve_custom_input,
)
from .store import ConversationEntry


class UsageCounterSource(Protocol):
    """Cumulative usage counters exposed by the context assembler."""

    uncached_input_tokens_this_session: int
    output_tokens_this_session: int
    cache_read_input_tokens_this_session: int
    cache_creation_input_tokens_this_session: int


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """One completed turn's token usage."""

    turn: int
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0
    model: str | None = None
    estimated_cost_usd: float | None = None

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


class UsageTracker:
    """Keep bounded usage deltas at real turn boundaries."""

    def __init__(self, source: UsageCounterSource, provider: str | None = None) -> None:
        self.source = source
        self.provider = provider
        self._baseline = self._counters()
        self._cost_baseline = self._baseline
        self._history: deque[UsageSnapshot] = deque(maxlen=8)
        self._cost_by_model: dict[str, UsageSnapshot] = {}
        self._completed_turns = 0

    @property
    def history(self) -> tuple[UsageSnapshot, ...]:
        return tuple(self._history)

    @property
    def cost_by_model(self) -> tuple[UsageSnapshot, ...]:
        return tuple(self._cost_by_model.values())

    def record(self, event_type: StreamEventType, model: str) -> None:
        current = self._counters()
        self._record_cost_delta(current, model)
        if event_type is StreamEventType.COMPACTION_END:
            self._baseline = current
        elif event_type is StreamEventType.TURN_END:
            self._completed_turns += 1
            snapshot = usage_delta(current, self._baseline, self._completed_turns, model)
            self._history.append(snapshot)
            self._baseline = current

    def record_compaction(self, model: str) -> None:
        """Record direct compaction usage without adding a trend snapshot."""

        current = self._counters()
        self._record_cost_delta(current, model)
        self._baseline = current

    def _record_cost_delta(self, current: Mapping[str, int], model: str) -> None:
        snapshot = usage_delta(current, self._cost_baseline, 0, model)
        self._cost_baseline = dict(current)
        if snapshot.cache_total or snapshot.output_tokens:
            if self.provider is not None:
                snapshot = replace(
                    snapshot,
                    estimated_cost_usd=_snapshot_cost(self.provider, snapshot),
                )
            self._accumulate_cost(snapshot)

    def _accumulate_cost(self, snapshot: UsageSnapshot) -> None:
        if snapshot.model is None:
            return
        previous = self._cost_by_model.get(snapshot.model)
        if previous is None:
            self._cost_by_model[snapshot.model] = snapshot
            return
        self._cost_by_model[snapshot.model] = UsageSnapshot(
            turn=previous.turn,
            input_tokens=previous.input_tokens + snapshot.input_tokens,
            output_tokens=previous.output_tokens + snapshot.output_tokens,
            cache_read_input_tokens=(
                previous.cache_read_input_tokens + snapshot.cache_read_input_tokens
            ),
            cache_creation_input_tokens=(
                previous.cache_creation_input_tokens
                + snapshot.cache_creation_input_tokens
            ),
            model=snapshot.model,
            estimated_cost_usd=(
                previous.estimated_cost_usd + snapshot.estimated_cost_usd
                if previous.estimated_cost_usd is not None
                and snapshot.estimated_cost_usd is not None
                else None
            ),
        )

    def _counters(self) -> dict[str, int]:
        return {
            "input_tokens": self.source.uncached_input_tokens_this_session,
            "output_tokens": self.source.output_tokens_this_session,
            "cache_read_input_tokens": self.source.cache_read_input_tokens_this_session,
            "cache_creation_input_tokens": self.source.cache_creation_input_tokens_this_session,
        }


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
    cache_write: float | None


# Unknown models remain valid and show no cost estimate. Prices are standard
# USD per million tokens.
UNPRICED_MODEL_IDS: dict[str, frozenset[str]] = {
    "claude": frozenset(),
    "codex": frozenset({"gpt-5.3-codex-spark", "gpt-reserve"}),
}

MODEL_PRICES: dict[str, dict[str, ModelPricing | None]] = {
    "claude": {
        "claude-fable-5": ModelPricing(10.0, 50.0, 1.0, 12.5),
        "claude-haiku-4-5-20251001": ModelPricing(1.0, 5.0, 0.1, 1.25),
        "claude-opus-4-6": ModelPricing(5.0, 25.0, 0.5, 6.25),
        "claude-opus-4-5": ModelPricing(5.0, 25.0, 0.5, 6.25),
        "claude-opus-4-7": ModelPricing(5.0, 25.0, 0.5, 6.25),
        "claude-opus-4-8": ModelPricing(5.0, 25.0, 0.5, 6.25),
        "claude-opus-5": ModelPricing(5.0, 25.0, 0.5, 6.25),
        "claude-sonnet-4-5-20250929": ModelPricing(3.0, 15.0, 0.3, 3.75),
        "claude-sonnet-4-6": ModelPricing(3.0, 15.0, 0.3, 3.75),
        "claude-sonnet-5": ModelPricing(2.0, 10.0, 0.2, 2.5),
        "claude-haiku-4-5": ModelPricing(1.0, 5.0, 0.1, 1.25),
    },
    "codex": {
        "codex-auto-review": ModelPricing(2.5, 15.0, 0.25, 0.0),
        "gpt-5.3-codex-spark": None,
        "gpt-5.4-mini": ModelPricing(0.75, 4.5, 0.075, 0.0),
        "gpt-5.5": ModelPricing(5.0, 30.0, 0.5, 0.0),
        "gpt-5.6-sol": ModelPricing(4.0, 20.0, 0.4, 0.0),
        "gpt-5.6-terra": ModelPricing(2.0, 12.0, 0.2, 0.0),
        "gpt-5.6-luna": ModelPricing(0.2, 1.2, 0.02, 0.0),
        "gpt-5.4": ModelPricing(2.5, 15.0, 0.25, 0.0),
        "gpt-reserve": None,
    },
}

MODEL_CONTEXT_WINDOWS: dict[str, dict[str, int | None]] = {
    "claude": {
        "claude-fable-5": 1_000_000,
        "claude-haiku-4-5-20251001": 200_000,
        "claude-opus-4-5": 200_000,
        "claude-opus-4-6": 1_000_000,
        "claude-opus-4-7": 1_000_000,
        "claude-opus-4-8": 1_000_000,
        "claude-opus-5": 1_000_000,
        "claude-sonnet-4-5-20250929": 200_000,
        "claude-sonnet-4-6": 1_000_000,
        "claude-sonnet-5": 1_000_000,
        "claude-haiku-4-5": 200_000,
    },
    "codex": {
        "codex-auto-review": 1_050_000,
        "gpt-5.3-codex-spark": None,
        "gpt-5.4-mini": 400_000,
        "gpt-5.5": 1_050_000,
        "gpt-5.6-sol": 1_050_000,
        "gpt-5.6-terra": 1_050_000,
        "gpt-5.6-luna": 1_050_000,
        "gpt-5.4": 1_050_000,
        "gpt-reserve": None,
    },
}

# Used when a model has no published window: unrecognized names, and the
# entries above that are deliberately None.
DEFAULT_TOKEN_BUDGET = 200_000


def context_window(provider: str, model: str) -> int | None:
    """Return a model's published context window, or None when unknown."""

    return MODEL_CONTEXT_WINDOWS.get(provider, {}).get(model)


def budget_for_model(provider: str, model: str) -> int:
    """Return the compaction budget to use for one model."""

    window = context_window(provider, model)
    return DEFAULT_TOKEN_BUDGET if window is None else window


def resolve_session_budget(
    stored_budget: int,
    stored_pin: bool,
    provider: str,
    model: str,
    override: int | None,
) -> tuple[int, bool]:
    """Choose a session's budget and whether the choice is pinned.

    An explicit --token-budget pins the value so later model changes never
    overwrite it. Otherwise the budget tracks the model's window.
    """

    if override is not None and override > 0:
        return override, True
    if stored_pin:
        return stored_budget, True
    return budget_for_model(provider, model), False


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

    source_messages, entries_folded = _compaction_source(marker, entries)
    marker_message, summary = _compaction_replacements(marker)
    source_tokens = sum(token_counter(message) for message in source_messages)
    return CompactionSummary(
        turn=turn,
        entries_folded=entries_folded,
        tokens_saved=max(
            0,
            source_tokens
            - token_counter(marker_message)
            - token_counter(summary),
        ),
    )


def _compaction_source(
    marker: ConversationEntry,
    entries: Sequence[ConversationEntry],
) -> tuple[list[Message], int]:
    start = marker.data["source_seq_start"]
    end = marker.data["source_seq_end"]
    replaces = marker.data.get("replaces", [])
    by_id = {entry.id: entry for entry in entries}
    source_messages: list[Message] = []
    entries_folded = 0
    consumed: set[str] = set()
    replaced_ranges: list[tuple[int, int]] = []
    for entry_id in replaces:
        entry = by_id.get(entry_id)
        if entry is None:
            continue
        consumed.add(entry.id)
        if entry.type == "message":
            source_messages.append(Message.from_dict(entry.data["message"]))
            entries_folded += 1
        elif entry.type == "compaction":
            source_messages.extend(_compaction_replacements(entry))
            replaced_ranges.append(
                (entry.data["source_seq_start"], entry.data["source_seq_end"])
            )
    for entry in entries:
        if entry.type != "message" or entry.id in consumed:
            continue
        if not (start <= entry.seq <= end):
            continue
        if any(lo <= entry.seq <= hi for lo, hi in replaced_ranges):
            continue
        if not _is_folded_message(entry):
            continue
        source_messages.append(Message.from_dict(entry.data["message"]))
        entries_folded += 1
    return source_messages, entries_folded


def _compaction_replacements(
    marker: ConversationEntry,
) -> tuple[Message, Message]:
    start = marker.data["source_seq_start"]
    end = marker.data["source_seq_end"]
    metadata = {
        "source_seq_start": start,
        "source_seq_end": end,
    }
    marker_message = Message(
        MessageRole.COMPACTION,
        [TextContent(f"[compaction marker: entries {start}–{end}]")],
        metadata=metadata,
    )
    summary = Message(
        MessageRole.ASSISTANT,
        [TextContent(marker.data["summary"])],
        metadata={"compaction_summary": True, **metadata},
    )
    return marker_message, summary


def _is_folded_message(entry: ConversationEntry) -> bool:
    """Return whether a message is source content, not a compaction replacement."""

    message = Message.from_dict(entry.data["message"])
    return message.role is not MessageRole.COMPACTION and not message.metadata.get(
        "compaction_summary"
    )


def usage_delta(
    current: Mapping[str, int],
    previous: Mapping[str, int],
    turn: int,
    model: str,
) -> UsageSnapshot:
    """Build one turn snapshot from cumulative counter deltas."""

    def delta(name: str) -> int:
        return max(0, current.get(name, 0) - previous.get(name, 0))

    return UsageSnapshot(
        turn=turn,
        model=model,
        input_tokens=delta("input_tokens"),
        output_tokens=delta("output_tokens"),
        cache_read_input_tokens=delta("cache_read_input_tokens"),
        cache_creation_input_tokens=delta("cache_creation_input_tokens"),
    )


LONG_CONTEXT_INPUT_THRESHOLD = 272_000
LONG_CONTEXT_MODELS = frozenset(
    {"codex-auto-review", "gpt-5.4", "gpt-5.5", "gpt-5.6-luna", "gpt-5.6-sol", "gpt-5.6-terra"}
)


def _snapshot_cost(provider: str, snapshot: UsageSnapshot) -> float | None:
    if snapshot.model is None:
        return None
    pricing = MODEL_PRICES.get(provider, {}).get(snapshot.model)
    if pricing is None:
        return None
    multiplier = (
        provider == "codex"
        and snapshot.model in LONG_CONTEXT_MODELS
        and snapshot.cache_total > LONG_CONTEXT_INPUT_THRESHOLD
    )
    input_multiplier = 2.0 if multiplier else 1.0
    output_multiplier = 1.5 if multiplier else 1.0
    return (
        snapshot.input_tokens * pricing.input * input_multiplier
        + snapshot.cache_read_input_tokens * pricing.cache_read * input_multiplier
        + snapshot.cache_creation_input_tokens * (pricing.cache_write or 0)
        + snapshot.output_tokens * pricing.output * output_multiplier
    ) / 1_000_000


def compaction_history(
    entries: Sequence[ConversationEntry],
    token_counter: Callable[[Message], int],
) -> tuple[CompactionSummary, ...]:
    """Convert durable compaction markers into ordered display data."""

    markers = [entry for entry in entries if entry.type == "compaction"]
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
        for marker in markers
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
    plan_mode: bool = False
    hooks: tuple[str, ...] = ()
    todo_counts: tuple[int, int, int, int] | None = None
    usage_history: tuple[UsageSnapshot, ...] = ()
    usage_cost_by_model: tuple[UsageSnapshot, ...] = ()
    compaction_history: tuple[CompactionSummary, ...] = ()
    model_window: int | None = None
    mcp_summary: str = "mcp: 0 mounted, 0 failed"


class SlashSession(Protocol):
    def slash_status(self) -> SlashStatus: ...

    async def slash_mcp(self, args: str) -> str: ...

    async def slash_mcp_prompt(
        self, name: str, arguments: dict[str, str]
    ) -> str: ...

    def slash_model(self, args: str) -> str: ...

    def slash_vim(self, args: str) -> str: ...

    def slash_plan(self, args: str) -> str | SlashModelInput: ...

    def slash_implement(self, args: str) -> str | SlashModelInput: ...

    def slash_paste(self, args: str) -> str: ...

    async def slash_compact(self) -> str: ...

    def slash_checkpoint(self, args: str) -> str: ...

    def slash_fork(self, args: str) -> str: ...

    async def slash_exec_macro(self, command: CustomCommand, args: str) -> str: ...


SlashResult = (
    str
    | SlashModelInput
    | SlashPromptError
    | Awaitable[str | SlashModelInput | SlashPromptError]
)
SlashHandler = Callable[[SlashSession, str], SlashResult]


@dataclass(frozen=True, slots=True)
class SlashCommand:
    """One named command in the input-time registry."""

    name: str
    handler: SlashHandler
    description: str = ""

    def run(self, session: SlashSession, args: str) -> SlashResult:
        return self.handler(session, args)


class SlashCommandRegistry:
    """Map registered command names to their handlers."""

    def __init__(self) -> None:
        self._commands: dict[str, SlashCommand] = {}
        self._custom_commands: dict[str, CustomCommand] = {}
        self._notices: list[str] = []
        self._warning_notices: set[str] = set()
        self._mcp_prompts = MCPPromptCommands(
            self._notices, self._warning_notices
        )

    def register(self, command: SlashCommand) -> None:
        if not command.name or any(character.isspace() for character in command.name):
            raise ValueError("slash command name must be one nonempty word")
        if command.name in self._commands:
            raise ValueError(f"slash command already registered: {command.name}")
        self._commands[command.name] = command

    @property
    def custom_commands(self) -> tuple[CustomCommand, ...]:
        """Return loaded custom commands in stable name order."""

        return tuple(self._custom_commands[name] for name in sorted(self._custom_commands))

    @property
    def notices(self) -> tuple[str, ...]:
        """Return non-fatal load notices for the session-start display."""

        return tuple(self._notices)

    @property
    def warning_notices(self) -> frozenset[str]:
        """Return notices that need prominent display."""

        return frozenset(self._warning_notices)

    @property
    def completion_entries(self) -> tuple[tuple[str, str, str], ...]:
        """Return command names, descriptions, and custom source badges."""

        builtins = tuple(
            (command.name, command.description, "")
            for command in self._commands.values()
        )
        custom = tuple(
            (command.name, command.description, command.source)
            for command in self.custom_commands
        )
        prompts = self._mcp_prompts.completion_entries()
        return builtins + custom + prompts

    def set_mcp_prompts(
        self, entries: Sequence[tuple[str, str, MCPPrompt]]
    ) -> None:
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

    def _dispatch(self, session: SlashSession, value: str) -> SlashResult | None:
        """Run a known command from the first line, or pass the input through."""

        first_line = value.split("\n", 1)[0]
        if not first_line.startswith("/") or first_line.startswith("//"):
            return None
        parts = first_line[1:].split(maxsplit=1)
        if not parts:
            return None
        command = self._commands.get(parts[0])
        if command is not None:
            return command.run(session, parts[1] if len(parts) == 2 else "")
        custom = self._custom_commands.get(parts[0])
        prompt = self._mcp_prompts.get(parts[0])
        if prompt is not None:
            return dispatch_prompt(
                session,
                parts[0],
                prompt[1],
                parts[1] if len(parts) == 2 else "",
            )
        if custom is None or custom.kind != "exec":
            return None
        return session.slash_exec_macro(custom, parts[1] if len(parts) == 2 else "")

    def dispatch(self, session: SlashSession, value: str) -> SlashResult | None:
        """Run a known command, returning an awaitable for async commands."""

        return self._dispatch(session, value)

    async def dispatch_async(
        self, session: SlashSession, value: str
    ) -> str | SlashModelInput | SlashPromptError | None:
        """Run a known command, awaiting it when the command is asynchronous."""

        result = self._dispatch(session, value)
        if result is None:
            return None
        if isinstance(result, (str, SlashModelInput, SlashPromptError)):
            return result
        return await result

    def input_for_model(self, value: str) -> str:
        """Expand a custom command or turn an escape into a literal slash."""
        return render_custom_input(value, self._custom_commands)

    async def resolve_for_model(
        self,
        value: str,
        inline_shell: InlineShellRunner,
    ) -> str | None:
        """Expand a prompt macro and resolve its inline shell spans."""

        return await resolve_custom_input(value, self._custom_commands, inline_shell)

    def needs_inline_shell_resolution(self, value: str) -> bool:
        """Return whether model resolution can wait for shell approval."""

        return needs_inline_shell_resolution(value, self._custom_commands)

    def exec_command_for(self, value: str) -> CustomCommand | None:
        """Return the custom execution command named by one input value."""

        first_line = value.split("\n", 1)[0]
        if not first_line.startswith("/") or first_line.startswith("//"):
            return None
        parts = first_line[1:].split(maxsplit=1)
        if not parts:
            return None
        name = parts[0]
        command = self._custom_commands.get(name)
        return command if command is not None and command.kind == "exec" else None

    def help_text(self) -> str:
        """Format the built-in and loaded custom commands for /help."""

        lines = ["built-in commands:"]
        for command in self._commands.values():
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
        return "\n".join(lines)


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
    cost_text = _format_estimated_cost(status)
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
        f"plan_mode: {'on' if status.plan_mode else 'off'}",
        f"retained_tail: {status.retained_tail}",
        f"tokens_used_this_session: {status.tokens_used_this_session}",
        f"tokens_in_current_context: {context_tokens}",
        f"compaction_marker_count: {status.compaction_marker_count}",
        f"checkpoint_count: {status.checkpoint_count}",
        f"live_pending_approvals: {len(status.pending_approvals)} ({pending})",
        status.mcp_summary,
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
        "transcript_navigation: ctrl+f find, ctrl+up/down users, pageup/pagedown scroll",
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
        pending, in_progress, completed, canceled = status.todo_counts
        lines.append(
            "todo: "
            f"pending={pending}, in_progress={in_progress}, completed={completed}, "
            f"canceled={canceled}"
        )
    return "\n".join(lines)


def _format_estimated_cost(status: SlashStatus) -> str:
    """Estimate cost only when each turn has a known model and rate."""

    usage = status.usage_cost_by_model or status.usage_history
    if not usage:
        if MODEL_PRICES.get(status.provider, {}).get(status.model) is None:
            return f"unavailable (unknown model: {status.model})"
        return "unavailable (model attribution unavailable)"
    if status.usage_cost_by_model and all(
        snapshot.estimated_cost_usd is not None for snapshot in status.usage_cost_by_model
    ):
        total = sum(
            snapshot.estimated_cost_usd or 0.0
            for snapshot in status.usage_cost_by_model
        )
        return f"${total:.6f}"
    total = 0.0
    for snapshot in usage:
        if snapshot.model is None:
            return "unavailable (model attribution unavailable)"
        pricing = MODEL_PRICES.get(status.provider, {}).get(snapshot.model)
        if pricing is None:
            return f"unavailable (unknown model: {snapshot.model})"
        if snapshot.cache_creation_input_tokens and pricing.cache_write is None:
            return f"unavailable (cache-write price unavailable for model: {snapshot.model})"
        total += _snapshot_cost(status.provider, snapshot) or 0.0
    return f"${total:.6f}"


def _run_status(session: SlashSession, args: str) -> str:
    del args
    return _format_status(session.slash_status())


async def _run_mcp(session: SlashSession, args: str) -> str:
    return await session.slash_mcp(args.strip())


def _run_model(session: SlashSession, args: str) -> str:
    return session.slash_model(args.strip())


def _run_vim(session: SlashSession, args: str) -> str:
    return session.slash_vim(args.strip())


def _run_plan(session: SlashSession, args: str) -> str | SlashModelInput:
    return session.slash_plan(args.strip())


def _run_implement(session: SlashSession, args: str) -> str | SlashModelInput:
    return session.slash_implement(args.strip())


async def _run_compact(session: SlashSession, args: str) -> str:
    del args
    return await session.slash_compact()


def _run_paste(session: SlashSession, args: str) -> str:
    return session.slash_paste(args)


def _run_checkpoint(session: SlashSession, args: str) -> str:
    return session.slash_checkpoint(args)


def _run_fork(session: SlashSession, args: str) -> str:
    return session.slash_fork(args)


def create_slash_registry(
    *,
    zeta_home: str | Path | None = None,
    project_dir: str | Path | None = None,
) -> SlashCommandRegistry:
    """Create the built-in registry."""

    registry = SlashCommandRegistry()
    registry.register(SlashCommand("status", _run_status, "show session status"))
    registry.register(SlashCommand("mcp", _run_mcp, "show MCP server status"))
    registry.register(SlashCommand("model", _run_model, "show or change the model"))
    registry.register(SlashCommand("vim", _run_vim, "show or change vim mode"))
    registry.register(
        SlashCommand("plan", _run_plan, "show or change plan mode")
    )
    registry.register(
        SlashCommand("implement", _run_implement, "implement the proposed plan")
    )
    registry.register(SlashCommand("paste", _run_paste, "paste an image"))
    registry.register(SlashCommand("compact", _run_compact, "compact the context"))
    registry.register(SlashCommand("checkpoint", _run_checkpoint, "save a checkpoint"))
    registry.register(SlashCommand("fork", _run_fork, "fork from a checkpoint"))
    registry.register(
        SlashCommand("help", lambda _session, _args: registry.help_text(), "list commands")
    )
    result = load_custom_commands(
        home=zeta_home,
        project_dir=project_dir or Path.cwd(),
    )
    registry._notices.extend(result.notices)
    for command in result.commands:
        registry.register_custom(command)
    return registry
