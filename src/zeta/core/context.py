"""Context assembly and durable conversation compaction."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Any

from ..compaction import (
    CompactionPolicy,
    SUMMARY_SOURCE_TOKEN_LIMIT,
    SummaryCompletionError as _SummaryCompletionError,
    SummaryInputTooLarge as _SummaryInputTooLarge,
)
from ..context_strategies import ContextTelemetry, budget_readout, context_strategies
from ..context_strategies.adaptive import (
    committed_messages as build_committed_messages,
    compaction_source,
    has_only_compacted_prefix,
    latest_user_index,
    message_digest,
    message_token_count,
    shrink_tail_boundary,
    strategy_records as build_strategy_records,
    truncate_tool_results,
)
from ..context_strategies.archive import context_blocks
from ..context_strategies.decisions import apply_persisted_decisions
from ..context_strategies.eviction_runtime import apply_eviction
from ..context_strategies.fold import fold_messages
from ..context_strategies.nudge import build_nudges
from ..protocol.types import (
    CompletionBackend,
    FAILED_TURN_MARKER,
    Message,
    MessageRole,
    StreamEvent,
    TextContent,
    ToolUseContent,
)
from .store import ConversationEntry, ConversationStore


SummaryCompletionError = _SummaryCompletionError
SummaryInputTooLarge = _SummaryInputTooLarge


class BudgetExceeded(RuntimeError):
    """The committed context cannot fit in the configured budget."""


class StaleBranchError(RuntimeError):
    """The active branch changed while a summary was in flight."""


@dataclass(frozen=True, slots=True)
class AssembledContext:
    messages: list[Message]
    token_count: int
    digest: str
    compacted: bool = False


@dataclass(frozen=True, slots=True)
class _ContextItem:
    entry: ConversationEntry | None
    message: Message
    fixed: bool = False


def _text_from_message(message: Message) -> str:
    return "".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


class ContextAssembler:
    """Build provider messages from the active branch and compact old entries."""

    def __init__(
        self,
        store: ConversationStore,
        *,
        token_budget: int = 200_000,
        retained_tail: int = 8,
        system_prompt: str | Message = "",
        backend: CompletionBackend | None = None,
        compaction_policy: CompactionPolicy | None = None,
        token_counter: Callable[[Message], int] | None = None,
        on_completion_success: Callable[[], None] | None = None,
        usage_sink: Callable[[Mapping[str, Any]], None] | None = None,
        telemetry_sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token budget must be positive")
        if retained_tail < 1:
            raise ValueError("retained tail must be at least one")
        self.store = store
        self.token_budget = token_budget
        self.retained_tail = retained_tail
        self.backend = backend
        self.compaction_policy = compaction_policy or CompactionPolicy(backend)
        self.token_counter = token_counter or message_token_count
        self.on_completion_success = on_completion_success
        self.usage_sink = usage_sink
        self.telemetry_sink = telemetry_sink
        self.strategies = context_strategies()
        self._experiment_telemetry = ContextTelemetry()
        self._nudge_thresholds: set[int] = set()
        self.last_compaction_telemetry: dict[str, Any] = {}
        self._descendant_usage = {
            "input_tokens": 0,
            "output_tokens": 0,
            "cache_read_input_tokens": 0,
            "cache_creation_input_tokens": 0,
        }
        self._descendant_usage_by_model: dict[str, dict[str, int]] = {}
        self.system_prompt = (
            system_prompt
            if isinstance(system_prompt, Message)
            else Message(MessageRole.SYSTEM, [TextContent(system_prompt)])
        )
        self.last_context: AssembledContext | None = None
        self.last_usage: dict[str, Any] = {}
        self._provider_token_total: int | None = None
        self._tokens_used_this_session = 0
        self._cache_read_input_tokens_this_session = 0
        self._cache_creation_input_tokens_this_session = 0
        self._uncached_input_tokens_this_session = 0
        self._output_tokens_this_session = 0

    @property
    def digest(self) -> str | None:
        return self.last_context.digest if self.last_context is not None else None

    def needs_compaction(self, *, force: bool = False) -> bool:
        """Return whether the next assembly must compact the active branch."""

        if force:
            return True

        branch = self.store.replay()
        items = self._visible_items(branch)
        system_prompt = self._system_prompt_message()
        messages = [
            *([] if system_prompt is None else [system_prompt]),
            *(item.message for item in items),
        ]
        return self.compaction_policy.should_compact(
            self._total_tokens(messages), self.token_budget
        )

    @property
    def token_count(self) -> int | None:
        return self.last_context.token_count if self.last_context is not None else None

    @property
    def tokens_used_this_session(self) -> int:
        return self._tokens_used_this_session

    @property
    def cache_read_input_tokens_this_session(self) -> int:
        return self._cache_read_input_tokens_this_session

    @property
    def cache_creation_input_tokens_this_session(self) -> int:
        return self._cache_creation_input_tokens_this_session

    @property
    def uncached_input_tokens_this_session(self) -> int:
        return self._uncached_input_tokens_this_session

    @property
    def output_tokens_this_session(self) -> int:
        return self._output_tokens_this_session

    @property
    def descendant_usage(self) -> dict[str, int]:
        return dict(self._descendant_usage)

    @property
    def descendant_usage_by_model(self) -> dict[str, dict[str, int]]:
        return {
            model: dict(counts)
            for model, counts in self._descendant_usage_by_model.items()
        }

    def record_descendant_usage(self, usage: Mapping[str, Any]) -> None:
        model = usage.get("_zeta_model")
        if type(model) is not str or not model:
            model = "unknown"
        model_usage = self._descendant_usage_by_model.setdefault(
            model, dict.fromkeys(self._descendant_usage, 0)
        )
        for key, fallback in (
            ("input_tokens", "prompt_tokens"),
            ("output_tokens", "completion_tokens"),
            ("cache_read_input_tokens", "cache_read_input_tokens"),
            ("cache_creation_input_tokens", "cache_creation_input_tokens"),
        ):
            value = usage.get(key, usage.get(fallback))
            if type(value) is int and value >= 0:
                self._descendant_usage[key] += value
                model_usage[key] += value
        if self.usage_sink is not None:
            self.usage_sink(usage)

    def _record_compaction_telemetry(self, telemetry: Mapping[str, Any]) -> None:
        self.last_compaction_telemetry = dict(telemetry)
        if self.telemetry_sink is not None:
            self.telemetry_sink(telemetry)

    def record_usage(self, usage: Mapping[str, Any]) -> None:
        self.last_usage = dict(usage)
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        cache_read_tokens = usage.get("cache_read_input_tokens")
        cache_creation_tokens = usage.get("cache_creation_input_tokens")
        if type(input_tokens) is int and input_tokens >= 0:
            self._uncached_input_tokens_this_session += input_tokens
        if type(output_tokens) is int and output_tokens >= 0:
            self._output_tokens_this_session += output_tokens
        if type(cache_read_tokens) is int and cache_read_tokens >= 0:
            self._cache_read_input_tokens_this_session += cache_read_tokens
        if type(cache_creation_tokens) is int and cache_creation_tokens >= 0:
            self._cache_creation_input_tokens_this_session += cache_creation_tokens
        total = usage.get("total_tokens")
        if type(total) is not int:
            known_tokens = [
                value
                for value in (
                    input_tokens,
                    output_tokens,
                    cache_read_tokens,
                    cache_creation_tokens,
                )
                if type(value) is int and value >= 0
            ]
            if known_tokens:
                total = sum(known_tokens)
        if type(total) is int and total >= 0:
            self._provider_token_total = total
            self._tokens_used_this_session += total
        if self.usage_sink is not None:
            self.usage_sink(usage)

    def observe_event(self, event: StreamEvent) -> None:
        usage = event.data.get("usage")
        if isinstance(usage, Mapping):
            self.record_usage(usage)

    async def assemble(
        self,
        *,
        backend: CompletionBackend | None = None,
        force: bool = False,
    ) -> list[Message]:
        context = await self.assemble_context(backend=backend, force=force)
        return list(context.messages)

    async def assemble_context(
        self,
        *,
        backend: CompletionBackend | None = None,
        force: bool = False,
    ) -> AssembledContext:
        branch = self.store.replay()
        branch_id = self._branch_id(branch)
        items = self._visible_items(branch)
        result_seqs = {
            id(item.message): item.entry.seq
            for item in items
            if item.entry is not None and item.message.tool_result is not None
        }
        boundary = self._tail_boundary(items)
        system_prompt = self._system_prompt_message()
        system_messages = [] if system_prompt is None else [system_prompt]
        latest_user = latest_user_index(items)
        committed_messages = build_committed_messages(
            items, boundary, system_messages, latest_user
        )
        committed_tokens = self._count(committed_messages)
        all_messages = [*system_messages, *(item.message for item in items)]
        total_tokens = self._total_tokens(all_messages)
        should_compact = force or self.compaction_policy.should_compact(
            total_tokens, self.token_budget
        )
        if not should_compact:
            return self._save(all_messages, False, items=items)

        adaptive_tail = committed_tokens > self.token_budget
        if adaptive_tail:
            boundary = shrink_tail_boundary(
                items,
                boundary,
                system_messages,
                self.token_budget,
                latest_user,
                self.token_counter,
            )
            committed_messages = build_committed_messages(
                items, boundary, system_messages, latest_user
            )
            committed_tokens = self._count(committed_messages)
        pinned_user = (
            items[latest_user]
            if latest_user is not None and latest_user < boundary
            else None
        )
        strategy_fixed = [
            item
            for item in items
            if item.message.metadata.get("context_strategy_fixed")
        ]
        if adaptive_tail and has_only_compacted_prefix(
            items, boundary, latest_user
        ):
            truncated = truncate_tool_results(
                committed_messages,
                self.token_budget,
                result_seqs,
                self.token_counter,
            )
            if truncated is not None:
                return self._save(truncated, False)
        candidates = [
            item
            for index, item in enumerate(items[:boundary])
            if not item.message.metadata.get("context_strategy_fixed")
            and index != latest_user
        ]
        evicted = await apply_eviction(
            self,
            items,
            system_messages,
            latest_user,
            branch_id,
            backend,
            max_source_tokens=min(SUMMARY_SOURCE_TOKEN_LIMIT, self.token_budget),
            stale_error=StaleBranchError,
        )
        if evicted is not None:
            return evicted
        if not candidates and adaptive_tail:
            truncated = truncate_tool_results(
                committed_messages,
                self.token_budget,
                result_seqs,
                self.token_counter,
            )
            if truncated is not None:
                return self._save(truncated, False)
        if not candidates and not (force and pinned_user is not None):
            if force:
                return self._save(all_messages, False, items=items)
            if committed_tokens > self.token_budget:
                self._raise_committed_budget(committed_tokens)
            raise BudgetExceeded("context exceeds budget and has no compactible range")
        # A forced compaction with only a pinned user summarizes an empty source.
        source_start, source_end, replaces = compaction_source(candidates, pinned_user)
        strategy_records = build_strategy_records(candidates, source_start)
        summary_messages = [item.message for item in candidates]
        if "fold" in self.strategies:
            folding = fold_messages(strategy_records, token_counter=self.token_counter)
            summary_messages = folding.messages
            self._experiment_telemetry.emit(
                "context_strategy",
                kind="fold",
                range=[source_start, source_end],
                items_folded=folding.items_folded,
                items_evicted=0,
                tokens_before=folding.tokens_before,
                tokens_after=folding.tokens_after,
            )
        compaction_started = perf_counter()
        self.last_compaction_telemetry = {}
        # Use the final request shape to limit a deterministic fallback. This
        # keeps fallback text inside the same budget as a model summary.
        max_source_tokens = min(
            SUMMARY_SOURCE_TOKEN_LIMIT,
            max(1, self.token_budget // 2, self.token_budget - 8_000),
        )
        retained_messages = [
            *(item.message for item in strategy_fixed),
            *([] if pinned_user is None else [pinned_user.message]),
            *(item.message for item in items[boundary:]),
        ]
        max_fallback_chars = self._summary_capacity(
            source_start,
            source_end,
            retained_messages,
            prefix_messages=system_messages,
            max_chars=max_source_tokens * 4,
        )
        summary = await self.compaction_policy.summarize_chunked(
            summary_messages,
            backend=backend or self.backend,
            system_prompt=system_prompt,
            max_source_tokens=max_source_tokens,
            max_fallback_chars=max_fallback_chars,
            on_success=self.on_completion_success,
            on_usage=self.record_usage,
            on_telemetry=self._record_compaction_telemetry,
        )
        if self._branch_id(self.store.replay()) != branch_id:
            raise StaleBranchError("active branch changed during compaction")
        marker_messages = self._marker_messages(source_start, source_end, summary)
        # Adaptive shrinking pins the latest user after the summary and before
        # the minimal valid tool tail. It is excluded from summary input.
        proposed_messages = [
            *system_messages,
            *marker_messages,
            *(item.message for item in strategy_fixed),
            *([] if pinned_user is None else [pinned_user.message]),
            *(item.message for item in items[boundary:]),
        ]
        proposed_items = [
            *(_ContextItem(None, message, fixed=True) for message in marker_messages),
            *strategy_fixed,
            *([] if pinned_user is None else [pinned_user]),
            *items[boundary:],
        ]
        # A successful summary starts the next reminder cycle. If the compacted
        # request is still large, its threshold belongs to that new cycle.
        self._nudge_thresholds.clear()
        proposed = self._context(
            self._with_strategy_tail(
                proposed_messages,
                proposed_items,
                compactions=self.store.compaction_marker_count() + 1,
            ),
            True,
        )
        if proposed.token_count > self.token_budget:
            truncated = truncate_tool_results(
                proposed_messages,
                self.token_budget,
                result_seqs,
                self.token_counter,
            )
            if truncated is not None:
                proposed = self._context(truncated, True)
        if proposed.token_count > self.token_budget:
            if committed_tokens > self.token_budget:
                self._raise_committed_budget(committed_tokens)
            raise BudgetExceeded(
                f"compacted context ({proposed.token_count} tokens) exceeds "
                f"the token budget ({self.token_budget})"
            )

        try:
            self.store.append_compaction_marker(
                summary,
                source_start,
                source_end,
                replaces=replaces,
                pinned_message=(None if pinned_user is None else pinned_user.message),
                expected_parent_id=branch_id,
            )
        except ValueError as exc:
            raise StaleBranchError("active branch changed during compaction") from exc
        chunk_count = self.last_compaction_telemetry.get("chunk_count", 1)
        self._experiment_telemetry.emit(
            "compaction",
            source_seq_start=source_start,
            source_seq_end=source_end,
            summary_chars=len(summary),
            duration_s=perf_counter() - compaction_started,
            map_calls=(
                chunk_count if type(chunk_count) is int and chunk_count > 1 else 0
            ),
        )
        self._provider_token_total = None
        self.last_context = proposed
        self._emit_request_telemetry(proposed)
        return proposed

    def _summary_capacity(
        self,
        source_start: int,
        source_end: int,
        retained_messages: Sequence[Message],
        *,
        prefix_messages: Sequence[Message] = (),
        max_chars: int,
    ) -> int:
        """Return the largest summary that fits with retained messages."""

        low, high = 0, max_chars
        while low < high:
            midpoint = (low + high + 1) // 2
            bounded_messages = [
                *prefix_messages,
                *self._marker_messages(source_start, source_end, "x" * midpoint),
                *retained_messages,
            ]
            if self._count(bounded_messages) <= self.token_budget:
                low = midpoint
            else:
                high = midpoint - 1
        return low

    def _raise_committed_budget(self, committed_tokens: int) -> None:
        raise BudgetExceeded(
            "system prompt and retained tail "
            f"({committed_tokens} tokens) exceed the token budget "
            f"({self.token_budget}); raise it with --token-budget"
        )

    def _context(self, messages: list[Message], compacted: bool) -> AssembledContext:
        return AssembledContext(
            messages=messages,
            token_count=self._count(messages),
            digest=message_digest(messages),
            compacted=compacted,
        )

    def _save(
        self,
        messages: list[Message],
        compacted: bool,
        *,
        items: Sequence[_ContextItem] = (),
    ) -> AssembledContext:
        context = self._context(self._with_strategy_tail(messages, items), compacted)
        self.last_context = context
        self._emit_request_telemetry(context)
        return context

    def _with_strategy_tail(
        self,
        messages: list[Message],
        items: Sequence[_ContextItem],
        *,
        compactions: int | None = None,
    ) -> list[Message]:
        tailed = self._with_budget_readout(messages, items, compactions=compactions)
        nudges = (
            build_nudges(
                estimated_tokens=self._count(messages),
                budget=self.token_budget,
                strategies=self.strategies,
                emitted=self._nudge_thresholds,
            )
            if "nudge" in self.strategies
            else []
        )
        if not nudges:
            return tailed

        budget_tail = tailed[-1] if len(tailed) > len(messages) else None
        additions = [*([] if budget_tail is None else [budget_tail]), *nudges]
        text = "\n".join(_text_from_message(message) for message in additions)
        thresholds = [message.metadata["context_nudge"] for message in nudges]
        metadata: dict[str, Any] = {
            "context_nudge": max(thresholds),
            "context_nudges": thresholds,
        }
        if budget_tail is not None:
            metadata["context_budget_readout"] = True
        return [
            *messages,
            Message(MessageRole.USER, [TextContent(text)], metadata=metadata),
        ]

    def _with_budget_readout(
        self,
        messages: list[Message],
        items: Sequence[_ContextItem],
        *,
        compactions: int | None = None,
    ) -> list[Message]:
        if "budget" not in self.strategies:
            return messages
        rows: list[tuple[str, str, int]] = []
        item_message_ids = {id(item.message) for item in items}
        visible_items = [
            *(
                _ContextItem(None, message, fixed=True)
                for message in messages
                if id(message) not in item_message_ids
            ),
            *items,
        ]
        for item in visible_items:
            message = item.message
            entry = item.entry
            source_start = message.metadata.get("source_seq_start")
            source_end = message.metadata.get("source_seq_end")
            if type(source_start) is int and type(source_end) is int:
                seq = f"{source_start}–{source_end}"
            else:
                seq = "?" if entry is None else str(entry.seq)
            tool_names = [
                block.tool_call.name
                for block in message.content
                if isinstance(block, ToolUseContent)
            ]
            if message.tool_result is not None:
                label = f"tool:{message.tool_result.tool_call_id}"
            elif tool_names:
                label = f"assistant:{','.join(tool_names)}"
            else:
                label = message.role.value
            rows.append((seq, label, self.token_counter(message)))
        readout = budget_readout(
            rows,
            estimated_tokens=self._count(messages),
            budget=self.token_budget,
            compactions=(
                self.store.compaction_marker_count()
                if compactions is None
                else compactions
            ),
        )
        return [
            *messages,
            Message(
                MessageRole.USER,
                [TextContent(readout)],
                metadata={"context_budget_readout": True},
            ),
        ]

    def _emit_request_telemetry(self, context: AssembledContext) -> None:
        self._experiment_telemetry.emit(
            "request",
            strategy=",".join(sorted(self.strategies)),
            est_tokens=context.token_count,
            budget=self.token_budget,
            compaction=context.compacted,
            budget_readout=any(
                message.metadata.get("context_budget_readout")
                for message in context.messages
            ),
        )

    def _count(self, messages: Sequence[Message]) -> int:
        return sum(
            self.token_counter(message)
            for message in messages
            if not message.metadata.get("context_budget_readout")
            and not message.metadata.get("context_nudge")
        )

    def _system_prompt_message(self) -> Message | None:
        if not _text_from_message(self.system_prompt).strip():
            return None
        return self.system_prompt

    def _total_tokens(self, messages: Sequence[Message]) -> int:
        estimated = self._count(messages)
        if self._provider_token_total is None:
            return estimated
        return max(estimated, self._provider_token_total)

    def _visible_items(
        self, entries: Sequence[ConversationEntry]
    ) -> list[_ContextItem]:
        items: list[_ContextItem] = []
        failed_tool_call_ids: set[str] = set()
        for entry in entries:
            if entry.type != "message":
                continue
            message = Message.from_dict(entry.data["message"])
            if message.metadata.get(FAILED_TURN_MARKER):
                failed_tool_call_ids.update(
                    block.tool_call.id
                    for block in message.content
                    if isinstance(block, ToolUseContent)
                )
                continue
            if (
                message.role is MessageRole.TOOL_RESULT
                and message.tool_result is not None
                and message.tool_result.tool_call_id in failed_tool_call_ids
            ):
                continue
            items.append(_ContextItem(entry, message))

        blocks = apply_persisted_decisions(
            entries,
            context_blocks(items),
            strategies=self.strategies,
            render_compaction=lambda marker: context_blocks(self._marker_items(marker)),
        )
        return [
            _ContextItem(block.entry, block.message, fixed=block.fixed)
            for block in blocks
        ]

    def _marker_items(self, entry: ConversationEntry) -> list[_ContextItem]:
        messages = self._marker_messages(
            entry.data["source_seq_start"],
            entry.data["source_seq_end"],
            entry.data["summary"],
        )
        items = [_ContextItem(entry, message, fixed=True) for message in messages]
        pinned = entry.data.get("pinned_message")
        if pinned is not None:
            items.append(_ContextItem(entry, Message.from_dict(pinned)))
        return items

    def _marker_messages(
        self,
        source_start: int,
        source_end: int,
        summary: str,
    ) -> list[Message]:
        marker_text = f"[compaction marker: entries {source_start}–{source_end}]"
        if "recall" in self.strategies:
            summary = (
                f"[compacted history seq {source_start}–{source_end}; "
                "use recall_history to retrieve exact messages]\n"
                f"{summary}"
            )
        metadata = {
            "source_seq_start": source_start,
            "source_seq_end": source_end,
        }
        return [
            Message(
                MessageRole.COMPACTION,
                [TextContent(marker_text)],
                metadata=metadata,
            ),
            Message(
                MessageRole.ASSISTANT,
                [TextContent(summary)],
                metadata={"compaction_summary": True, **metadata},
            ),
        ]

    @staticmethod
    def _branch_id(entries: Sequence[ConversationEntry]) -> str | None:
        return entries[-1].id if entries else None

    def _tail_boundary(self, items: Sequence[_ContextItem]) -> int:
        tail_start = max(0, len(items) - self.retained_tail)
        while tail_start > 0 and self._is_tool_result(items[tail_start]):
            result = items[tail_start].message.tool_result
            if result is None:
                break
            call_id = result.tool_call_id
            call_index = self._find_tool_call(items, call_id, tail_start)
            if call_index is None:
                break
            tail_start = call_index
        return tail_start

    @staticmethod
    def _is_tool_result(item: _ContextItem) -> bool:
        return (
            item.message.role is MessageRole.TOOL_RESULT
            and item.message.tool_result is not None
        )

    @staticmethod
    def _find_tool_call(
        items: Sequence[_ContextItem], call_id: str, before: int
    ) -> int | None:
        for index in range(before - 1, -1, -1):
            message = items[index].message
            if any(
                isinstance(block, ToolUseContent) and block.tool_call.id == call_id
                for block in message.content
            ):
                return index
        return None
