"""Context assembly and durable conversation compaction."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from typing import Any

from .store import ConversationEntry, ConversationStore
from ..context_eviction import (
    EVICTION_KIND,
    HYSTERESIS_RATIO,
    TARGET_RATIO,
    evict_messages,
    eviction_view,
)
from ..compaction import (
    CompactionPolicy,
    SUMMARY_SOURCE_TOKEN_LIMIT,
    SummaryCompletionError as _SummaryCompletionError,
    SummaryInputTooLarge as _SummaryInputTooLarge,
)
from ..protocol.types import (
    CompletionBackend,
    FAILED_TURN_MARKER,
    flatten_tool_content,
    Message,
    MessageRole,
    StreamEvent,
    TextContent,
    ToolResult,
    ToolUseContent,
)


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


IMAGE_TOKEN_ESTIMATE = 1024


def _message_token_count(message: Message) -> int:
    """Estimate text tokens and charge a small fixed amount per image.

    Base64 is transport data, not text. Without image dimensions, use a fixed
    estimate that keeps images near the 4 MiB transport cap usable.
    """

    value = message.to_dict()
    image_count = 0
    content = value.get("content")
    if isinstance(content, list):
        for index, block in enumerate(content):
            if isinstance(block, dict) and block.get("type") == "image":
                content[index] = {
                    key: item for key, item in block.items() if key != "data"
                }
                image_count += 1
    tool_result = value.get("tool_result")
    if isinstance(tool_result, dict):
        blocks = tool_result.get("content_blocks")
        if isinstance(blocks, list):
            tool_result["content_blocks"] = [
                {key: item for key, item in block.items() if key != "data"}
                if isinstance(block, dict) and block.get("type") == "image"
                else block
                for block in blocks
            ]
            image_count += sum(
                isinstance(block, dict) and block.get("type") == "image"
                for block in blocks
            )
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return max(1, ceil(len(encoded) / 4) + image_count * IMAGE_TOKEN_ESTIMATE)


def _digest(messages: Sequence[Message]) -> str:
    encoded = json.dumps(
        [message.to_dict() for message in messages],
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def _text_from_message(message: Message) -> str:
    parts: list[str] = []
    for block in message.content:
        if isinstance(block, TextContent):
            parts.append(block.text)
    return "".join(parts)




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
        compaction: str = "summary",
    ) -> None:
        if token_budget <= 0:
            raise ValueError("token budget must be positive")
        if retained_tail < 1:
            raise ValueError("retained tail must be at least one")
        if compaction not in {"summary", "evict"}:
            raise ValueError("compaction must be 'summary' or 'evict'")
        self.store = store
        self.compaction = compaction
        self.token_budget = token_budget
        self.retained_tail = retained_tail
        self.backend = backend
        self.compaction_policy = compaction_policy or CompactionPolicy(backend)
        self.token_counter = token_counter or _message_token_count
        self.on_completion_success = on_completion_success
        self.usage_sink = usage_sink
        self.telemetry_sink = telemetry_sink
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
        latest_user = self._latest_user_index(items)
        committed_messages = self._committed_messages(
            items, boundary, system_messages, latest_user
        )
        committed_tokens = self._count(committed_messages)
        all_messages = [*system_messages, *(item.message for item in items)]
        total_tokens = self._total_tokens(all_messages)
        should_compact = force or self.compaction_policy.should_compact(
            total_tokens, self.token_budget
        )
        if not should_compact:
            return self._save(
                all_messages,
                False,
            )
        adaptive_tail = committed_tokens > self.token_budget
        if adaptive_tail:
            boundary = self._shrink_tail_boundary(
                items,
                boundary,
                system_messages,
                self.token_budget,
                latest_user,
            )
            committed_messages = self._committed_messages(
                items, boundary, system_messages, latest_user
            )
            committed_tokens = self._count(committed_messages)

        pinned_user = (
            items[latest_user]
            if latest_user is not None and latest_user < boundary
            else None
        )
        prefix_has_uncompacted_items = any(
            not item.fixed and index != latest_user
            for index, item in enumerate(items[:boundary])
        )
        if (
            adaptive_tail
            and not prefix_has_uncompacted_items
            and any(item.fixed for item in items[:boundary])
        ):
            truncated = self._truncate_tool_results(
                committed_messages,
                self.token_budget,
                result_seqs,
            )
            if truncated is not None:
                return self._save(truncated, False)
        candidates = [
            item for index, item in enumerate(items[:boundary]) if index != latest_user
        ]
        if self.compaction == "evict":
            evicted = self._evict_context(
                branch=branch,
                branch_id=branch_id,
                items=items,
                latest_user=latest_user,
                system_messages=system_messages,
            )
            if evicted is not None:
                return evicted
        if not candidates and adaptive_tail:
            truncated = self._truncate_tool_results(
                committed_messages,
                self.token_budget,
                result_seqs,
            )
            if truncated is not None:
                return self._save(truncated, False)
        if not candidates and not (force and pinned_user is not None):
            if force:
                return self._save(all_messages, False)
            if committed_tokens > self.token_budget:
                self._raise_committed_budget(committed_tokens)
            raise BudgetExceeded("context exceeds budget and has no compactible range")
        # Manual compaction remains explicit even when pinning leaves no
        # eligible source. Summarizing an empty range preserves completion
        # errors and usage without duplicating the pinned user in prompt/replay.
        source_entries = {
            item.entry.id: item.entry
            for item in (*candidates, pinned_user)
            if item is not None and item.entry is not None
        }
        source_ranges = [
            (
                entry.data["source_seq_start"],
                entry.data["source_seq_end"],
            )
            if entry.type == "compaction"
            else (entry.seq, entry.seq)
            for entry in source_entries.values()
        ]
        source_start = min(start for start, _ in source_ranges)
        source_end = max(end for _, end in source_ranges)
        replaces = [
            entry.id for entry in source_entries.values() if entry.type == "compaction"
        ]
        # Bound fallback characters with the same message accounting as the
        # final check. A zero bound is valid when no fallback prefix can fit;
        # the final check can then raise BudgetExceeded.
        max_source_tokens = min(
            SUMMARY_SOURCE_TOKEN_LIMIT,
            max(1, self.token_budget // 2, self.token_budget - 8_000),
        )
        low, high = 0, max_source_tokens * 4
        while low < high:
            midpoint = (low + high + 1) // 2
            bounded_messages = [
                *system_messages,
                *self._marker_messages(source_start, source_end, "x" * midpoint),
                *([] if pinned_user is None else [pinned_user.message]),
                *(item.message for item in items[boundary:]),
            ]
            if self._count(bounded_messages) <= self.token_budget:
                low = midpoint
            else:
                high = midpoint - 1
        summary = await self.compaction_policy.summarize_chunked(
            [item.message for item in candidates],
            backend=backend or self.backend,
            system_prompt=system_prompt,
            max_fallback_chars=low,
            # Product backends can expose smaller windows than model cards.
            # Bound each request and summarize larger ranges in chunks.
            # Keep the budget-relative bound for small configured budgets.
            max_source_tokens=max_source_tokens,
            on_success=self.on_completion_success,
            on_usage=self.record_usage,
            on_telemetry=self._record_compaction_telemetry,
        )
        if self._branch_id(self.store.replay()) != branch_id:
            raise StaleBranchError("active branch changed during compaction")

        marker_messages = self._marker_messages(source_start, source_end, summary)
        # When adaptive shrinking crosses the latest user message, the durable
        # marker replays it immediately after the summary. This deliberately
        # moves it ahead of the minimal valid tool tail while preserving the
        # user request verbatim and keeping every tool call/result pair intact.
        proposed_messages = [
            *system_messages,
            *marker_messages,
            *([] if pinned_user is None else [pinned_user.message]),
            *(item.message for item in items[boundary:]),
        ]
        proposed = self._context(proposed_messages, True)
        if proposed.token_count > self.token_budget:
            truncated = self._truncate_tool_results(
                proposed_messages, self.token_budget, result_seqs
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
        self._provider_token_total = None
        self.last_context = proposed
        return proposed

    def _evict_context(
        self,
        *,
        branch: Sequence[ConversationEntry],
        branch_id: str | None,
        items: Sequence[_ContextItem],
        latest_user: int | None,
        system_messages: Sequence[Message],
    ) -> AssembledContext | None:
        """Persist one deterministic eviction view, or request summary fallback."""

        active_markers = self._active_markers(branch)
        previous_ends = [
            marker.data["source_seq_end"]
            for marker in active_markers
            if marker.data.get("kind") == EVICTION_KIND
        ]
        if previous_ends:
            previous_end = max(previous_ends)
            growth = sum(
                self.token_counter(Message.from_dict(entry.data["message"]))
                for entry in branch
                if entry.type == "message" and entry.seq > previous_end
            )
            if growth < max(1, int(self.token_budget * HYSTERESIS_RATIO)):
                return None

        candidates = [
            item for index, item in enumerate(items) if index != latest_user
        ]
        latest_user_item = items[latest_user] if latest_user is not None else None
        source_entries = {
            item.entry.id: item.entry
            for item in (*candidates, latest_user_item)
            if item is not None and item.entry is not None
        }
        if not source_entries:
            return None
        source_ranges = [
            (
                entry.data["source_seq_start"],
                entry.data["source_seq_end"],
            )
            if entry.type == "compaction"
            else (entry.seq, entry.seq)
            for entry in source_entries.values()
        ]
        source_start = min(start for start, _ in source_ranges)
        source_end = max(end for _, end in source_ranges)
        replaces = [
            entry.id for entry in source_entries.values() if entry.type == "compaction"
        ]
        records = [
            (
                int(item.message.metadata.get("source_seq", item.entry.seq)),
                item.message,
            )
            for item in candidates
            if item.entry is not None
        ]
        fixed_messages = [
            *system_messages,
            *([] if latest_user_item is None else [latest_user_item.message]),
        ]
        result = evict_messages(
            records,
            fixed_tokens=self._count(fixed_messages),
            target_tokens=max(1, int(self.token_budget * TARGET_RATIO)),
            token_counter=self.token_counter,
        )
        if not result.reached_target or not result.items_evicted:
            return None
        if self._branch_id(self.store.replay()) != branch_id:
            raise StaleBranchError("active branch changed during eviction")
        telemetry = {
            "kind": EVICTION_KIND,
            "eviction_count": 1,
            "items_evicted": result.items_evicted,
            "tokens_before": result.tokens_before,
            "tokens_after": result.tokens_after,
        }
        try:
            self.store.append_compaction_marker(
                "[deterministic semantic eviction view]",
                source_start,
                source_end,
                replaces=replaces,
                pinned_message=(
                    None if latest_user_item is None else latest_user_item.message
                ),
                expected_parent_id=branch_id,
                kind=EVICTION_KIND,
                view=eviction_view(records, result),
                telemetry=telemetry,
            )
        except ValueError as exc:
            raise StaleBranchError("active branch changed during eviction") from exc
        visible = self._visible_items(self.store.replay())
        proposed = self._context(
            [*system_messages, *(item.message for item in visible)], True
        )
        if proposed.token_count > self.token_budget:
            raise BudgetExceeded(
                "persisted eviction view exceeds the configured token budget"
            )
        self._record_compaction_telemetry(telemetry)
        self._provider_token_total = None
        self.last_context = proposed
        return proposed

    @staticmethod
    def _active_markers(
        entries: Sequence[ConversationEntry],
    ) -> list[ConversationEntry]:
        markers = [entry for entry in entries if entry.type == "compaction"]
        superseded = {
            marker_id
            for marker in markers
            for marker_id in marker.data.get("replaces", [])
        }
        return [marker for marker in markers if marker.id not in superseded]

    def _committed_messages(
        self,
        items: Sequence[_ContextItem],
        boundary: int,
        system_messages: Sequence[Message],
        latest_user: int | None,
    ) -> list[Message]:
        committed = [item.message for item in items if item.fixed]
        committed.extend(
            item.message
            for index, item in enumerate(items)
            if not item.fixed and (index >= boundary or index == latest_user)
        )
        return [*system_messages, *committed]

    def _shrink_tail_boundary(
        self,
        items: Sequence[_ContextItem],
        boundary: int,
        system_messages: Sequence[Message],
        target: int,
        latest_user: int | None,
    ) -> int:
        minimum = self._minimum_tail_boundary(items)
        for candidate in range(boundary + 1, minimum + 1):
            if not self._is_valid_tail_boundary(items, candidate):
                continue
            messages = self._committed_messages(
                items, candidate, system_messages, latest_user
            )
            if self._count(messages) <= target:
                return candidate
        return minimum

    @staticmethod
    def _minimum_tail_boundary(items: Sequence[_ContextItem]) -> int:
        if not items:
            return 0
        last = len(items) - 1
        message = items[last].message
        if message.role is not MessageRole.TOOL_RESULT or message.tool_result is None:
            return last
        call_id = message.tool_result.tool_call_id
        call_index = ContextAssembler._find_tool_call(items, call_id, len(items))
        return last if call_index is None else call_index

    @staticmethod
    def _is_valid_tail_boundary(items: Sequence[_ContextItem], boundary: int) -> bool:
        call_indexes: dict[str, int] = {}
        for index, item in enumerate(items):
            for block in item.message.content:
                if isinstance(block, ToolUseContent):
                    call_indexes[block.tool_call.id] = index
        return all(
            item.message.tool_result is None
            or call_indexes.get(item.message.tool_result.tool_call_id, -1) >= boundary
            for item in items[boundary:]
        )

    @staticmethod
    def _latest_user_index(items: Sequence[_ContextItem]) -> int | None:
        return next(
            (
                index
                for index in range(len(items) - 1, -1, -1)
                if items[index].message.role is MessageRole.USER
            ),
            None,
        )

    def _truncate_tool_results(
        self,
        messages: Sequence[Message],
        target: int,
        result_seqs: Mapping[int, int],
    ) -> list[Message] | None:
        if self._count(messages) <= target:
            return list(messages)
        result = list(messages)
        candidates: list[tuple[int, int, int, str]] = []
        for index, message in enumerate(messages):
            tool_result = message.tool_result
            seq = result_seqs.get(id(message))
            if tool_result is None or seq is None:
                continue
            content = (
                flatten_tool_content(tool_result.content_blocks, detailed_images=True)
                if tool_result.content_blocks is not None
                else tool_result.content
            )
            candidates.append((-len(content), seq, index, content))
        for _, seq, index, content in sorted(candidates):
            if self._count(result) <= target:
                break
            original = result[index].tool_result
            if original is None:
                continue
            low = 0
            high = len(content)
            best: Message | None = None
            while low <= high:
                shown = (low + high) // 2
                replacement = self._truncated_tool_message(
                    result[index], content, shown, seq
                )
                proposed = [*result[:index], replacement, *result[index + 1 :]]
                if self._count(proposed) <= target:
                    best = replacement
                    low = shown + 1
                else:
                    high = shown - 1
            result[index] = best or self._truncated_tool_message(
                result[index], content, 0, seq
            )
        return result if self._count(result) <= target else None

    @staticmethod
    def _truncated_tool_message(
        message: Message,
        content: str,
        shown: int,
        seq: int,
    ) -> Message:
        head_size = (shown + 1) // 2
        tail_size = shown - head_size
        excerpt = content[:head_size]
        if tail_size:
            excerpt += "\n…\n" + content[len(content) - tail_size :]
        marker = (
            f"[output truncated for context: showed {shown} of {len(content)} chars; "
            f"full output is in the session log at seq {seq}]"
        )
        excerpt = f"{excerpt}\n{marker}" if excerpt else marker
        original = message.tool_result
        if original is None:
            return message
        return Message(
            message.role,
            [],
            tool_result=ToolResult(
                original.tool_call_id,
                excerpt,
                is_error=original.is_error,
                is_canceled=original.is_canceled,
            ),
            metadata=dict(message.metadata),
        )

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
            digest=_digest(messages),
            compacted=compacted,
        )

    def _save(self, messages: list[Message], compacted: bool) -> AssembledContext:
        context = self._context(messages, compacted)
        self.last_context = context
        return context

    def _count(self, messages: Sequence[Message]) -> int:
        return sum(self.token_counter(message) for message in messages)

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
        all_markers = [entry for entry in entries if entry.type == "compaction"]
        superseded_ids: set[str] = set()
        for marker in all_markers:
            superseded_ids.update(marker.data.get("replaces", []))
        markers = [entry for entry in all_markers if entry.id not in superseded_ids]
        compacted_ranges = [
            (entry.data["source_seq_start"], entry.data["source_seq_end"])
            for entry in markers
        ]
        markers_by_start = {entry.data["source_seq_start"]: entry for entry in markers}
        items: list[_ContextItem] = []
        emitted_marker_ids: set[str] = set()
        failed_tool_call_ids: set[str] = set()
        for entry in entries:
            marker = markers_by_start.get(entry.seq)
            if marker is not None:
                items.extend(self._marker_items(marker))
                emitted_marker_ids.add(marker.id)
            if entry.type == "compaction":
                continue
            if entry.type in {"warning", "checkpoint", "fork"}:
                continue
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
            if any(start <= entry.seq <= end for start, end in compacted_ranges):
                continue
            if (
                message.role is MessageRole.TOOL_RESULT
                and message.tool_result is not None
                and message.tool_result.tool_call_id in failed_tool_call_ids
            ):
                continue
            items.append(_ContextItem(entry, message))
        for marker in markers:
            if marker.id not in emitted_marker_ids:
                items.extend(self._marker_items(marker))
        return items

    @staticmethod
    def _marker_items(entry: ConversationEntry) -> list[_ContextItem]:
        if entry.data.get("kind") == EVICTION_KIND:
            items = [
                _ContextItem(entry, ContextAssembler._eviction_view_message(item))
                for item in entry.data["view"]
            ]
        else:
            messages = ContextAssembler._marker_messages(
                entry.data["source_seq_start"],
                entry.data["source_seq_end"],
                entry.data["summary"],
            )
            items = [_ContextItem(entry, message, fixed=True) for message in messages]
        pinned = entry.data.get("pinned_message")
        if pinned is not None:
            items.append(_ContextItem(entry, Message.from_dict(pinned)))
        return items

    @staticmethod
    def _eviction_view_message(item: Mapping[str, Any]) -> Message:
        message = Message.from_dict(item["message"])
        return Message(
            message.role,
            list(message.content),
            tool_result=message.tool_result,
            metadata={"source_seq": item["seq"], **message.metadata},
        )

    @staticmethod
    def _marker_messages(
        source_start: int,
        source_end: int,
        summary: str,
    ) -> list[Message]:
        marker_text = f"[compaction marker: entries {source_start}–{source_end}]"
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
