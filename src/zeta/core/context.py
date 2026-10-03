"""Context assembly and durable conversation compaction."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import ceil
from time import perf_counter
from pathlib import Path
from typing import Any

from .store import ConversationEntry, ConversationStore
from ..compaction import fallback_summary
from ..protocol.types import (
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    FAILED_TURN_MARKER,
    flatten_tool_content,
    ImageContent,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolResult,
    ToolUseContent,
)


class BudgetExceeded(RuntimeError):
    """The committed context cannot fit in the configured budget."""


class SummaryCompletionError(RuntimeError):
    """The no-tools completion did not produce a usable summary."""


class SummaryInputTooLarge(SummaryCompletionError):
    """The source range is too large for a safe summary request."""


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
SUMMARY_SOURCE_TOKEN_LIMIT = 64_000
SUMMARY_RETRIES = 2
RETRYABLE_SUMMARY_STREAM_ERRORS = {"Codex output item completed with open blocks"}
_logger = logging.getLogger(__name__)


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


def _strip_thinking(message: Message) -> Message:
    content = [
        block
        for block in message.content
        if isinstance(block, (ImageContent, TextContent, ToolUseContent))
    ]
    return Message(
        message.role,
        content,
        tool_result=message.tool_result,
        metadata=dict(message.metadata),
    )


def _summary_message(message: Message) -> dict[str, Any]:
    """Serialize a message without copying image base64 into a summary prompt."""

    value = message.to_dict()
    # Provider replay and control metadata are not conversation content.
    value.pop("metadata", None)
    content = value.get("content")
    if isinstance(content, list):
        for index, block in enumerate(message.content):
            if not isinstance(block, ImageContent):
                continue
            filename = Path(block.path).name if block.path else "clipboard image"
            size = block.size if block.size is not None else "unknown"
            content[index] = {
                "type": "text",
                "text": (
                    f"[image attachment] filename={filename} "
                    f"media_type={block.mime_type} bytes={size}"
                ),
            }
    tool_result = value.get("tool_result")
    if not isinstance(tool_result, dict):
        return value
    blocks = tool_result.get("content_blocks")
    if not isinstance(blocks, list) or not any(
        isinstance(block, dict) and block.get("type") == "image" for block in blocks
    ):
        return value
    tool_result["content"] = flatten_tool_content(blocks, detailed_images=True)
    tool_result.pop("content_blocks", None)
    return value


def _has_tool_call(message: Message) -> bool:
    return any(isinstance(block, ToolUseContent) for block in message.content)


class CompactionPolicy:
    """Summarize a selected range with a completion that has no tools."""

    def __init__(
        self,
        backend: CompletionBackend | None = None,
        *,
        summary_prompt: str = (
            "Summarize the conversation range below. Preserve decisions, facts, "
            "open work, and tool results. Return only the summary."
        ),
    ) -> None:
        self.backend = backend
        self.summary_prompt = summary_prompt

    async def summarize(
        self,
        messages: Sequence[Message],
        *,
        backend: CompletionBackend | None = None,
        system_prompt: Message | None = None,
        max_source_tokens: int | None = None,
        on_success: Callable[[], None] | None = None,
        on_usage: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> str:
        completion_backend = backend or self.backend
        if completion_backend is None:
            raise SummaryCompletionError("compaction requires a completion backend")
        sanitized_messages = [_strip_thinking(message) for message in messages]
        source = json.dumps(
            [_summary_message(message) for message in sanitized_messages],
            sort_keys=True,
            separators=(",", ":"),
        )
        source_tokens = max(1, ceil(len(source) / 4))
        if max_source_tokens is not None and source_tokens > max_source_tokens:
            raise SummaryInputTooLarge(
                f"summary source is too large: {source_tokens} tokens "
                f"exceeds {max_source_tokens}"
            )
        return await self._complete_source(
            source,
            max_chars=max_source_tokens * 4 if max_source_tokens is not None else 3_000,
            backend=backend,
            system_prompt=system_prompt,
            on_success=on_success,
            on_usage=on_usage,
        )

    async def summarize_chunked(
        self,
        messages: Sequence[Message],
        *,
        backend: CompletionBackend | None = None,
        system_prompt: Message | None = None,
        max_source_tokens: int = SUMMARY_SOURCE_TOKEN_LIMIT,
        max_fallback_tokens: int | None = None,
        on_success: Callable[[], None] | None = None,
        on_usage: Callable[[Mapping[str, Any]], None] | None = None,
        on_telemetry: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> str:
        """Reduce a large range without sending the whole range in one request."""

        started = perf_counter()
        max_chars = max_source_tokens * 4
        fallback_max_chars = (
            max_chars if max_fallback_tokens is None else max_fallback_tokens * 4
        )
        usage_totals = {
            key: 0
            for key in (
                "input_tokens",
                "output_tokens",
                "cache_read_input_tokens",
                "cache_creation_input_tokens",
                "total_tokens",
            )
        }
        models: set[str] = set()
        max_models = 32
        retries = 0
        fallback_count = 0

        def record_retry() -> None:
            nonlocal retries
            retries += 1

        def record_fallback() -> None:
            nonlocal fallback_count
            fallback_count += 1

        def record_usage(usage: Mapping[str, Any]) -> None:
            sanitized: dict[str, Any] = {}
            for key in usage_totals:
                value = usage.get(key)
                if type(value) is int and value >= 0:
                    usage_totals[key] += value
                    sanitized[key] = value
            model = usage.get("_zeta_model")
            if type(model) is str and model:
                if len(models) < max_models:
                    models.add(model)
                sanitized["_zeta_model"] = model
            if on_usage is not None and sanitized:
                on_usage(sanitized)

        def record_telemetry(chunks: int, mapped: float, reduced: float) -> None:
            if on_telemetry is None:
                return
            telemetry = {
                "source_size": sum(map(len, sources)),
                "chunk_count": chunks,
                "map_seconds": mapped,
                "reduce_seconds": reduced,
                "total_seconds": perf_counter() - started,
                "retries": retries,
                "output_tokens": usage_totals["output_tokens"],
                "models": sorted(models),
            }
            if fallback_count:
                telemetry["fallback_count"] = fallback_count
            on_telemetry(telemetry)

        if max_chars < 16:
            raise SummaryInputTooLarge("summary source limit is too small")
        rows = [
            json.dumps(
                _summary_message(_strip_thinking(message)),
                sort_keys=True,
                separators=(",", ":"),
            )
            for message in messages
        ]
        sources: list[str] = []
        fallback_sources: list[str] = []
        group: list[str] = []
        group_chars = 2
        for row in rows:
            if len(row) + 2 > max_chars:
                if group:
                    grouped_source = f"[{','.join(group)}]"
                    sources.append(grouped_source)
                    fallback_sources.append(grouped_source)
                    group = []
                    group_chars = 2
                fragments = [
                    row[start : start + max_chars]
                    for start in range(0, len(row), max_chars)
                ]
                sources.extend(fragments)
                # Keep provider chunking unchanged, but use valid structured
                # messages if any fragment needs the deterministic fallback.
                fallback_sources.extend(f"[{row}]" for _ in fragments)
                continue
            added = len(row) + int(bool(group))
            if group and group_chars + added > max_chars:
                grouped_source = f"[{','.join(group)}]"
                sources.append(grouped_source)
                fallback_sources.append(grouped_source)
                group = []
                group_chars = 2
                added = len(row)
            group.append(row)
            group_chars += added
        if group:
            grouped_source = f"[{','.join(group)}]"
            sources.append(grouped_source)
            fallback_sources.append(grouped_source)
        if not sources:
            sources.append("[]")
            fallback_sources.append("[]")
        if len(sources) == 1:
            result = await self._summarize_source(
                sources[0],
                max_chars,
                backend,
                system_prompt,
                on_success,
                record_usage,
                record_retry,
                record_fallback,
                fallback_source=fallback_sources[0],
                fallback_max_chars=fallback_max_chars,
            )
            record_telemetry(1, 0.0, perf_counter() - started)
            return result
        map_started = perf_counter()
        semaphore = asyncio.Semaphore(3)
        map_failed = asyncio.Event()

        async def map_one(source: str, fallback_source: str) -> str:
            async with semaphore:
                if map_failed.is_set():
                    raise asyncio.CancelledError
                try:
                    return await self._summarize_source(
                        source,
                        max_chars,
                        backend,
                        system_prompt,
                        on_success,
                        record_usage,
                        record_retry,
                        record_fallback,
                        fallback_source=fallback_source,
                        fallback_max_chars=fallback_max_chars,
                    )
                except BaseException:
                    map_failed.set()
                    raise

        fallbacks_before_map = fallback_count
        tasks = [
            asyncio.create_task(map_one(source, fallback_source))
            for source, fallback_source in zip(sources, fallback_sources, strict=True)
        ]
        try:
            # gather preserves source order even when provider streams finish out of order.
            summaries = await asyncio.gather(*tasks)
        except BaseException:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        map_seconds = perf_counter() - map_started
        combined = json.dumps(summaries, separators=(",", ":"))
        reduce_started = perf_counter()
        if len(combined) >= sum(map(len, sources)):
            if fallback_count == fallbacks_before_map:
                raise SummaryCompletionError(
                    "compaction summaries did not reduce source"
                )
            record_fallback()
            result = fallback_summary(
                f"[{','.join(rows)}]", max_chars=fallback_max_chars
            )
        else:
            result = await self._summarize_source(
                combined,
                max_chars,
                backend,
                system_prompt,
                on_success,
                record_usage,
                record_retry,
                record_fallback,
                fallback_source=f"[{','.join(rows)}]",
                fallback_max_chars=fallback_max_chars,
            )
        record_telemetry(len(sources), map_seconds, perf_counter() - reduce_started)
        return result

    async def _summarize_source(
        self,
        source: str,
        max_chars: int,
        backend: CompletionBackend | None,
        system_prompt: Message | None,
        on_success: Callable[[], None] | None,
        on_usage: Callable[[Mapping[str, Any]], None] | None,
        on_retry: Callable[[], None] | None = None,
        on_fallback: Callable[[], None] | None = None,
        depth: int = 0,
        fallback_source: str | None = None,
        fallback_max_chars: int | None = None,
    ) -> str:
        if depth >= 8:
            raise SummaryCompletionError(
                "summary source exceeds provider context limit"
            )
        if len(source) > max_chars:
            split_fallbacks = 0

            def record_split_fallback() -> None:
                nonlocal split_fallbacks
                split_fallbacks += 1
                if on_fallback is not None:
                    on_fallback()

            parts = [
                source[start : start + max_chars]
                for start in range(0, len(source), max_chars)
            ]
            summaries = [
                await self._summarize_source(
                    part,
                    max_chars,
                    backend,
                    system_prompt,
                    on_success,
                    on_usage,
                    on_retry,
                    record_split_fallback,
                    depth + 1,
                    fallback_source=fallback_source,
                    fallback_max_chars=fallback_max_chars,
                )
                for part in parts
            ]
            combined = json.dumps(summaries, separators=(",", ":"))
            if len(combined) >= len(source):
                if not split_fallbacks:
                    raise SummaryCompletionError(
                        "compaction summaries did not reduce source"
                    )
                record_split_fallback()
                return fallback_summary(
                    fallback_source or source,
                    max_chars=(
                        max_chars
                        if fallback_max_chars is None
                        else min(max_chars, fallback_max_chars)
                    ),
                )
            return await self._summarize_source(
                combined,
                max_chars,
                backend,
                system_prompt,
                on_success,
                on_usage,
                on_retry,
                record_split_fallback,
                depth + 1,
                fallback_source=fallback_source,
                fallback_max_chars=fallback_max_chars,
            )
        try:
            return await self._complete_source(
                source,
                max_chars=max_chars,
                backend=backend,
                system_prompt=system_prompt,
                on_success=on_success,
                on_usage=on_usage,
                on_retry=on_retry,
                on_fallback=on_fallback,
                fallback_source=fallback_source,
                fallback_max_chars=fallback_max_chars,
            )
        except SummaryCompletionError as exc:
            if (
                getattr(exc, "code", None) != "context_length_exceeded"
                or max_chars < 64
            ):
                raise
            return await self._summarize_source(
                source,
                max_chars // 2,
                backend,
                system_prompt,
                on_success,
                on_usage,
                on_retry,
                on_fallback,
                depth + 1,
                fallback_source=fallback_source,
                fallback_max_chars=fallback_max_chars,
            )

    async def _complete_source(
        self,
        source: str,
        *,
        max_chars: int,
        backend: CompletionBackend | None,
        system_prompt: Message | None,
        on_success: Callable[[], None] | None,
        on_usage: Callable[[Mapping[str, Any]], None] | None,
        on_retry: Callable[[], None] | None = None,
        on_fallback: Callable[[], None] | None = None,
        fallback_source: str | None = None,
        fallback_max_chars: int | None = None,
    ) -> str:
        for attempt in range(SUMMARY_RETRIES + 1):
            try:
                return await self._complete_source_once(
                    source,
                    backend=backend,
                    system_prompt=system_prompt,
                    on_success=on_success,
                    on_usage=on_usage,
                    on_retry=on_retry,
                    retry=attempt > 0,
                )
            except SummaryCompletionError as exc:
                empty = getattr(exc, "code", None) == "empty_summary"
                if attempt >= SUMMARY_RETRIES:
                    if not empty:
                        raise
                    _logger.warning(
                        "compaction model returned no summary after %d attempts; "
                        "using automatic fallback",
                        attempt + 1,
                    )
                    if on_fallback is not None:
                        on_fallback()
                    return fallback_summary(
                        fallback_source or source,
                        max_chars=(
                            max_chars
                            if fallback_max_chars is None
                            else min(max_chars, fallback_max_chars)
                        ),
                    )
                if not empty and not (
                    getattr(exc, "code", None) == "stream_error"
                    and str(exc) in RETRYABLE_SUMMARY_STREAM_ERRORS
                ):
                    raise
                if on_retry is not None:
                    on_retry()

    async def _complete_source_once(
        self,
        source: str,
        *,
        backend: CompletionBackend | None,
        system_prompt: Message | None,
        on_success: Callable[[], None] | None,
        on_usage: Callable[[Mapping[str, Any]], None] | None,
        on_retry: Callable[[], None] | None,
        retry: bool,
    ) -> str:
        completion_backend = backend or self.backend
        if completion_backend is None:
            raise SummaryCompletionError("compaction requires a completion backend")
        instruction = self.summary_prompt
        if retry:
            instruction += " Return a non-empty summary."
        prompt = Message(
            MessageRole.USER,
            [TextContent(f"{instruction}\n\n{source}")],
        )
        summary_messages: list[Message] = []
        if system_prompt is not None:
            summary_messages.append(_strip_thinking(system_prompt))
        summary_messages.append(prompt)

        partial: list[ContentBlock] = []
        completed: Message | None = None
        summary_usage: dict[str, Any] = {}
        model: str | None = None
        completion = None
        try:
            try:
                completion = completion_backend.complete(summary_messages, [])
                async for event in completion:
                    if event.type is StreamEventType.RETRY and on_retry is not None:
                        on_retry()
                    event_model = event.data.get("model")
                    if type(event_model) is str and event_model:
                        model = event_model
                    usage = event.data.get("usage")
                    if isinstance(usage, Mapping):
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_input_tokens",
                            "cache_creation_input_tokens",
                            "total_tokens",
                        ):
                            value = usage.get(key)
                            if type(value) is int and value >= 0:
                                summary_usage[key] = summary_usage.get(key, 0) + value
                    if event.type is StreamEventType.ERROR:
                        info = (
                            event.error
                            if isinstance(event.error, ErrorInfo)
                            else ErrorInfo(
                                "backend_error",
                                "provider emitted an invalid error event",
                            )
                        )
                        failure = SummaryCompletionError(info.message)
                        failure.code = info.code
                        failure.status_code = info.status_code
                        raise failure
                    if event.type is StreamEventType.MESSAGE_UPDATE:
                        if event.content is not None:
                            partial.append(event.content)
                        if event.delta is not None:
                            partial.append(TextContent(event.delta))
                    if (
                        event.type is StreamEventType.MESSAGE_END
                        and event.message is not None
                    ):
                        completed = event.message
            except Exception as exc:
                if isinstance(exc, SummaryCompletionError):
                    raise
                code = getattr(exc, "code", None)
                if type(code) is str:
                    failure = SummaryCompletionError(str(exc))
                    failure.code = code
                    failure.status_code = getattr(exc, "status_code", None)
                    raise failure from exc
                raise SummaryCompletionError("summary completion failed") from exc
        except BaseException:
            close = getattr(completion, "aclose", None)
            if close is not None:
                try:
                    await close()
                except BaseException:  # noqa: BLE001, S110 - preserve the primary exception
                    pass
            raise
        else:
            close = getattr(completion, "aclose", None)
            if close is not None:
                await close()

        if on_usage is not None and summary_usage:
            if model is not None:
                summary_usage["_zeta_model"] = model
            on_usage(summary_usage)
        result = completed or Message(MessageRole.ASSISTANT, partial)
        if _has_tool_call(result):
            raise SummaryCompletionError("summary completion returned a tool call")
        summary = _text_from_message(result).strip()
        if not summary:
            failure = SummaryCompletionError(
                "summary completion returned an empty summary"
            )
            failure.code = "empty_summary"
            raise failure
        if on_success is not None:
            on_success()
        return summary

    @staticmethod
    def should_compact(total_tokens: int, budget: int) -> bool:
        return total_tokens > budget


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
        # The fallback output must fit beside the same fixed messages used by
        # the final budget check. If even its prefix cannot fit, the later
        # BudgetExceeded is intentional: no useful fallback can fit.
        fixed_messages = [
            *system_messages,
            *self._marker_messages(source_start, source_end, ""),
            *([] if pinned_user is None else [pinned_user.message]),
            *(item.message for item in items[boundary:]),
        ]
        fallback_tokens = max(0, self.token_budget - self._count(fixed_messages))
        summary = await self.compaction_policy.summarize_chunked(
            [item.message for item in candidates],
            backend=backend or self.backend,
            system_prompt=system_prompt,
            max_fallback_tokens=fallback_tokens,
            # Product backends can expose smaller windows than model cards.
            # Bound each request and summarize larger ranges in chunks.
            # Keep the budget-relative bound for small configured budgets.
            max_source_tokens=min(
                SUMMARY_SOURCE_TOKEN_LIMIT,
                max(1, self.token_budget // 2, self.token_budget - 8_000),
            ),
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
