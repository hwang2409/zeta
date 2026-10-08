"""Bounded summary fallback for eviction when deterministic views cannot fit."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

FALLBACK_SUMMARY_PREFIX = "[automatic fallback summary: model returned no summary]"
_FALLBACK_TRUNCATION_MARKER = "[fallback summary truncated]"


def _bounded_summary(lines: list[str], max_chars: int) -> str:
    """Join fallback lines without exceeding the caller's output bound."""

    if max_chars < 0:
        raise ValueError("fallback summary bound must not be negative")
    summary = "\n".join(lines)
    if len(summary) <= max_chars:
        return summary
    suffix = f"\n{_FALLBACK_TRUNCATION_MARKER}"
    body_start = f"{FALLBACK_SUMMARY_PREFIX}\n"
    body_chars = max_chars - len(body_start) - len(suffix)
    if body_chars < 0:
        return FALLBACK_SUMMARY_PREFIX[:max_chars]
    body = "\n".join(lines[1:])[:body_chars]
    return f"{body_start}{body}{suffix}"


def _tool_result_line(result: Mapping[object, object]) -> str:
    """Return a bounded head/tail excerpt of one serialized tool result."""

    content = result.get("content")
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        text = "".join(
            block.get("text", "")
            for block in content
            if isinstance(block, Mapping) and type(block.get("text")) is str
        )
    else:
        text = ""
    text = " ".join(text.split())
    if len(text) > 400:
        text = f"{text[:200]} … {text[-200:]}"
    label = "tool_result error" if result.get("is_error") is True else "tool_result"
    return f"{label}: {text}"


def fallback_summary(source: str, *, max_chars: int) -> str:
    """Build a deterministic fallback within the exact output character bound."""

    lines = [FALLBACK_SUMMARY_PREFIX]
    try:
        rows = json.loads(source)
    except (json.JSONDecodeError, TypeError):
        digest = hashlib.sha256(source.encode()).hexdigest()
        return _bounded_summary(
            [FALLBACK_SUMMARY_PREFIX, f"source_sha256={digest}"], max_chars
        )
    rows = rows if isinstance(rows, list) else []
    for row in rows:
        if not isinstance(row, Mapping):
            lines.append(f"summary: {str(row)[:400]}")
            continue
        role = str(row.get("role", "unknown"))[:100]
        content = row.get("content")
        blocks = content if isinstance(content, list) else []
        text = "".join(
            block.get("text", "")
            for block in blocks
            if isinstance(block, Mapping) and type(block.get("text")) is str
        )
        nonempty = [line for line in text.splitlines() if line.strip()]
        if role == "user":
            lines.append(f"user: {text[:400]}")
        elif nonempty:
            excerpt = nonempty[0][:200]
            if len(nonempty) > 1:
                excerpt += f" … {nonempty[-1][:200]}"
            lines.append(f"{role}: {excerpt}")
        result = row.get("tool_result")
        if isinstance(result, Mapping):
            lines.append(_tool_result_line(result))
        for block in blocks:
            call = block.get("tool_call") if isinstance(block, Mapping) else None
            if isinstance(call, Mapping):
                args = json.dumps(call.get("arguments", {}), sort_keys=True)
                digest = hashlib.sha256(args.encode()).hexdigest()[:16]
                name = str(call.get("name", "unknown"))[:400]
                lines.append(f"tool: {name} args_sha256={digest}")
    return _bounded_summary(lines, max_chars)


import asyncio
import logging
from collections.abc import Callable, Sequence
from math import ceil
from pathlib import Path
from time import perf_counter
from typing import Any

from .protocol.types import (
    CompletionBackend,
    ContentBlock,
    ErrorInfo,
    ImageContent,
    Message,
    MessageRole,
    StreamEventType,
    TextContent,
    ToolUseContent,
    flatten_tool_content,
)

SUMMARY_SOURCE_TOKEN_LIMIT = 64_000
SUMMARY_RETRIES = 2
RETRYABLE_SUMMARY_STREAM_ERRORS = {"Codex output item completed with open blocks"}
_logger = logging.getLogger(__name__)


class SummaryCompletionError(RuntimeError):
    """The no-tools completion did not produce a usable summary."""


class SummaryInputTooLarge(SummaryCompletionError):
    """The source range is too large for a safe summary request."""


def _text_from_message(message: Message) -> str:
    return "".join(
        block.text for block in message.content if isinstance(block, TextContent)
    )


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
        max_fallback_chars: int | None = None,
        on_success: Callable[[], None] | None = None,
        on_usage: Callable[[Mapping[str, Any]], None] | None = None,
        on_telemetry: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> str:
        """Reduce a large range without sending the whole range in one request."""

        started = perf_counter()
        max_chars = max_source_tokens * 4
        fallback_max_chars = (
            max_chars if max_fallback_chars is None else max_fallback_chars
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
                    if event.type is StreamEventType.ASSISTANT_RESET:
                        partial.clear()
                        completed = None
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
