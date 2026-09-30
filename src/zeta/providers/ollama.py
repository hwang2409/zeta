"""Ollama /api/chat streaming backend (text and tool calls only)."""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

import httpx

from ..protocol.types import (
    CompletionBackend,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)
from .transport import (
    DEFAULT_STREAM_STALL_RETRIES,
    DEFAULT_STREAM_STALL_SECONDS,
    retry_provider_completion,
    retryable_provider_error,
    stall_retry_kwargs,
    stall_watchdog,
    wait_for_response_headers,
)
from .usage import normalize_usage

OLLAMA_API_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"


class OllamaError(RuntimeError):
    """A malformed or failed Ollama request/stream."""

    def __init__(
        self, message: str, *, retryable: bool = False, is_stall: bool = False
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.is_stall = is_stall


def _content(message: Message) -> str:
    parts = [block.text for block in message.content if isinstance(block, TextContent)]
    if any(
        not isinstance(block, (TextContent, ToolUseContent))
        for block in message.content
    ):
        raise OllamaError(
            "Ollama provider supports text only; images/reasoning are unsupported"
        )
    return "".join(parts)


def _messages(messages: Sequence[Message]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    prior_calls: dict[str, ToolCall] = {}
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            result.append({"role": "system", "content": _content(message)})
        elif message.role is MessageRole.TOOL_RESULT:
            if message.tool_result is None:
                raise OllamaError("tool result message is missing its result")
            call_id = message.tool_result.tool_call_id
            call = prior_calls.get(call_id)
            if call is None:
                raise OllamaError(
                    f"tool result {call_id!r} does not match a prior tool call"
                )
            result.append(
                {
                    "role": "tool",
                    "tool_name": call.name,
                    "content": message.tool_result.content,
                }
            )
        elif message.role in (MessageRole.USER, MessageRole.ASSISTANT):
            row: dict[str, Any] = {
                "role": message.role.value,
                "content": _content(message),
            }
            calls = [
                block.tool_call
                for block in message.content
                if isinstance(block, ToolUseContent)
            ]
            if calls:
                for call in calls:
                    if call.id in prior_calls:
                        raise OllamaError(f"tool call id {call.id!r} is ambiguous")
                    prior_calls[call.id] = call
                row["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": c.arguments}}
                    for c in calls
                ]
            result.append(row)
        else:
            text = _content(message)
            if text:
                result.append({"role": "user", "content": text})
    return result


def _tools(schemas: Sequence[ToolSchema]) -> list[dict[str, Any]]:
    tools = []
    for schema in schemas:
        name = schema.get("name")
        if type(name) is not str or not name:
            raise OllamaError("tool schema name must be a nonempty string")
        parameters = schema.get("input_schema", schema.get("parameters", {}))
        if not isinstance(parameters, Mapping):
            raise OllamaError(f"tool schema {name!r} parameters must be an object")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": schema.get("description", ""),
                    "parameters": dict(parameters),
                },
            }
        )
    return tools


class OllamaBackend(CompletionBackend):
    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = OLLAMA_API_URL,
        client: httpx.AsyncClient | None = None,
        timeout: float | None = None,
        stall_seconds: float = DEFAULT_STREAM_STALL_SECONDS,
        stall_retries: int = DEFAULT_STREAM_STALL_RETRIES,
    ) -> None:
        if not base_url or any(ch.isspace() for ch in base_url):
            raise ValueError("Ollama base URL must be a nonempty URL")
        if stall_seconds < 0 or stall_retries < 0:
            raise ValueError("Ollama stall settings must be nonnegative")
        self.model, self.base_url, self.client = model, base_url.rstrip("/"), client
        self.stall_seconds, self.stall_retries = stall_seconds, stall_retries
        # The shared watchdog owns read-stall timing. Keep HTTPX from racing it.
        self.timeout = (
            httpx.Timeout(None, connect=10.0, write=10.0, pool=10.0)
            if timeout is None
            else httpx.Timeout(timeout, read=None)
        )

    def complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        return self._complete(messages, tool_schemas)

    async def _complete(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        async def refresh() -> str:
            return ""

        attempts = retry_provider_completion(
            lambda: self._complete_once(messages, tool_schemas),
            lambda _token: self._complete_once(messages, tool_schemas),
            refresh,
            lambda _error: False,
            lambda error: error,
            lambda event: event.type is StreamEventType.MESSAGE_START,
            retryable_provider_error,
            lambda number, delay, error: StreamEvent(
                StreamEventType.RETRY,
                data={
                    "text": f"retrying ({number}/3) in {delay:.1f}s",
                    "retry": number,
                    "delay": delay,
                },
            ),
            lambda _error, _retries: None,
            **stall_retry_kwargs(self.stall_retries),
        )
        try:
            async for event in attempts:
                yield event
        finally:
            await attempts.aclose()

    async def _complete_once(
        self, messages: Sequence[Message], tool_schemas: Sequence[ToolSchema]
    ) -> AsyncIterator[StreamEvent]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": _messages(messages),
            "stream": True,
        }
        tools = _tools(tool_schemas)
        if tools:
            payload["tools"] = tools
        client = self.client or httpx.AsyncClient(timeout=self.timeout)
        try:
            stream = client.stream(
                "POST", f"{self.base_url}/api/chat", json=payload, timeout=self.timeout
            )
            response = await wait_for_response_headers(
                stream.__aenter__(), self.stall_seconds, "Ollama", OllamaError
            )
            try:
                if response.status_code >= 400:
                    body = (await response.aread())[:500].decode("utf-8", "replace")
                    raise OllamaError(
                        f"Ollama HTTP {response.status_code}: {body}",
                        retryable=response.status_code >= 500,
                    )
                text = ""
                started = False
                calls: list[ToolCall] = []
                usage: dict[str, Any] = {}
                stop_reason: str | None = None
                done = False
                async for line in stall_watchdog(
                    response.aiter_lines(),
                    seconds=self.stall_seconds,
                    on_stall=lambda elapsed: OllamaError(
                        f"Ollama stream stalled for {elapsed:.0f}s", is_stall=True
                    ),
                ):
                    if not line.strip():
                        continue
                    try:
                        item = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise OllamaError(
                            "Ollama stream contained malformed JSON"
                        ) from exc
                    if not isinstance(item, Mapping):
                        raise OllamaError("Ollama stream item must be an object")
                    if "error" in item:
                        raise OllamaError(f"Ollama error: {item['error']}")
                    message = item.get("message")
                    if not isinstance(message, Mapping):
                        raise OllamaError("Ollama stream message is invalid")
                    if not started:
                        started = True
                        yield StreamEvent(
                            StreamEventType.MESSAGE_START, data={"model": self.model}
                        )
                    chunk = message.get("content", "")
                    if type(chunk) is not str:
                        raise OllamaError("Ollama text delta is not a string")
                    if chunk:
                        text += chunk
                        yield StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=chunk)
                    if "tool_calls" in message and not isinstance(
                        message["tool_calls"], list
                    ):
                        raise OllamaError("Ollama tool_calls must be an array")
                    raw_calls = message.get("tool_calls", [])
                    if raw_calls:
                        for raw in raw_calls:
                            if not isinstance(raw, Mapping) or not isinstance(
                                raw.get("function"), Mapping
                            ):
                                raise OllamaError("Ollama tool call is malformed")
                            function = raw["function"]
                            name, args = (
                                function.get("name"),
                                function.get("arguments", {}),
                            )
                            if type(name) is not str or not name:
                                raise OllamaError("Ollama tool call name is invalid")
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except json.JSONDecodeError as exc:
                                    raise OllamaError(
                                        "Ollama tool arguments are malformed JSON"
                                    ) from exc
                            if not isinstance(args, Mapping):
                                raise OllamaError(
                                    "Ollama tool arguments must be an object"
                                )
                            call = ToolCall(f"ollama-{uuid.uuid4()}", name, dict(args))
                            calls.append(call)
                            yield StreamEvent(
                                StreamEventType.MESSAGE_UPDATE, tool_call=call
                            )
                    if "done" in item and type(item["done"]) is not bool:
                        raise OllamaError("Ollama done must be a boolean")
                    if item.get("done", False):
                        done = True
                        done_reason = item.get("done_reason")
                        if calls:
                            stop_reason = "tool_use"
                        elif done_reason == "length":
                            stop_reason = "max_tokens"
                        else:
                            stop_reason = "end_turn"
                        if type(item.get("prompt_eval_count")) is int:
                            usage["prompt_tokens"] = item["prompt_eval_count"]
                        if type(item.get("eval_count")) is int:
                            usage["completion_tokens"] = item["eval_count"]
                        break
                if not done:
                    raise OllamaError("Ollama stream ended before done")
                content = [TextContent(text)] if text else []
                content.extend(ToolUseContent(call) for call in calls)
                yield StreamEvent(
                    StreamEventType.MESSAGE_END,
                    message=Message(MessageRole.ASSISTANT, content),
                    data={"usage": normalize_usage(usage), "stop_reason": stop_reason},
                )
            finally:
                await stream.__aexit__(None, None, None)
        except httpx.TransportError as exc:
            raise OllamaError(
                f"Ollama connection failed: {exc}", retryable=True
            ) from exc
        finally:
            if self.client is None:
                await client.aclose()


__all__ = ["DEFAULT_OLLAMA_MODEL", "OLLAMA_API_URL", "OllamaBackend", "OllamaError"]
