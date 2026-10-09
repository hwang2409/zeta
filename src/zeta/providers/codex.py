"""ChatGPT plan OAuth and Responses SSE support for Codex."""

from __future__ import annotations

import asyncio
import json
import re
import uuid
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..codex import (
    CODEX_API_URL,
    DEFAULT_CODEX_MODEL,
    DEFAULT_CODEX_REDIRECT_URI,
    CodexAuthError,
    CodexBackendError,
    CodexCredentialStore,
    CodexHTTPError,
    CodexStreamError,
    build_authorization_url,
    codex_request_headers,
    exchange_authorization_code,
    extract_account_id,
)
from ..oauth import error_body_excerpt
from typing import Any

import httpx

from .codex_payload import (
    _cache_affinity_json,
    build_responses_payload,
)
from .stream_diagnostics import StreamDiagnostics
from .stream_errors import decode_stream_error
from .transport import (
    DEFAULT_STREAM_STALL_RETRIES,
    DEFAULT_STREAM_STALL_SECONDS,
    StreamFinished,
    cleanup_transport,
    is_control_exception,
    provider_retry_notice,
    request_error,
    retry_after_seconds,
    retry_provider_completion,
    sse_lines,
    stall_retry_kwargs,
    task_is_cancelling,
    wait_for_response_headers,
)
from .usage import normalize_usage
from ..protocol.types import (
    CompletionBackend,
    ContentBlock,
    Message,
    MessageRole,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)

_COMPATIBILITY_EXPORTS = (
    DEFAULT_CODEX_REDIRECT_URI,
    build_authorization_url,
    exchange_authorization_code,
    extract_account_id,
)

BlockKey = tuple[int, str, int]

_CODEX_RESULT_TOKENS = re.compile(
    r"\bresulted in ([0-9][0-9,]*) tokens?\b", re.IGNORECASE
)


def _provider_prompt_tokens(detail: Mapping[str, Any]) -> int | None:
    prompt_tokens = detail.get("prompt_tokens")
    if type(prompt_tokens) is int and prompt_tokens >= 0:
        return prompt_tokens
    message = detail.get("message")
    match = _CODEX_RESULT_TOKENS.search(message) if type(message) is str else None
    return int(match.group(1).replace(",", "")) if match is not None else None


@dataclass(slots=True)
class _BlockState:
    kind: str
    state: str = "active"
    text: str = ""
    arguments: str = ""
    text_done: bool = False


@dataclass(slots=True)
class _ReassemblyState:
    mismatches: int = 0

    def reconcile(self, completed: object, streamed: object) -> None:
        if completed != streamed:
            self.mismatches += 1


@dataclass(slots=True)
class _ItemState:
    kind: str
    item_id: str
    state: str = "active"
    name: str = ""
    call_id: str = ""
    text: str = ""
    arguments: str = ""
    summary_text: str = ""
    raw_text: str = ""
    encrypted_content: str | None = None
    completed_item: dict[str, Any] | None = None
    completion_status: object = None
    blocks: set[BlockKey] = field(default_factory=set)


class _SSEDecoder:
    def __init__(self) -> None:
        self.event = "message"
        self.data: list[str] = []

    def feed(self, line: str) -> tuple[str, dict[str, Any]] | None:
        if not line:
            return self._finish()
        if line.startswith(":"):
            return None
        field, _, value = line.partition(":")
        value = value.removeprefix(" ")
        if field == "event":
            self.event = value
        elif field == "data":
            self.data.append(value)
        return None

    def finish(self) -> tuple[str, dict[str, Any]] | None:
        return self._finish()

    def _finish(self) -> tuple[str, dict[str, Any]] | None:
        if not self.data:
            self.event = "message"
            return None
        event = self.event
        value = "\n".join(self.data)
        self.event = "message"
        self.data = []
        if value == "[DONE]":
            return event, {"type": "done"}
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, TypeError) as exc:
            raise CodexStreamError("Codex returned invalid SSE JSON") from exc
        if not isinstance(payload, dict):
            raise CodexStreamError("Codex SSE payload is not an object")
        return event, payload


class CodexBackend(CompletionBackend):
    """One-completion Codex Responses streaming backend."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_CODEX_MODEL,
        base_url: str = CODEX_API_URL,
        token_store: CodexCredentialStore | None = None,
        client: httpx.AsyncClient | None = None,
        diagnostics_path: str | Path | None = None,
        stall_seconds: float = DEFAULT_STREAM_STALL_SECONDS,
        stall_retries: int = DEFAULT_STREAM_STALL_RETRIES,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.token_store = token_store or CodexCredentialStore()
        self.client = client
        self.diagnostics_path = (
            Path(diagnostics_path)
            if diagnostics_path is not None
            else self.token_store.path.parent / "logs" / "stream-diagnostics.jsonl"
        )
        self.stall_seconds, self.stall_retries = stall_seconds, stall_retries

    def complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        return self._complete(messages, tool_schemas)

    async def _complete(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
    ) -> AsyncIterator[StreamEvent]:
        attempts = retry_provider_completion(
            lambda: self._complete_once(messages, tool_schemas),
            lambda token: self._complete_once(messages, tool_schemas, token=token),
            self._refresh_token,
            lambda error: (
                isinstance(error, CodexAuthError) and error.status_code == 401
            ),
            lambda error: CodexAuthError(
                "Codex authentication failed after token refresh; run `zeta login --provider codex`",
                status_code=401,
            ),
            provider_retry_notice,
            self._record_retry_exhausted,
            **stall_retry_kwargs(self.stall_retries),
        )
        try:
            async for event in attempts:
                yield event
        finally:
            await attempts.aclose()

    async def _refresh_token(self) -> str:
        client = self.client or httpx.AsyncClient(timeout=None)
        try:
            return await self.token_store.refresh_token(client)
        finally:
            if self.client is None:
                await client.aclose()

    async def _complete_once(
        self,
        messages: Sequence[Message],
        tool_schemas: Sequence[ToolSchema],
        *,
        token: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        client = self.client or httpx.AsyncClient(timeout=None)
        stream_context: Any = None
        entered = False
        primary_exception: BaseException | None = None
        try:
            access_token = (
                token
                if token is not None
                else await self.token_store.access_token(client)
            )
            payload = build_responses_payload(
                messages,
                tool_schemas,
                model=self.model,
            )
            static_json = _cache_affinity_json(payload, messages, self.model)
            cache_key = str(uuid.uuid5(uuid.NAMESPACE_OID, static_json))
            # ponytail: pre-5.6 key routes one prefix; shard above ~15 requests/min.
            # GPT-5.6 routes its cache without a payload key; keep session-id.
            if not self.model.startswith("gpt-5.6-"):
                payload["prompt_cache_key"] = cache_key
            headers = codex_request_headers(access_token)
            headers["session-id"] = cache_key
            stream_context = client.stream(
                "POST", self.base_url, headers=headers, json=payload
            )
            response = await wait_for_response_headers(
                stream_context.__aenter__(),
                self.stall_seconds,
                "Codex",
                CodexStreamError,
            )
            entered = True
            try:
                if response.status_code >= 400:
                    body = await _read_error_body(response)
                    raise _http_error(response.status_code, body, response.headers)
                async for event in _decode_response(
                    response,
                    self.stall_seconds,
                    diagnostics_path=self.diagnostics_path,
                ):
                    yield event
            except CodexBackendError as exc:
                primary_exception = exc
            except httpx.HTTPError as exc:
                primary_exception = request_error(exc, CodexHTTPError)
            except BaseException as exc:
                if task_is_cancelling() and not is_control_exception(exc):
                    primary_exception = asyncio.CancelledError()
                    primary_exception.__cause__ = exc
                else:
                    primary_exception = exc
        except CodexBackendError as exc:
            primary_exception = exc
        except httpx.HTTPError as exc:
            primary_exception = request_error(exc, CodexHTTPError)
        except BaseException as exc:
            primary_exception = exc
        finally:
            primary_exception = await cleanup_transport(
                stream_context=stream_context,
                entered=entered,
                client=client,
                owns_client=self.client is None,
                primary_exception=primary_exception,
                http_error_type=CodexHTTPError,
            )
            if primary_exception is not None:
                raise primary_exception

    def _record_retry_exhausted(self, error: RuntimeError, retries: int) -> None:
        StreamDiagnostics.record_retry_exhausted(
            self.diagnostics_path,
            error,
            retries=retries,
        )


async def _read_error_body(response: httpx.Response, limit: int = 8192) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk[: limit - len(body)])
        if len(body) >= limit:
            break
    return bytes(body)


def _http_error(
    status_code: int,
    body: bytes,
    headers: Mapping[str, str] | None = None,
) -> CodexBackendError:
    error_type = CodexAuthError if status_code in {401, 403} else CodexHTTPError
    excerpt = error_body_excerpt(body)
    detail = f": {excerpt}" if excerpt else ""
    if error_type is CodexAuthError:
        return error_type(
            f"Codex HTTP request failed ({status_code}){detail}",
            status_code=status_code,
        )
    retry_after = retry_after_seconds(headers)
    error = error_type(
        f"Codex HTTP request failed ({status_code}){detail}",
        status_code=status_code,
        retry_after=retry_after,
    )
    if status_code == 400:
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, TypeError):
            payload = None
        if isinstance(payload, Mapping):
            detail = payload.get("error")
            if (
                isinstance(detail, Mapping)
                and detail.get("code") == "context_length_exceeded"
            ):
                error.code = "context_length_exceeded"
                error.provider_prompt_tokens = _provider_prompt_tokens(detail)
    return error


async def _decode_response(
    response: httpx.Response,
    stall_seconds: float = 0.0,
    *,
    diagnostics_path: Path | None = None,
) -> AsyncIterator[StreamEvent]:
    decoder = _SSEDecoder()
    response_state = "not-started"
    items: dict[int, _ItemState] = {}
    blocks: dict[BlockKey, _BlockState] = {}
    usage: dict[str, Any] = {}
    response_data: dict[str, Any] = {}
    reassembly = _ReassemblyState()
    finished = StreamFinished()
    async for line in sse_lines(
        response,
        stall_seconds,
        "Codex",
        CodexStreamError,
        finished=finished,
    ):
        record = decoder.feed(line)
        if record is None:
            continue
        event, payload = record
        translated, response_state = _translate_event(
            event,
            payload,
            response_state,
            items,
            blocks,
            usage,
            response_data,
            reassembly,
        )
        if translated is not None:
            if translated.type is StreamEventType.MESSAGE_END:
                finished.value = True
            yield translated
    record = decoder.finish()
    if record is not None:
        translated, response_state = _translate_event(
            record[0],
            record[1],
            response_state,
            items,
            blocks,
            usage,
            response_data,
            reassembly,
        )
        if translated is not None:
            if translated.type is StreamEventType.MESSAGE_END:
                finished.value = True
            yield translated
    if response_state != "stopped":
        if any(
            item.kind == "function_call" and item.state != "stopped"
            for item in items.values()
        ):
            raise _stream_inconsistent(
                "Codex stream ended with an incomplete function call"
            )
        raise CodexStreamError("Codex stream ended before response completion")
    if diagnostics_path is not None and reassembly.mismatches:
        StreamDiagnostics.record_reassembly_mismatches(
            diagnostics_path, count=reassembly.mismatches
        )


def _translate_event(
    event: str,
    payload: Mapping[str, Any],
    response_state: str,
    items: dict[int, _ItemState],
    blocks: dict[BlockKey, _BlockState],
    usage: dict[str, Any],
    response_data: dict[str, Any],
    reassembly: _ReassemblyState | None = None,
) -> tuple[StreamEvent | None, str]:
    reassembly = reassembly or _ReassemblyState()
    event_type = payload.get("type", event)
    if type(event_type) is not str:
        raise CodexStreamError("Codex SSE event type is invalid")
    if event_type == "done":
        return None, response_state
    if event_type in {"keepalive", "response.in_progress", "response.metadata"}:
        # No-op liveness events; tolerate them before response.created too.
        if response_state == "stopped":
            raise CodexStreamError("Codex event follows response completion")
        return None, response_state
    if event_type == "error":
        detail = payload.get("error")
        if not isinstance(detail, Mapping):
            detail = {key: value for key, value in payload.items() if key != "type"}
        error = decode_stream_error(detail)
        failure = CodexStreamError(
            error.message,
            code=error.code,
            status_code=error.status_code,
            retryable=error.retry_reason is not None,
            retry_reason=error.retry_reason,
        )
        failure.provider_prompt_tokens = _provider_prompt_tokens(detail)
        raise failure
    if event_type == "response.created":
        if response_state != "not-started":
            raise CodexStreamError("Codex response.created is duplicated")
        response = payload.get("response")
        if not isinstance(response, Mapping):
            raise CodexStreamError("Codex response.created is invalid")
        response_data.update(
            {key: response[key] for key in ("id", "model", "status") if key in response}
        )
        return StreamEvent(
            StreamEventType.MESSAGE_START, data=dict(response_data)
        ), "started"
    _require_response_started(response_state, event_type)
    if response_state == "stopped":
        raise CodexStreamError("Codex event follows response completion")
    if event_type in {
        "response.completed",
        "response.done",
        "response.incomplete",
        "response.failed",
    }:
        if response_state != "started":
            raise CodexStreamError("Codex response completion has no active response")
        return (
            _decode_terminal_response(
                event_type,
                payload,
                items,
                blocks,
                usage,
                response_data,
            ),
            "stopped",
        )
    if event_type == "response.output_item.added":
        index = _output_index(payload)
        if index in items:
            raise CodexStreamError("Codex output item is duplicated")
        item = payload.get("item")
        if not isinstance(item, Mapping):
            raise CodexStreamError("Codex output item is invalid")
        kind = item.get("type")
        item_id = item.get("id")
        if kind not in {"message", "reasoning", "function_call"}:
            raise CodexStreamError("unsupported Codex output item type")
        if type(item_id) is not str or not item_id:
            raise CodexStreamError("Codex output item id is invalid")
        item_state = _ItemState(
            kind=kind,
            item_id=item_id,
            name=item.get("name") if type(item.get("name")) is str else "",
            call_id=item.get("call_id") if type(item.get("call_id")) is str else "",
            encrypted_content=(
                item.get("encrypted_content")
                if type(item.get("encrypted_content")) is str
                else None
            ),
        )
        if kind == "function_call" and (not item_state.name or not item_state.call_id):
            raise CodexStreamError("Codex function call metadata is incomplete")
        items[index] = item_state
        if kind == "function_call":
            blocks[(index, "tool_call", -1)] = _BlockState("tool_call")
            item_state.blocks.add((index, "tool_call", -1))
        return None, response_state
    if event_type == "response.content_part.added":
        index = _output_index(payload)
        item = _active_item(items, index, payload)
        content_index = _content_index(payload)
        part = payload.get("part")
        kind = part.get("type") if isinstance(part, Mapping) else None
        if item.kind == "reasoning" and kind == "reasoning_text":
            raw_key = (index, "thinking_raw", content_index)
            if raw_key in blocks:
                raise CodexStreamError("Codex content block is duplicated")
            blocks[raw_key] = _BlockState("thinking_raw")
            item.blocks.add(raw_key)
            return None, response_state
        if item.kind != "message" or kind not in {"output_text", "refusal"}:
            raise CodexStreamError("Codex content part has the wrong item type")
        message_key = (index, "message", content_index)
        if message_key in blocks:
            raise CodexStreamError("Codex content block is duplicated")
        blocks[message_key] = _BlockState(kind)
        item.blocks.add(message_key)
        return None, response_state
    if event_type == "response.reasoning_summary_part.added":
        index = _output_index(payload)
        item = _active_item(items, index, payload)
        if item.kind != "reasoning":
            raise CodexStreamError("Codex reasoning part has the wrong item type")
        part = payload.get("part")
        if part is not None and (
            not isinstance(part, Mapping) or part.get("type") != "summary_text"
        ):
            raise CodexStreamError("Codex reasoning part has the wrong item type")
        summary_index = payload.get("summary_index")
        if type(summary_index) is not int or summary_index < 0:
            raise CodexStreamError("Codex reasoning summary index is invalid")
        key = (index, "thinking", summary_index)
        if key in blocks:
            raise CodexStreamError("Codex reasoning block is duplicated")
        blocks[key] = _BlockState("thinking")
        item.blocks.add(key)
        return None, response_state
    if event_type in {
        "response.output_text.delta",
        "response.refusal.delta",
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
        "response.reasoning_text.delta",
    }:
        return _translate_delta(event_type, payload, items, blocks), response_state
    if event_type in {
        "response.output_text.done",
        "response.refusal.done",
        "response.content_part.done",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.done",
        "response.function_call_arguments.done",
    }:
        _finish_block(event_type, payload, items, blocks, reassembly)
        return None, response_state
    if event_type == "response.reasoning_summary_part.done":
        _finish_reasoning_summary_part(payload, items, blocks, reassembly)
        return None, response_state
    if event_type == "response.output_item.done":
        index = _output_index(payload)
        item = _active_item(items, index, payload)
        complete = payload.get("item")
        if not isinstance(complete, Mapping):
            raise CodexStreamError("Codex completed output item is invalid")
        _merge_completed_item(item, complete, blocks, reassembly)
        # The completed item is authoritative and has just been checked against
        # every streamed block, so it closes blocks whose own stop event the
        # server omitted (seen live for message parts and reasoning parts).
        for key in item.blocks:
            blocks[key].state = "stopped"
        item.state = "stopped"
        if item.kind == "function_call" and (
            item.completion_status is None or item.completion_status == "completed"
        ):
            return (
                StreamEvent(
                    StreamEventType.MESSAGE_UPDATE,
                    data={"tool_call_completed": True, "index": index},
                ),
                response_state,
            )
        return None, response_state
    raise CodexStreamError("unsupported Codex SSE event type")


def _require_response_started(state: str, event_type: str) -> None:
    if state == "not-started":
        raise CodexStreamError("Codex event precedes response.created")


def _output_index(payload: Mapping[str, Any]) -> int:
    value = payload.get("output_index")
    if type(value) is not int or value < 0:
        raise CodexStreamError("Codex output index is invalid")
    return value


def _content_index(payload: Mapping[str, Any]) -> int:
    value = payload.get("content_index")
    if type(value) is not int or value < 0:
        raise CodexStreamError("Codex content index is invalid")
    return value


def _active_item(
    items: Mapping[int, _ItemState],
    index: int,
    payload: Mapping[str, Any] | None = None,
) -> _ItemState:
    item = items.get(index)
    if item is None:
        raise CodexStreamError(f"Codex event references unknown output item: {index}")
    if item.state != "active":
        raise CodexStreamError(f"Codex event references stopped output item: {index}")
    if payload is not None:
        _validate_item_identity(payload, item)
    return item


def _validate_item_identity(payload: Mapping[str, Any], item: _ItemState) -> None:
    if "item_id" in payload and payload["item_id"] != item.item_id:
        raise CodexStreamError("Codex event item id does not match output item")


def _translate_delta(
    event_type: str,
    payload: Mapping[str, Any],
    items: Mapping[int, _ItemState],
    blocks: dict[BlockKey, _BlockState],
) -> StreamEvent:
    index = _output_index(payload)
    item = _active_item(items, index, payload)
    if event_type == "response.function_call_arguments.delta":
        key = (index, "tool_call", -1)
        block = blocks.get(key)
        if item.kind != "function_call" or block is None or block.state != "active":
            raise CodexStreamError("Codex tool delta references an inactive block")
        delta = payload.get("delta")
        if type(delta) is not str:
            raise CodexStreamError("Codex tool delta is invalid")
        block.arguments += delta
        item.arguments += delta
        arguments = _parse_partial_object(block.arguments)
        return StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            tool_call=ToolCall(item.call_id, item.name, arguments),
            data={"tool_call_delta": delta, "index": index},
        )
    if event_type in {
        "response.reasoning_summary_text.delta",
        "response.reasoning_text.delta",
    }:
        if event_type == "response.reasoning_text.delta":
            content_index = _content_index(payload)
            key = (index, "thinking_raw", content_index)
            block = blocks.get(key)
            if block is None:
                block = _BlockState("thinking_raw")
                blocks[key] = block
                item.blocks.add(key)
            if item.kind != "reasoning" or block.kind != "thinking_raw":
                raise CodexStreamError(
                    "Codex reasoning delta references an inactive block"
                )
            if block.state != "active" or block.text_done:
                raise CodexStreamError(
                    "Codex reasoning delta references an inactive block"
                )
            delta = payload.get("delta")
            if type(delta) is not str:
                raise CodexStreamError("Codex reasoning delta is invalid")
            block.text += delta
            item.raw_text += delta
            # Keep raw reasoning in provider state for Codex replay. Do not
            # attach it to the shared stream because it is never display-safe.
            return StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                data={"index": (index, "raw", content_index)},
            )
        summary_index = payload.get("summary_index")
        if type(summary_index) is not int or summary_index < 0:
            raise CodexStreamError("Codex reasoning summary index is invalid")
        key = (index, "thinking", summary_index)
        block = blocks.get(key)
        if (
            item.kind != "reasoning"
            or block is None
            or block.state != "active"
            or block.text_done
        ):
            raise CodexStreamError("Codex reasoning delta references an inactive block")
        delta = payload.get("delta")
        if type(delta) is not str:
            raise CodexStreamError("Codex reasoning delta is invalid")
        block.text += delta
        item.summary_text += delta
        return StreamEvent(
            StreamEventType.MESSAGE_UPDATE,
            content=ThinkingContent(delta),
            data={"index": (index, "summary", summary_index)},
        )
    content_index = _content_index(payload)
    expected_kind = (
        "refusal" if event_type == "response.refusal.delta" else "output_text"
    )
    key = (index, "message", content_index)
    block = blocks.get(key)
    if (
        item.kind != "message"
        or block is None
        or block.kind != expected_kind
        or block.state != "active"
    ):
        raise CodexStreamError("Codex text delta references an inactive block")
    delta = payload.get("delta")
    if type(delta) is not str:
        raise CodexStreamError("Codex text delta is invalid")
    block.text += delta
    item.text += delta
    return StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=delta)


def _finish_block(
    event_type: str,
    payload: Mapping[str, Any],
    items: Mapping[int, _ItemState],
    blocks: dict[BlockKey, _BlockState],
    reassembly: _ReassemblyState,
) -> None:
    index = _output_index(payload)
    item = _active_item(items, index, payload)
    if event_type == "response.function_call_arguments.done":
        key = (index, "tool_call", -1)
        expected_kind = "tool_call"
    elif event_type == "response.reasoning_summary_text.done":
        summary_index = payload.get("summary_index")
        if type(summary_index) is not int or summary_index < 0:
            raise CodexStreamError("Codex reasoning summary index is invalid")
        key = (index, "thinking", summary_index)
        expected_kind = "thinking"
    elif event_type == "response.reasoning_text.done":
        key = (index, "thinking_raw", _content_index(payload))
        expected_kind = "thinking_raw"
    else:
        content_index = _content_index(payload)
        expected_kind = ""
        if event_type == "response.content_part.done":
            part = payload.get("part")
            wire_kind = part.get("type") if isinstance(part, Mapping) else None
            expected_kind = (
                "thinking_raw" if wire_kind == "reasoning_text" else wire_kind or ""
            )
            message_key = (index, "message", content_index)
            candidates = (
                (index, expected_kind, content_index),
                message_key,
                (index, "thinking_raw", content_index),
            )
            key = next(
                (candidate for candidate in candidates if candidate in blocks),
                candidates[0],
            )
        else:
            expected_kind = (
                "refusal" if event_type == "response.refusal.done" else "output_text"
            )
            key = (index, "message", content_index)
    block = blocks.get(key)
    if block is None or (
        event_type != "response.content_part.done" and block.kind != expected_kind
    ):
        raise CodexStreamError("Codex block stop references an unknown block")
    if block.state != "active":
        raise CodexStreamError("Codex block stop is duplicated")
    if event_type == "response.content_part.done":
        part = payload.get("part")
        wire_kind = "reasoning_text" if block.kind == "thinking_raw" else block.kind
        if part is not None and (
            not isinstance(part, Mapping) or part.get("type") != wire_kind
        ):
            raise CodexStreamError("Codex content part has the wrong item type")
    if event_type.endswith(".done"):
        complete_text = (
            payload.get("refusal")
            if event_type == "response.refusal.done"
            else payload.get("text")
        )
        if type(complete_text) is str and block.kind in {
            "output_text",
            "refusal",
            "thinking",
            "thinking_raw",
        }:
            _reconcile_completed_text(block, complete_text, reassembly)
            _rebuild_item_text(item, blocks)
        complete_args = payload.get("arguments")
        if type(complete_args) is str and block.kind == "tool_call":
            if block.arguments and complete_args != block.arguments:
                raise _stream_inconsistent(
                    "Codex completed arguments do not match deltas"
                )
            block.arguments = complete_args
            item.arguments = complete_args
    if event_type == "response.reasoning_summary_text.done":
        if block.text_done:
            raise CodexStreamError("Codex reasoning text stop is duplicated")
        block.text_done = True
    elif event_type == "response.reasoning_text.done":
        if block.text_done:
            raise CodexStreamError("Codex reasoning text stop is duplicated")
        block.text_done = True
    elif event_type in {
        "response.content_part.done",
        "response.function_call_arguments.done",
    }:
        block.state = "stopped"


def _reconcile_completed_text(
    block: _BlockState,
    complete_text: str,
    reassembly: _ReassemblyState,
) -> None:
    if block.text:
        reassembly.reconcile(complete_text, block.text)
    block.text = complete_text


def _rebuild_item_text(
    item: _ItemState, blocks: Mapping[BlockKey, _BlockState]
) -> None:
    def joined(kind: str) -> str:
        return "".join(
            blocks[key].text
            for key in sorted(item.blocks)
            if key[1] == kind
        )

    if item.kind == "message":
        item.text = joined("message")
    elif item.kind == "reasoning":
        item.summary_text = joined("thinking")
        item.raw_text = joined("thinking_raw")


def _require_tool_match(actual: Any, streamed: Any, error: str) -> None:
    if actual != streamed:
        raise _stream_inconsistent(error)


def _stream_inconsistent(message: str) -> CodexStreamError:
    return CodexStreamError(
        message,
        retryable=True,
        retry_reason="stream_inconsistent",
    )


def _format_provider_status(status: object) -> str:
    if type(status) is str:
        bounded = status.encode("utf-8")[:32].decode("utf-8", errors="ignore")
        return repr(bounded)
    return f"<{type(status).__name__}>"


def _decode_terminal_response(
    event_type: str,
    payload: Mapping[str, Any],
    items: Mapping[int, _ItemState],
    blocks: Mapping[BlockKey, _BlockState],
    usage: dict[str, Any],
    response_data: dict[str, Any],
) -> StreamEvent:
    response = payload.get("response")
    if event_type == "response.failed":
        detail = response.get("error") if isinstance(response, Mapping) else None
        if not isinstance(detail, Mapping):
            detail = {}
        error = decode_stream_error(detail)
        failure = CodexStreamError(
            error.message,
            code=error.code,
            status_code=error.status_code,
            retryable=error.retry_reason is not None,
            retry_reason=error.retry_reason,
        )
        failure.provider_prompt_tokens = _provider_prompt_tokens(detail)
        raise failure
    if response is not None and not isinstance(response, Mapping):
        raise CodexStreamError("Codex response completion is invalid")
    if any(item.state != "stopped" for item in items.values()):
        raise _stream_inconsistent("Codex response completed with open items")
    if any(block.state != "stopped" for block in blocks.values()):
        raise _stream_inconsistent("Codex response completed with open blocks")

    response_status = response.get("status") if isinstance(response, Mapping) else None
    incomplete = event_type == "response.incomplete"
    if incomplete:
        if response_status is not None and response_status != "incomplete":
            raise CodexStreamError(
                "Codex incomplete response status "
                f"{_format_provider_status(response_status)} is invalid"
            )
        details = (
            response.get("incomplete_details")
            if isinstance(response, Mapping)
            else None
        )
        reason = details.get("reason") if isinstance(details, Mapping) else None
        if reason == "max_output_tokens":
            stop_reason = "max_tokens"
        elif reason == "content_filter":
            stop_reason = "content_filter"
        else:
            raise CodexStreamError(
                f"Codex response was incomplete: {_format_provider_status(reason)}"
            )
    else:
        if response_status is not None and response_status != "completed":
            raise _stream_inconsistent(
                "Codex response completion status "
                f"{_format_provider_status(response_status)} is invalid"
            )
        invalid_item = next(
            (
                item
                for item in items.values()
                if item.completion_status is not None
                and item.completion_status != "completed"
            ),
            None,
        )
        if invalid_item is not None:
            raise _stream_inconsistent(
                f"Codex completed {invalid_item.kind} status "
                f"{_format_provider_status(invalid_item.completion_status)} is invalid"
            )
        stop_reason = "end_turn"

    if isinstance(response, Mapping):
        response_data.update(
            {key: response[key] for key in ("id", "status") if key in response}
        )
        response_usage = response.get("usage")
        if response_usage is not None and not isinstance(response_usage, Mapping):
            raise CodexStreamError("Codex response usage is invalid")
        if isinstance(response_usage, Mapping):
            usage.update(response_usage)

    content: list[ContentBlock] = []
    output_items: list[dict[str, Any]] = []
    for index in sorted(items):
        item = items[index]
        if item.completed_item is None:
            raise CodexStreamError("Codex output item has no completed item")
        if (
            incomplete
            and item.kind == "function_call"
            and item.completion_status != "completed"
        ):
            continue
        content.extend(_complete_item(item))
        output_items.append(dict(item.completed_item))
    return StreamEvent(
        StreamEventType.MESSAGE_END,
        message=Message(
            MessageRole.ASSISTANT,
            content,
            metadata={"codex_output_items": output_items},
        ),
        data={
            "usage": normalize_usage(usage),
            **response_data,
            "stop_reason": stop_reason,
        },
    )


def _finish_reasoning_summary_part(
    payload: Mapping[str, Any],
    items: Mapping[int, _ItemState],
    blocks: dict[BlockKey, _BlockState],
    reassembly: _ReassemblyState,
) -> None:
    index = _output_index(payload)
    item = _active_item(items, index, payload)
    summary_index = payload.get("summary_index")
    if type(summary_index) is not int or summary_index < 0:
        raise CodexStreamError("Codex reasoning summary index is invalid")
    key = (index, "thinking", summary_index)
    block = blocks.get(key)
    if item.kind != "reasoning" or block is None or block.kind != "thinking":
        raise CodexStreamError(
            "Codex reasoning summary stop references an unknown block"
        )
    if block.state != "active":
        raise CodexStreamError("Codex reasoning summary stop is duplicated")
    part = payload.get("part")
    if part is not None and not isinstance(part, Mapping):
        raise CodexStreamError("Codex reasoning summary part is invalid")
    if isinstance(part, Mapping) and part.get("type") != "summary_text":
        raise CodexStreamError("Codex reasoning part has the wrong item type")
    complete_text = part.get("text") if isinstance(part, Mapping) else None
    if complete_text is not None and type(complete_text) is not str:
        raise CodexStreamError("Codex reasoning summary text is invalid")
    if isinstance(complete_text, str):
        _reconcile_completed_text(block, complete_text, reassembly)
        _rebuild_item_text(item, blocks)
    block.state = "stopped"


def _merge_completed_item(
    item: _ItemState,
    complete: Mapping[str, Any],
    blocks: Mapping[BlockKey, _BlockState],
    reassembly: _ReassemblyState,
) -> None:
    if complete.get("id") != item.item_id:
        raise CodexStreamError("Codex completed item id does not match output item")
    if complete.get("type") != item.kind:
        raise CodexStreamError("Codex completed item type does not match output item")
    item.completion_status = complete.get("status")
    if item.kind == "message":
        if complete.get("role") != "assistant":
            raise CodexStreamError("Codex completed message metadata is invalid")
        content = complete.get("content")
        if not isinstance(content, list) or not content:
            raise CodexStreamError("Codex completed message content is invalid")
        completed_parts: list[tuple[str, str]] = []
        for part in content:
            if not isinstance(part, Mapping):
                raise CodexStreamError("Codex completed message part is invalid")
            part_type = part.get("type")
            text_key = "text" if part_type == "output_text" else "refusal"
            if (
                part_type not in {"output_text", "refusal"}
                or type(part.get(text_key)) is not str
            ):
                raise CodexStreamError("Codex completed message part is invalid")
            completed_parts.append((part_type, part[text_key]))
        streamed_parts = [
            (blocks[key].kind, blocks[key].text)
            for key in sorted(item.blocks)
            if key[1] == "message"
        ]
        reassembly.reconcile(completed_parts, streamed_parts)
        item.text = "".join(text for _, text in completed_parts)
    elif item.kind == "reasoning":
        summary = complete.get("summary")
        if not isinstance(summary, list):
            raise CodexStreamError("Codex completed reasoning metadata is invalid")
        completed_summary: list[str] = []
        for part in summary:
            if (
                not isinstance(part, Mapping)
                or part.get("type") != "summary_text"
                or type(part.get("text")) is not str
            ):
                raise CodexStreamError("Codex completed reasoning summary is invalid")
            completed_summary.append(part["text"])
        streamed_summary = [
            blocks[key].text
            for key in sorted(item.blocks)
            if key[1] == "thinking"
        ]
        reassembly.reconcile(completed_summary, streamed_summary)
        item.summary_text = "".join(completed_summary)

        content = complete.get("content")
        completed_raw: list[str] = []
        if content is not None:
            if not isinstance(content, list):
                raise CodexStreamError("Codex completed reasoning content is invalid")
            for part in content:
                if (
                    not isinstance(part, Mapping)
                    or part.get("type") != "reasoning_text"
                    or type(part.get("text")) is not str
                ):
                    raise CodexStreamError(
                        "Codex completed reasoning content is invalid"
                    )
                completed_raw.append(part["text"])
        streamed_raw = [
            blocks[key].text
            for key in sorted(item.blocks)
            if key[1] == "thinking_raw"
        ]
        reassembly.reconcile(completed_raw, streamed_raw)
        item.raw_text = "".join(completed_raw)
        encrypted = complete.get("encrypted_content")
        if encrypted is not None and (type(encrypted) is not str or not encrypted):
            raise CodexStreamError("Codex completed reasoning metadata is invalid")
        if type(encrypted) is str:
            item.encrypted_content = encrypted
    else:
        _require_tool_match(
            complete.get("call_id"),
            item.call_id,
            "Codex completed tool metadata is invalid",
        )
        _require_tool_match(
            complete.get("name"), item.name, "Codex completed tool metadata is invalid"
        )
        arguments = complete.get("arguments")
        if type(arguments) is not str:
            raise CodexStreamError("Codex completed tool arguments are invalid")
        _require_tool_match(
            arguments,
            item.arguments,
            "Codex completed tool does not match its deltas",
        )
        item.arguments = arguments
    item.completed_item = dict(complete)


def _complete_item(item: _ItemState) -> list[ContentBlock]:
    if item.kind == "message":
        return [TextContent(item.text)] if item.text else []
    if item.kind == "reasoning":
        if not item.summary_text and not item.encrypted_content and not item.raw_text:
            return []
        return [ThinkingContent(item.summary_text, item.encrypted_content)]
    arguments = _parse_complete_object(item.arguments)
    return [ToolUseContent(ToolCall(item.call_id, item.name, arguments))]


def _parse_partial_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_complete_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value or "{}")
    except json.JSONDecodeError as exc:
        raise CodexStreamError("Codex tool arguments are invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise CodexStreamError("Codex tool arguments are not an object")
    return parsed
