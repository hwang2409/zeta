"""ChatGPT plan OAuth and Responses SSE support for Codex."""

from __future__ import annotations

import asyncio
import base64
import binascii
import json
import math
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from .auth import OAuthCredentialStore, OAuthTokens
from .transport import (
    cleanup_transport,
    is_control_exception,
    request_error,
    task_is_cancelling,
)
from .types import (
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

CODEX_API_URL = "https://chatgpt.com/backend-api/codex/responses"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
DEFAULT_CODEX_MODEL = "gpt-5.4"
JWT_AUTH_CLAIM = "https://api.openai.com/auth"


class CodexBackendError(RuntimeError):
    """Base class for errors that the agent loop can report."""

    code = "backend_error"


class CodexAuthError(CodexBackendError):
    """Raised when ChatGPT subscription credentials are missing or invalid."""

    code = "auth_error"


class CodexHTTPError(CodexBackendError):
    """Raised when the ChatGPT backend returns an unsuccessful response."""

    code = "http_error"


class CodexStreamError(CodexBackendError):
    """Raised when a Responses SSE stream violates its lifecycle contract."""

    code = "stream_error"


def _first_string(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if type(candidate) is str and candidate:
            return candidate
    return None


def _extract_codex_tokens(value: Any) -> OAuthTokens:
    if not isinstance(value, Mapping):
        raise TypeError("Codex credentials are not an object")
    nested = value.get("tokens")
    if isinstance(nested, Mapping):
        value = nested
    access = _first_string(value, "access_token", "accessToken", "access")
    refresh = _first_string(value, "refresh_token", "refreshToken", "refresh")
    if not access or not refresh:
        raise ValueError("Codex credentials are incomplete")
    payload = _jwt_payload(access)
    expiry = payload.get("exp")
    expires_at = (
        float(expiry)
        if type(expiry) in {int, float} and math.isfinite(float(expiry))
        else 0.0
    )
    return OAuthTokens(access, refresh, expires_at)


class CodexCredentialStore(OAuthCredentialStore):
    """Owns zeta's Codex OAuth file and reads ~/.codex/auth.json only to bootstrap."""

    auth_error_type = CodexAuthError
    http_error_type = CodexHTTPError
    provider_label = "Codex"

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        codex_auth: str | Path | None = None,
        token_url: str = CODEX_TOKEN_URL,
    ) -> None:
        super().__init__(path or Path.home() / ".zeta" / "codex-oauth.json", token_url=token_url)
        self.codex_auth = Path(codex_auth or Path.home() / ".codex" / "auth.json")

    def bootstrap(self) -> OAuthTokens | None:
        if not self.codex_auth.exists():
            return None
        try:
            with self.codex_auth.open() as handle:
                return _extract_codex_tokens(json.load(handle))
        except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CodexAuthError("Codex credentials could not be read") from exc

    async def refresh(self, refresh_token: str, client: httpx.AsyncClient) -> OAuthTokens:
        try:
            response = await client.post(
                self.token_url,
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": refresh_token,
                    "client_id": CODEX_CLIENT_ID,
                },
                headers={"accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise CodexAuthError("Codex OAuth token refresh failed") from exc
        if response.status_code >= 400:
            raise CodexHTTPError(
                f"Codex OAuth token refresh failed with HTTP {response.status_code}"
            )
        try:
            value = response.json()
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise CodexAuthError("Codex OAuth token response is invalid") from exc
        if not isinstance(value, Mapping):
            raise CodexAuthError("Codex OAuth token response is invalid")
        access = _first_string(value, "access_token", "accessToken", "access")
        refresh = _first_string(value, "refresh_token", "refreshToken", "refresh")
        expires_in = value.get("expires_in", value.get("expiresIn"))
        if not access or not refresh or type(expires_in) not in {int, float}:
            raise CodexAuthError("Codex OAuth token response is invalid")
        return OAuthTokens(access, refresh, time.time() + float(expires_in) - 300)


def extract_account_id(access_token: str) -> str:
    """Derive the ChatGPT account id from the access token claim."""

    try:
        payload = _jwt_payload(access_token)
        account = payload[JWT_AUTH_CLAIM]["chatgpt_account_id"]
    except (
        KeyError,
        UnicodeDecodeError,
        ValueError,
        TypeError,
        binascii.Error,
        json.JSONDecodeError,
    ):
        raise CodexAuthError("Codex access token has no ChatGPT account id") from None
    if type(account) is not str or not account:
        raise CodexAuthError("Codex access token has no ChatGPT account id")
    return account


def _jwt_payload(access_token: str) -> Mapping[str, Any]:
    parts = access_token.split(".")
    if len(parts) != 3:
        raise ValueError
    encoded = parts[1] + "=" * (-len(parts[1]) % 4)
    payload = json.loads(base64.urlsafe_b64decode(encoded).decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError
    return payload


def _wire_text(blocks: Sequence[ContentBlock], *, output: bool) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            result.append({"type": "output_text" if output else "input_text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            if output:
                reasoning: dict[str, Any] = {
                    "type": "reasoning",
                    "summary": [{"type": "summary_text", "text": block.text}],
                }
                if block.signature:
                    reasoning["encrypted_content"] = block.signature
                result.append(reasoning)
        elif isinstance(block, ToolUseContent):
            if output:
                result.append(
                    {
                        "type": "function_call",
                        "call_id": block.tool_call.id,
                        "name": block.tool_call.name,
                        "arguments": json.dumps(
                            block.tool_call.arguments, separators=(",", ":")
                        ),
                    }
                )
            else:
                raise CodexHTTPError("tool calls are not valid user content")
        else:
            raise CodexHTTPError("unsupported zeta content block")
    return result


def build_responses_payload(
    messages: Sequence[Message],
    tool_schemas: Sequence[ToolSchema],
    *,
    model: str,
    max_output_tokens: int,
) -> dict[str, Any]:
    instructions: list[str] = []
    input_items: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            instructions.extend(
                block.text for block in message.content if isinstance(block, TextContent)
            )
        elif message.role is MessageRole.TOOL_RESULT:
            if message.tool_result is None:
                raise CodexHTTPError("tool result message is missing its result")
            input_items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.tool_result.tool_call_id,
                    "output": message.tool_result.content,
                }
            )
        else:
            output = message.role is MessageRole.ASSISTANT
            if output and "codex_output_items" in message.metadata:
                replayed = message.metadata["codex_output_items"]
                if type(replayed) is not list or any(
                    not isinstance(item, Mapping) for item in replayed
                ):
                    raise CodexHTTPError("Codex replay output items are invalid")
                input_items.extend(dict(item) for item in replayed)
                continue
            wire_blocks = _wire_text(message.content, output=output)
            if not output:
                input_items.append({"role": "user", "content": wire_blocks})
                continue
            text_blocks: list[dict[str, Any]] = []
            for block in wire_blocks:
                if block.get("type") == "output_text":
                    text_blocks.append({"type": "input_text", "text": block["text"]})
                    continue
                if text_blocks:
                    input_items.append({"role": "assistant", "content": text_blocks})
                    text_blocks = []
                input_items.append(block)
            if text_blocks:
                input_items.append({"role": "assistant", "content": text_blocks})

    tools = []
    for schema in tool_schemas:
        name = schema.get("name")
        if type(name) is not str or not name:
            raise CodexHTTPError("tool schema name must be a nonempty string")
        parameters = schema.get("parameters", schema.get("input_schema"))
        if not isinstance(parameters, Mapping):
            raise CodexHTTPError("tool schema parameters must be an object")
        tool: dict[str, Any] = {
            "type": "function",
            "name": name,
            "parameters": dict(parameters),
            "strict": False,
        }
        description = schema.get("description")
        if type(description) is str:
            tool["description"] = description
        tools.append(tool)

    payload: dict[str, Any] = {
        "model": model,
        "store": False,
        "stream": True,
        "max_output_tokens": max_output_tokens,
        "instructions": "\n\n".join(instructions) or "You are a helpful assistant.",
        "input": input_items,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning": {"summary": "auto"},
        "include": ["reasoning.encrypted_content"],
    }
    if tools:
        payload["tools"] = tools
    return payload


@dataclass(slots=True)
class _BlockState:
    kind: str
    state: str = "active"
    text: str = ""
    arguments: str = ""
    text_done: bool = False


@dataclass(slots=True)
class _ItemState:
    kind: str
    item_id: str
    state: str = "active"
    name: str = ""
    call_id: str = ""
    text: str = ""
    arguments: str = ""
    thinking: str = ""
    encrypted_content: str | None = None
    completed_item: dict[str, Any] | None = None
    blocks: set[tuple[int, int]] = field(default_factory=set)


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
        max_output_tokens: int = 8192,
        base_url: str = CODEX_API_URL,
        token_store: CodexCredentialStore | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.max_output_tokens = max_output_tokens
        self.base_url = base_url.rstrip("/")
        self.token_store = token_store or CodexCredentialStore()
        self.client = client

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
        client = self.client or httpx.AsyncClient(timeout=None)
        stream_context: Any = None
        entered = False
        primary_exception: BaseException | None = None
        try:
            access_token = await self.token_store.access_token(client)
            account_id = extract_account_id(access_token)
            payload = build_responses_payload(
                messages,
                tool_schemas,
                model=self.model,
                max_output_tokens=self.max_output_tokens,
            )
            headers = {
                "accept": "text/event-stream",
                "authorization": f"Bearer {access_token}",
                "chatgpt-account-id": account_id,
                "content-type": "application/json",
                "originator": "zeta",
                "openai-beta": "responses=experimental",
                "user-agent": "zeta/0.1",
            }
            stream_context = client.stream(
                "POST", self.base_url, headers=headers, json=payload
            )
            response = await stream_context.__aenter__()
            entered = True
            try:
                if response.status_code >= 400:
                    body = await _read_error_body(response)
                    raise _http_error(response.status_code, body)
                async for event in _decode_response(response):
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


async def _read_error_body(response: httpx.Response, limit: int = 8192) -> bytes:
    body = bytearray()
    async for chunk in response.aiter_bytes():
        body.extend(chunk[: limit - len(body)])
        if len(body) >= limit:
            break
    return bytes(body)


def _http_error(status_code: int, body: bytes) -> CodexBackendError:
    error_type = CodexAuthError if status_code in {401, 403} else CodexHTTPError
    return error_type(f"Codex HTTP request failed ({status_code})")


async def _decode_response(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    decoder = _SSEDecoder()
    response_state = "not-started"
    items: dict[int, _ItemState] = {}
    blocks: dict[tuple[int, int], _BlockState] = {}
    usage: dict[str, Any] = {}
    response_data: dict[str, Any] = {}
    async for line in response.aiter_lines():
        record = decoder.feed(line)
        if record is None:
            continue
        event, payload = record
        translated, response_state = _translate_event(
            event, payload, response_state, items, blocks, usage, response_data
        )
        if translated is not None:
            yield translated
    record = decoder.finish()
    if record is not None:
        translated, response_state = _translate_event(
            record[0], record[1], response_state, items, blocks, usage, response_data
        )
        if translated is not None:
            yield translated
    if response_state != "stopped":
        raise CodexStreamError("Codex stream ended before response completion")


def _translate_event(
    event: str,
    payload: Mapping[str, Any],
    response_state: str,
    items: dict[int, _ItemState],
    blocks: dict[tuple[int, int], _BlockState],
    usage: dict[str, Any],
    response_data: dict[str, Any],
) -> tuple[StreamEvent | None, str]:
    event_type = payload.get("type", event)
    if type(event_type) is not str:
        raise CodexStreamError("Codex SSE event type is invalid")
    if event_type == "done":
        return None, response_state
    if event_type in {"keepalive", "response.in_progress", "response.metadata"}:
        _require_response_started(response_state, event_type)
        if response_state == "stopped":
            raise CodexStreamError(
                f"Codex event follows response completion: {event_type}"
            )
        return None, response_state
    if event_type == "error":
        detail = payload.get("error")
        if not isinstance(detail, Mapping) and not isinstance(payload.get("message"), str):
            raise CodexStreamError("Codex stream error payload is invalid")
        raise CodexStreamError("Codex response reported an error")
    if event_type == "response.created":
        if response_state != "not-started":
            raise CodexStreamError("Codex response.created is duplicated")
        response = payload.get("response")
        if not isinstance(response, Mapping):
            raise CodexStreamError("Codex response.created is invalid")
        response_data.update(
            {
                key: response[key]
                for key in ("id", "model", "status")
                if key in response
            }
        )
        return StreamEvent(StreamEventType.MESSAGE_START, data=dict(response_data)), "started"
    _require_response_started(response_state, event_type)
    if response_state == "stopped":
        raise CodexStreamError(f"Codex event follows response completion: {event_type}")
    if event_type == "response.failed":
        raise CodexStreamError("Codex response failed")
    if event_type == "response.incomplete":
        raise CodexStreamError("Codex response was incomplete")
    if event_type in {"response.completed", "response.done"}:
        if response_state != "started":
            raise CodexStreamError("Codex response completion has no active response")
        if any(item.state != "stopped" for item in items.values()):
            raise CodexStreamError("Codex response completed with open items")
        if any(block.state != "stopped" for block in blocks.values()):
            raise CodexStreamError("Codex response completed with open blocks")
        response = payload.get("response")
        if response is not None and not isinstance(response, Mapping):
            raise CodexStreamError("Codex response completion is invalid")
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
            content.extend(_complete_item(item))
            if item.completed_item is None:
                raise CodexStreamError("Codex output item has no completed item")
            output_items.append(dict(item.completed_item))
        return (
            StreamEvent(
                StreamEventType.MESSAGE_END,
                message=Message(
                    MessageRole.ASSISTANT,
                    content,
                    metadata={"codex_output_items": output_items},
                ),
                data={"usage": dict(usage), **response_data},
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
            raise CodexStreamError(f"unsupported Codex output item: {kind}")
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
            blocks[(index, -1)] = _BlockState("tool_call")
            item_state.blocks.add((index, -1))
        return None, response_state
    if event_type == "response.content_part.added":
        index = _output_index(payload)
        item = _active_item(items, index, payload)
        content_index = _content_index(payload)
        key = (index, content_index)
        if key in blocks:
            raise CodexStreamError("Codex content block is duplicated")
        part = payload.get("part")
        kind = part.get("type") if isinstance(part, Mapping) else None
        if item.kind != "message" or kind != "output_text":
            raise CodexStreamError("Codex content part has the wrong item type")
        blocks[key] = _BlockState("text")
        item.blocks.add(key)
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
        key = (index, summary_index)
        if key in blocks:
            raise CodexStreamError("Codex reasoning block is duplicated")
        blocks[key] = _BlockState("thinking")
        item.blocks.add(key)
        return None, response_state
    if event_type in {
        "response.output_text.delta",
        "response.reasoning_summary_text.delta",
        "response.function_call_arguments.delta",
        "response.reasoning_text.delta",
    }:
        return _translate_delta(event_type, payload, items, blocks), response_state
    if event_type in {
        "response.output_text.done",
        "response.content_part.done",
        "response.reasoning_summary_text.done",
        "response.reasoning_text.done",
        "response.function_call_arguments.done",
    }:
        _finish_block(event_type, payload, items, blocks)
        return None, response_state
    if event_type == "response.reasoning_summary_part.done":
        _finish_reasoning_summary_part(payload, items, blocks)
        return None, response_state
    if event_type == "response.output_item.done":
        index = _output_index(payload)
        item = _active_item(items, index, payload)
        complete = payload.get("item")
        if not isinstance(complete, Mapping):
            raise CodexStreamError("Codex completed output item is invalid")
        _merge_completed_item(item, complete)
        if any(blocks[key].state != "stopped" for key in item.blocks):
            raise CodexStreamError("Codex output item completed with open blocks")
        item.state = "stopped"
        return None, response_state
    raise CodexStreamError(f"unsupported Codex SSE event: {event_type}")


def _require_response_started(state: str, event_type: str) -> None:
    if state == "not-started":
        raise CodexStreamError(f"Codex event precedes response.created: {event_type}")


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
    blocks: dict[tuple[int, int], _BlockState],
) -> StreamEvent:
    index = _output_index(payload)
    item = _active_item(items, index, payload)
    if event_type == "response.function_call_arguments.delta":
        key = (index, -1)
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
            key = (index, content_index)
            block = blocks.get(key)
            if block is None:
                block = _BlockState("thinking_raw")
                blocks[key] = block
                item.blocks.add(key)
            if item.kind != "reasoning" or block.kind != "thinking_raw":
                raise CodexStreamError("Codex reasoning delta references an inactive block")
            if block.state != "active" or block.text_done:
                raise CodexStreamError("Codex reasoning delta references an inactive block")
            delta = payload.get("delta")
            if type(delta) is not str:
                raise CodexStreamError("Codex reasoning delta is invalid")
            block.text += delta
            item.thinking += delta
            return StreamEvent(StreamEventType.MESSAGE_UPDATE, content=ThinkingContent(delta))
        summary_index = payload.get("summary_index")
        if type(summary_index) is not int or summary_index < 0:
            raise CodexStreamError("Codex reasoning summary index is invalid")
        key = (index, summary_index)
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
        item.thinking += delta
        return StreamEvent(StreamEventType.MESSAGE_UPDATE, content=ThinkingContent(delta))
    content_index = _content_index(payload)
    key = (index, content_index)
    block = blocks.get(key)
    if item.kind != "message" or block is None or block.state != "active":
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
    blocks: dict[tuple[int, int], _BlockState],
) -> None:
    index = _output_index(payload)
    item = _active_item(items, index, payload)
    if event_type == "response.function_call_arguments.done":
        key = (index, -1)
        expected_kind = "tool_call"
    elif event_type == "response.reasoning_summary_text.done":
        summary_index = payload.get("summary_index")
        if type(summary_index) is not int or summary_index < 0:
            raise CodexStreamError("Codex reasoning summary index is invalid")
        key = (index, summary_index)
        expected_kind = "thinking"
    elif event_type == "response.reasoning_text.done":
        key = (index, _content_index(payload))
        expected_kind = "thinking_raw"
    else:
        key = (index, _content_index(payload))
        expected_kind = "text"
    block = blocks.get(key)
    if block is None or block.kind != expected_kind:
        raise CodexStreamError("Codex block stop references an unknown block")
    if block.state != "active":
        raise CodexStreamError("Codex block stop is duplicated")
    if event_type == "response.content_part.done":
        part = payload.get("part")
        if part is not None and (
            not isinstance(part, Mapping) or part.get("type") != "output_text"
        ):
            raise CodexStreamError("Codex content part has the wrong item type")
    if event_type.endswith(".done"):
        complete_text = payload.get("text")
        if type(complete_text) is str and block.kind in {"text", "thinking_raw"}:
            if block.text and complete_text != block.text:
                raise CodexStreamError("Codex completed text does not match its deltas")
            block.text = complete_text
        complete_args = payload.get("arguments")
        if type(complete_args) is str and block.kind == "tool_call":
            if block.arguments and complete_args != block.arguments:
                raise CodexStreamError("Codex completed arguments do not match deltas")
            block.arguments = complete_args
            item.arguments = complete_args
    if event_type == "response.reasoning_summary_text.done":
        if block.text_done:
            raise CodexStreamError("Codex reasoning text stop is duplicated")
        block.text_done = True
    elif event_type in {
        "response.content_part.done",
        "response.reasoning_text.done",
        "response.function_call_arguments.done",
    }:
        block.state = "stopped"


def _finish_reasoning_summary_part(
    payload: Mapping[str, Any],
    items: Mapping[int, _ItemState],
    blocks: dict[tuple[int, int], _BlockState],
) -> None:
    index = _output_index(payload)
    item = _active_item(items, index, payload)
    summary_index = payload.get("summary_index")
    if type(summary_index) is not int or summary_index < 0:
        raise CodexStreamError("Codex reasoning summary index is invalid")
    key = (index, summary_index)
    block = blocks.get(key)
    if item.kind != "reasoning" or block is None or block.kind != "thinking":
        raise CodexStreamError("Codex reasoning summary stop references an unknown block")
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
        if block.text and complete_text != block.text:
            raise CodexStreamError("Codex completed reasoning does not match its deltas")
        if not block.text:
            block.text = complete_text
            item.thinking += complete_text
    block.state = "stopped"


def _merge_completed_item(item: _ItemState, complete: Mapping[str, Any]) -> None:
    if complete.get("id") != item.item_id:
        raise CodexStreamError("Codex completed item id does not match output item")
    if complete.get("type") != item.kind:
        raise CodexStreamError("Codex completed item type does not match output item")
    item.completed_item = dict(complete)
    if item.kind == "message":
        content = complete.get("content")
        if content is not None and not isinstance(content, list):
            raise CodexStreamError("Codex completed message content is invalid")
        if isinstance(content, list):
            complete_text_parts: list[str] = []
            for part in content:
                if (
                    not isinstance(part, Mapping)
                    or part.get("type") != "output_text"
                    or type(part.get("text")) is not str
                ):
                    raise CodexStreamError("Codex completed message part is invalid")
                complete_text_parts.append(part["text"])
            complete_text = "".join(complete_text_parts)
            if complete_text and item.text and complete_text != item.text:
                raise CodexStreamError("Codex completed item does not match its deltas")
            if complete_text:
                item.text = complete_text
    elif item.kind == "reasoning":
        encrypted = complete.get("encrypted_content")
        if type(encrypted) is str:
            item.encrypted_content = encrypted
    else:
        arguments = complete.get("arguments")
        if type(arguments) is str:
            if item.arguments and arguments != item.arguments:
                raise CodexStreamError("Codex completed tool does not match its deltas")
            item.arguments = arguments


def _complete_item(item: _ItemState) -> list[ContentBlock]:
    if item.kind == "message":
        return [TextContent(item.text)] if item.text else []
    if item.kind == "reasoning":
        if not item.thinking and not item.encrypted_content:
            return []
        return [ThinkingContent(item.thinking, item.encrypted_content)]
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
