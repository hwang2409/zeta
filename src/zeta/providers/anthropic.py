"""Anthropic Messages backend and Claude subscription OAuth support."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import httpx

from .auth import OAuthCredentialStore, OAuthTokens, error_body_excerpt
from .transport import (
    cleanup_transport,
    is_control_exception,
    request_error,
    retry_auth_completion,
    task_is_cancelling,
)
from .usage import normalize_usage
from ..prompts import load_identity
from ..types import (
    CompletionBackend,
    ContentBlock,
    flatten_tool_content,
    Message,
    MessageRole,
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolSchema,
    ToolUseContent,
)


CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"
AUTHORIZE_URL = "https://claude.ai/oauth/authorize"
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
API_URL = "https://api.anthropic.com/v1/messages"
DEFAULT_REDIRECT_URI = "http://localhost:53692/callback"
OAUTH_BETA = "oauth-2025-04-20"
CLAUDE_CODE_BETA = "claude-code-20250219"
OAUTH_SCOPES = (
    "org:create_api_key user:profile user:inference user:sessions:claude_code "
    "user:mcp_servers user:file_upload"
)


class AnthropicBackendError(RuntimeError):
    """Base class for errors that the agent loop can report as backend errors."""

    code = "backend_error"


class AnthropicAuthError(AnthropicBackendError):
    """Raised when Claude subscription credentials are missing or invalid."""

    code = "auth_error"

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class AnthropicHTTPError(AnthropicBackendError):
    """Raised when Anthropic returns an unsuccessful HTTP response."""

    code = "http_error"


class AnthropicStreamError(AnthropicBackendError):
    """Raised when an Anthropic SSE stream is invalid or ends early."""

    code = "stream_error"


def _first_string(value: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        candidate = value.get(key)
        if type(candidate) is str and candidate:
            return candidate
    return None


def _credential_candidates() -> tuple[Path, ...]:
    claude_dir = Path.home() / ".claude"
    return (claude_dir / ".credentials.json", claude_dir / "credentials.json")


def _keychain_claude_tokens() -> OAuthTokens | None:
    if sys.platform != "darwin":
        return None
    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            check=False,
            env={"PATH": os.defpath},
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0 or not isinstance(result.stdout, str) or not result.stdout.strip():
        return None
    try:
        return _extract_claude_tokens(json.loads(result.stdout))
    except (AnthropicAuthError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _extract_claude_tokens(value: Any) -> OAuthTokens:
    if not isinstance(value, Mapping):
        raise ValueError("Claude credentials are not an object")
    nested = value.get("claudeAiOauth")
    if isinstance(nested, Mapping):
        return OAuthTokens.from_mapping(nested, error_type=AnthropicAuthError)
    nested = value.get("oauth")
    if isinstance(nested, Mapping):
        return OAuthTokens.from_mapping(nested, error_type=AnthropicAuthError)
    return OAuthTokens.from_mapping(value, error_type=AnthropicAuthError)


class AnthropicCredentialStore(OAuthCredentialStore):
    """Owns zeta's OAuth file and reads Claude credentials only for bootstrap."""

    def __init__(
        self,
        path: str | Path | None = None,
        *,
        claude_credentials: str | Path | None = None,
        token_url: str = TOKEN_URL,
    ) -> None:
        super().__init__(
            path or Path.home() / ".zeta" / "anthropic-oauth.json",
            token_url=token_url,
        )
        self.claude_credentials = (
            Path(claude_credentials) if claude_credentials is not None else None
        )

    auth_error_type = AnthropicAuthError
    http_error_type = AnthropicHTTPError
    provider_label = "Claude"

    def bootstrap(self) -> OAuthTokens | None:
        candidates = (
            (self.claude_credentials,)
            if self.claude_credentials is not None
            else _credential_candidates()
        )
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with candidate.open() as handle:
                    return _extract_claude_tokens(json.load(handle))
            except self.auth_error_type:
                raise
            except (OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
                raise self.auth_error_type(
                    f"{self.provider_label} credentials could not be read"
                ) from exc
        return _keychain_claude_tokens()

    async def refresh(self, refresh_token: str, client: httpx.AsyncClient) -> OAuthTokens:
        try:
            response = await client.post(
                self.token_url,
                json={
                    "grant_type": "refresh_token",
                    "client_id": CLIENT_ID,
                    "refresh_token": refresh_token,
                },
                headers={"accept": "application/json"},
            )
        except httpx.HTTPError as exc:
            raise self.auth_error_type(
                f"{self.provider_label} OAuth token refresh failed"
            ) from exc
        if response.status_code >= 400:
            raise self.http_error_type(
                f"{self.provider_label} OAuth token refresh failed with HTTP "
                f"{response.status_code}"
            )
        try:
            return _tokens_from_response(response.json(), refresh_token)
        except (TypeError, ValueError, json.JSONDecodeError) as exc:
            raise self.auth_error_type(
                f"{self.provider_label} OAuth token response is invalid"
            ) from exc


def build_authorization_url(
    state: str,
    code_challenge: str,
    redirect_uri: str = DEFAULT_REDIRECT_URI,
) -> str:
    """Build the Claude Pro/Max PKCE authorization URL."""

    if not state or not code_challenge or not redirect_uri:
        raise AnthropicAuthError("Claude OAuth PKCE parameters are incomplete")
    return f"{AUTHORIZE_URL}?{urlencode({'code': 'true', 'client_id': CLIENT_ID, 'response_type': 'code', 'scope': OAUTH_SCOPES, 'code_challenge': code_challenge, 'code_challenge_method': 'S256', 'state': state, 'redirect_uri': redirect_uri})}"


async def exchange_authorization_code(
    client: httpx.AsyncClient,
    code: str,
    state: str,
    code_verifier: str,
    redirect_uri: str,
    *,
    token_url: str = TOKEN_URL,
) -> OAuthTokens:
    if not code or not state or not code_verifier or not redirect_uri:
        raise AnthropicAuthError("Claude OAuth code exchange parameters are incomplete")
    try:
        response = await client.post(
            token_url,
            json={
                "grant_type": "authorization_code",
                "client_id": CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            headers={"accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise AnthropicAuthError("Claude OAuth code exchange failed") from exc
    if response.status_code >= 400:
        raise AnthropicHTTPError(
            f"Claude OAuth code exchange failed with HTTP {response.status_code}"
        )
    try:
        return _tokens_from_response(response.json(), None)
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise AnthropicAuthError("Claude OAuth code response is invalid") from exc


def _tokens_from_response(value: Any, fallback_refresh: str | None) -> OAuthTokens:
    if not isinstance(value, Mapping):
        raise ValueError("token response is not an object")
    access_token = _first_string(value, "access_token", "accessToken", "access")
    refresh_token = _first_string(value, "refresh_token", "refreshToken", "refresh")
    expires_in = value.get("expires_in", value.get("expiresIn"))
    if not access_token or not (refresh_token or fallback_refresh):
        raise ValueError("token response is incomplete")
    if type(expires_in) not in {int, float}:
        raise ValueError("token response expiry is invalid")
    return OAuthTokens(
        access_token,
        refresh_token or fallback_refresh or "",
        time.time() + float(expires_in) - 300,
    )


@dataclass(slots=True)
class _BlockState:
    kind: str
    text: str = ""
    signature: str = ""
    call_id: str = ""
    name: str = ""
    input_json: str = ""
    initial_input: dict[str, Any] | None = None
    redacted_data: str = ""


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
        value = value[1:] if value.startswith(" ") else value
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
            raise AnthropicStreamError("Anthropic returned invalid SSE JSON") from exc
        if not isinstance(payload, dict):
            raise AnthropicStreamError("Anthropic SSE payload is not an object")
        return event, payload


class AnthropicBackend(CompletionBackend):
    """One-completion Anthropic Messages streaming backend."""

    def __init__(
        self,
        *,
        model: str = "claude-sonnet-4-6",
        max_tokens: int = 8192,
        base_url: str = API_URL,
        token_store: AnthropicCredentialStore | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self.base_url = base_url.rstrip("/")
        self.token_store = token_store or AnthropicCredentialStore()
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
        attempts = retry_auth_completion(
            lambda: self._complete_once(messages, tool_schemas),
            lambda token: self._complete_once(messages, tool_schemas, token=token),
            self._refresh_token,
            lambda error: isinstance(error, AnthropicAuthError) and error.status_code == 401,
            lambda error: AnthropicAuthError(
                "Anthropic authentication failed after token refresh; run `zeta login`",
                status_code=401,
            ),
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
        stream_context = None
        entered = False
        primary_exception: BaseException | None = None
        try:
            token = token if token is not None else await self.token_store.access_token(client)
            payload = build_messages_payload(
                messages,
                tool_schemas,
                model=self.model,
                max_tokens=self.max_tokens,
            )
            identity = {
                "type": "text",
                "text": "You are Claude Code, Anthropic's official CLI for Claude.",
                "cache_control": {"type": "ephemeral"},
            }
            system = payload.get("system", [])
            if not any(
                block.get("text") == load_identity()
                for block in system
                if isinstance(block, dict)
            ):
                if system:
                    system[-1].pop("cache_control", None)
                system = [
                    *system,
                    {
                        "type": "text",
                        "text": load_identity(),
                        "cache_control": {"type": "ephemeral"},
                    },
                ]
            payload["system"] = [identity, *system]
            headers = {
                "accept": "text/event-stream",
                "anthropic-beta": f"{CLAUDE_CODE_BETA},{OAUTH_BETA}",
                "anthropic-version": "2023-06-01",
                "authorization": f"Bearer {token}",
                "content-type": "application/json",
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
            except AnthropicBackendError as exc:
                primary_exception = exc
            except httpx.HTTPError as exc:
                primary_exception = request_error(exc, AnthropicHTTPError)
            except BaseException as exc:
                if task_is_cancelling() and not is_control_exception(exc):
                    primary_exception = asyncio.CancelledError()
                    primary_exception.__cause__ = exc
                else:
                    primary_exception = exc
        except httpx.HTTPError as exc:
            primary_exception = request_error(exc, AnthropicHTTPError)
        except BaseException as exc:
            primary_exception = exc
        finally:
            primary_exception = await cleanup_transport(
                stream_context=stream_context,
                entered=entered,
                client=client,
                owns_client=self.client is None,
                primary_exception=primary_exception,
                http_error_type=AnthropicHTTPError,
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


def _http_error(status_code: int, body: bytes) -> AnthropicHTTPError:
    message = error_body_excerpt(body) or "request failed"
    error_type = AnthropicAuthError if status_code in {401, 403} else AnthropicHTTPError
    if error_type is AnthropicAuthError:
        return error_type(
            f"Anthropic HTTP {status_code}: {message}", status_code=status_code
        )
    return error_type(f"Anthropic HTTP {status_code}: {message}")


async def _decode_response(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    decoder = _SSEDecoder()
    blocks: dict[int, _BlockState] = {}
    active_blocks: set[int] = set()
    stopped_blocks: set[int] = set()
    usage: dict[str, Any] = {}
    stop_reason: str | None = None
    finished = False
    message_state = "not-started"
    async for line in response.aiter_lines():
        record = decoder.feed(line)
        if record is None:
            continue
        event, payload = record
        event_type = payload.get("type", event)
        if type(event_type) is str:
            message_state = _advance_message_state(message_state, event_type)
        translated = _translate_event(
            event, payload, blocks, active_blocks, stopped_blocks, usage
        )
        if translated is None:
            if payload.get("type") == "message_delta":
                delta = payload.get("delta")
                if not isinstance(delta, Mapping):
                    raise AnthropicStreamError("Anthropic message delta is invalid")
                stop_reason = delta.get("stop_reason")
            continue
        if translated.type is StreamEventType.MESSAGE_END:
            translated = StreamEvent(
                translated.type,
                message=translated.message,
                data={
                    **translated.data,
                    "usage": normalize_usage(usage),
                    "stop_reason": stop_reason,
                },
            )
            finished = True
        yield translated
    record = decoder.finish()
    if record is not None and record[1].get("type") != "done":
        event_type = record[1].get("type", record[0])
        if type(event_type) is str:
            message_state = _advance_message_state(message_state, event_type)
        translated = _translate_event(
            record[0], record[1], blocks, active_blocks, stopped_blocks, usage
        )
        if translated is not None:
            if translated.type is StreamEventType.MESSAGE_END:
                translated = StreamEvent(
                    translated.type,
                    message=translated.message,
                    data={
                        **translated.data,
                        "usage": normalize_usage(usage),
                        "stop_reason": stop_reason,
                    },
                )
                yield translated
                return
            yield translated
    if not finished:
        raise AnthropicStreamError("Anthropic stream ended before message_stop")


def _translate_event(
    event: str,
    payload: Mapping[str, Any],
    blocks: dict[int, _BlockState],
    active_blocks: set[int],
    stopped_blocks: set[int],
    usage: dict[str, Any],
) -> StreamEvent | None:
    event_type = payload.get("type", event)
    if type(event_type) is not str:
        raise AnthropicStreamError("Anthropic SSE event type is invalid")
    if event_type == "error":
        detail = payload.get("error")
        if not isinstance(detail, Mapping):
            raise AnthropicStreamError("Anthropic stream error payload is invalid")
        message = detail.get("message")
        if type(message) is not str:
            raise AnthropicStreamError("Anthropic stream error message is invalid")
        raise AnthropicStreamError(
            error_body_excerpt(message.encode()) or "Anthropic stream error"
        )
    if event_type == "message_start":
        message = payload.get("message", {})
        if isinstance(message, Mapping):
            initial_usage = message.get("usage")
            if initial_usage is None:
                pass
            elif isinstance(initial_usage, Mapping):
                usage.update(initial_usage)
            else:
                raise AnthropicStreamError("Anthropic message usage is invalid")
            return StreamEvent(
                StreamEventType.MESSAGE_START,
                data={key: message[key] for key in ("id", "model", "role") if key in message},
            )
        raise AnthropicStreamError("Anthropic message_start is invalid")
    if event_type == "content_block_start":
        index = _index(payload)
        if index in blocks:
            raise AnthropicStreamError("Anthropic content block index is duplicated")
        block = payload.get("content_block")
        if not isinstance(block, Mapping):
            raise AnthropicStreamError("Anthropic content block is invalid")
        kind = block.get("type")
        if kind == "text":
            blocks[index] = _BlockState("text")
        elif kind == "thinking":
            blocks[index] = _BlockState("thinking")
        elif kind == "redacted_thinking":
            data = block.get("data")
            if type(data) is not str or not data:
                raise AnthropicStreamError("Anthropic redacted thinking is invalid")
            blocks[index] = _BlockState("redacted_thinking", redacted_data=data)
        elif kind == "tool_use":
            initial_input = block.get("input")
            if initial_input is not None and not isinstance(initial_input, Mapping):
                raise AnthropicStreamError("Anthropic tool input is invalid")
            call_id = block.get("id")
            name = block.get("name")
            if type(call_id) is not str or not call_id:
                raise AnthropicStreamError("Anthropic tool call id is invalid")
            if type(name) is not str or not name:
                raise AnthropicStreamError("Anthropic tool call name is invalid")
            blocks[index] = _BlockState(
                "tool_use",
                call_id=call_id,
                name=name,
                initial_input=(
                    dict(initial_input)
                    if isinstance(initial_input, Mapping)
                    else None
                ),
            )
        else:
            raise AnthropicStreamError("unsupported Anthropic content block type")
        active_blocks.add(index)
        return None
    if event_type == "content_block_delta":
        index = _index(payload)
        if index in stopped_blocks:
            raise AnthropicStreamError(
                f"Anthropic delta references stopped block index: {index}"
            )
        if index not in active_blocks:
            raise AnthropicStreamError(
                f"Anthropic delta references unknown block index: {index}"
            )
        block = blocks.get(index)
        if block is None:
            raise AnthropicStreamError(
                f"Anthropic delta references unknown block index: {index}"
            )
        delta = payload.get("delta")
        if not isinstance(delta, Mapping):
            raise AnthropicStreamError("Anthropic content delta is invalid")
        kind = delta.get("type")
        if kind == "text_delta":
            if block.kind != "text":
                raise AnthropicStreamError("Anthropic text delta has the wrong block type")
            text = _required_string(delta, "text")
            block.text += text
            return StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=text)
        if kind == "thinking_delta":
            if block.kind != "thinking":
                raise AnthropicStreamError(
                    "Anthropic thinking delta has the wrong block type"
                )
            text = _required_string(delta, "thinking")
            block.text += text
            return StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                content=ThinkingContent(text),
            )
        if kind == "signature_delta":
            if block.kind != "thinking":
                raise AnthropicStreamError("Anthropic signature is outside thinking")
            signature = _required_string(delta, "signature")
            block.signature += signature
            return StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                data={"thinking_signature_delta": signature, "index": index},
            )
        if kind == "input_json_delta":
            if block.kind != "tool_use":
                raise AnthropicStreamError(
                    "Anthropic tool delta has the wrong block type"
                )
            partial = _required_string(delta, "partial_json")
            block.input_json += partial
            arguments = _parse_partial_object(block.input_json)
            call = ToolCall(block.call_id, block.name, arguments)
            return StreamEvent(
                StreamEventType.MESSAGE_UPDATE,
                tool_call=call,
                data={"tool_call_delta": partial, "index": index},
            )
        raise AnthropicStreamError("unsupported Anthropic content delta type")
    if event_type == "content_block_stop":
        index = _index(payload)
        if index in stopped_blocks:
            raise AnthropicStreamError(
                f"Anthropic content block stop is duplicated: {index}"
            )
        if index not in active_blocks:
            raise AnthropicStreamError(
                f"Anthropic content block stop references unknown index: {index}"
            )
        active_blocks.remove(index)
        stopped_blocks.add(index)
        return None
    if event_type == "message_delta":
        delta = payload.get("delta")
        if not isinstance(delta, Mapping):
            raise AnthropicStreamError("Anthropic message delta is invalid")
        stop_reason = delta.get("stop_reason")
        if stop_reason is not None and type(stop_reason) is not str:
            raise AnthropicStreamError("Anthropic stop reason is invalid")
        message_usage = payload.get("usage")
        if message_usage is None:
            pass
        elif isinstance(message_usage, Mapping):
            usage.update(message_usage)
        else:
            raise AnthropicStreamError("Anthropic message usage is invalid")
        return None
    if event_type == "message_stop":
        if active_blocks:
            raise AnthropicStreamError("Anthropic message stop has open content blocks")
        content: list[ContentBlock] = []
        for index in sorted(blocks):
            block = blocks[index]
            if block.kind == "text":
                content.append(TextContent(block.text))
            elif block.kind == "thinking":
                if not block.signature:
                    raise AnthropicStreamError(
                        "Anthropic thinking block is missing its signature"
                    )
                content.append(ThinkingContent(block.text, block.signature))
            elif block.kind == "redacted_thinking":
                content.append(RedactedThinkingContent(block.redacted_data))
            else:
                arguments = (
                    _parse_complete_object(block.input_json)
                    if block.input_json
                    else block.initial_input or {}
                )
                if not block.call_id or not block.name:
                    raise AnthropicStreamError("Anthropic tool call is incomplete")
                content.append(ToolUseContent(ToolCall(block.call_id, block.name, arguments)))
        return StreamEvent(
            StreamEventType.MESSAGE_END,
            message=Message(MessageRole.ASSISTANT, content),
            data={},
        )
    return None


def _advance_message_state(state: str, event_type: str) -> str:
    if event_type == "done":
        return state
    if state == "stopped":
        raise AnthropicStreamError("Anthropic event follows message_stop")
    if event_type == "message_start":
        if state != "not-started":
            raise AnthropicStreamError("Anthropic message_start is duplicated")
        return "started"
    if state == "not-started":
        if event_type == "error":
            return state
        raise AnthropicStreamError("Anthropic event precedes message_start")
    if event_type == "message_stop":
        return "stopped"
    return state


def _index(payload: Mapping[str, Any]) -> int:
    index = payload.get("index")
    if type(index) is not int or index < 0:
        raise AnthropicStreamError("Anthropic content block index is invalid")
    return index


def _required_string(value: Mapping[str, Any], key: str) -> str:
    result = value.get(key)
    if type(result) is not str:
        raise AnthropicStreamError(f"Anthropic delta field {key} is invalid")
    return result


def _parse_partial_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _parse_complete_object(value: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise AnthropicStreamError("Anthropic tool arguments are invalid JSON") from exc
    if not isinstance(parsed, dict):
        raise AnthropicStreamError("Anthropic tool arguments are not an object")
    return parsed


def _wire_content(blocks: Sequence[ContentBlock]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for block in blocks:
        if isinstance(block, TextContent):
            result.append({"type": "text", "text": block.text})
        elif isinstance(block, ThinkingContent):
            if not block.signature:
                raise AnthropicHTTPError(
                    "thinking block is missing its Anthropic signature"
                )
            result.append(
                {
                    "type": "thinking",
                    "thinking": block.text,
                    "signature": block.signature,
                }
            )
        elif isinstance(block, RedactedThinkingContent):
            result.append({"type": "redacted_thinking", "data": block.data})
        elif isinstance(block, ToolUseContent):
            result.append(
                {
                    "type": "tool_use",
                    "id": block.tool_call.id,
                    "name": block.tool_call.name,
                    "input": block.tool_call.arguments,
                }
            )
        else:
            raise AnthropicHTTPError("unsupported zeta content block")
    return result


def build_messages_payload(
    messages: Sequence[Message],
    tool_schemas: Sequence[ToolSchema],
    *,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    system: list[dict[str, Any]] = []
    wire_messages: list[dict[str, Any]] = []
    for message in messages:
        if message.role is MessageRole.SYSTEM:
            content = _wire_content(message.content)
            if content and any(
                block.get("type") != "text" or block.get("text", "").strip()
                for block in content
            ):
                system.extend(content)
            continue
        if message.role is MessageRole.TOOL_RESULT:
            if message.tool_result is None:
                raise AnthropicHTTPError("tool result message is missing its result")
            content = [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_result.tool_call_id,
                    "content": (
                        flatten_tool_content(message.tool_result.content_blocks)
                        if message.tool_result.content_blocks is not None
                        else message.tool_result.content
                    ),
                    "is_error": message.tool_result.is_error,
                }
            ]
            wire_messages.append({"role": "user", "content": content})
            continue
        role = "assistant" if message.role is MessageRole.ASSISTANT else "user"
        wire_messages.append({"role": role, "content": _wire_content(message.content)})

    if system:
        system[-1]["cache_control"] = {"type": "ephemeral"}
    tools = [_wire_tool_schema(schema) for schema in tool_schemas]
    if tools:
        tools[-1]["cache_control"] = {"type": "ephemeral"}
    payload: dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": wire_messages,
        "stream": True,
    }
    if system:
        payload["system"] = system
    if tools:
        payload["tools"] = tools
    for message in reversed(wire_messages):
        if message["role"] != "user":
            continue
        content = message["content"]
        if isinstance(content, list) and content:
            last_block = content[-1]
            if last_block.get("type") in {"text", "tool_result"}:
                last_block["cache_control"] = {"type": "ephemeral"}
        break
    return payload


def _wire_tool_schema(schema: ToolSchema) -> dict[str, Any]:
    name = schema.get("name")
    if type(name) is not str or not name:
        raise AnthropicHTTPError("tool schema name must be a nonempty string")
    input_schema = schema.get("input_schema", schema.get("parameters"))
    if not isinstance(input_schema, Mapping):
        input_schema = {
            key: value
            for key, value in schema.items()
            if key not in {"name", "description", "cache_control"}
        }
    result: dict[str, Any] = {"name": name, "input_schema": dict(input_schema)}
    description = schema.get("description")
    if type(description) is str:
        result["description"] = description
    return result
