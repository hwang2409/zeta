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
from urllib.parse import urlencode

import httpx

from .auth import OAuthCredentialStore, OAuthTokens, error_body_excerpt
from .codex_errors import (
    CodexAuthError,
    CodexBackendError,
    CodexHTTPError,
    CodexStreamError,
)
from .codex_payload import build_responses_payload
from .transport import (
    cleanup_transport,
    is_control_exception,
    request_error,
    retry_auth_completion,
    task_is_cancelling,
)
from ..types import (
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
CODEX_AUTHORIZE_URL = "https://auth.openai.com/oauth/authorize"
CODEX_TOKEN_URL = "https://auth.openai.com/oauth/token"
CODEX_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
CODEX_OAUTH_SCOPES = "openid profile email offline_access"
DEFAULT_CODEX_REDIRECT_URI = "http://localhost:1455/auth/callback"
DEFAULT_CODEX_MODEL = "gpt-5.6-luna"
JWT_AUTH_CLAIM = "https://api.openai.com/auth"


def build_authorization_url(
    state: str,
    code_challenge: str,
    redirect_uri: str = DEFAULT_CODEX_REDIRECT_URI,
) -> str:
    """Build the ChatGPT plan OAuth PKCE authorization URL."""

    if not state or not code_challenge or not redirect_uri:
        raise CodexAuthError("Codex OAuth PKCE parameters are incomplete")
    params = {
        "client_id": CODEX_CLIENT_ID,
        "response_type": "code",
        "scope": CODEX_OAUTH_SCOPES,
        "code_challenge": code_challenge,
        "code_challenge_method": "S256",
        "id_token_add_organizations": "true",
        "codex_cli_simplified_flow": "true",
        "state": state,
        "originator": "codex_cli_rs",
        "redirect_uri": redirect_uri,
    }
    return f"{CODEX_AUTHORIZE_URL}?{urlencode(params)}"


async def exchange_authorization_code(
    client: httpx.AsyncClient,
    code: str,
    state: str,
    code_verifier: str,
    redirect_uri: str,
    *,
    token_url: str = CODEX_TOKEN_URL,
) -> OAuthTokens:
    """Exchange a ChatGPT plan authorization code for OAuth tokens."""

    if not code or not state or not code_verifier or not redirect_uri:
        raise CodexAuthError("Codex OAuth code exchange parameters are incomplete")
    try:
        response = await client.post(
            token_url,
            data={
                "grant_type": "authorization_code",
                "client_id": CODEX_CLIENT_ID,
                "code": code,
                "state": state,
                "redirect_uri": redirect_uri,
                "code_verifier": code_verifier,
            },
            headers={"accept": "application/json"},
        )
    except httpx.HTTPError as exc:
        raise CodexAuthError("Codex OAuth code exchange failed") from exc
    if response.status_code >= 400:
        raise CodexHTTPError(
            f"Codex OAuth code exchange failed with HTTP {response.status_code}"
        )
    try:
        value = response.json()
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise CodexAuthError("Codex OAuth code response is invalid") from exc
    if not isinstance(value, Mapping):
        raise CodexAuthError("Codex OAuth code response is invalid")
    access = _first_string(value, "access_token", "accessToken", "access")
    refresh = _first_string(value, "refresh_token", "refreshToken", "refresh")
    expires_in = value.get("expires_in", value.get("expiresIn"))
    if not access or not refresh or type(expires_in) not in {int, float}:
        raise CodexAuthError("Codex OAuth code response is invalid")
    return OAuthTokens(access, refresh, time.time() + float(expires_in) - 300)


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
        refresh_keys = ("refresh_token", "refreshToken", "refresh")
        refresh_present = any(key in value for key in refresh_keys)
        if not access or type(expires_in) not in {int, float} or (
            refresh_present and not refresh
        ):
            raise CodexAuthError("Codex OAuth token response is invalid")
        return OAuthTokens(
            access,
            refresh or refresh_token,
            time.time() + float(expires_in) - 300,
        )


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


BlockKey = tuple[int, str, int]


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
    summary_text: str = ""
    raw_text: str = ""
    encrypted_content: str | None = None
    completed_item: dict[str, Any] | None = None
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
    ) -> None:
        self.model = model
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
        attempts = retry_auth_completion(
            lambda: self._complete_once(messages, tool_schemas),
            lambda token: self._complete_once(messages, tool_schemas, token=token),
            self._refresh_token,
            lambda error: isinstance(error, CodexAuthError) and error.status_code == 401,
            lambda error: CodexAuthError(
                "Codex authentication failed after token refresh; run `zeta login`",
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
        stream_context: Any = None
        entered = False
        primary_exception: BaseException | None = None
        try:
            access_token = (
                token if token is not None else await self.token_store.access_token(client)
            )
            account_id = extract_account_id(access_token)
            payload = build_responses_payload(
                messages,
                tool_schemas,
                model=self.model,
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
    excerpt = error_body_excerpt(body)
    detail = f": {excerpt}" if excerpt else ""
    if error_type is CodexAuthError:
        return error_type(
            f"Codex HTTP request failed ({status_code}){detail}",
            status_code=status_code,
        )
    return error_type(f"Codex HTTP request failed ({status_code}){detail}")


async def _decode_response(response: httpx.Response) -> AsyncIterator[StreamEvent]:
    decoder = _SSEDecoder()
    response_state = "not-started"
    items: dict[int, _ItemState] = {}
    blocks: dict[BlockKey, _BlockState] = {}
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
    blocks: dict[BlockKey, _BlockState],
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
            raise CodexStreamError("Codex event follows response completion")
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
        raise CodexStreamError("Codex event follows response completion")
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
            if "status" in response and response["status"] != "completed":
                raise CodexStreamError("Codex response completion status is invalid")
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
        _merge_completed_item(item, complete, blocks)
        for key in item.blocks:
            block = blocks[key]
            if block.kind == "thinking_raw" and block.text_done:
                block.state = "stopped"
        if any(blocks[key].state != "stopped" for key in item.blocks):
            raise CodexStreamError("Codex output item completed with open blocks")
        item.state = "stopped"
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
                raise CodexStreamError("Codex reasoning delta references an inactive block")
            if block.state != "active" or block.text_done:
                raise CodexStreamError("Codex reasoning delta references an inactive block")
            delta = payload.get("delta")
            if type(delta) is not str:
                raise CodexStreamError("Codex reasoning delta is invalid")
            block.text += delta
            item.raw_text += delta
            item.thinking += delta
            return StreamEvent(StreamEventType.MESSAGE_UPDATE, content=ThinkingContent(delta))
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
        item.thinking += delta
        return StreamEvent(StreamEventType.MESSAGE_UPDATE, content=ThinkingContent(delta))
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
            key = next((candidate for candidate in candidates if candidate in blocks), candidates[0])
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
            not isinstance(part, Mapping)
            or part.get("type") != wire_kind
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
            error = (
                "Codex completed reasoning does not match its deltas"
                if block.kind in {"thinking", "thinking_raw"}
                else "Codex completed text does not match its deltas"
            )
            if _reconcile_completed_text(block, complete_text, error):
                if block.kind == "thinking":
                    item.summary_text += complete_text
                    item.thinking += complete_text
                elif block.kind == "thinking_raw":
                    item.raw_text += complete_text
                    item.thinking += complete_text
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
    block: _BlockState, complete_text: str, error: str
) -> bool:
    if block.text and complete_text != block.text:
        raise CodexStreamError(error)
    if block.text:
        return False
    block.text = complete_text
    return True


def _require_completed_match(actual: Any, streamed: Any, error: str) -> None:
    if actual != streamed:
        raise CodexStreamError(error)


def _finish_reasoning_summary_part(
    payload: Mapping[str, Any],
    items: Mapping[int, _ItemState],
    blocks: dict[BlockKey, _BlockState],
) -> None:
    index = _output_index(payload)
    item = _active_item(items, index, payload)
    summary_index = payload.get("summary_index")
    if type(summary_index) is not int or summary_index < 0:
        raise CodexStreamError("Codex reasoning summary index is invalid")
    key = (index, "thinking", summary_index)
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
        if _reconcile_completed_text(
            block,
            complete_text,
            "Codex completed reasoning does not match its deltas",
        ):
            item.summary_text += complete_text
            item.thinking += complete_text
    block.state = "stopped"


def _merge_completed_item(
    item: _ItemState,
    complete: Mapping[str, Any],
    blocks: Mapping[BlockKey, _BlockState],
) -> None:
    if complete.get("id") != item.item_id:
        raise CodexStreamError("Codex completed item id does not match output item")
    if complete.get("type") != item.kind:
        raise CodexStreamError("Codex completed item type does not match output item")
    if complete.get("status") != "completed":
        raise CodexStreamError(f"Codex completed {item.kind} status is invalid")
    if item.kind == "message":
        if complete.get("role") != "assistant":
            raise CodexStreamError("Codex completed message metadata is invalid")
        content = complete.get("content")
        if not isinstance(content, list) or not content:
            raise CodexStreamError("Codex completed message content is invalid")
        streamed_parts = sorted(
            (key[2], blocks[key])
            for key in item.blocks
            if key[1] == "message"
        )
        if len(content) != len(streamed_parts) or any(
            index != position
            for position, (index, _) in enumerate(streamed_parts)
        ):
            raise CodexStreamError("Codex completed message parts do not match blocks")
        for position, part in enumerate(content):
            if not isinstance(part, Mapping):
                raise CodexStreamError("Codex completed message part is invalid")
            part_type = part.get("type")
            text_key = "text" if part_type == "output_text" else "refusal"
            if (
                part_type not in {"output_text", "refusal"}
                or type(part.get(text_key)) is not str
            ):
                raise CodexStreamError("Codex completed message part is invalid")
            block = streamed_parts[position][1]
            _require_completed_match(
                part_type,
                block.kind,
                "Codex completed message parts do not match blocks",
            )
            _require_completed_match(
                part[text_key],
                block.text,
                "Codex completed message parts do not match deltas",
            )
        item.text = "".join(
            part["text"] if part.get("type") == "output_text" else part["refusal"]
            for part in content
        )
    elif item.kind == "reasoning":
        summary = complete.get("summary")
        if not isinstance(summary, list):
            raise CodexStreamError("Codex completed reasoning metadata is invalid")
        for part in summary:
            if (
                not isinstance(part, Mapping)
                or part.get("type") != "summary_text"
                or type(part.get("text")) is not str
            ):
                raise CodexStreamError("Codex completed reasoning summary is invalid")
        streamed_summary = sorted(
            (key[2], blocks[key])
            for key in item.blocks
            if key[1] == "thinking"
        )
        if len(summary) != len(streamed_summary) or any(
            index != position
            for position, (index, _) in enumerate(streamed_summary)
        ):
            raise CodexStreamError("Codex completed reasoning does not match its deltas")
        for position, part in enumerate(summary):
            block = streamed_summary[position][1]
            _require_completed_match(
                part["type"],
                "summary_text",
                "Codex completed reasoning does not match its deltas",
            )
            _require_completed_match(
                part["text"],
                block.text,
                "Codex completed reasoning does not match its deltas",
            )
        content = complete.get("content")
        complete_raw_parts: list[str] = []
        if content is not None:
            if not isinstance(content, list):
                raise CodexStreamError("Codex completed reasoning content is invalid")
            for part in content:
                if (
                    not isinstance(part, Mapping)
                    or part.get("type") != "reasoning_text"
                    or type(part.get("text")) is not str
                ):
                    raise CodexStreamError("Codex completed reasoning content is invalid")
                complete_raw_parts.append(part["text"])
        streamed_raw = sorted(
            (key[2], blocks[key])
            for key in item.blocks
            if key[1] == "thinking_raw"
        )
        if len(complete_raw_parts) != len(streamed_raw) or any(
            index != position for position, (index, _) in enumerate(streamed_raw)
        ):
            raise CodexStreamError("Codex completed reasoning does not match its deltas")
        for position, part in enumerate(complete_raw_parts):
            _require_completed_match(
                part,
                streamed_raw[position][1].text,
                "Codex completed reasoning does not match its deltas",
            )
        encrypted = complete.get("encrypted_content")
        if encrypted is not None and (type(encrypted) is not str or not encrypted):
            raise CodexStreamError("Codex completed reasoning metadata is invalid")
        if type(encrypted) is str:
            item.encrypted_content = encrypted
    else:
        _require_completed_match(
            complete.get("call_id"),
            item.call_id,
            "Codex completed tool metadata is invalid",
        )
        _require_completed_match(
            complete.get("name"), item.name, "Codex completed tool metadata is invalid"
        )
        arguments = complete.get("arguments")
        if type(arguments) is not str:
            raise CodexStreamError("Codex completed tool arguments are invalid")
        _require_completed_match(
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
