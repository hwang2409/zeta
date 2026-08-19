import asyncio
import json
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from zeta.anthropic import (
    AnthropicAuthError,
    AnthropicBackend,
    AnthropicCredentialStore,
    AnthropicStreamError,
    OAuthTokens,
    build_authorization_url,
    build_messages_payload,
)
from zeta.types import (
    Message,
    MessageRole,
    RedactedThinkingContent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolUseContent,
)


SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg-1","model":"claude-test","role":"assistant","usage":{"input_tokens":12}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}

event: content_block_start
data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":1}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":4}}

event: message_stop
data: {"type":"message_stop"}

event: message_start
data: {"type":"message_start","message":{"id":"ignored"}}
"""


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_stream_maps_thinking_text_usage_and_stops_at_one_completion(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=SSE,
            request=request,
        )

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    backend = AnthropicBackend(
        client=client,
        token_store=store,
        base_url="https://test.invalid/v1/messages",
    )

    events = [
        event
        async for event in backend.complete(
            [
                Message(MessageRole.SYSTEM, [TextContent("keep this system prompt")]),
                Message(MessageRole.USER, [TextContent("hi")]),
            ],
            [],
        )
    ]

    assert len(requests) == 1
    assert requests[0].headers["authorization"] == "Bearer access-test"
    assert "x-api-key" not in requests[0].headers
    request_payload = json.loads(requests[0].content)
    assert request_payload["model"] == "claude-sonnet-4-6"
    assert request_payload["system"][0]["text"].startswith("You are Claude Code")
    assert request_payload["system"][1]["text"] == "keep this system prompt"
    assert [event.type for event in events] == [
        StreamEventType.MESSAGE_START,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_END,
    ]
    assert events[-1].data["usage"] == {
        "input_tokens": 12,
        "output_tokens": 4,
    }
    assert events[-1].message is not None
    assert events[-1].message.content == [
        ThinkingContent("plan", "sig-1"),
        TextContent("hello"),
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_tool_call_delta_and_single_completion_boundary(tmp_path: Path) -> None:
    stream = """event: message_start
data: {"type":"message_start","message":{"id":"msg-2","usage":{"input_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"tool-1","name":"read","input":{}}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"README.md\\"}"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"tool_use"},"usage":{"output_tokens":2}}

event: message_stop
data: {"type":"message_stop"}
"""

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=stream,
            request=request,
        )

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    events = [
        event
        async for event in AnthropicBackend(
            client=client,
            token_store=store,
            base_url="https://test.invalid/v1/messages",
        ).complete([], [])
    ]

    assert len([event for event in events if event.type is StreamEventType.MESSAGE_END]) == 1
    assert events[-1].message is not None
    assert events[-1].message.content == [
        ToolUseContent(ToolCall("tool-1", "read", {"path": "README.md"}))
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_expired_claude_login_refreshes_into_zeta_store(tmp_path: Path) -> None:
    claude_path = tmp_path / "claude" / ".credentials.json"
    claude_path.parent.mkdir()
    claude_path.write_text(
        json.dumps(
            {
                "claudeAiOauth": {
                    "accessToken": "expired-access",
                    "refreshToken": "refresh-test",
                    "expiresAt": 1,
                }
            }
        )
    )
    original_claude_credentials = claude_path.read_text()
    token_requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        token_requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "refreshed-access",
                "refresh_token": "refreshed-refresh",
                "expires_in": 3600,
            },
            request=request,
        )

    store = AnthropicCredentialStore(
        tmp_path / "zeta" / "anthropic.json",
        claude_credentials=claude_path,
        token_url="https://test.invalid/oauth/token",
    )
    client = client_for(handler)

    assert await store.access_token(client) == "refreshed-access"
    assert len(token_requests) == 1
    assert store.read() is not None
    assert oct((tmp_path / "zeta" / "anthropic.json").stat().st_mode & 0o777) == "0o600"
    assert claude_path.read_text() == original_claude_credentials
    await client.aclose()


def test_payload_caches_stable_prefix_and_maps_tool_results() -> None:
    payload = build_messages_payload(
        [
            Message(MessageRole.SYSTEM, [TextContent("stable")]),
            Message(MessageRole.USER, [TextContent("run")]),
        ],
        [{"name": "read", "description": "read a file", "parameters": {"type": "object"}}],
        model="claude-test",
        max_tokens=100,
    )

    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][0]["input_schema"] == {"type": "object"}
    assert payload["messages"][-1]["role"] == "user"
    assert payload["messages"][-1]["content"][0]["text"] == "run"
    assert payload["messages"][-1]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }


def test_authorization_url_contains_validated_redirect_uri() -> None:
    url = build_authorization_url("state", "challenge", "http://localhost/callback")
    assert "redirect_uri=http%3A%2F%2Flocalhost%2Fcallback" in url


def test_signed_thinking_blocks_use_anthropic_wire_types() -> None:
    payload = build_messages_payload(
        [
            Message(
                MessageRole.ASSISTANT,
                [
                    ThinkingContent("plan", "sig-1"),
                    RedactedThinkingContent("opaque"),
                ],
            )
        ],
        [],
        model="claude-test",
        max_tokens=100,
    )

    assert payload["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]


@pytest.mark.asyncio
async def test_http_failure_is_typed(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    with pytest.raises(AnthropicAuthError):
        await anext(AnthropicBackend(client=client, token_store=store).complete([], []))
    await client.aclose()


@pytest.mark.asyncio
async def test_truncated_stream_is_typed(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"type":"message_start","message":{}}\n',
            request=request,
        )

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    with pytest.raises(AnthropicStreamError):
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]
    await client.aclose()


@pytest.mark.asyncio
async def test_malformed_nested_sse_is_typed(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"type":"message_delta","delta":"bad"}\n\n',
            request=request,
        )

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    with pytest.raises(AnthropicStreamError):
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]
    await client.aclose()


@pytest.mark.asyncio
async def test_concurrent_refreshes_share_rotating_token_lock(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": "rotated-access",
                "refresh_token": "rotated-refresh",
                "expires_in": 3600,
            },
            request=request,
        )

    store = AnthropicCredentialStore(tmp_path / "zeta.json", token_url="https://test.invalid/token")
    store.save(OAuthTokens("expired-access", "rotating-refresh", 1))
    client = client_for(handler)

    assert await asyncio.gather(store.access_token(client), store.access_token(client)) == [
        "rotated-access",
        "rotated-access",
    ]
    assert len(requests) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_cancellation_survives_failing_owned_client_cleanup(
    tmp_path: Path,
) -> None:
    stopped = asyncio.Event()

    class Response:
        status_code = 200

        async def aiter_lines(self):
            await stopped.wait()
            yield ""

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            raise RuntimeError("stream cleanup failed")

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

        async def aclose(self):
            raise RuntimeError("cleanup failed")

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    with patch("zeta.anthropic.httpx.AsyncClient", return_value=Client()):
        task = asyncio.create_task(
            anext(AnthropicBackend(token_store=store).complete([], []))
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
