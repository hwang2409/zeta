import asyncio
import base64
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

import zeta.providers.anthropic as anthropic_module
from zeta.providers.anthropic import (
    AnthropicAuthError,
    AnthropicBackend,
    AnthropicCredentialStore,
    AnthropicHTTPError,
    ANTHROPIC_MAX_IMAGE_BYTES,
    AnthropicStreamError,
    OAuthTokens,
    build_authorization_url,
    build_messages_payload,
)
from zeta.core.loop import AgentLoop
from zeta.core.store import ConversationStore
from zeta.types import (
    Message,
    MessageRole,
    RedactedThinkingContent,
    StreamEvent,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
    image_dimensions,
)
from zeta.prompts import load_identity


SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg-1","model":"claude-test","role":"assistant","usage":{"input_tokens":12}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig-1"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

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
"""


def client_for(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def png_block(*, caption: str | None = None, data: bytes | None = None) -> dict[str, object]:
    png = data or bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
    )
    block: dict[str, object] = {
        "type": "image",
        "data": base64.b64encode(png).decode(),
        "mimeType": "image/png",
    }
    if caption is not None:
        block["caption"] = caption
    return block


def webp_data(chunk_type: bytes, chunk_data: bytes) -> bytes:
    body = b"WEBP" + chunk_type + len(chunk_data).to_bytes(4, "little") + chunk_data
    return b"RIFF" + (len(body) + 4).to_bytes(4, "little") + body


@pytest.mark.parametrize(
    ("chunk_type", "chunk_data", "dimensions"),
    [
        (b"VP8X", b"\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00", (1, 1)),
        (b"VP8L", b"/\x00\x00\x00\x00", (1, 1)),
        (b"VP8 ", b"\x00\x00\x00\x9d\x01\x2a\x01\x00\x01\x00", (1, 1)),
    ],
)
def test_image_dimensions_supports_webp_headers(
    chunk_type: bytes, chunk_data: bytes, dimensions: tuple[int, int]
) -> None:
    data = webp_data(chunk_type, chunk_data)
    block = {
        "type": "image",
        "data": base64.b64encode(data).decode(),
        "mimeType": "image/webp",
    }
    assert image_dimensions(block) == dimensions


def test_anthropic_sends_valid_image_without_dimensions_natively() -> None:
    block = {
        "type": "image",
        "data": base64.b64encode(b"\xff\xd8\xff").decode(),
        "mimeType": "image/jpeg",
    }
    payload = build_messages_payload(
        [Message(MessageRole.TOOL_RESULT, tool_result=ToolResult("call-1", "stale", content_blocks=[block]))],
        [],
        model="claude-test",
        max_tokens=100,
    )

    content = payload["messages"][0]["content"][0]["content"]
    assert isinstance(content, list)
    assert content[0]["type"] == "image"


def test_anthropic_falls_back_for_oversized_image_dimensions() -> None:
    data = bytearray(bytes.fromhex(
        "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
        "0000000d49444154789c6360f8cf00000004000101a2e0c4b00000000049454e44ae426082"
    ))
    data[16:20] = (8001).to_bytes(4, "big")
    block = png_block(data=bytes(data))
    payload = build_messages_payload(
        [Message(MessageRole.TOOL_RESULT, tool_result=ToolResult("call-1", "stale", content_blocks=[block]))],
        [],
        model="claude-test",
        max_tokens=100,
    )

    content = payload["messages"][0]["content"][0]["content"]
    assert isinstance(content, str)
    assert "dimensions are 8001x1" in content
    assert "limit is 8000x8000" in content


def test_anthropic_sends_supported_tool_images_as_native_blocks() -> None:
    payload = build_messages_payload(
        [
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult(
                    "call-1",
                    "stale",
                    content_blocks=[
                        {"type": "text", "text": "answer", "truncated": False, "full_size": 6},
                        png_block(caption="plot"),
                    ],
                ),
            )
        ],
        [],
        model="claude-test",
        max_tokens=100,
    )

    content = payload["messages"][0]["content"][0]["content"]
    assert content[0] == {"type": "text", "text": "answer"}
    assert content[1] == {"type": "text", "text": "caption: plot"}
    assert content[2]["type"] == "image"
    assert content[2]["source"]["media_type"] == "image/png"


@pytest.mark.parametrize(
    ("block", "note"),
    [
        (png_block(data=b"x" * (ANTHROPIC_MAX_IMAGE_BYTES + 1)), "limit is"),
        ({**png_block(), "mimeType": "image/tiff"}, "unsupported media type"),
    ],
)
def test_anthropic_falls_back_for_images_outside_native_limits(
    block: dict[str, object], note: str
) -> None:
    payload = build_messages_payload(
        [Message(MessageRole.TOOL_RESULT, tool_result=ToolResult("call-1", "stale", content_blocks=[block]))],
        [],
        model="claude-test",
        max_tokens=100,
    )

    content = payload["messages"][0]["content"][0]["content"]
    assert isinstance(content, str)
    assert note in content
    assert block["data"] not in content


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
    assert request_payload["max_tokens"] == 16384
    assert request_payload["thinking"] == {
        "type": "enabled",
        "budget_tokens": 8192,
    }
    assert "interleaved-thinking-2025-05-14" in requests[0].headers["anthropic-beta"]
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
async def test_agent_loop_sends_one_zeta_identity_after_oauth_spoof(
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
    loop = AgentLoop(backend, ConversationStore(tmp_path / "sessions"))

    async for _ in loop.run_turn("hi"):
        pass

    payload = json.loads(requests[0].content)
    assert payload["system"][0]["text"].startswith("You are Claude Code")
    assert payload["system"][1]["text"] == load_identity()
    assert sum(block["text"].startswith("You are zeta") for block in payload["system"]) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_preserves_anthropic_cache_usage_fields(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=SSE.replace(
                '"input_tokens":12',
                '"input_tokens":12,"cache_read_input_tokens":8,'
                '"cache_creation_input_tokens":2',
            ),
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

    assert events[-1].data["usage"] == {
        "input_tokens": 12,
        "cache_read_input_tokens": 8,
        "cache_creation_input_tokens": 2,
        "output_tokens": 4,
    }
    await client.aclose()


@pytest.mark.asyncio
async def test_401_refreshes_token_and_retries_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    statuses = iter((401, 200))
    refreshes: list[httpx.AsyncClient] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        status = next(statuses)
        if status == 200:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=SSE,
                request=request,
            )
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("stale-access", "refresh-test", 4_000_000_000))

    async def refresh_token(client: httpx.AsyncClient) -> str:
        refreshes.append(client)
        return "fresh-access"

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    client = client_for(handler)
    events = [
        event
        async for event in AnthropicBackend(
            client=client, token_store=store, base_url="https://test.invalid/v1/messages"
        ).complete([], [])
    ]

    assert len(refreshes) == 1
    assert len(requests) == 2
    assert requests[0].headers["authorization"] == "Bearer stale-access"
    assert requests[1].headers["authorization"] == "Bearer fresh-access"
    assert events[-1].message is not None
    await client.aclose()


@pytest.mark.asyncio
async def test_second_401_fails_loudly_without_a_retry_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    refreshes: list[httpx.AsyncClient] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("stale-access", "refresh-test", 4_000_000_000))

    async def refresh_token(client: httpx.AsyncClient) -> str:
        refreshes.append(client)
        return "fresh-access"

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    client = client_for(handler)
    with pytest.raises(AnthropicAuthError, match=r"Anthropic.*zeta login"):
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]

    assert len(requests) == 2
    assert len(refreshes) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_401_refresh_failure_propagates_without_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("stale-access", "refresh-test", 4_000_000_000))

    async def refresh_token(client: httpx.AsyncClient) -> str:
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    client = client_for(handler)
    with pytest.raises(RuntimeError, match="refresh failed"):
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]

    assert len(requests) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_non_401_error_does_not_refresh_or_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[httpx.Request] = []
    refreshes: list[httpx.AsyncClient] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "server"}}, request=request)

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))

    async def refresh_token(client: httpx.AsyncClient) -> str:
        refreshes.append(client)
        return "fresh-access"

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    client = client_for(handler)
    with pytest.raises(AnthropicHTTPError, match="500"):
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]

    assert len(requests) == 1
    assert refreshes == []
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
        max_tokens=4096,
        thinking_budget=2048,
    )

    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][0]["input_schema"] == {"type": "object"}
    assert payload["messages"][-1]["role"] == "user"
    assert payload["messages"][-1]["content"][0]["text"] == "run"
    assert payload["messages"][-1]["content"][0]["cache_control"] == {
        "type": "ephemeral"
    }


def test_anthropic_flattens_non_text_tool_blocks_at_provider_boundary() -> None:
    payload = build_messages_payload(
        [
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult(
                    "call-1",
                    "stale",
                    content_blocks=[
                        {
                            "type": "image",
                            "data": "aGVsbG8=",
                            "mimeType": "image/png",
                        },
                        {
                            "type": "resource",
                            "resource": {
                                "uri": "file:///tmp/note.txt",
                                "text": "note",
                            },
                        },
                    ],
                ),
            )
        ],
        [],
        model="claude-test",
        max_tokens=4096,
        thinking_budget=2048,
    )

    assert payload["messages"][0]["content"][0]["content"] == (
        "[image block] media_type=image/png bytes=5 fallback=invalid image data\n"
        "[resource: file:///tmp/note.txt]"
    )


@pytest.mark.asyncio
async def test_backend_keeps_four_cache_breakpoints_governed(tmp_path: Path) -> None:
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
                Message(MessageRole.SYSTEM, [TextContent("stable")]),
                Message(MessageRole.USER, [TextContent("run")]),
                Message(
                    MessageRole.TOOL_RESULT,
                    tool_result=ToolResult("call-1", "result"),
                ),
            ],
            [{"name": "read", "parameters": {"type": "object"}}],
        )
    ]

    del events
    payload = json.loads(requests[0].content)
    assert payload["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert payload["system"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["tools"][-1]["cache_control"] == {"type": "ephemeral"}
    assert payload["messages"][-1]["content"][-1]["cache_control"] == {
        "type": "ephemeral"
    }
    assert sum(
        isinstance(value, dict) and "cache_control" in value
        for section in (payload["system"], payload["tools"], payload["messages"])
        for value in section
        if isinstance(value, dict)
    ) == 3
    assert sum(
        isinstance(block, dict) and "cache_control" in block
        for message in payload["messages"]
        for block in message["content"]
    ) == 1
    await client.aclose()


def test_empty_system_prompt_is_omitted_from_payload() -> None:
    payload = build_messages_payload(
        [
            Message(MessageRole.SYSTEM, [TextContent("  ")]),
            Message(MessageRole.USER, [TextContent("run")]),
        ],
        [],
        model="claude-test",
        max_tokens=4096,
        thinking_budget=2048,
    )

    assert "system" not in payload


def test_authorization_url_contains_validated_redirect_uri() -> None:
    url = build_authorization_url("state", "challenge", "http://localhost/callback")
    assert "redirect_uri=http%3A%2F%2Flocalhost%2Fcallback" in url


def test_claude_keychain_bootstrap_reads_oauth_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[tuple[object, ...], dict[str, object]]] = []

    def run(*args: object, **kwargs: object) -> object:
        calls.append((args, kwargs))
        return type("Result", (), {"returncode": 0, "stdout": json.dumps({
            "claudeAiOauth": {
                "accessToken": "keychain-access",
                "refreshToken": "keychain-refresh",
                "expiresAt": 4_000_000_000,
            }
        })})()

    monkeypatch.setattr(anthropic_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(anthropic_module.sys, "platform", "darwin")
    monkeypatch.setattr(anthropic_module.subprocess, "run", run)
    tokens = AnthropicCredentialStore(tmp_path / "zeta.json").bootstrap()

    assert tokens == OAuthTokens("keychain-access", "keychain-refresh", 4_000_000_000)
    assert calls == [
        (
            (["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],),
            {
                "capture_output": True,
                "check": False,
                "env": {"PATH": anthropic_module.os.defpath},
                "text": True,
                "timeout": 2,
            },
        )
    ]


@pytest.mark.parametrize(
    "result",
    [
        type("Result", (), {"returncode": 1, "stdout": ""})(),
        type("Result", (), {"returncode": 0, "stdout": "not json"})(),
    ],
)
def test_claude_keychain_bootstrap_treats_invalid_output_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, result: object
) -> None:
    monkeypatch.setattr(anthropic_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(anthropic_module.sys, "platform", "darwin")
    monkeypatch.setattr(anthropic_module.subprocess, "run", lambda *args, **kwargs: result)

    assert AnthropicCredentialStore(tmp_path / "zeta.json").bootstrap() is None


def test_claude_keychain_bootstrap_treats_timeout_as_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def run(*args: object, **kwargs: object) -> object:
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(anthropic_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(anthropic_module.sys, "platform", "darwin")
    monkeypatch.setattr(anthropic_module.subprocess, "run", run)

    assert AnthropicCredentialStore(tmp_path / "zeta.json").bootstrap() is None


def test_claude_file_bootstrap_wins_over_keychain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    (claude_dir / ".credentials.json").write_text(json.dumps({
        "claudeAiOauth": {
            "accessToken": "file-access",
            "refreshToken": "file-refresh",
            "expiresAt": 4_000_000_000,
        }
    }))
    monkeypatch.setattr(anthropic_module.Path, "home", lambda: tmp_path)
    monkeypatch.setattr(anthropic_module.sys, "platform", "darwin")
    monkeypatch.setattr(
        anthropic_module.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("keychain should not be queried"),
    )

    assert AnthropicCredentialStore(tmp_path / "zeta.json").bootstrap() == OAuthTokens(
        "file-access", "file-refresh", 4_000_000_000
    )


def test_anthropic_http_error_includes_safe_truncated_body() -> None:
    body = json.dumps(
        {
            "error": {
                "message": "unsupported request " + "x" * 400,
                "access_token": "anthropic-secret",
            }
        }
    ).encode()

    error = anthropic_module._http_error(400, body)

    assert "unsupported request" in str(error)
    assert "anthropic-secret" not in str(error)
    assert len(anthropic_module.error_body_excerpt(body)) == 300


def test_anthropic_http_error_redacts_markers_in_valid_json_values() -> None:
    body = json.dumps(
        {
            "error": {
                "message": (
                    "access-token=access-secret refresh-token=refresh-secret "
                    "authorization=authorization-secret"
                )
            }
        }
    ).encode()

    error = anthropic_module._http_error(400, body)

    assert "access-secret" not in str(error)
    assert "refresh-secret" not in str(error)
    assert "authorization-secret" not in str(error)


@pytest.mark.asyncio
async def test_anthropic_sse_error_redacts_authorization_marker(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"type":"error","error":{"type":"api_error",'
                '"message":"authorization=authorization-secret"}}\n\n'
            ),
            request=request,
        )

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    with pytest.raises(AnthropicStreamError) as raised:
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]

    assert "authorization-secret" not in str(raised.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_sse_error_redacts_bearer_authorization(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"type":"error","error":{"type":"api_error",'
                '"message":"authorization: Bearer sse-authorization-secret"}}\n\n'
            ),
            request=request,
        )

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    with pytest.raises(AnthropicStreamError) as raised:
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]

    assert "sse-authorization-secret" not in str(raised.value)
    assert "Bearer" not in str(raised.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_anthropic_sse_error_redacts_multiline_authorization(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=(
                'data: {"type":"error","error":\n'
                'data: {"type":"api_error","message":"authorization: Bearer\\n'
                'newline-sse-marker"}}\n\n'
            ),
            request=request,
        )

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    with pytest.raises(AnthropicStreamError) as raised:
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]

    assert "newline-sse-marker" not in str(raised.value)
    assert "Bearer" not in str(raised.value)
    await client.aclose()


@pytest.mark.parametrize("probe", ["block", "delta", "follows", "precedes"])
def test_anthropic_provider_types_do_not_enter_errors(probe: str) -> None:
    marker = f"anthropic-{probe}-marker"
    with pytest.raises(AnthropicStreamError) as raised:
        if probe == "block":
            anthropic_module._translate_event(
                "message",
                {
                    "type": "content_block_start",
                    "index": 0,
                    "content_block": {"type": marker},
                },
                {},
                set(),
                set(),
                {},
            )
        elif probe == "delta":
            anthropic_module._translate_event(
                "message",
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": marker},
                },
                {0: anthropic_module._BlockState("text")},
                {0},
                set(),
                {},
            )
        elif probe == "follows":
            anthropic_module._advance_message_state("stopped", marker)
        else:
            anthropic_module._advance_message_state("not-started", marker)

    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)


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
        max_tokens=4096,
        thinking_budget=2048,
    )

    assert payload["messages"][0]["content"] == [
        {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
        {"type": "redacted_thinking", "data": "opaque"},
    ]


def test_thinking_tool_turn_replays_assistant_blocks_before_tool_result() -> None:
    payload = build_messages_payload(
        [
            Message(MessageRole.USER, [TextContent("inspect this")]),
            Message(
                MessageRole.ASSISTANT,
                [
                    ThinkingContent("plan", "sig-1"),
                    RedactedThinkingContent("opaque"),
                    ToolUseContent(ToolCall("call-1", "read", {"path": "note.txt"})),
                ],
            ),
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult("call-1", "contents"),
            ),
        ],
        [{"name": "read", "parameters": {"type": "object"}}],
        model="claude-test",
        max_tokens=4096,
        thinking_budget=2048,
    )

    assert payload["messages"] == [
        {"role": "user", "content": [{"type": "text", "text": "inspect this"}]},
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "plan", "signature": "sig-1"},
                {"type": "redacted_thinking", "data": "opaque"},
                {
                    "type": "tool_use",
                    "id": "call-1",
                    "name": "read",
                    "input": {"path": "note.txt"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call-1",
                    "content": "contents",
                    "is_error": False,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
    ]


@pytest.mark.asyncio
async def test_http_failure_is_typed(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    with pytest.raises(AnthropicHTTPError):
        await anext(AnthropicBackend(client=client, token_store=store).complete([], []))
    await client.aclose()


@pytest.mark.asyncio
async def test_request_entry_transport_failure_is_typed(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret connection details", request=request)

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    with pytest.raises(AnthropicHTTPError) as raised:
        await anext(AnthropicBackend(client=client, token_store=store).complete([], []))
    assert "secret connection details" not in str(raised.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_early_stream_end_is_salvaged(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        'data: {"type":"message_start","message":{}}\n',
    )

    assert events[-1].type is StreamEventType.MESSAGE_END
    assert events[-1].data["truncated"] is True
    assert events[-1].message == Message(MessageRole.ASSISTANT)


@pytest.mark.asyncio
async def test_message_stop_salvages_open_text_block(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )

    assert events[-1].message == Message(
        MessageRole.ASSISTANT, [TextContent("partial")]
    )
    assert events[-1].data["truncated"] is True


@pytest.mark.asyncio
async def test_message_stop_salvages_parseable_open_tool_block(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-1","name":"read"}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"README.md\\"}"}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )

    assert events[-1].message == Message(
        MessageRole.ASSISTANT,
        [ToolUseContent(ToolCall("call-1", "read", {"path": "README.md"}))],
    )
    assert events[-1].data["truncated"] is True


@pytest.mark.asyncio
async def test_message_stop_drops_unparseable_open_tool_block(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-1","name":"read"}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"README.md\\""}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )

    assert events[-1].message == Message(MessageRole.ASSISTANT)
    assert events[-1].data == {
        "truncated": True,
        "dropped_tool_calls": 1,
        "usage": {},
        "stop_reason": None,
    }


@pytest.mark.asyncio
async def test_message_stop_salvages_mixed_open_blocks(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"partial"}}',
                "",
                'data: {"type":"content_block_start","index":1,"content_block":{"type":"tool_use","id":"call-1","name":"read"}}',
                "",
                'data: {"type":"content_block_delta","index":1,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"README.md\\""}}',
                "",
                'data: {"type":"content_block_start","index":2,"content_block":{"type":"redacted_thinking","data":"opaque"}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )

    assert events[-1].message == Message(
        MessageRole.ASSISTANT,
        [TextContent("partial"), RedactedThinkingContent("opaque")],
    )
    assert events[-1].data["dropped_tool_calls"] == 1


@pytest.mark.asyncio
async def test_eof_after_empty_text_start_omits_text_block(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        'data: {"type":"message_start","message":{}}\n\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n',
    )

    assert events[-1].message == Message(MessageRole.ASSISTANT)
    assert events[-1].data["truncated"] is True


@pytest.mark.asyncio
async def test_eof_after_tool_start_drops_tool_call(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        'data: {"type":"message_start","message":{}}\n\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-1","name":"read"}}\n',
    )

    assert events[-1].message == Message(MessageRole.ASSISTANT)
    assert events[-1].data["dropped_tool_calls"] == 1


@pytest.mark.asyncio
async def test_eof_keeps_open_redacted_thinking(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        'data: {"type":"message_start","message":{}}\n\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"redacted_thinking","data":"opaque"}}\n',
    )

    assert events[-1].message == Message(
        MessageRole.ASSISTANT, [RedactedThinkingContent("opaque")]
    )


@pytest.mark.asyncio
async def test_message_stop_keeps_unsigned_thinking_for_display(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"plan"}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"partial-sig"}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )

    assert events[-1].message == Message(
        MessageRole.ASSISTANT, [ThinkingContent("plan")]
    )
    assert events[-1].data["truncated"] is True


@pytest.mark.asyncio
async def test_early_stream_end_salvages_block_without_message_stop(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"cut off"}}',
            ]
        ),
    )

    assert events[-1].message == Message(
        MessageRole.ASSISTANT, [TextContent("cut off")]
    )
    assert events[-1].data["truncated"] is True


def test_salvaged_context_omits_unsigned_thinking_on_replay() -> None:
    payload = build_messages_payload(
        [
            Message(MessageRole.USER, [TextContent("continue")]),
            Message(
                MessageRole.ASSISTANT,
                [ThinkingContent("partial plan"), TextContent("partial answer")],
            ),
        ],
        [],
        model="claude-test",
        max_tokens=4096,
        thinking_budget=2048,
    )

    assert payload["messages"][-1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "partial answer"}],
    }


@pytest.mark.asyncio
async def test_closed_unsigned_thinking_remains_strict(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_closed_unsigned_thinking_with_open_sibling_remains_strict(
    tmp_path: Path,
) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_closed_malformed_tool_with_open_sibling_remains_strict(
    tmp_path: Path,
) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"call-1","name":"read"}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":"{\\"path\\":\\"README.md\\""}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_eof_message_delta_preserves_stop_reason_and_usage(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        'data: {"type":"message_start","message":{}}\n\n'
        'data: {"type":"message_delta","delta":{"stop_reason":"max_tokens"},"usage":{"output_tokens":7}}',
    )

    assert events[-1].data["stop_reason"] == "max_tokens"
    assert events[-1].data["usage"] == {"output_tokens": 7}


@pytest.mark.asyncio
async def test_two_turn_replay_omits_empty_salvaged_text_block(tmp_path: Path) -> None:
    truncated = (
        'data: {"type":"message_start","message":{}}\n\n'
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}\n'
    )
    requests: list[dict[str, object]] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        requests.append(payload)
        stream = truncated if len(requests) == 1 else SSE
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=stream,
            request=request,
        )

    store = ConversationStore(tmp_path / "sessions")
    token_store = AnthropicCredentialStore(tmp_path / "zeta.json")
    token_store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    client = client_for(handler)
    loop = AgentLoop(
        AnthropicBackend(client=client, token_store=token_store),
        store,
        tool_schemas=[],
    )

    [event async for event in loop.run_turn("first")]
    [event async for event in loop.run_turn("second")]

    assert len(requests) == 2
    assert all(
        message != {"role": "assistant", "content": []}
        for message in requests[1]["messages"]
    )
    assert requests[1]["messages"][-1]["content"][-1]["text"] == "second"
    await client.aclose()


@pytest.mark.asyncio
async def test_malformed_nested_sse_is_typed(tmp_path: Path) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text='data: {"type":"message_start","message":{}}\n\ndata: {"type":"message_delta","delta":"bad"}\n\ndata: {"type":"message_stop"}\n\n',
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
    with patch("zeta.providers.anthropic.httpx.AsyncClient", return_value=Client()):
        task = asyncio.create_task(
            anext(AnthropicBackend(token_store=store).complete([], []))
        )
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_cancellation_during_owned_client_close_is_not_swallowed(
    tmp_path: Path,
) -> None:
    close_started = asyncio.Event()
    never = asyncio.Event()

    class Response:
        status_code = 200

        async def aiter_lines(self):
            for line in SSE.splitlines():
                yield line

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

        async def aclose(self):
            close_started.set()
            await never.wait()

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    async def consume() -> list[object]:
        return [event async for event in AnthropicBackend(token_store=store).complete([], [])]

    with patch("zeta.providers.anthropic.httpx.AsyncClient", return_value=Client()):
        task = asyncio.create_task(consume())
        await close_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task


@pytest.mark.asyncio
async def test_consumer_aclose_suppresses_failing_stream_cleanup(
    tmp_path: Path,
) -> None:
    class Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"type":"message_start","message":{}}'
            yield ""
            await asyncio.Event().wait()
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
            raise RuntimeError("client cleanup failed")

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    with patch("zeta.providers.anthropic.httpx.AsyncClient", return_value=Client()):
        stream = AnthropicBackend(token_store=store).complete([], [])
        await anext(stream)
        await stream.aclose()


@pytest.mark.asyncio
async def test_response_cleanup_cancellation_wins_over_stream_error(
    tmp_path: Path,
) -> None:
    cleanup_started = asyncio.Event()
    never = asyncio.Event()

    class Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"type":"message_delta","delta":"bad"}'
            yield ""

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            cleanup_started.set()
            await never.wait()

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    task = asyncio.create_task(
        anext(AnthropicBackend(client=Client(), token_store=store).complete([], []))
    )
    await cleanup_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as raised:
        await task
    assert isinstance(raised.value.__cause__, AnthropicStreamError)


@pytest.mark.asyncio
async def test_client_cleanup_cancellation_wins_over_stream_error(
    tmp_path: Path,
) -> None:
    cleanup_started = asyncio.Event()
    never = asyncio.Event()

    class Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"type":"message_delta","delta":"bad"}'
            yield ""

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

        async def aclose(self):
            cleanup_started.set()
            await never.wait()

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    with patch("zeta.providers.anthropic.httpx.AsyncClient", return_value=Client()):
        task = asyncio.create_task(
            anext(AnthropicBackend(token_store=store).complete([], []))
        )
        await cleanup_started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as raised:
            await task
    assert isinstance(raised.value.__cause__, AnthropicStreamError)


@pytest.mark.asyncio
async def test_response_cleanup_http_error_is_typed(tmp_path: Path) -> None:
    class Response:
        status_code = 200

        async def aiter_lines(self):
            for line in SSE.splitlines():
                yield line

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            raise httpx.ConnectError("response cleanup secret")

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    with pytest.raises(AnthropicHTTPError) as raised:
        [
            event
            async for event in AnthropicBackend(
                client=Client(), token_store=store
            ).complete([], [])
        ]
    assert "response cleanup secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_client_cleanup_http_error_is_typed(tmp_path: Path) -> None:
    class Response:
        status_code = 200

        async def aiter_lines(self):
            for line in SSE.splitlines():
                yield line

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

        async def aclose(self):
            raise httpx.TimeoutException("client cleanup secret")

    store = AnthropicCredentialStore(tmp_path / "zeta.json")
    store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    with patch("zeta.providers.anthropic.httpx.AsyncClient", return_value=Client()):
        with pytest.raises(AnthropicHTTPError) as raised:
            [
                event
                async for event in AnthropicBackend(token_store=store).complete([], [])
            ]
    assert "client cleanup secret" not in str(raised.value)


@pytest.mark.asyncio
async def test_cancel_mid_thinking_drops_partial_block_before_resume(
    tmp_path: Path,
) -> None:
    thinking_started = asyncio.Event()
    never = asyncio.Event()
    calls = 0

    class Response:
        status_code = 200

        def __init__(self, lines, *, wait_after=True):
            self.lines = lines
            self.wait_after = wait_after

        async def aiter_lines(self):
            for line in self.lines:
                yield line
            if self.wait_after:
                await never.wait()

    class Stream:
        def __init__(self, response):
            self.response = response

        async def __aenter__(self):
            return self.response

        async def __aexit__(self, exc_type, exc_value, traceback):
            return False

    class Client:
        def stream(self, method, url, *, headers, json):
            nonlocal calls
            calls += 1
            if calls == 1:
                return Stream(
                    Response(
                        [
                            'data: {"type":"message_start","message":{}}',
                            "",
                            'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
                            "",
                            'data: {"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"partial"}}',
                            "",
                        ]
                    )
                )
            return Stream(
                Response(
                    [
                        'data: {"type":"message_start","message":{}}',
                        "",
                        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                        "",
                        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"resumed"}}',
                        "",
                        'data: {"type":"content_block_stop","index":0}',
                        "",
                        'data: {"type":"message_stop"}',
                        "",
                    ],
                    wait_after=False,
                )
            )

    store = ConversationStore(tmp_path / "sessions")
    token_store = AnthropicCredentialStore(tmp_path / "zeta.json")
    token_store.save(OAuthTokens("access-test", "refresh-test", 4_000_000_000))
    backend = AnthropicBackend(client=Client(), token_store=token_store)
    loop = AgentLoop(backend, store)

    task = asyncio.create_task(
        consume_loop_turn(loop, thinking_started)
    )
    await thinking_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    messages = store.messages()
    assert len(messages) == 1
    assert calls == 1
    events = [event async for event in loop.run_turn("resume")]
    assert events[-1].type.name == "AGENT_END"
    assert calls == 2


async def _assert_malformed_stream_raises(tmp_path: Path, stream: str) -> None:
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
    with pytest.raises(AnthropicStreamError):
        [event async for event in AnthropicBackend(client=client, token_store=store).complete([], [])]
    await client.aclose()


async def _collect_anthropic_events(tmp_path: Path, stream: str) -> list[StreamEvent]:
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
        async for event in AnthropicBackend(client=client, token_store=store).complete([], [])
    ]
    await client.aclose()
    return events


async def consume_loop_turn(loop: AgentLoop, thinking_started: asyncio.Event) -> None:
    async for event in loop.run_turn("start"):
        if event.type.name == "MESSAGE_UPDATE" and event.content is not None:
            thinking_started.set()


@pytest.mark.asyncio
async def test_delta_without_block_start_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"bad"}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_message_stop_without_message_start_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_duplicate_message_start_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"message_start","message":{}}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_block_before_message_start_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_delta_before_message_start_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"bad"}}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_event_after_message_stop_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"message_stop"}',
                "",
                'data: {"type":"message_start","message":{}}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_delta_after_block_stop_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"bad"}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_message_stop_with_open_block_is_salvaged(tmp_path: Path) -> None:
    events = await _collect_anthropic_events(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )

    assert events[-1].message == Message(MessageRole.ASSISTANT)
    assert events[-1].data["truncated"] is True


@pytest.mark.asyncio
async def test_stop_without_block_start_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_duplicate_block_stop_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_duplicate_block_start_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_unknown_delta_type_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"unknown_delta"}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_unknown_block_type_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"unknown_block"}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_text_delta_inside_thinking_block_is_rejected(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"bad"}}',
                "",
                'data: {"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"sig"}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )


@pytest.mark.asyncio
async def test_tool_identifiers_must_be_strings(tmp_path: Path) -> None:
    await _assert_malformed_stream_raises(
        tmp_path,
        "\n".join(
            [
                'data: {"type":"message_start","message":{}}',
                "",
                'data: {"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":["tool-1"],"name":{"value":"read"},"input":{}}}',
                "",
                'data: {"type":"content_block_stop","index":0}',
                "",
                'data: {"type":"message_stop"}',
                "",
            ]
        ),
    )
