import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

import zeta.providers.transport as transport_module
from tests.support.credentials import anthropic_credentials, codex_credentials
from zeta.protocol.types import StreamEventType
from zeta.providers.anthropic import (
    AnthropicAuthError,
    AnthropicBackend,
    AnthropicHTTPError,
    OAuthTokens,
)
from zeta.providers.codex import CodexAuthError, CodexBackend, CodexHTTPError

ANTHROPIC_SSE = """event: message_start
data: {"type":"message_start","message":{"id":"msg-1","model":"claude-test","role":"assistant","usage":{"input_tokens":1}}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hello"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}

event: message_stop
data: {"type":"message_stop"}
"""


def _codex_access_token(account_id: str) -> str:
    def encode(value: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    claims = {
        "exp": 4_000_000_000,
        "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
    }
    return ".".join(map(encode, ({"alg": "none"}, claims, {"signature": "fixture"})))


def _codex_sse() -> str:
    events = [
        {"type": "response.created", "response": {"id": "response-test"}},
        {"type": "response.output_item.added", "output_index": 0, "item": {"type": "message", "id": "message-test", "role": "assistant"}},
        {"type": "response.content_part.added", "output_index": 0, "content_index": 0, "part": {"type": "output_text"}},
        {"type": "response.output_text.delta", "output_index": 0, "content_index": 0, "delta": "hello"},
        {"type": "response.output_text.done", "output_index": 0, "content_index": 0, "text": "hello"},
        {"type": "response.content_part.done", "output_index": 0, "content_index": 0},
        {"type": "response.output_item.done", "output_index": 0, "item": {"type": "message", "id": "message-test", "status": "completed", "role": "assistant", "content": [{"type": "output_text", "text": "hello"}]}},
        {"type": "response.completed", "response": {"id": "response-test", "status": "completed", "usage": {"input_tokens": 1, "output_tokens": 1}}},
    ]
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events)


@dataclass(frozen=True)
class ProviderContract:
    name: str
    backend_type: type[Any]
    auth_error_type: type[RuntimeError]
    auth_error_match: str
    http_error_type: type[RuntimeError]
    stale_token: str
    fresh_token: str
    success_sse: str

    def store(self, path: Path):
        credentials = (anthropic_credentials if self.name == "anthropic" else codex_credentials)(path)
        credentials.save(OAuthTokens(self.stale_token, "refresh-test", 4_000_000_000))
        return credentials


PROVIDERS = [
    pytest.param(
        ProviderContract("anthropic", AnthropicBackend, AnthropicAuthError, r"Anthropic.*zeta login", AnthropicHTTPError, "stale-access", "fresh-access", ANTHROPIC_SSE),
        id="anthropic",
    ),
    pytest.param(
        ProviderContract("codex", CodexBackend, CodexAuthError, r"Codex authentication failed after token refresh; run `zeta login --provider codex`", CodexHTTPError, _codex_access_token("stale-account"), _codex_access_token("fresh-account"), _codex_sse()),
        id="codex",
    ),
]


def _setup(provider: ProviderContract, tmp_path: Path, handler: Any):
    store = provider.store(tmp_path / f"{provider.name}.json")
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    backend = provider.backend_type(
        client=client,
        token_store=store,
        diagnostics_path=tmp_path / "logs" / "stream-diagnostics.jsonl",
        base_url=f"https://test.invalid/{provider.name}",
    )
    return store, client, backend


async def _collect(backend: Any):
    return [event async for event in backend.complete([], [])]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_401_refreshes_token_and_retries_once(provider, tmp_path, monkeypatch):
    requests, refreshes = [], []
    statuses = iter((401, 200))

    async def handler(request):
        requests.append(request)
        if next(statuses) == 200:
            return httpx.Response(200, text=provider.success_sse, request=request)
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store, client, backend = _setup(provider, tmp_path, handler)

    async def refresh_token(refresh_client):
        refreshes.append(refresh_client)
        return provider.fresh_token

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    events = await _collect(backend)
    assert refreshes == [client]
    assert len(requests) == 2
    assert requests[0].headers["authorization"] == f"Bearer {provider.stale_token}"
    assert requests[1].headers["authorization"] == f"Bearer {provider.fresh_token}"
    assert events[-1].message is not None
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_second_401_fails_loudly_without_a_retry_loop(provider, tmp_path, monkeypatch):
    requests, refreshes = [], []

    async def handler(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store, client, backend = _setup(provider, tmp_path, handler)

    async def refresh_token(refresh_client):
        refreshes.append(refresh_client)
        return provider.fresh_token

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    with pytest.raises(provider.auth_error_type, match=provider.auth_error_match):
        await _collect(backend)
    assert len(requests) == 2
    assert refreshes == [client]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_401_refresh_failure_propagates_without_retry(provider, tmp_path, monkeypatch):
    requests = []

    async def handler(request):
        requests.append(request)
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store, client, backend = _setup(provider, tmp_path, handler)

    async def refresh_token(_client):
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    with pytest.raises(RuntimeError, match="refresh failed"):
        await _collect(backend)
    assert len(requests) == 1
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_server_error_retries_without_refresh(provider, tmp_path, monkeypatch):
    requests, refreshes = [], []

    async def handler(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"message": "server"}}, request=request)

    store, client, backend = _setup(provider, tmp_path, handler)

    async def refresh_token(refresh_client):
        refreshes.append(refresh_client)
        return provider.fresh_token

    async def no_sleep(_delay):
        pass

    monkeypatch.setattr(transport_module.asyncio, "sleep", no_sleep)
    monkeypatch.setattr(store, "refresh_token", refresh_token)
    with pytest.raises(provider.http_error_type, match="500"):
        await _collect(backend)
    assert len(requests) == 5
    assert refreshes == []
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", PROVIDERS)
async def test_disconnect_after_headers_retries_before_first_event(provider, tmp_path, monkeypatch):
    requests = []

    async def no_sleep(_delay):
        pass

    monkeypatch.setattr(transport_module.asyncio, "sleep", no_sleep)

    class DisconnectStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            raise httpx.ReadError("peer closed", request=requests[-1])
            yield b""

    async def handler(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(200, stream=DisconnectStream(), request=request)
        return httpx.Response(200, text=provider.success_sse, request=request)

    _store, client, backend = _setup(provider, tmp_path, handler)
    events = await _collect(backend)
    assert len(requests) == 2
    assert [event.type for event in events].count(StreamEventType.RETRY) == 1
    assert not (tmp_path / "logs" / "stream-diagnostics.jsonl").exists()
    await client.aclose()
