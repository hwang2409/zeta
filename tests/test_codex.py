import asyncio
import base64
import json
from pathlib import Path

import httpx
import pytest

from zeta.anthropic import OAuthTokens
from zeta.codex import (
    DEFAULT_CODEX_MODEL,
    CodexAuthError,
    CodexBackend,
    CodexCredentialStore,
    CodexHTTPError,
    CodexStreamError,
    build_responses_payload,
    extract_account_id,
)
from zeta.types import (
    Message,
    MessageRole,
    StreamEventType,
    TextContent,
    ThinkingContent,
    ToolCall,
    ToolResult,
    ToolUseContent,
)


def access_token(account_id: str = "account-test") -> str:
    def encode(value: object) -> str:
        return base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")

    return ".".join(
        (
            encode({"alg": "none"}),
            encode({"https://api.openai.com/auth": {"chatgpt_account_id": account_id}}),
            encode({"signature": "fixture"}),
        )
    )


def event(event_type: str, **values: object) -> dict[str, object]:
    return {"type": event_type, **values}


def sse(events: list[dict[str, object]]) -> str:
    return "".join(
        f"event: {value['type']}\ndata: {json.dumps(value)}\n\n" for value in events
    )


def message_stream() -> list[dict[str, object]]:
    return [
        event(
            "response.created",
            response={"id": "response-test", "model": DEFAULT_CODEX_MODEL},
        ),
        event(
            "response.output_item.added",
            output_index=0,
            item={"type": "message", "id": "message-test", "role": "assistant"},
        ),
        event(
            "response.content_part.added",
            output_index=0,
            content_index=0,
            part={"type": "output_text"},
        ),
        event(
            "response.output_text.delta",
            output_index=0,
            content_index=0,
            delta="hello",
        ),
        event(
            "response.output_text.done",
            output_index=0,
            content_index=0,
            text="hello",
        ),
        event("response.content_part.done", output_index=0, content_index=0),
        event(
            "response.output_item.done",
            output_index=0,
            item={
                "type": "message",
                "id": "message-test",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        ),
        event(
            "response.completed",
            response={
                "id": "response-test",
                "status": "completed",
                "usage": {"input_tokens": 3, "output_tokens": 2},
            },
        ),
    ]


def client_for(stream: str):
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=stream,
            request=request,
        )

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def store_for(path: Path, token: str | None = None) -> CodexCredentialStore:
    store = CodexCredentialStore(path)
    store.save(OAuthTokens(token or access_token(), "refresh-fixture", 4_000_000_000))
    return store


@pytest.mark.asyncio
async def test_responses_stream_maps_text_usage_and_has_one_completion_boundary(
    tmp_path: Path,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            text=sse(message_stream()),
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    events = [
        item
        async for item in CodexBackend(
            client=client,
            token_store=store_for(tmp_path / "codex.json"),
            base_url="https://test.invalid/codex/responses",
        ).complete([Message(MessageRole.USER, [TextContent("hi")])], [])
    ]

    assert len(requests) == 1
    assert requests[0].headers["chatgpt-account-id"] == "account-test"
    assert requests[0].headers["authorization"].startswith("Bearer ")
    assert [item.type for item in events] == [
        StreamEventType.MESSAGE_START,
        StreamEventType.MESSAGE_UPDATE,
        StreamEventType.MESSAGE_END,
    ]
    assert len([item for item in events if item.type is StreamEventType.MESSAGE_END]) == 1
    assert events[-1].data["usage"] == {"input_tokens": 3, "output_tokens": 2}
    assert events[-1].message is not None
    assert events[-1].message.content == [TextContent("hello")]
    await client.aclose()


@pytest.mark.asyncio
async def test_responses_stream_maps_reasoning_and_tool_call_items(tmp_path: Path) -> None:
    stream = sse(
        [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={"type": "reasoning", "id": "reasoning-test"},
            ),
            event(
                "response.reasoning_summary_part.added",
                output_index=0,
                summary_index=0,
            ),
            event(
                "response.reasoning_summary_text.delta",
                output_index=0,
                summary_index=0,
                delta="plan",
            ),
            event(
                "response.reasoning_summary_text.done",
                output_index=0,
                summary_index=0,
                text="plan",
            ),
            event(
                "response.output_item.done",
                output_index=0,
                item={"type": "reasoning", "id": "reasoning-test"},
            ),
            event(
                "response.output_item.added",
                output_index=1,
                item={
                    "type": "function_call",
                    "id": "function-test",
                    "call_id": "call-test",
                    "name": "read",
                },
            ),
            event(
                "response.function_call_arguments.delta",
                output_index=1,
                delta='{"path":"README.md"}',
            ),
            event(
                "response.function_call_arguments.done",
                output_index=1,
                arguments='{"path":"README.md"}',
            ),
            event(
                "response.output_item.done",
                output_index=1,
                item={
                    "type": "function_call",
                    "id": "function-test",
                    "call_id": "call-test",
                    "name": "read",
                    "arguments": '{"path":"README.md"}',
                },
            ),
            event("response.completed", response={"usage": {"total_tokens": 8}}),
        ]
    )
    client = client_for(stream)
    events = [
        item
        async for item in CodexBackend(
            client=client, token_store=store_for(tmp_path / "codex.json")
        ).complete([], [])
    ]

    assert events[-1].message is not None
    assert events[-1].message.content == [
        ThinkingContent("plan"),
        ToolUseContent(ToolCall("call-test", "read", {"path": "README.md"})),
    ]
    await client.aclose()


def test_payload_maps_plan_messages_and_tools() -> None:
    payload = build_responses_payload(
        [
            Message(MessageRole.SYSTEM, [TextContent("system")]),
            Message(MessageRole.USER, [TextContent("run")]),
            Message(
                MessageRole.TOOL_RESULT,
                tool_result=ToolResult("call-test", "done"),
            ),
        ],
        [],
        model=DEFAULT_CODEX_MODEL,
        max_output_tokens=100,
    )
    assert payload["model"] == DEFAULT_CODEX_MODEL
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["instructions"] == "system"
    assert payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "run"}]},
        {"type": "function_call_output", "call_id": "call-test", "output": "done"},
    ]


def test_extract_account_id_ignores_account_id_outside_access_claim() -> None:
    assert extract_account_id(access_token("derived-account")) == "derived-account"
    with pytest.raises(CodexAuthError):
        extract_account_id("not-a-jwt")


@pytest.mark.asyncio
async def test_bootstrap_reads_codex_store_without_writing_codex_auth(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    auth_path.write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": access_token(),
                    "refresh_token": "refresh-fixture",
                },
                "account_id": "wrong-account",
            }
        )
    )
    original = auth_path.read_text()
    store = CodexCredentialStore(
        tmp_path / "zeta" / "codex.json", codex_auth=auth_path
    )
    client = client_for(sse(message_stream()))
    assert await store.access_token(client) == access_token()
    assert auth_path.read_text() == original
    assert oct((tmp_path / "zeta" / "codex.json").stat().st_mode & 0o777) == "0o600"
    await client.aclose()


@pytest.mark.asyncio
async def test_expired_codex_token_refreshes_under_shared_store_lock(tmp_path: Path) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "access_token": access_token("refreshed-account"),
                "refresh_token": "rotated-refresh",
                "expires_in": 3600,
            },
            request=request,
        )

    store = CodexCredentialStore(tmp_path / "codex.json", token_url="https://test.invalid/token")
    store.save(OAuthTokens(access_token(), "rotating-refresh", 1))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await asyncio.gather(store.access_token(client), store.access_token(client)) == [
        access_token("refreshed-account"),
        access_token("refreshed-account"),
    ]
    assert len(requests) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_http_failure_and_transport_failure_are_typed(tmp_path: Path) -> None:
    async def http_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(http_handler))
    with pytest.raises(CodexAuthError):
        await anext(
            CodexBackend(client=client, token_store=store_for(tmp_path / "codex.json")).complete([], [])
        )
    await client.aclose()

    async def transport_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("transport secret", request=request)

    client = httpx.AsyncClient(transport=httpx.MockTransport(transport_handler))
    with pytest.raises(CodexHTTPError) as raised:
        await anext(
            CodexBackend(client=client, token_store=store_for(tmp_path / "transport.json")).complete([], [])
        )
    assert "transport secret" not in str(raised.value)
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("before_start", "precedes response.created"),
        ("duplicate_start", "response.created is duplicated"),
        ("unknown_event", "unsupported Codex SSE event"),
        ("unknown_delta_item", "unknown output item"),
        ("stopped_delta", "inactive block"),
        ("duplicate_stop", "block stop is duplicated"),
        ("unknown_stop", "unknown block"),
        ("open_item_at_end", "open items"),
    ],
)
async def test_each_malformed_stream_fails_at_its_named_gate(
    tmp_path: Path, mutation: str, message: str
) -> None:
    events = message_stream()
    if mutation == "before_start":
        events = events[1:]
    elif mutation == "duplicate_start":
        events.insert(1, events[0].copy())
    elif mutation == "unknown_event":
        events.insert(1, event("response.unknown"))
    elif mutation == "unknown_delta_item":
        events[3] = {**events[3], "output_index": 1}
    elif mutation == "stopped_delta":
        events.insert(6, events[3].copy())
    elif mutation == "duplicate_stop":
        events.insert(6, events[5].copy())
    elif mutation == "unknown_stop":
        events.insert(6, event("response.content_part.done", output_index=0, content_index=1))
    elif mutation == "open_item_at_end":
        events.pop(6)
    client = client_for(sse(events))
    with pytest.raises(CodexStreamError, match=message):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / f"{mutation}.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_cancellation_wins_over_failing_cleanup(tmp_path: Path) -> None:
    cleanup_started = asyncio.Event()
    never = asyncio.Event()

    class Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"type":"response.created","response":{}}'
            yield ""
            yield 'data: {"type":"response.unknown"}'
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

    store = store_for(tmp_path / "codex.json")
    async def consume() -> list[StreamEventType]:
        return [
            item.type
            async for item in CodexBackend(client=Client(), token_store=store).complete([], [])
        ]

    task = asyncio.create_task(consume())
    await asyncio.wait_for(cleanup_started.wait(), timeout=1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError) as raised:
        await task
    assert isinstance(raised.value.__cause__, CodexStreamError)
