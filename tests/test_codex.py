import asyncio
import base64
import http.server
import json
import multiprocessing
import threading
from pathlib import Path

import httpx
import pytest

import zeta.codex as codex_module
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
            encode(
                {
                    "exp": 4_000_000_000,
                    "https://api.openai.com/auth": {"chatgpt_account_id": account_id},
                }
            ),
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


def malformed_events(mutation: str) -> list[dict[str, object]]:
    created = event("response.created", response={"id": "response-test"})
    item = event(
        "response.output_item.added",
        output_index=0,
        item={"type": "message", "id": "message-test", "role": "assistant"},
    )
    part = event(
        "response.content_part.added",
        output_index=0,
        content_index=0,
        part={"type": "output_text"},
    )
    delta = event(
        "response.output_text.delta", output_index=0, content_index=0, delta="hello"
    )
    text_done = event(
        "response.output_text.done", output_index=0, content_index=0, text="hello"
    )
    part_done = event("response.content_part.done", output_index=0, content_index=0)
    if mutation == "before_start":
        return [item]
    if mutation == "duplicate_start":
        return [created, created.copy()]
    if mutation == "unknown_event":
        return [created, event("response.unknown")]
    if mutation == "unknown_delta_item":
        return [created, item, part, {**delta, "output_index": 1}]
    if mutation == "stopped_delta":
        return [created, item, part, delta, text_done, part_done, delta.copy()]
    if mutation == "duplicate_stop":
        return [created, item, part, delta, text_done, part_done, part_done.copy()]
    if mutation == "unknown_stop":
        return [created, item, part, delta, text_done, part_done, event(
            "response.content_part.done", output_index=0, content_index=1
        )]
    if mutation == "open_item_at_end":
        return [created, item, part, delta, text_done, part_done, event("response.completed")]
    raise AssertionError(mutation)


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
                "response.reasoning_summary_part.done",
                output_index=0,
                summary_index=0,
                part={"type": "summary_text", "text": "plan"},
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


@pytest.mark.asyncio
async def test_reasoning_summary_stop_only_and_raw_reasoning_text_are_durable(
    tmp_path: Path,
) -> None:
    stream = sse(
        [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={"type": "reasoning", "id": "reasoning-summary"},
            ),
            event(
                "response.reasoning_summary_part.added",
                output_index=0,
                summary_index=0,
                part={"type": "summary_text"},
            ),
            event(
                "response.reasoning_summary_part.done",
                output_index=0,
                summary_index=0,
                part={"type": "summary_text", "text": "stop-only"},
            ),
            event("response.output_item.done", output_index=0),
            event(
                "response.output_item.added",
                output_index=1,
                item={"type": "reasoning", "id": "reasoning-raw"},
            ),
            event(
                "response.reasoning_text.delta",
                output_index=1,
                content_index=0,
                delta="raw",
            ),
            event(
                "response.reasoning_text.done",
                output_index=1,
                content_index=0,
                text="raw",
            ),
            event("response.output_item.done", output_index=1),
            event("response.completed"),
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
        ThinkingContent("stop-only"),
        ThinkingContent("raw"),
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


def test_payload_preserves_assistant_output_item_order() -> None:
    payload = build_responses_payload(
        [
            Message(
                MessageRole.ASSISTANT,
                [
                    ThinkingContent("plan", "opaque"),
                    TextContent("answer"),
                    ToolUseContent(ToolCall("call-test", "read", {"path": "README.md"})),
                ],
            )
        ],
        [],
        model=DEFAULT_CODEX_MODEL,
        max_output_tokens=100,
    )

    assert [item.get("type", item.get("role")) for item in payload["input"]] == [
        "reasoning",
        "assistant",
        "function_call",
    ]


@pytest.mark.asyncio
async def test_encrypted_only_reasoning_round_trips_into_payload(tmp_path: Path) -> None:
    stream = sse(
        [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={
                    "type": "reasoning",
                    "id": "reasoning-test",
                    "encrypted_content": "opaque",
                },
            ),
            event(
                "response.output_item.done",
                output_index=0,
                item={"type": "reasoning", "id": "reasoning-test"},
            ),
            event("response.completed"),
        ]
    )
    client = client_for(stream)
    events = [
        item
        async for item in CodexBackend(
            client=client, token_store=store_for(tmp_path / "codex.json")
        ).complete([], [])
    ]

    message = events[-1].message
    assert message is not None
    assert message.content == [ThinkingContent("", "opaque")]
    payload = build_responses_payload(
        [message], [], model=DEFAULT_CODEX_MODEL, max_output_tokens=100
    )
    assert payload["input"][0]["encrypted_content"] == "opaque"
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation", ["delta", "completed_id", "completed_type", "completed_part_type"]
)
async def test_stream_rejects_mismatched_item_identity(
    tmp_path: Path, mutation: str
) -> None:
    events = message_stream()
    if mutation == "delta":
        events[3]["item_id"] = "other-item"
    elif mutation == "completed_id":
        events[6]["item"]["id"] = "other-item"  # type: ignore[index]
    elif mutation == "completed_type":
        events[6]["item"]["type"] = "reasoning"  # type: ignore[index]
    else:
        events[5]["part"] = {"type": "input_text"}

    client = client_for(sse(events))
    with pytest.raises(CodexStreamError):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / f"{mutation}.json")
            ).complete([], [])
        ]
    await client.aclose()


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


def test_bootstrap_requires_refresh_when_jwt_expiry_is_missing_or_past(
    tmp_path: Path,
) -> None:
    auth_path = tmp_path / "auth.json"
    header, _, signature = access_token().split(".")
    for name, payload in (
        ("missing", {"https://api.openai.com/auth": {"chatgpt_account_id": "a"}}),
        ("expired", {"exp": 1, "https://api.openai.com/auth": {"chatgpt_account_id": "a"}}),
    ):
        encoded_payload = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
        token = f"{header}.{encoded_payload}.{signature}"
        auth_path.write_text(
            json.dumps({"tokens": {"access_token": token, "refresh_token": "refresh"}})
        )
        store = CodexCredentialStore(tmp_path / f"{name}.json", codex_auth=auth_path)
        assert store.bootstrap() is not None
        assert store.bootstrap().expires_at <= 1


def _refresh_process(path: str, token_url: str, results: object) -> None:
    async def run() -> None:
        client = httpx.AsyncClient()
        try:
            token = await CodexCredentialStore(path, token_url=token_url).access_token(client)
            results.put(token)
        finally:
            await client.aclose()

    asyncio.run(run())


def test_expired_codex_token_refreshes_once_across_two_processes(tmp_path: Path) -> None:
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_POST(self) -> None:
            self.server.refresh_count += 1
            body = json.dumps(
                {
                    "access_token": access_token("refreshed-account"),
                    "refresh_token": "rotated-refresh",
                    "expires_in": 3600,
                }
            ).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.refresh_count = 0
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        path = tmp_path / "codex.json"
        CodexCredentialStore(path).save(OAuthTokens(access_token(), "refresh", 1))
        context = multiprocessing.get_context("spawn")
        results = context.Queue()
        processes = [
            context.Process(
                target=_refresh_process,
                args=(str(path), f"http://127.0.0.1:{server.server_port}/token", results),
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        for process in processes:
            process.join(timeout=5)
            assert process.exitcode == 0
        assert [results.get(timeout=1), results.get(timeout=1)] == [
            access_token("refreshed-account"),
            access_token("refreshed-account"),
        ]
        assert server.refresh_count == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=1)


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
        return httpx.Response(
            401,
            json={"error": {"message": "access-token-secret-value"}},
            request=request,
        )

    client = httpx.AsyncClient(transport=httpx.MockTransport(http_handler))
    with pytest.raises(CodexAuthError) as raised:
        await anext(
            CodexBackend(client=client, token_store=store_for(tmp_path / "codex.json")).complete([], [])
        )
    assert "access-token-secret-value" not in str(raised.value)
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
async def test_stream_error_does_not_expose_provider_body(tmp_path: Path) -> None:
    stream = sse([event("response.created", response={"id": "response-test"}), event(
        "error", error={"message": "access-token-secret-value"}
    )])
    client = client_for(stream)
    with pytest.raises(CodexStreamError) as raised:
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "codex.json")
            ).complete([], [])
        ]
    assert "access-token-secret-value" not in str(raised.value)
    await client.aclose()


@pytest.mark.asyncio
async def test_events_after_response_completion_are_rejected(tmp_path: Path) -> None:
    events = message_stream() + [event("keepalive")]
    client = client_for(sse(events))
    with pytest.raises(CodexStreamError, match="follows response completion"):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "codex.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_output_item_done_rejects_open_blocks(tmp_path: Path) -> None:
    events = message_stream()[:4] + [event("response.output_item.done", output_index=0)]
    client = client_for(sse(events))
    with pytest.raises(CodexStreamError, match="open blocks"):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "codex.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_non_http_stream_exception_is_not_remapped(tmp_path: Path) -> None:
    class Response:
        status_code = 200

        async def aiter_lines(self):
            yield 'data: {"type":"response.created","response":{}}'
            yield ""
            raise ValueError("stream parser bug")

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            return None

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

    with pytest.raises(ValueError, match="stream parser bug"):
        [
            item
            async for item in CodexBackend(
                client=Client(), token_store=store_for(tmp_path / "codex.json")
            ).complete([], [])
        ]


@pytest.mark.asyncio
async def test_non_http_cleanup_exception_is_not_remapped(tmp_path: Path) -> None:
    class Response:
        status_code = 200

        async def aiter_lines(self):
            for line in sse(message_stream()).splitlines():
                yield line

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            raise ValueError("cleanup bug")

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

    with pytest.raises(ValueError, match="cleanup bug"):
        [
            item
            async for item in CodexBackend(
                client=Client(), token_store=store_for(tmp_path / "codex.json")
            ).complete([], [])
        ]


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
    events = malformed_events(mutation)
    client = client_for(sse(events))
    with pytest.raises(CodexStreamError, match=message):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / f"{mutation}.json")
            ).complete([], [])
        ]
    await client.aclose()


class _CleanupStream:
    def __init__(self, events: list[dict[str, object]], cleanup_error: type[BaseException] | None) -> None:
        self.events = events
        self.cleanup_error = cleanup_error

    async def __aenter__(self):
        events = self.events

        class Response:
            status_code = 200

            async def aiter_lines(response_self):
                for line in sse(events).splitlines():
                    yield line

        return Response()

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.cleanup_error is not None:
            raise self.cleanup_error()


class _CleanupClient:
    def __init__(
        self,
        events: list[dict[str, object]],
        stream_error: type[BaseException] | None,
        client_error: type[BaseException] | None,
    ) -> None:
        self.events = events
        self.stream_error = stream_error
        self.client_error = client_error

    def stream(self, method, url, *, headers, json):
        return _CleanupStream(self.events, self.stream_error)

    async def aclose(self):
        if self.client_error is not None:
            raise self.client_error()


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", ["response", "client"])
@pytest.mark.parametrize("primary", [False, True])
@pytest.mark.parametrize("cleanup_error", [asyncio.CancelledError, GeneratorExit])
async def test_cleanup_control_exception_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    scope: str,
    primary: bool,
    cleanup_error: type[BaseException],
) -> None:
    events = (
        [
            event("response.created", response={"id": "response-test"}),
            event("response.unknown"),
        ]
        if primary
        else message_stream()
    )
    stream_cleanup = cleanup_error if scope == "response" else None
    client_cleanup = cleanup_error if scope == "client" else None
    client = _CleanupClient(events, stream_cleanup, client_cleanup)
    if scope == "client":
        monkeypatch.setattr(codex_module.httpx, "AsyncClient", lambda timeout=None: client)
        backend = CodexBackend(token_store=store_for(tmp_path / "codex.json"))
    else:
        backend = CodexBackend(
            client=client,
            token_store=store_for(tmp_path / "codex.json"),
        )
    with pytest.raises(cleanup_error) as raised:
        [item async for item in backend.complete([], [])]
    if primary:
        assert isinstance(raised.value.__cause__, CodexStreamError)
    else:
        assert raised.value.__cause__ is None


@pytest.mark.asyncio
async def test_consumer_aclose_finishes_without_leaking_cleanup(
    tmp_path: Path,
) -> None:
    cleanup_finished = asyncio.Event()
    never = asyncio.Event()

    class Response:
        status_code = 200

        async def aiter_lines(self):
            for line in sse(message_stream()[:1]).splitlines():
                yield line
            await never.wait()

    class Stream:
        async def __aenter__(self):
            return Response()

        async def __aexit__(self, exc_type, exc_value, traceback):
            cleanup_finished.set()

    class Client:
        def stream(self, method, url, *, headers, json):
            return Stream()

    iterator = CodexBackend(
        client=Client(), token_store=store_for(tmp_path / "codex.json")
    ).complete([], [])
    assert (await anext(iterator)).type is StreamEventType.MESSAGE_START
    await asyncio.wait_for(iterator.aclose(), timeout=1)
    assert cleanup_finished.is_set()


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
