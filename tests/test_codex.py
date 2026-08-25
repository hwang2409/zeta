import asyncio
import base64
import http.server
import json
import multiprocessing
import threading
from pathlib import Path

import httpx
import pytest

import zeta.providers.codex as codex_module
from zeta.providers.anthropic import OAuthTokens
from zeta.providers.codex import (
    DEFAULT_CODEX_MODEL,
    CodexAuthError,
    CodexBackend,
    CodexCredentialStore,
    CodexHTTPError,
    CodexStreamError,
    build_responses_payload,
    extract_account_id,
)
from zeta.core.context import ContextAssembler
from zeta.core.slash import SlashStatus, _format_status
from zeta.core.store import ConversationStore
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


def test_codex_flattens_non_text_tool_blocks_at_provider_boundary() -> None:
    payload = build_responses_payload(
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
        model="codex-test",
    )

    assert payload["input"][0]["output"] == (
        "[image block]\n[resource: file:///tmp/note.txt]"
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


def message_stream(
    usage: dict[str, object] | None = None,
) -> list[dict[str, object]]:
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
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        ),
        event(
            "response.completed",
            response={
                "id": "response-test",
                "status": "completed",
                "usage": usage or {"input_tokens": 3, "output_tokens": 2},
            },
        ),
    ]


def reasoning_content_stream(
    *, summary: str = "", raw: str = "raw", completed_raw: str | None = None
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        event("response.created", response={"id": "response-test"}),
        event(
            "response.output_item.added",
            output_index=0,
            item={"type": "reasoning", "id": "reasoning-test"},
        ),
    ]
    if summary:
        events.extend(
            [
                event(
                    "response.reasoning_summary_part.added",
                    output_index=0,
                    summary_index=0,
                    part={"type": "summary_text"},
                ),
                event(
                    "response.reasoning_summary_text.delta",
                    output_index=0,
                    summary_index=0,
                    delta=summary,
                ),
                event(
                    "response.reasoning_summary_text.done",
                    output_index=0,
                    summary_index=0,
                    text=summary,
                ),
                event(
                    "response.reasoning_summary_part.done",
                    output_index=0,
                    summary_index=0,
                    part={"type": "summary_text", "text": summary},
                ),
            ]
        )
    events.extend(
        [
            event(
                "response.content_part.added",
                output_index=0,
                content_index=0,
                part={"type": "reasoning_text"},
            ),
            event(
                "response.reasoning_text.delta",
                output_index=0,
                content_index=0,
                delta=raw,
            ),
            event(
                "response.reasoning_text.done",
                output_index=0,
                content_index=0,
                text=raw,
            ),
            event(
                "response.content_part.done",
                output_index=0,
                content_index=0,
                part={"type": "reasoning_text"},
            ),
            event(
                "response.output_item.done",
                output_index=0,
                item={
                    "type": "reasoning",
                    "id": "reasoning-test",
                    "status": "completed",
                    "summary": (
                        [{"type": "summary_text", "text": summary}]
                        if summary
                        else []
                    ),
                    "content": [
                        {
                            "type": "reasoning_text",
                            "text": raw if completed_raw is None else completed_raw,
                        }
                    ],
                },
            ),
            event("response.completed"),
        ]
    )
    return events


def message_parts_stream(completed_item: dict[str, object]) -> list[dict[str, object]]:
    return [
        event("response.created", response={"id": "response-test"}),
        event(
            "response.output_item.added",
            output_index=0,
            item={"type": "message", "id": "message-test", "role": "assistant"},
        ),
        event(
            "response.content_part.added",
            output_index=0,
            content_index=0,
            part={"type": "refusal"},
        ),
        event(
            "response.refusal.delta",
            output_index=0,
            content_index=0,
            delta="first",
        ),
        event(
            "response.refusal.done",
            output_index=0,
            content_index=0,
            refusal="first",
        ),
        event(
            "response.content_part.done",
            output_index=0,
            content_index=0,
            part={"type": "refusal"},
        ),
        event(
            "response.content_part.added",
            output_index=0,
            content_index=1,
            part={"type": "output_text"},
        ),
        event(
            "response.output_text.delta",
            output_index=0,
            content_index=1,
            delta="second",
        ),
        event(
            "response.output_text.done",
            output_index=0,
            content_index=1,
            text="second",
        ),
        event(
            "response.content_part.done",
            output_index=0,
            content_index=1,
            part={"type": "output_text"},
        ),
        event("response.output_item.done", output_index=0, item=completed_item),
        event("response.completed"),
    ]


def reasoning_two_part_stream(
    *,
    summary_stream: tuple[str, ...] = (),
    summary_complete: tuple[str, ...] | None = None,
    raw_stream: tuple[str, ...] = (),
    raw_complete: tuple[str, ...] | None = None,
) -> list[dict[str, object]]:
    events: list[dict[str, object]] = [
        event("response.created", response={"id": "response-test"}),
        event(
            "response.output_item.added",
            output_index=0,
            item={"type": "reasoning", "id": "reasoning-test"},
        ),
    ]
    for index, text in enumerate(summary_stream):
        events.extend(
            [
                event(
                    "response.reasoning_summary_part.added",
                    output_index=0,
                    summary_index=index,
                    part={"type": "summary_text"},
                ),
                event(
                    "response.reasoning_summary_text.delta",
                    output_index=0,
                    summary_index=index,
                    delta=text,
                ),
                event(
                    "response.reasoning_summary_text.done",
                    output_index=0,
                    summary_index=index,
                    text=text,
                ),
                event(
                    "response.reasoning_summary_part.done",
                    output_index=0,
                    summary_index=index,
                    part={"type": "summary_text", "text": text},
                ),
            ]
        )
    for index, text in enumerate(raw_stream):
        events.extend(
            [
                event(
                    "response.content_part.added",
                    output_index=0,
                    content_index=index,
                    part={"type": "reasoning_text"},
                ),
                event(
                    "response.reasoning_text.delta",
                    output_index=0,
                    content_index=index,
                    delta=text,
                ),
                event(
                    "response.reasoning_text.done",
                    output_index=0,
                    content_index=index,
                    text=text,
                ),
                event(
                    "response.content_part.done",
                    output_index=0,
                    content_index=index,
                    part={"type": "reasoning_text"},
                ),
            ]
        )
    complete_summary = summary_stream if summary_complete is None else summary_complete
    complete_raw = raw_stream if raw_complete is None else raw_complete
    events.extend(
        [
            event(
                "response.output_item.done",
                output_index=0,
                item={
                    "type": "reasoning",
                    "id": "reasoning-test",
                    "status": "completed",
                    "summary": [
                        {"type": "summary_text", "text": text}
                        for text in complete_summary
                    ],
                    "content": [
                        {"type": "reasoning_text", "text": text}
                        for text in complete_raw
                    ],
                },
            ),
            event("response.completed"),
        ]
    )
    return events


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
async def test_responses_stream_normalizes_cached_input_usage(tmp_path: Path) -> None:
    usage = {
        "input_tokens": 20,
        "output_tokens": 4,
        "input_tokens_details": {"cached_tokens": 17},
    }
    client = client_for(sse(message_stream(usage)))
    events = [
        item
        async for item in CodexBackend(
            client=client,
            token_store=store_for(tmp_path / "codex.json"),
            base_url="https://test.invalid/codex/responses",
        ).complete([], [])
    ]

    assert events[-1].data["usage"] == {
        **usage,
        "input_tokens": 3,
        "cache_read_input_tokens": 17,
    }
    assert "cache_creation_input_tokens" not in events[-1].data["usage"]
    assembler = ContextAssembler(ConversationStore(tmp_path / "sessions"))
    assembler.record_usage(events[-1].data["usage"])
    assert assembler.uncached_input_tokens_this_session == 3
    assert assembler.cache_read_input_tokens_this_session == 17
    assert assembler.output_tokens_this_session == 4
    assert assembler.tokens_used_this_session == 24
    status = SlashStatus(
        session_id="session-1",
        provider="codex",
        model="test",
        retained_tail=1,
        tokens_used_this_session=assembler.tokens_used_this_session,
        tokens_in_current_context=None,
        compaction_marker_count=0,
        pending_approvals=(),
        cache_read_input_tokens=assembler.cache_read_input_tokens_this_session,
        cache_creation_input_tokens=0,
        uncached_input_tokens=assembler.uncached_input_tokens_this_session,
        output_tokens_this_session=assembler.output_tokens_this_session,
    )
    assert "prompt_cache_hit_rate: 85.0%" in _format_status(status)
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
                text=sse(message_stream()),
                request=request,
            )
        return httpx.Response(401, json={"error": {"message": "expired"}}, request=request)

    store = store_for(tmp_path / "codex.json", access_token("stale-account"))

    async def refresh_token(client: httpx.AsyncClient) -> str:
        refreshes.append(client)
        return access_token("fresh-account")

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    events = [
        item
        async for item in CodexBackend(
            client=client, token_store=store, base_url="https://test.invalid/codex/responses"
        ).complete([], [])
    ]

    assert len(refreshes) == 1
    assert len(requests) == 2
    assert requests[0].headers["authorization"] == f"Bearer {access_token('stale-account')}"
    assert requests[1].headers["authorization"] == f"Bearer {access_token('fresh-account')}"
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

    store = store_for(tmp_path / "codex.json", access_token("stale-account"))

    async def refresh_token(client: httpx.AsyncClient) -> str:
        refreshes.append(client)
        return access_token("fresh-account")

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(
        CodexAuthError,
        match=r"Codex authentication failed after token refresh; run `zeta login --provider codex`",
    ):
        [item async for item in CodexBackend(client=client, token_store=store).complete([], [])]

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

    store = store_for(tmp_path / "codex.json", access_token("stale-account"))

    async def refresh_token(client: httpx.AsyncClient) -> str:
        raise RuntimeError("refresh failed")

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(RuntimeError, match="refresh failed"):
        [item async for item in CodexBackend(client=client, token_store=store).complete([], [])]

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

    store = store_for(tmp_path / "codex.json")

    async def refresh_token(client: httpx.AsyncClient) -> str:
        refreshes.append(client)
        return access_token("fresh-account")

    monkeypatch.setattr(store, "refresh_token", refresh_token)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    with pytest.raises(CodexHTTPError, match="500"):
        [item async for item in CodexBackend(client=client, token_store=store).complete([], [])]

    assert len(requests) == 1
    assert refreshes == []
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
                item={
                    "type": "reasoning",
                    "id": "reasoning-test",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "plan"}],
                },
            ),
            event(
                "response.output_item.added",
                output_index=1,
                item={
                    "type": "function_call",
                    "id": "function-test",
                    "status": "completed",
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
                    "status": "completed",
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
async def test_responses_stream_maps_refusal_text_and_replays_item(tmp_path: Path) -> None:
    completed_item = {
        "type": "message",
        "id": "message-test",
        "status": "completed",
        "role": "assistant",
        "content": [{"type": "refusal", "refusal": "cannot help"}],
    }
    stream = sse(
        [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={"type": "message", "id": "message-test", "role": "assistant"},
            ),
            event(
                "response.content_part.added",
                output_index=0,
                content_index=0,
                part={"type": "refusal"},
            ),
            event(
                "response.refusal.delta",
                output_index=0,
                content_index=0,
                delta="cannot help",
            ),
            event(
                "response.refusal.done",
                output_index=0,
                content_index=0,
                refusal="cannot help",
            ),
            event(
                "response.content_part.done",
                output_index=0,
                content_index=0,
                part={"type": "refusal"},
            ),
            event(
                "response.output_item.done",
                output_index=0,
                item=completed_item,
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
    assert message.content == [TextContent("cannot help")]
    replayed = Message.from_dict(message.to_dict())
    payload = build_responses_payload(
        [replayed], [], model=DEFAULT_CODEX_MODEL
    )
    assert payload["input"] == [completed_item]
    await client.aclose()


@pytest.mark.asyncio
async def test_message_parts_allow_refusal_before_output_text(tmp_path: Path) -> None:
    completed_item = {
        "type": "message",
        "id": "message-test",
        "status": "completed",
        "role": "assistant",
        "content": [
            {"type": "refusal", "refusal": "first"},
            {"type": "output_text", "text": "second"},
        ],
    }
    stream = sse(
        [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={"type": "message", "id": "message-test", "role": "assistant"},
            ),
            event(
                "response.content_part.added",
                output_index=0,
                content_index=0,
                part={"type": "refusal"},
            ),
            event(
                "response.refusal.delta",
                output_index=0,
                content_index=0,
                delta="first",
            ),
            event(
                "response.refusal.done",
                output_index=0,
                content_index=0,
                refusal="first",
            ),
            event(
                "response.content_part.done",
                output_index=0,
                content_index=0,
                part={"type": "refusal"},
            ),
            event(
                "response.content_part.added",
                output_index=0,
                content_index=1,
                part={"type": "output_text"},
            ),
            event(
                "response.output_text.delta",
                output_index=0,
                content_index=1,
                delta="second",
            ),
            event(
                "response.output_text.done",
                output_index=0,
                content_index=1,
                text="second",
            ),
            event(
                "response.content_part.done",
                output_index=0,
                content_index=1,
                part={"type": "output_text"},
            ),
            event("response.output_item.done", output_index=0, item=completed_item),
            event("response.completed"),
        ]
    )
    client = client_for(stream)
    events = [
        item
        async for item in CodexBackend(
            client=client, token_store=store_for(tmp_path / "ordered-parts.json")
        ).complete([], [])
    ]

    assert events[-1].message is not None
    assert events[-1].message.content == [TextContent("firstsecond")]
    replayed = Message.from_dict(events[-1].message.to_dict())
    payload = build_responses_payload(
        [replayed], [], model=DEFAULT_CODEX_MODEL
    )
    assert payload["input"] == [completed_item]
    await client.aclose()


@pytest.mark.asyncio
async def test_message_parts_reject_shared_index_across_kinds(tmp_path: Path) -> None:
    stream = sse(
        [
            event("response.created", response={"id": "response-test"}),
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
                "response.content_part.added",
                output_index=0,
                content_index=0,
                part={"type": "refusal"},
            ),
        ]
    )
    client = client_for(stream)
    with pytest.raises(CodexStreamError, match="content block is duplicated"):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "shared-index.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_rejects_message_text_moved_between_parts(tmp_path: Path) -> None:
    completed_item = {
        "type": "message",
        "id": "message-test",
        "status": "completed",
        "role": "assistant",
        "content": [
            {"type": "refusal", "refusal": "firstsecond"},
            {"type": "output_text", "text": ""},
        ],
    }
    client = client_for(sse(message_parts_stream(completed_item)))
    with pytest.raises(CodexStreamError, match="message parts do not match deltas"):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "moved-message.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_reasoning_summary_and_content_parts_decode_together(tmp_path: Path) -> None:
    client = client_for(sse(reasoning_content_stream(summary="plan")))
    events = [
        item
        async for item in CodexBackend(
            client=client, token_store=store_for(tmp_path / "combined.json")
        ).complete([], [])
    ]

    assert events[-1].message is not None
    assert events[-1].message.content == [ThinkingContent("planraw")]
    await client.aclose()


@pytest.mark.asyncio
async def test_reasoning_text_round_trips_into_payload(tmp_path: Path) -> None:
    client = client_for(sse(reasoning_content_stream()))
    events = [
        item
        async for item in CodexBackend(
            client=client, token_store=store_for(tmp_path / "raw.json")
        ).complete([], [])
    ]

    message = events[-1].message
    assert message is not None
    payload = build_responses_payload(
        [Message.from_dict(message.to_dict())],
        [],
        model=DEFAULT_CODEX_MODEL,
    )
    assert payload["input"] == [
        {
            "type": "reasoning",
            "id": "reasoning-test",
            "status": "completed",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "raw"}],
        }
    ]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_rejects_completed_raw_reasoning_mismatch(tmp_path: Path) -> None:
    client = client_for(
        sse(reasoning_content_stream(raw="streamed", completed_raw="replayed"))
    )
    with pytest.raises(
        CodexStreamError, match="completed reasoning does not match its deltas"
    ):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "raw-mismatch.json")
            ).complete([], [])
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
            event(
                "response.output_item.done",
                output_index=0,
                item={
                    "type": "reasoning",
                    "id": "reasoning-summary",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "stop-only"}],
                },
            ),
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
            event(
                "response.output_item.done",
                output_index=1,
                item={
                    "type": "reasoning",
                    "id": "reasoning-raw",
                    "status": "completed",
                    "summary": [],
                    "content": [{"type": "reasoning_text", "text": "raw"}],
                },
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
    )
    assert payload["model"] == DEFAULT_CODEX_MODEL
    assert DEFAULT_CODEX_MODEL == "gpt-5.6-luna"
    assert payload["stream"] is True
    assert payload["store"] is False
    assert payload["instructions"] == "system"
    assert payload["input"] == [
        {"role": "user", "content": [{"type": "input_text", "text": "run"}]},
        {"type": "function_call_output", "call_id": "call-test", "output": "done"},
    ]


def test_empty_codex_instructions_keep_provider_fallback() -> None:
    payload = build_responses_payload([], [], model=DEFAULT_CODEX_MODEL)

    assert payload["instructions"] == "You are a helpful assistant."


def test_codex_http_error_includes_safe_truncated_body() -> None:
    body = json.dumps(
        {
            "detail": "unsupported request " + "x" * 400,
            "refresh_token": "codex-secret",
        }
    ).encode()

    error = codex_module._http_error(400, body)

    assert "unsupported request" in str(error)
    assert "codex-secret" not in str(error)
    assert len(codex_module.error_body_excerpt(body)) == 300


def test_codex_http_error_redacts_markers_in_valid_json_values() -> None:
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

    error = codex_module._http_error(400, body)

    assert "access-secret" not in str(error)
    assert "refresh-secret" not in str(error)
    assert "authorization-secret" not in str(error)


def test_payload_maps_name_only_tool_schema() -> None:
    payload = build_responses_payload(
        [Message(MessageRole.USER, [TextContent("run")])],
        [{"name": "read"}],
        model=DEFAULT_CODEX_MODEL,
    )

    assert payload["tools"] == [
        {
            "type": "function",
            "name": "read",
            "parameters": {"type": "object", "properties": {}},
            "strict": False,
        }
    ]


@pytest.mark.parametrize(
    "schema", [{"name": "read", "parameters": []}, {"name": "read", "input_schema": "bad"}]
)
def test_payload_rejects_malformed_tool_schema(schema: dict[str, object]) -> None:
    with pytest.raises(CodexHTTPError, match="parameters must be an object"):
        build_responses_payload(
            [], [schema], model=DEFAULT_CODEX_MODEL
        )


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
    )

    assert [item.get("type", item.get("role")) for item in payload["input"]] == [
        "reasoning",
        "assistant",
        "function_call",
    ]
    assert payload["input"][1]["content"][0]["type"] == "input_text"


def test_payload_replays_completed_codex_items_verbatim() -> None:
    output_items = [
        {
            "type": "reasoning",
            "id": "reasoning-test",
            "status": "completed",
            "summary": [],
            "encrypted_content": "opaque",
        },
        {
            "type": "message",
            "id": "message-test",
            "status": "completed",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "answer"}],
        },
        {
            "type": "function_call",
            "id": "function-test",
            "status": "completed",
            "call_id": "call-test",
            "name": "read",
            "arguments": "{}",
        },
    ]
    message = Message(
        MessageRole.ASSISTANT,
        [ThinkingContent("", "opaque"), TextContent("answer")],
        metadata={"codex_output_items": output_items},
    )

    persisted = Message.from_dict(message.to_dict())
    payload = build_responses_payload(
        [persisted], [], model=DEFAULT_CODEX_MODEL
    )

    assert payload["input"] == output_items


@pytest.mark.asyncio
async def test_encrypted_only_reasoning_round_trips_into_payload(tmp_path: Path) -> None:
    completed_item = {
        "type": "reasoning",
        "id": "reasoning-test",
        "status": "completed",
        "summary": [],
        "encrypted_content": "opaque",
    }
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
                item=completed_item,
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
    message = Message.from_dict(message.to_dict())
    payload = build_responses_payload(
        [message], [], model=DEFAULT_CODEX_MODEL
    )
    assert payload["input"] == [completed_item]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_rejects_completed_reasoning_summary_mismatch(tmp_path: Path) -> None:
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
                delta="streamed",
            ),
            event(
                "response.reasoning_summary_text.done",
                output_index=0,
                summary_index=0,
                text="streamed",
            ),
            event(
                "response.reasoning_summary_part.done",
                output_index=0,
                summary_index=0,
                part={"type": "summary_text", "text": "streamed"},
            ),
            event(
                "response.output_item.done",
                output_index=0,
                item={
                    "type": "reasoning",
                    "id": "reasoning-test",
                    "status": "completed",
                    "summary": [{"type": "summary_text", "text": "replayed"}],
                },
            ),
            event("response.completed"),
        ]
    )
    client = client_for(stream)
    with pytest.raises(
        CodexStreamError, match="completed reasoning does not match its deltas"
    ):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "summary-mismatch.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_rejects_summary_text_moved_between_parts(tmp_path: Path) -> None:
    client = client_for(
        sse(
            reasoning_two_part_stream(
                summary_stream=("a", "b"), summary_complete=("ab", "")
            )
        )
    )
    with pytest.raises(
        CodexStreamError, match="completed reasoning does not match its deltas"
    ):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "moved-summary.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_rejects_raw_text_moved_between_parts(tmp_path: Path) -> None:
    client = client_for(
        sse(
            reasoning_two_part_stream(
                raw_stream=("a", "b"), raw_complete=("ab", "")
            )
        )
    )
    with pytest.raises(
        CodexStreamError, match="completed reasoning does not match its deltas"
    ):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "moved-raw.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("mutation", ["delta", "done", "completed"])
async def test_stream_rejects_cross_kind_message_parts(
    tmp_path: Path, mutation: str
) -> None:
    events = message_stream()
    if mutation == "delta":
        events[3]["type"] = "response.refusal.delta"
    elif mutation == "done":
        events[4]["type"] = "response.refusal.done"
    else:
        events[6]["item"]["content"] = [  # type: ignore[index]
            {"type": "refusal", "refusal": "hello"}
        ]

    client = client_for(sse(events))
    with pytest.raises(CodexStreamError):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / f"{mutation}.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutation",
    ["delta", "completed_id", "completed_type", "completed_part_type"],
)
async def test_stream_rejects_invalid_item_identity(
    tmp_path: Path, mutation: str
) -> None:
    events = message_stream()
    if mutation == "delta":
        events[3]["item_id"] = "other-item"
    elif mutation == "completed_id":
        events[6]["item"]["id"] = "other-item"  # type: ignore[index]
    elif mutation == "completed_type":
        events[6]["item"]["type"] = "reasoning"  # type: ignore[index]
    elif mutation == "completed_part_type":
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


@pytest.mark.asyncio
async def test_stream_rejects_missing_completed_item(tmp_path: Path) -> None:
    events = message_stream()
    events[6].pop("item")
    client = client_for(sse(events))
    with pytest.raises(CodexStreamError, match="completed output item is invalid"):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "missing-item.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_rejects_in_progress_completed_message(tmp_path: Path) -> None:
    events = message_stream()
    events[6]["item"]["status"] = "in_progress"  # type: ignore[index]
    client = client_for(sse(events))
    with pytest.raises(CodexStreamError, match="completed message status is invalid"):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "in-progress.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["reasoning", "function_call"])
async def test_stream_rejects_in_progress_completed_items(
    tmp_path: Path, kind: str
) -> None:
    if kind == "reasoning":
        item = {
            "type": "reasoning",
            "id": "reasoning-test",
            "status": "in_progress",
            "summary": [],
        }
        events = [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={"type": "reasoning", "id": "reasoning-test"},
            ),
            event("response.output_item.done", output_index=0, item=item),
            event("response.completed"),
        ]
    else:
        item = {
            "type": "function_call",
            "id": "function-test",
            "status": "in_progress",
            "call_id": "call-test",
            "name": "read",
            "arguments": "{}",
        }
        events = [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={
                    "type": "function_call",
                    "id": "function-test",
                    "call_id": "call-test",
                    "name": "read",
                },
            ),
            event(
                "response.function_call_arguments.delta",
                output_index=0,
                delta="{}",
            ),
            event(
                "response.function_call_arguments.done",
                output_index=0,
                arguments="{}",
            ),
            event("response.output_item.done", output_index=0, item=item),
            event("response.completed"),
        ]

    client = client_for(sse(events))
    with pytest.raises(CodexStreamError, match="completed .* status is invalid"):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / f"{kind}.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
async def test_stream_rejects_non_completed_response_status(tmp_path: Path) -> None:
    events = message_stream()
    events[7]["response"]["status"] = "in_progress"  # type: ignore[index]
    client = client_for(sse(events))
    with pytest.raises(CodexStreamError, match="response completion status is invalid"):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / "response-status.json")
            ).complete([], [])
        ]
    await client.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["message", "reasoning", "tool"])
async def test_stream_rejects_incomplete_completed_items(tmp_path: Path, kind: str) -> None:
    if kind == "message":
        events = message_stream()
        events[6]["item"] = {"type": "message", "id": "message-test"}
    elif kind == "reasoning":
        events = [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={"type": "reasoning", "id": "reasoning-test"},
            ),
            event(
                "response.output_item.done",
                output_index=0,
                item={"type": "reasoning", "id": "reasoning-test"},
            ),
            event("response.completed"),
        ]
    else:
        events = [
            event("response.created", response={"id": "response-test"}),
            event(
                "response.output_item.added",
                output_index=0,
                item={
                    "type": "function_call",
                    "id": "function-test",
                    "call_id": "call-a",
                    "name": "read",
                },
            ),
            event(
                "response.function_call_arguments.delta",
                output_index=0,
                delta="{}",
            ),
            event(
                "response.function_call_arguments.done",
                output_index=0,
                arguments="{}",
            ),
            event(
                "response.output_item.done",
                output_index=0,
                item={
                    "type": "function_call",
                    "id": "function-test",
                    "status": "completed",
                    "call_id": "call-b",
                    "name": "write",
                    "arguments": "{}",
                },
            ),
            event("response.completed"),
        ]

    client = client_for(sse(events))
    with pytest.raises(CodexStreamError):
        [
            item
            async for item in CodexBackend(
                client=client, token_store=store_for(tmp_path / f"{kind}.json")
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

    store = CodexCredentialStore(
        tmp_path / "codex.json", token_url="https://test.invalid/token"
    )
    store.save(OAuthTokens(access_token(), "rotating-refresh", 1))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    assert await asyncio.gather(store.access_token(client), store.access_token(client)) == [
        access_token("refreshed-account"),
        access_token("refreshed-account"),
    ]
    assert len(requests) == 1
    await client.aclose()


@pytest.mark.asyncio
async def test_codex_refresh_keeps_existing_token_when_response_does_not_rotate(
    tmp_path: Path,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "access_token": access_token("refreshed-account"),
                "expires_in": 3600,
            },
            request=request,
        )

    store = CodexCredentialStore(
        tmp_path / "codex.json", token_url="https://test.invalid/token"
    )
    store.save(OAuthTokens(access_token(), "non-rotating-refresh", 1))
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))

    assert await store.access_token(client) == access_token("refreshed-account")
    refreshed = store.read()
    assert refreshed is not None
    assert refreshed.refresh_token == "non-rotating-refresh"
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
    with pytest.raises(CodexHTTPError) as raised:
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
    events = message_stream()[:4] + [
        event(
            "response.output_item.done",
            output_index=0,
            item={
                "type": "message",
                "id": "message-test",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "hello"}],
            },
        )
    ]
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


@pytest.mark.parametrize("probe", ["unsupported", "before_start"])
def test_codex_provider_types_do_not_enter_errors(probe: str) -> None:
    marker = f"codex-{probe}-marker"
    with pytest.raises(CodexStreamError) as raised:
        codex_module._translate_event(
            marker,
            {"type": marker},
            "started" if probe == "unsupported" else "not-started",
            {},
            {},
            {},
            {},
        )

    assert marker not in str(raised.value)
    assert marker not in repr(raised.value)


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
