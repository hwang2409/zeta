from __future__ import annotations

from pathlib import Path

import httpx
import pytest

import zeta.tools.fetch as fetch_tool
import zeta.tools.websearch as websearch
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.tools import ToolRegistry
from zeta.types import ToolCall

_ASYNC_CLIENT = httpx.AsyncClient


def _mock_client(monkeypatch: pytest.MonkeyPatch, handler):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (fetch_tool.socket.AF_INET, fetch_tool.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )
    monkeypatch.setattr(
        fetch_tool.httpx,
        "AsyncClient",
        lambda **kwargs: _ASYNC_CLIENT(transport=transport, **kwargs),
    )


async def _execute_fetch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    response: httpx.Response,
    *,
    arguments: dict[str, object] | None = None,
    max_output_chars: int = 50_000,
):
    def handler(request: httpx.Request) -> httpx.Response:
        response.request = request
        return response

    _mock_client(monkeypatch, handler)
    registry = ToolRegistry(tmp_path, max_output_chars=max_output_chars)
    return await registry.execute(
        ToolCall("fetch-1", "fetch", arguments or {"url": "example.com"})
    )


@pytest.mark.asyncio
async def test_fetch_html_extracts_text_and_expands_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/html"},
            text='<h1>Title</h1><script>bad()</script><p>Hello <a href="/next">there</a>.</p>',
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == (
        "Title\nHello there (https://example.com/next)."
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("content_type", "body"),
    [("text/plain", "plain text"), ("application/json", '{"ok": true}')],
)
async def test_fetch_passes_text_and_json_through(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_type: str,
    body: str,
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": content_type}, text=body),
    )

    assert result["content"][0]["text"] == body


@pytest.mark.asyncio
async def test_fetch_refuses_binary_and_large_responses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    binary = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": "application/pdf"}, content=b"pdf"),
    )
    large = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, content=b"123456"),
        arguments={"url": "example.com", "max_bytes": 5},
    )

    assert binary["isError"] is True
    assert "application/pdf" in binary["content"][0]["text"]
    assert large["isError"] is True
    assert "response too large" in large["content"][0]["text"]


@pytest.mark.asyncio
async def test_real_network_connections_are_blocked() -> None:
    async with httpx.AsyncClient() as client:
        with pytest.raises(AssertionError, match="real network"):
            await client.get("https://example.com")


@pytest.mark.asyncio
async def test_fetch_streams_body_and_ignores_lying_content_length(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-length": "5", "content-type": "text/plain"},
            content=b"123456",
        ),
        arguments={"url": "example.com", "max_bytes": 5},
    )

    assert result["isError"] is True
    assert "more than 5 bytes" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_validates_each_redirect_and_uses_final_url_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    responses = {
        "https://example.com/start": httpx.Response(
            302, headers={"location": "http://example.com/final"}
        ),
        "http://example.com/final": httpx.Response(
            200, headers={"content-type": "text/plain"}, text="done"
        ),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        response = responses[str(request.url)]
        response.request = request
        return response

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/start"})
    )

    assert result["isError"] is False
    assert result["content"][0]["text"].startswith(
        "notice: http URL is not encrypted\n\ndone"
    )


@pytest.mark.asyncio
async def test_fetch_refuses_non_http_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(302, headers={"location": "file:///tmp/secret"})
        response.request = request
        return response

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/start"})
    )

    assert result["isError"] is True
    assert "http or https" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_caps_manual_redirects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        response = httpx.Response(
            302,
            headers={"location": "https://example.com/loop"},
            request=request,
        )
        return response

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/start"})
    )

    assert result["isError"] is True
    assert "redirect limit exceeded (5)" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_allows_private_target_with_notice_and_blocks_metadata(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def private_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/plain"}, text="local", request=request)

    _mock_client(monkeypatch, private_handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (fetch_tool.socket.AF_INET, fetch_tool.socket.SOCK_STREAM, 6, "", ("192.168.1.5", 0))
        ],
    )
    allowed = await ToolRegistry(tmp_path).execute(
        ToolCall("fetch-1", "fetch", {"url": "http://printer.local"})
    )
    blocked = await ToolRegistry(tmp_path).execute(
        ToolCall("fetch-2", "fetch", {"url": "http://169.254.169.254/"})
    )

    assert allowed["isError"] is False
    assert "target resolves to a private or loopback address" in allowed["content"][0]["text"]
    assert blocked["isError"] is True
    assert "cloud metadata" in blocked["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_output_is_capped_with_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, text="x" * 50_100),
    )

    block = result["content"][0]
    assert block["truncated"] is True
    assert block["full_size"] == 50_100
    assert block["text"].endswith("\n...[output truncated]")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error",
    [
        httpx.TimeoutException("slow"),
        httpx.TooManyRedirects("loop", request=httpx.Request("GET", "https://x.test")),
    ],
)
async def test_fetch_timeout_and_redirect_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, error: Exception
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        raise error

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("fetch-1", "fetch", {"url": "example.com"})
    )

    assert result["isError"] is True
    expected = "timed out" if isinstance(error, httpx.TimeoutException) else "redirect limit"
    assert expected in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [404, 500])
async def test_fetch_http_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(status, text="failure"),
    )

    assert result["isError"] is True
    assert f"HTTP {status}" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_uses_declared_charset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/plain; charset=iso-8859-1"},
            content="café".encode("iso-8859-1"),
        ),
    )

    assert result["content"][0]["text"] == "café"


@pytest.mark.asyncio
async def test_websearch_parses_saved_duckduckgo_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = Path(__file__).parent / "fixtures" / "duckduckgo.html"
    body = fixture.read_text(encoding="utf-8")

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["q"] == "zeta"
        return httpx.Response(200, headers={"content-type": "text/html"}, text=body)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert result["structuredContent"] == {
        "results": [
            {
                "title": "First result",
                "url": "https://example.com/one",
                "snippet": "A short description.",
            }
        ]
    }


@pytest.mark.asyncio
async def test_websearch_empty_results_and_parse_failure() -> None:
    assert websearch.parse_search_results(
        '<div class="no-results__container result__title"><span class="no-results">'
        '<div class="no-results__message"><h1>No results found for <strong>query</strong>'
        '</h1></div></span></div>',
        max_results=8,
    ) == []
    with pytest.raises(ValueError, match="search backend failed"):
        websearch.parse_search_results(
            "<html><body>No results found</body></html>", max_results=8
        )
    with pytest.raises(ValueError, match="search backend failed"):
        websearch.parse_search_results(
            '<div class="no-results__container result__title"><span class="no-results">'
            '<div class="no-results__message"><h1>temporarily blocked</h1></div></span></div>',
            max_results=8,
        )


@pytest.mark.asyncio
async def test_discovery_and_approval_gate_network_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    async def forbidden(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        raise AssertionError("network handler should not run")

    _mock_client(monkeypatch, forbidden)
    store = ConversationStore(tmp_path)
    policy = ApprovalPolicy(always_deny={"fetch", "websearch"}, store=store)
    registry = ToolRegistry(
        tmp_path,
        approval_policy=policy,
        approval_store=store,
    )

    assert {"fetch", "websearch"} <= registry.definitions_by_name.keys()
    fetch_result = await registry.execute(
        ToolCall("fetch-1", "fetch", {"url": "example.com"})
    )
    search_result = await registry.execute(
        ToolCall("search-1", "websearch", {"query": "zeta"})
    )

    assert fetch_result["content"][0]["text"] == "tool execution denied"
    assert search_result["content"][0]["text"] == "tool execution denied"
    assert calls == 0
