from __future__ import annotations

import asyncio
import gzip
import zlib
from pathlib import Path

import httpx
import pytest

import zeta.tools.fetch as fetch_tool
import zeta.tools.websearch as websearch
from zeta.core.approval import ApprovalPolicy
from zeta.core.store import ConversationStore
from zeta.tools import ToolRegistry
from zeta.types import ToolCall, flatten_tool_content

_ASYNC_CLIENT = httpx.AsyncClient


class _ChunkedByteStream(httpx.AsyncByteStream):
    def __init__(self, content: bytes, *, first_chunk_size: int | None = None) -> None:
        self.content = content
        self.first_chunk_size = first_chunk_size

    async def __aiter__(self):
        split_at = self.first_chunk_size or max(1, len(self.content) // 2)
        yield self.content[:split_at]
        yield self.content[split_at:]

    async def aclose(self) -> None:
        return None


def _raw_deflate(value: bytes) -> bytes:
    compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
    return compressor.compress(value) + compressor.flush()


def _mock_client(
    monkeypatch: pytest.MonkeyPatch,
    handler,
    *,
    client_kwargs: dict[str, object] | None = None,
):
    transport = httpx.MockTransport(handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (fetch_tool.socket.AF_INET, fetch_tool.socket.SOCK_STREAM, 6, "", ("93.184.216.34", 0))
        ],
    )
    def make_client(**kwargs):
        if client_kwargs is not None:
            client_kwargs.update(kwargs)
        return _ASYNC_CLIENT(transport=transport, **kwargs)

    monkeypatch.setattr(fetch_tool.httpx, "AsyncClient", make_client)


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
async def test_fetch_decodes_gzip_response_from_raw_stream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(gzip.compress(b"compressed response")),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "compressed response"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("encoding", "compressed", "expected"),
    [
        ("gzip", gzip.compress(b"gzip response"), "gzip response"),
        ("deflate", zlib.compress(b"zlib response"), "zlib response"),
        (
            "deflate",
            _raw_deflate(b"raw response"),
            "raw response",
        ),
    ],
)
async def test_fetch_decodes_supported_content_encodings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    encoding: str,
    compressed: bytes,
    expected: str,
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": encoding,
            },
            stream=_ChunkedByteStream(compressed),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == expected


@pytest.mark.asyncio
async def test_fetch_decodes_raw_deflate_with_one_byte_first_chunk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "deflate",
            },
            stream=_ChunkedByteStream(
                _raw_deflate(b"raw response"), first_chunk_size=1
            ),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "raw response"


@pytest.mark.asyncio
async def test_fetch_marks_truncated_gzip_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    compressed = gzip.compress(b"truncated response")
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(compressed[:-1]),
        ),
    )

    assert result["isError"] is True
    assert result["content"][0]["truncated"] is True
    assert "stream ended early" in result["content"][0]["text"]


@pytest.mark.asyncio
@pytest.mark.parametrize("content_encoding", ["identity", "x-custom"])
async def test_fetch_preserves_uncompressed_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    content_encoding: str,
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/plain", "content-encoding": content_encoding},
            stream=_ChunkedByteStream(b"identity response"),
        ),
    )

    assert result["isError"] is False
    assert result["content"][0]["text"] == "identity response"


@pytest.mark.asyncio
async def test_fetch_rejects_multiple_content_encodings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip, br",
            },
            stream=_ChunkedByteStream(b"body"),
        ),
    )

    assert result["isError"] is True
    assert "multiple content encodings" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_fetch_aborts_when_decompressed_body_exceeds_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={
                "content-type": "text/plain",
                "content-encoding": "gzip",
            },
            stream=_ChunkedByteStream(gzip.compress(b"x" * 100)),
        ),
        arguments={"url": "example.com", "max_bytes": 32},
    )

    assert result["isError"] is True
    assert result["content"][0]["truncated"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("with_stream", [False, True])
async def test_parallel_fetches_cancel_on_registry_abort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    with_stream: bool,
) -> None:
    started = asyncio.Event()
    blocked = asyncio.Event()
    started_count = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal started_count
        del request
        started_count += 1
        if started_count == 2:
            started.set()
        await blocked.wait()
        raise AssertionError("blocked transport was not canceled")

    _mock_client(monkeypatch, handler)
    registry = ToolRegistry(tmp_path)
    calls = [
        ToolCall("fetch-a", "fetch", {"url": "example.com/a"}),
        ToolCall("fetch-b", "fetch", {"url": "example.com/b"}),
    ]
    if with_stream:
        async def run_streaming() -> list[object]:
            return await asyncio.gather(
                *(
                    registry.execute(call, _stream_sink=lambda event: None)
                    for call in calls
                )
            )

        task = asyncio.create_task(run_streaming())
    else:
        task = asyncio.create_task(registry.execute_many(calls))

    await asyncio.wait_for(started.wait(), timeout=1)
    registry.abort()
    results = await asyncio.wait_for(task, timeout=1)

    assert [result["content"][0]["text"] for result in results] == [
        "tool execution canceled",
        "tool execution canceled",
    ]


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
        requested_url = (
            f"{request.url.scheme}://{request.headers['host']}{request.url.path}"
        )
        response = responses[requested_url]
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


@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:169.254.169.254]/",
        "http://[::169.254.169.254]/",
    ],
)
def test_fetch_blocks_mapped_cloud_metadata_addresses(url: str) -> None:
    with pytest.raises(ValueError, match="cloud metadata"):
        fetch_tool._validate_target(url)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "url",
    [
        "http://[::ffff:192.168.1.5]/",
        "http://[::192.168.1.5]/",
    ],
)
async def test_fetch_notices_mapped_private_addresses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, url: str
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": "text/plain"}, text="local"),
        arguments={"url": url},
    )

    assert result["isError"] is False
    assert (
        "target resolves to a private or loopback address"
        in result["content"][0]["text"]
    )


@pytest.mark.asyncio
async def test_fetch_pins_connection_to_first_validated_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookups = iter(
        [
            ("192.168.1.5",),
            ("93.184.216.34",),
        ]
    )
    lookup_count = 0

    def getaddrinfo(*args, **kwargs):
        nonlocal lookup_count
        lookup_count += 1
        address = next(lookups)[0]
        return [
            (
                fetch_tool.socket.AF_INET,
                fetch_tool.socket.SOCK_STREAM,
                6,
                "",
                (address, 0),
            )
        ]

    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, text="local", request=request)

    client_kwargs: dict[str, object] = {}
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.invalid")
    _mock_client(monkeypatch, handler, client_kwargs=client_kwargs)
    monkeypatch.setattr(fetch_tool.socket, "getaddrinfo", getaddrinfo)
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("fetch-1", "fetch", {"url": "https://example.com/"})
    )

    assert result["isError"] is False
    assert lookup_count == 1
    assert client_kwargs["trust_env"] is False
    assert str(requests[0].url) == "https://192.168.1.5/"
    assert requests[0].headers["host"] == "example.com"
    assert requests[0].extensions["sni_hostname"] == "example.com"
    assert (
        "target resolves to a private or loopback address"
        in result["content"][0]["text"]
    )


@pytest.mark.asyncio
async def test_fetch_output_is_paginated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, text="x" * 50_100),
        max_output_chars=10_000,
    )

    block = result["content"][0]
    assert block["truncated"] is True
    assert block["full_size"] == 50_100
    assert block["full_size_chars"] == 50_100
    assert block["text"] == "x" * 10_000
    assert block["next_offset"] == 10_000


@pytest.mark.asyncio
async def test_fetch_pagination_reassembles_extracted_readable_text(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    html = (
        "<article><h1>Title</h1><p>Readable article text. "
        + "word " * 12
        + "</p></article>"
    )
    expected = fetch_tool._readable_content("https://example.com", "text/html", html)
    offset = 0
    pages: list[str] = []

    while True:
        result = await _execute_fetch(
            tmp_path,
            monkeypatch,
            httpx.Response(200, headers={"content-type": "text/html"}, text=html),
            arguments={"url": "example.com", "offset": offset},
            max_output_chars=17,
        )
        block = result["content"][0]
        pages.append(block["text"])
        assert block["full_size"] == len(expected)
        if not block["truncated"]:
            break
        assert block["next_offset"] == offset + len(block["text"])
        offset = block["next_offset"]

    assert len(pages) > 2
    assert "".join(pages) == expected


@pytest.mark.asyncio
async def test_fetch_pagination_overshoot_returns_empty_page(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "readable body"
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(200, headers={"content-type": "text/plain"}, text=body),
        arguments={"url": "example.com", "offset": len(body) + 100},
    )

    block = result["content"][0]
    assert block["text"] == ""
    assert block["truncated"] is False
    assert block["full_size"] == len(body)
    assert "next_offset" not in block


@pytest.mark.asyncio
async def test_fetch_pagination_repeats_notices_without_advancing_offset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = "readable body " * 10

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text=body,
            request=request,
        )

    _mock_client(monkeypatch, handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                fetch_tool.socket.AF_INET,
                fetch_tool.socket.SOCK_STREAM,
                6,
                "",
                ("192.168.1.5", 0),
            )
        ],
    )
    notice = (
        "notice: http URL is not encrypted\n"
        "notice: target resolves to a private or loopback address\n\n"
    )
    offset = 0
    body_pages: list[str] = []

    while True:
        result = await ToolRegistry(tmp_path, max_output_chars=128).execute(
            ToolCall(
                "fetch-1",
                "fetch",
                {"url": "http://printer.local", "offset": offset},
            )
        )
        block = result["content"][0]
        assert block["text"].startswith(notice)
        page = block["text"][len(notice) :]
        body_pages.append(page)
        assert page == body[offset : offset + len(page)]
        if not block["truncated"]:
            break
        assert block["next_offset"] == offset + len(page)
        offset = block["next_offset"]

    assert "".join(body_pages) == body


@pytest.mark.asyncio
async def test_fetch_pagination_reports_utf8_bytes_and_character_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = await _execute_fetch(
        tmp_path,
        monkeypatch,
        httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="éé",
        ),
        arguments={"url": "example.com", "offset": 0},
        max_output_chars=2,
    )

    block = result["content"][0]
    assert block["text"] == "é"
    assert block["full_size"] == 4
    assert block["full_size_chars"] == 2
    assert block["next_offset"] == 1
    assert flatten_tool_content([block]) == (
        "é\n[truncated: full_size_chars=2 chars; next_offset=1]"
    )


@pytest.mark.asyncio
async def test_fetch_pagination_rejects_cap_that_cannot_fit_notice(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/plain"},
            text="readable body",
            request=request,
        )

    _mock_client(monkeypatch, handler)
    monkeypatch.setattr(
        fetch_tool.socket,
        "getaddrinfo",
        lambda *args, **kwargs: [
            (
                fetch_tool.socket.AF_INET,
                fetch_tool.socket.SOCK_STREAM,
                6,
                "",
                ("192.168.1.5", 0),
            )
        ],
    )
    result = await ToolRegistry(tmp_path, max_output_chars=34).execute(
        ToolCall(
            "fetch-1",
            "fetch",
            {"url": "http://printer.local", "offset": 0},
        )
    )

    block = result["content"][0]
    assert result["isError"] is True
    assert "output limit too small" in block["text"]
    assert "next_offset" not in block


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
        assert request.method == "POST"
        assert request.url.path == "/html/"
        assert request.content == b"q=zeta"
        assert request.headers["accept-language"] == "en-US,en;q=0.9"
        assert request.headers["user-agent"].startswith("Mozilla/5.0")
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
async def test_websearch_falls_back_to_lite_for_provider_challenge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    challenge = (
        Path(__file__).parent / "fixtures" / "duckduckgo_challenge.html"
    ).read_text(encoding="utf-8")
    lite = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if len(requests) == 1:
            assert request.headers["host"] == "html.duckduckgo.com"
            return httpx.Response(
                200, headers={"content-type": "text/html"}, text=challenge
            )
        assert request.headers["host"] == "lite.duckduckgo.com"
        assert request.method == "POST"
        assert request.content == b"q=zeta"
        return httpx.Response(200, headers={"content-type": "text/html"}, text=lite)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert result["structuredContent"]["results"][0]["title"] == (
        "Login / Sign up - zeta"
    )
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_websearch_falls_back_to_lite_for_parser_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lite = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )
    calls = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(
                200,
                headers={"content-type": "text/html"},
                text="<html><body>unexpected response</body></html>",
            )
        return httpx.Response(200, headers={"content-type": "text/html"}, text=lite)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path).execute(
        ToolCall("search-1", "websearch", {"query": "zeta", "max_results": 1})
    )

    assert result["isError"] is False
    assert calls == 2


def test_websearch_detects_captured_provider_challenge() -> None:
    body = (Path(__file__).parent / "fixtures" / "duckduckgo_challenge.html").read_text(
        encoding="utf-8"
    )

    with pytest.raises(
        websearch.SearchProviderChallengeError,
        match="search provider served a no-results/challenge page",
    ):
        websearch.parse_search_results(body, max_results=8)


def test_websearch_parses_lite_fixture() -> None:
    body = (Path(__file__).parent / "fixtures" / "duckduckgo_lite.html").read_text(
        encoding="utf-8"
    )

    assert websearch.parse_lite_search_results(body, max_results=1) == [
        {
            "title": "Login / Sign up - zeta",
            "url": "https://zeta-ai.io/en/login",
            "snippet": (
                "The No.1 AI chat! Over 13 hours of weekly use — and it's free. "
                "Not using zeta yet? Everyone else is!"
            ),
        }
    ]


@pytest.mark.asyncio
async def test_websearch_output_keeps_registry_truncation_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    body = (
        '<a class="result__a" href="https://example.com/one">First result</a>'
        '<div class="result__snippet">A short description.</div>'
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, text=body)

    _mock_client(monkeypatch, handler)
    result = await ToolRegistry(tmp_path, max_output_chars=64).execute(
        ToolCall("search-1", "websearch", {"query": "zeta"})
    )

    block = result["content"][0]
    assert block["truncated"] is True
    assert block["text"].endswith("\n...[output truncated]")


@pytest.mark.asyncio
async def test_websearch_empty_results_and_parse_failure() -> None:
    empty = (Path(__file__).parent / "fixtures" / "duckduckgo_empty.html").read_text(
        encoding="utf-8"
    )
    assert websearch.parse_search_results(empty, max_results=8) == []
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
