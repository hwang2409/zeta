from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

from zeta.codex import CodexLoginRequiredError
from zeta.core.abort import AbortSignal
from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools import ToolRegistry, websearch
from zeta.tools.websearch import codex
from zeta.tools.websearch.codex import CodexSearchResult


@pytest.fixture
def registry(tmp_path: Path) -> ToolRegistry:
    return ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())


async def execute(registry: ToolRegistry) -> dict:
    return await registry.execute(ToolCall("search", "websearch", {"query": "zeta"}))


_REAL_ASYNC_CLIENT = httpx.AsyncClient


def install_codex_transport(
    monkeypatch: pytest.MonkeyPatch,
    *,
    body: bytes = b"",
    status: int = 200,
    stream: httpx.AsyncByteStream | None = None,
) -> None:
    async def access_token(_store, _client):
        return "test-token"

    def handler(_request: httpx.Request) -> httpx.Response:
        if stream is not None:
            return httpx.Response(status, stream=stream)
        return httpx.Response(status, content=body)

    def client_factory(*args, **kwargs):
        kwargs["transport"] = httpx.MockTransport(handler)
        return _REAL_ASYNC_CLIENT(*args, **kwargs)

    monkeypatch.setattr(codex.CodexCredentialStore, "access_token", access_token)
    if hasattr(codex, "codex_request_headers"):
        monkeypatch.setattr(codex, "codex_request_headers", lambda _token: {})
    else:
        monkeypatch.setattr(codex, "extract_account_id", lambda _token: "test")
    monkeypatch.setattr(codex.httpx, "AsyncClient", client_factory)


def sse(*events: dict) -> bytes:
    return (
        b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events)
        + b"data: [DONE]\n\n"
    )


def install_ddg_result(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []

    async def ddg(query, max_results):
        calls.append(query)
        return [{"title": "D", "url": "https://d.example", "snippet": "s"}]

    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    return calls


@pytest.mark.asyncio
async def test_websearch_no_codex_login_uses_duckduckgo(monkeypatch, registry):
    async def no_login(query, signal):
        raise CodexLoginRequiredError("no Codex OAuth login found; log in first")

    async def ddg(query, max_results):
        return [{"title": "D", "url": "https://d.example", "snippet": "s"}]

    monkeypatch.setattr(codex, "search", no_login)
    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    result = await execute(registry)
    assert result["structuredContent"]["backend"] == "duckduckgo"


@pytest.mark.asyncio
async def test_websearch_codex_success_returns_answer_and_sources(
    monkeypatch, registry
):
    fixture = (Path(__file__).parent / "fixtures/codex_search_success.sse").read_text()
    assert "response.output_text.delta" in fixture

    async def hosted(query, signal):
        return CodexSearchResult(
            "The answer is 42.",
            [{"title": "Example source", "url": "https://example.com/source"}],
        )

    monkeypatch.setattr(codex, "search", hosted)
    result = await execute(registry)
    assert result["structuredContent"] == {
        "backend": "codex",
        "answer": "The answer is 42.",
        "sources": [{"title": "Example source", "url": "https://example.com/source"}],
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429])
async def test_websearch_codex_http_error_falls_back_to_duckduckgo(
    monkeypatch, registry, status
):
    async def hosted(query, signal):
        raise codex.CodexHTTPError(
            f"Codex search failed with HTTP {status}", status_code=status
        )

    async def ddg(query, max_results):
        return []

    monkeypatch.setattr(codex, "search", hosted)
    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    result = await execute(registry)
    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert str(status) in result["structuredContent"]["codex_failure"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [codex.CodexStreamError("stream failed"), codex.CodexBackendError("timed out")],
)
async def test_websearch_codex_failures_fall_back(monkeypatch, registry, failure):
    async def hosted(query, signal):
        raise failure

    async def ddg(query, max_results):
        return []

    monkeypatch.setattr(codex, "search", hosted)
    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    result = await execute(registry)
    assert result["structuredContent"]["backend"] == "duckduckgo"


@pytest.mark.asyncio
async def test_websearch_both_fail_error_has_both_reasons(monkeypatch, registry):
    async def hosted(query, signal):
        raise codex.CodexHTTPError("Codex is unavailable", status_code=500)

    async def ddg(query, max_results):
        raise ValueError("DuckDuckGo is unavailable")

    monkeypatch.setattr(codex, "search", hosted)
    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    result = await execute(registry)
    assert result["isError"] is True
    assert "Codex is unavailable" in result["content"][0]["text"]
    assert "DuckDuckGo is unavailable" in result["content"][0]["text"]


@pytest.mark.asyncio
async def test_websearch_abort_does_not_fall_back(monkeypatch, registry):
    signal = AbortSignal()
    signal.abort()
    called = False

    async def hosted(query, received):
        raise ValueError("should not be converted to fallback")

    async def ddg(query, max_results):
        nonlocal called
        called = True
        return []

    monkeypatch.setattr(codex, "search", hosted)
    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    with pytest.raises(ValueError, match="should not be converted"):
        await websearch._websearch(registry, {"query": "zeta"}, signal)
    assert not called


@pytest.mark.asyncio
async def test_websearch_login_checked_per_call(monkeypatch, registry):
    calls = 0

    async def hosted(query, signal):
        nonlocal calls
        calls += 1
        if calls == 1:
            return CodexSearchResult("ok", [])
        raise CodexLoginRequiredError("no Codex OAuth login found; log in first")

    async def ddg(query, max_results):
        return []

    monkeypatch.setattr(codex, "search", hosted)
    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    assert (await execute(registry))["structuredContent"]["backend"] == "codex"
    second = await execute(registry)
    assert second["structuredContent"]["backend"] == "duckduckgo"
    assert "codex_failure" not in second["structuredContent"]


def test_websearch_tool_description_backend_neutral(registry):
    websearch.register(registry)
    text = registry.definitions_by_name["websearch"].description
    assert "Search the web" in text
    assert "Codex" not in text
    assert "DuckDuckGo" not in text


@pytest.mark.asyncio
async def test_websearch_receipt_shows_backend_from_real_tool_content(
    monkeypatch, registry
):
    from zeta.tui.render import _receipt_arguments

    async def hosted(query, signal):
        return CodexSearchResult(
            "answer", [{"title": "source", "url": "https://example.com"}]
        )

    monkeypatch.setattr(codex, "search", hosted)
    result = await execute(registry)
    call = ToolCall("search", "websearch", {"query": "zeta"})
    content = result["content"][0]["text"]
    summary = _receipt_arguments(call, content)
    assert "codex" in summary
    assert "1 results" in summary


@pytest.mark.asyncio
async def test_codex_incomplete_stream_falls_back(monkeypatch, registry):
    install_codex_transport(
        monkeypatch,
        body=sse(
            {"type": "response.output_text.delta", "delta": "partial"},
            {"type": "response.incomplete"},
        ),
    )
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert calls == ["zeta"]
    assert "stream failed" in result["structuredContent"]["codex_failure"]


@pytest.mark.asyncio
async def test_codex_eof_before_completion_falls_back(monkeypatch, registry):
    install_codex_transport(
        monkeypatch,
        body=sse({"type": "response.output_text.delta", "delta": "partial"}),
    )
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert calls == ["zeta"]
    assert "before completion" in result["structuredContent"]["codex_failure"]


@pytest.mark.asyncio
async def test_codex_whitespace_answer_falls_back(monkeypatch, registry):
    install_codex_transport(
        monkeypatch,
        body=sse(
            {"type": "response.output_text.delta", "delta": " \n\t"},
            {"type": "response.completed"},
        ),
    )
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert calls == ["zeta"]
    assert "no answer" in result["structuredContent"]["codex_failure"]


@pytest.mark.asyncio
async def test_codex_failed_status_falls_back(monkeypatch, registry):
    install_codex_transport(
        monkeypatch,
        body=sse(
            {"type": "response.output_text.delta", "delta": "partial"},
            {"type": "response.completed", "response": {"status": "failed"}},
        ),
    )
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert calls == ["zeta"]
    assert (
        "did not complete successfully" in result["structuredContent"]["codex_failure"]
    )


@pytest.mark.asyncio
async def test_codex_fixture_stream_success(monkeypatch, registry):
    fixture = (Path(__file__).parent / "fixtures/codex_search_success.sse").read_bytes()
    install_codex_transport(monkeypatch, body=fixture)

    result = await execute(registry)

    assert result["structuredContent"] == {
        "backend": "codex",
        "answer": "The answer is 42.",
        "sources": [{"title": "Example source", "url": "https://example.com/source"}],
    }


@pytest.mark.asyncio
async def test_codex_many_sources_truncates_not_fails(monkeypatch, registry):
    sources = [
        {"title": f"Source {index}", "url": f"https://example.com/{index}"}
        for index in range(50)
    ]
    install_codex_transport(
        monkeypatch,
        body=sse(
            {"type": "response.output_text.delta", "delta": "answer"},
            {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "output": [
                        {
                            "type": "web_search_call",
                            "action": {"sources": sources},
                        }
                    ],
                },
            },
        ),
    )
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    content = result["structuredContent"]
    assert content["backend"] == "codex"
    assert len(content["sources"]) == codex.CODEX_SEARCH_MAX_SOURCES
    assert content["sources_truncated"] == 30
    assert "(30 more sources omitted)" in result["content"][0]["text"]
    assert calls == []


@pytest.mark.asyncio
async def test_codex_long_answer_truncates_not_fails(monkeypatch, registry):
    limit = codex.CODEX_SEARCH_MAX_ANSWER_BYTES
    install_codex_transport(
        monkeypatch,
        body=sse(
            {
                "type": "response.output_text.delta",
                "delta": "x" * (limit + 1),
            },
            {"type": "response.output_text.delta", "delta": "still consumed"},
            {"type": "response.completed"},
        ),
    )
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    content = result["structuredContent"]
    assert content["backend"] == "codex"
    assert len(content["answer"].encode()) <= limit
    assert content["answer_truncated"] is True
    assert "[answer truncated]" in result["content"][0]["text"]
    assert calls == []


@pytest.mark.asyncio
async def test_codex_deadline_covers_credential_acquisition(monkeypatch, registry):
    install_codex_transport(
        monkeypatch,
        body=sse(
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.completed"},
        ),
    )

    async def slow_access_token(_store, _client):
        await asyncio.sleep(0.05)
        return "test-token"

    monkeypatch.setattr(codex.CodexCredentialStore, "access_token", slow_access_token)
    monkeypatch.setattr(codex, "CODEX_SEARCH_DEADLINE", 0.001)
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert calls == ["zeta"]
    assert "timed out" in result["structuredContent"]["codex_failure"]


@pytest.mark.asyncio
async def test_concurrent_searches_share_one_credential_acquisition(
    monkeypatch, registry
):
    install_codex_transport(monkeypatch)
    acquisition_calls = 0
    active_acquisitions = 0
    release_acquisition = asyncio.Event()
    acquisitions_finished = asyncio.Event()

    async def slow_access_token(_store, _client):
        nonlocal acquisition_calls, active_acquisitions
        acquisition_calls += 1
        active_acquisitions += 1
        try:
            await release_acquisition.wait()
            return "test-token"
        finally:
            active_acquisitions -= 1
            if active_acquisitions == 0:
                acquisitions_finished.set()

    monkeypatch.setattr(codex.CodexCredentialStore, "access_token", slow_access_token)
    monkeypatch.setattr(codex, "CODEX_SEARCH_DEADLINE", 0.01)
    ddg_calls = install_ddg_result(monkeypatch)

    try:
        searches = [execute(registry) for _ in range(8)]
        results = await asyncio.gather(*searches)

        assert all(
            result["structuredContent"]["backend"] == "duckduckgo"
            for result in results
        )
        assert len(ddg_calls) == 8
        assert acquisition_calls == 1
        assert active_acquisitions == 1
    finally:
        release_acquisition.set()
        await asyncio.wait_for(acquisitions_finished.wait(), timeout=0.2)


@pytest.mark.asyncio
async def test_inflight_acquisition_cleared_after_completion(monkeypatch, registry):
    install_codex_transport(
        monkeypatch,
        body=sse(
            {"type": "response.output_text.delta", "delta": "answer"},
            {"type": "response.completed"},
        ),
    )
    acquisition_calls = 0
    release_first = asyncio.Event()
    first_finished = asyncio.Event()

    async def access_token(_store, _client):
        nonlocal acquisition_calls
        acquisition_calls += 1
        if acquisition_calls == 1:
            await release_first.wait()
            first_finished.set()
        return "test-token"

    monkeypatch.setattr(codex.CodexCredentialStore, "access_token", access_token)
    monkeypatch.setattr(codex, "CODEX_SEARCH_DEADLINE", 0.001)
    install_ddg_result(monkeypatch)

    first = await execute(registry)
    assert first["structuredContent"]["backend"] == "duckduckgo"
    release_first.set()
    await asyncio.wait_for(first_finished.wait(), timeout=0.2)
    await asyncio.sleep(0)

    monkeypatch.setattr(codex, "CODEX_SEARCH_DEADLINE", 0.2)
    second = await execute(registry)

    assert second["structuredContent"]["backend"] == "codex"
    assert acquisition_calls == 2


@pytest.mark.asyncio
async def test_codex_deadline_does_not_cancel_token_refresh(monkeypatch, registry):
    install_codex_transport(monkeypatch)
    refresh_started = asyncio.Event()
    refresh_persisted = asyncio.Event()
    refresh_cancelled = False

    async def slow_access_token(_store, _client):
        nonlocal refresh_cancelled
        refresh_started.set()
        try:
            await asyncio.sleep(0.05)
            refresh_persisted.set()
            return "test-token"
        except asyncio.CancelledError:
            refresh_cancelled = True
            raise

    monkeypatch.setattr(codex.CodexCredentialStore, "access_token", slow_access_token)
    monkeypatch.setattr(codex, "CODEX_SEARCH_DEADLINE", 0.001)
    install_ddg_result(monkeypatch)

    result = await execute(registry)

    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert "timed out" in result["structuredContent"]["codex_failure"]
    await asyncio.wait_for(refresh_started.wait(), timeout=0.2)
    await asyncio.wait_for(refresh_persisted.wait(), timeout=0.2)
    assert not refresh_cancelled


@pytest.mark.asyncio
async def test_codex_user_abort_during_credential_acquisition_no_fallback(
    monkeypatch, registry
):
    install_codex_transport(monkeypatch)
    refresh_started = asyncio.Event()
    allow_refresh_to_finish = asyncio.Event()
    refresh_persisted = asyncio.Event()
    refresh_cancelled = False

    async def slow_access_token(_store, _client):
        nonlocal refresh_cancelled
        refresh_started.set()
        try:
            await allow_refresh_to_finish.wait()
            refresh_persisted.set()
            return "test-token"
        except asyncio.CancelledError:
            refresh_cancelled = True
            raise

    monkeypatch.setattr(codex.CodexCredentialStore, "access_token", slow_access_token)
    calls = install_ddg_result(monkeypatch)
    task = asyncio.create_task(
        websearch._websearch(registry, {"query": "zeta"}, AbortSignal())
    )
    await asyncio.wait_for(refresh_started.wait(), timeout=0.2)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    allow_refresh_to_finish.set()
    await asyncio.wait_for(refresh_persisted.wait(), timeout=0.2)

    assert calls == []
    assert not refresh_cancelled


@pytest.mark.asyncio
async def test_codex_overall_deadline_falls_back(monkeypatch, registry):
    class SlowStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            await asyncio.sleep(0.05)
            yield sse(
                {"type": "response.output_text.delta", "delta": "answer"},
                {"type": "response.completed"},
            )

    install_codex_transport(monkeypatch, stream=SlowStream())
    monkeypatch.setattr(codex, "CODEX_SEARCH_DEADLINE", 0.001, raising=False)
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert calls == ["zeta"]
    assert "timed out" in result["structuredContent"]["codex_failure"]


@pytest.mark.asyncio
async def test_codex_user_abort_during_stream_no_fallback(monkeypatch, registry):
    signal = AbortSignal()

    class AbortingStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield sse({"type": "response.output_text.delta", "delta": "partial"})
            signal.abort()
            yield sse({"type": "response.completed"})

    install_codex_transport(monkeypatch, stream=AbortingStream())
    calls = install_ddg_result(monkeypatch)

    with pytest.raises(asyncio.CancelledError):
        await websearch._websearch(registry, {"query": "zeta"}, signal)
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429])
async def test_codex_http_401_and_429_fall_back_through_transport(
    monkeypatch, registry, status
):
    install_codex_transport(monkeypatch, status=status)
    calls = install_ddg_result(monkeypatch)

    result = await execute(registry)

    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert calls == ["zeta"]
    assert str(status) in result["structuredContent"]["codex_failure"]


def test_websearch_does_not_import_providers():
    environment = os.environ.copy()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import zeta.tools.websearch; "
                "print([m for m in sys.modules if m.startswith('zeta.providers')])"
            ),
        ],
        capture_output=True,
        text=True,
        env=environment,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
