from __future__ import annotations

from pathlib import Path

import pytest

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


@pytest.mark.asyncio
async def test_websearch_no_codex_login_uses_duckduckgo(monkeypatch, registry):
    async def no_login(query, signal):
        raise codex.CodexAuthError("no Codex OAuth login found; log in first")
    async def ddg(query, max_results):
        return [{"title": "D", "url": "https://d.example", "snippet": "s"}]
    monkeypatch.setattr(codex, "search", no_login)
    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    result = await execute(registry)
    assert result["structuredContent"]["backend"] == "duckduckgo"


@pytest.mark.asyncio
async def test_websearch_codex_success_returns_answer_and_sources(monkeypatch, registry):
    fixture = (Path(__file__).parent / "fixtures/codex_search_success.sse").read_text()
    assert "response.output_text.delta" in fixture
    async def hosted(query, signal):
        return CodexSearchResult("The answer is 42.", [{"title": "Example source", "url": "https://example.com/source"}])
    monkeypatch.setattr(codex, "search", hosted)
    result = await execute(registry)
    assert result["structuredContent"] == {"backend": "codex", "answer": "The answer is 42.", "sources": [{"title": "Example source", "url": "https://example.com/source"}]}


@pytest.mark.asyncio
@pytest.mark.parametrize("status", [401, 429])
async def test_websearch_codex_http_error_falls_back_to_duckduckgo(monkeypatch, registry, status):
    async def hosted(query, signal):
        raise codex.CodexHTTPError(f"Codex search failed with HTTP {status}", status_code=status)
    async def ddg(query, max_results):
        return []
    monkeypatch.setattr(codex, "search", hosted)
    monkeypatch.setattr(websearch, "_ddg_search", ddg)
    result = await execute(registry)
    assert result["structuredContent"]["backend"] == "duckduckgo"
    assert str(status) in result["structuredContent"]["codex_failure"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [codex.CodexStreamError("stream failed"), codex.CodexBackendError("timed out")])
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
        raise codex.CodexAuthError("no Codex OAuth login found; log in first")
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
    assert "Codex" in text and "DuckDuckGo" in text
    assert "only" not in text.lower()


@pytest.mark.asyncio
async def test_websearch_receipt_shows_backend_from_real_tool_content(monkeypatch, registry):
    from zeta.tui.render import _receipt_arguments

    async def hosted(query, signal):
        return CodexSearchResult("answer", [{"title": "source", "url": "https://example.com"}])

    monkeypatch.setattr(codex, "search", hosted)
    result = await execute(registry)
    call = ToolCall("search", "websearch", {"query": "zeta"})
    content = result["content"][0]["text"]
    summary = _receipt_arguments(call, content)
    assert "codex" in summary
    assert "1 results" in summary
