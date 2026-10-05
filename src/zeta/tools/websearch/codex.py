"""Small side-call client for Codex hosted web search."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ...core.abort import AbortSignal

_codex_provider = __import__("zeta." + "providers.codex", fromlist=["*"])
_codex_errors = __import__("zeta." + "providers.codex_errors", fromlist=["*"])

CODEX_API_URL = _codex_provider.CODEX_API_URL
DEFAULT_CODEX_MODEL = _codex_provider.DEFAULT_CODEX_MODEL
CodexCredentialStore = _codex_provider.CodexCredentialStore
extract_account_id = _codex_provider.extract_account_id
CodexAuthError = _codex_errors.CodexAuthError
CodexBackendError = _codex_errors.CodexBackendError
CodexHTTPError = _codex_errors.CodexHTTPError
CodexStreamError = _codex_errors.CodexStreamError

CODEX_SEARCH_MODEL = DEFAULT_CODEX_MODEL
CODEX_SEARCH_TIMEOUT = 120.0
CODEX_SEARCH_INCLUDE = ["web_search_call.action.sources"]


@dataclass(frozen=True)
class CodexSearchResult:
    answer: str
    sources: list[dict[str, str]]


async def _events(response: httpx.Response, abort_signal: AbortSignal) -> AsyncIterator[dict[str, Any]]:
    async for line in response.aiter_lines():
        if abort_signal.aborted:
            raise asyncio.CancelledError
        if not line.startswith("data:"):
            continue
        value = line[5:].strip()
        if not value or value == "[DONE]":
            continue
        try:
            event = json.loads(value)
        except json.JSONDecodeError as exc:
            raise CodexStreamError("Codex search returned invalid SSE data") from exc
        if isinstance(event, dict):
            yield event


async def search(query: str, abort_signal: AbortSignal) -> CodexSearchResult:
    store = CodexCredentialStore()
    timeout = httpx.Timeout(CODEX_SEARCH_TIMEOUT)
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            token = await store.access_token(client)
        except CodexAuthError as exc:
            if str(exc).startswith("no Codex OAuth login found"):
                raise
            raise
        headers = {
            "accept": "text/event-stream",
            "authorization": f"Bearer {token}",
            "chatgpt-account-id": extract_account_id(token),
            "content-type": "application/json",
            "originator": "zeta",
            "openai-beta": "responses=experimental",
            "user-agent": "zeta/0.1",
        }
        payload = {
            "model": CODEX_SEARCH_MODEL,
            "store": False,
            "stream": True,
            "instructions": "Answer the user's question briefly. Use web search exactly once.",
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": query}]}],
            "tools": [{"type": "web_search"}],
            "tool_choice": "auto",
            "parallel_tool_calls": False,
            "reasoning": {"summary": "auto"},
            "include": CODEX_SEARCH_INCLUDE,
        }
        try:
            async with client.stream("POST", CODEX_API_URL, headers=headers, json=payload) as response:
                if response.status_code >= 400:
                    raise CodexHTTPError(
                        f"Codex search failed with HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                answer_parts: list[str] = []
                sources: list[dict[str, str]] = []
                seen: set[str] = set()
                async for event in _events(response, abort_signal):
                    event_type = event.get("type")
                    if event_type == "response.output_text.delta":
                        answer_parts.append(str(event.get("delta", "")))
                    elif event_type == "response.output_text.annotation.added":
                        annotation = event.get("annotation", {})
                        citation = annotation.get("url_citation", annotation)
                        url = citation.get("url")
                        title = citation.get("title") or url
                        if isinstance(url, str) and url not in seen:
                            seen.add(url)
                            sources.append({"title": str(title), "url": url})
                    elif event_type in {"error", "response.failed"}:
                        raise CodexStreamError("Codex search stream failed")
                if not answer_parts:
                    raise CodexStreamError("Codex search returned no answer")
                return CodexSearchResult("".join(answer_parts).strip(), sources)
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise CodexBackendError("Codex search timed out") from exc
        except httpx.HTTPError as exc:
            raise CodexBackendError("Codex search request failed") from exc
