"""Small side-call client for Codex hosted web search."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import httpx

from ...codex import (
    CODEX_API_URL,
    DEFAULT_CODEX_MODEL,
    CodexAuthError,
    CodexBackendError,
    CodexCredentialStore,
    CodexHTTPError,
    CodexStreamError,
    codex_request_headers,
)
from ...core.abort import AbortSignal

CODEX_SEARCH_MODEL = DEFAULT_CODEX_MODEL
CODEX_SEARCH_TIMEOUT = 120.0
CODEX_SEARCH_INCLUDE = ["web_search_call.action.sources"]
CODEX_SEARCH_DEADLINE = 30.0
CODEX_SEARCH_MAX_ANSWER_BYTES = 16_384
CODEX_SEARCH_MAX_SOURCES = 20


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
        headers = codex_request_headers(token)
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
            async with asyncio.timeout(CODEX_SEARCH_DEADLINE):
                async with client.stream("POST", CODEX_API_URL, headers=headers, json=payload) as response:
                    if response.status_code >= 400:
                        raise CodexHTTPError(f"Codex search failed with HTTP {response.status_code}", status_code=response.status_code)
                    answer_parts: list[str] = []
                    answer_bytes = 0
                    sources: list[dict[str, str]] = []
                    seen: set[str] = set()
                    completed = False
                    async for event in _events(response, abort_signal):
                        event_type = event.get("type")
                        if event_type == "response.output_text.delta":
                            delta = str(event.get("delta", ""))
                            answer_bytes += len(delta.encode("utf-8"))
                            if answer_bytes > CODEX_SEARCH_MAX_ANSWER_BYTES:
                                raise CodexStreamError("Codex search answer exceeded size limit")
                            answer_parts.append(delta)
                        elif event_type == "response.output_text.annotation.added":
                            annotation = event.get("annotation", {})
                            citation = annotation.get("url_citation", annotation)
                            url = citation.get("url")
                            title = citation.get("title") or url
                            if isinstance(url, str) and url not in seen:
                                if len(sources) >= CODEX_SEARCH_MAX_SOURCES:
                                    raise CodexStreamError("Codex search returned too many sources")
                                seen.add(url)
                                sources.append({"title": str(title), "url": url})
                        elif event_type in {"error", "response.failed", "response.incomplete"}:
                            raise CodexStreamError("Codex search stream failed")
                        elif event_type in {"response.completed", "response.done"}:
                            status = event.get("response", event).get("status", "completed")
                            if status not in {"completed", "succeeded"}:
                                raise CodexStreamError("Codex search stream did not complete successfully")
                            completed = True
                    if not completed:
                        raise CodexStreamError("Codex search stream ended before completion")
                    answer = "".join(answer_parts).strip()
                    if not answer:
                        raise CodexStreamError("Codex search returned no answer")
                    return CodexSearchResult(answer, sources)
        except (httpx.TimeoutException, TimeoutError) as exc:
            raise CodexBackendError("Codex search timed out") from exc
        except httpx.HTTPError as exc:
            raise CodexBackendError("Codex search request failed") from exc
