"""Small side-call client for Codex hosted web search."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from ...codex import (
    CODEX_API_URL,
    DEFAULT_CODEX_MODEL,
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
# Keep deduplication memory bounded while processing an untrusted stream.
CODEX_SEARCH_MAX_SEEN_SOURCES = 512


@dataclass(frozen=True)
class CodexSearchResult:
    answer: str
    sources: list[dict[str, str]]
    sources_truncated: int = 0
    answer_truncated: bool = False


async def _events(
    response: httpx.Response, abort_signal: AbortSignal
) -> AsyncIterator[dict[str, Any]]:
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


def _source_values(event: dict[str, Any]) -> list[dict[str, Any]]:
    """Return sources from documented web-search action payloads."""

    items: list[Any] = []
    item = event.get("item")
    if isinstance(item, dict):
        items.append(item)
    response = event.get("response")
    if isinstance(response, dict) and isinstance(response.get("output"), list):
        items.extend(response["output"])

    values: list[dict[str, Any]] = []
    for candidate in items:
        if (
            not isinstance(candidate, dict)
            or candidate.get("type") != "web_search_call"
        ):
            continue
        action = candidate.get("action")
        if not isinstance(action, dict) or not isinstance(action.get("sources"), list):
            continue
        values.extend(value for value in action["sources"] if isinstance(value, dict))
    return values


async def _acquire_token(store: CodexCredentialStore, timeout: httpx.Timeout) -> str:
    """Acquire and persist credentials in a task that is safe from caller cancellation."""

    async with httpx.AsyncClient(timeout=timeout) as client:
        return await store.access_token(client)


_credential_tasks: dict[Path, asyncio.Task[str]] = {}


def _credential_task_done(path: Path, task: asyncio.Task[str]) -> None:
    if _credential_tasks.get(path) is task:
        del _credential_tasks[path]
    if not task.cancelled():
        task.exception()


def _start_credential_task(
    store: CodexCredentialStore, timeout: httpx.Timeout
) -> asyncio.Task[str]:
    path = store.path.expanduser().resolve()
    if task := _credential_tasks.get(path):
        return task
    task = asyncio.create_task(_acquire_token(store, timeout))
    _credential_tasks[path] = task
    task.add_done_callback(lambda completed: _credential_task_done(path, completed))
    return task


async def search(query: str, abort_signal: AbortSignal) -> CodexSearchResult:
    store = CodexCredentialStore()
    timeout = httpx.Timeout(CODEX_SEARCH_TIMEOUT)
    try:
        async with asyncio.timeout(CODEX_SEARCH_DEADLINE):
            # A deadline or caller cancellation stops this search, but shield lets a
            # rotating OAuth refresh finish and persist its replacement token.
            token = await asyncio.shield(_start_credential_task(store, timeout))
            headers = codex_request_headers(token)
            payload = {
                "model": CODEX_SEARCH_MODEL,
                "store": False,
                "stream": True,
                "instructions": "Answer the user's question briefly. Use web search exactly once.",
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": query}],
                    }
                ],
                "tools": [{"type": "web_search"}],
                "tool_choice": "auto",
                "parallel_tool_calls": False,
                "reasoning": {"summary": "auto"},
                "include": CODEX_SEARCH_INCLUDE,
            }
            async with (
                httpx.AsyncClient(timeout=timeout) as client,
                client.stream(
                    "POST", CODEX_API_URL, headers=headers, json=payload
                ) as response,
            ):
                if response.status_code >= 400:
                    raise CodexHTTPError(
                        f"Codex search failed with HTTP {response.status_code}",
                        status_code=response.status_code,
                    )
                answer_parts: list[str] = []
                answer_bytes = 0
                answer_truncated = False
                sources: list[dict[str, str]] = []
                seen: set[str] = set()
                sources_truncated = 0

                def add_source(value: dict[str, Any]) -> None:
                    nonlocal sources_truncated
                    url = value.get("url")
                    title = value.get("title") or url
                    if not isinstance(url, str) or url in seen:
                        return
                    if len(seen) >= CODEX_SEARCH_MAX_SEEN_SOURCES:
                        # The source is unique among the bounded set we retained.
                        sources_truncated += 1
                        return
                    seen.add(url)
                    if len(sources) >= CODEX_SEARCH_MAX_SOURCES:
                        sources_truncated += 1
                    else:
                        sources.append({"title": str(title), "url": url})

                completed = 0
                async for event in _events(response, abort_signal):
                    event_type = event.get("type")
                    if event_type == "response.output_text.delta":
                        delta = str(event.get("delta", ""))
                        if answer_bytes < CODEX_SEARCH_MAX_ANSWER_BYTES:
                            remaining = CODEX_SEARCH_MAX_ANSWER_BYTES - answer_bytes
                            encoded = delta.encode("utf-8")
                            chunk = encoded[:remaining].decode("utf-8", errors="ignore")
                            answer_parts.append(chunk)
                            answer_bytes += len(chunk.encode("utf-8"))
                            if len(chunk.encode("utf-8")) < len(encoded):
                                answer_truncated = True
                        elif delta:
                            answer_truncated = True
                    elif event_type == "response.output_text.annotation.added":
                        annotation = event.get("annotation", {})
                        citation = annotation.get("url_citation", annotation)
                        if isinstance(citation, dict):
                            add_source(citation)
                    elif event_type in {
                        "error",
                        "response.failed",
                        "response.incomplete",
                    }:
                        raise CodexStreamError("Codex search stream failed")
                    for source in _source_values(event):
                        add_source(source)
                    if event_type in {"response.completed", "response.done"}:
                        status = event.get("response", event).get("status", "completed")
                        if status not in {"completed", "succeeded"}:
                            raise CodexStreamError(
                                "Codex search stream did not complete successfully"
                            )
                        completed += 1
                        if completed > 1:
                            raise CodexStreamError(
                                "Codex search returned multiple completion events"
                            )
                if completed != 1:
                    raise CodexStreamError(
                        "Codex search stream ended before completion"
                    )
                answer = "".join(answer_parts).strip()
                if not answer:
                    raise CodexStreamError("Codex search returned no answer")
                return CodexSearchResult(
                    answer, sources, sources_truncated, answer_truncated
                )
    except (httpx.TimeoutException, TimeoutError) as exc:
        raise CodexBackendError("Codex search timed out") from exc
    except httpx.HTTPError as exc:
        raise CodexBackendError("Codex search request failed") from exc
