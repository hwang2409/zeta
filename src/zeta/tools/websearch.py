"""Search the web through DuckDuckGo's keyless HTML endpoint."""

from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from typing import Any, TypedDict
from urllib.parse import parse_qs, unquote, urlsplit

from ..core.abort import AbortSignal
from ..types import StructuredToolResult
from .fetch import MAX_RESPONSE_BYTES, get_response, output_block, response_text
from .registry import ToolRegistry, _success_result

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"


class SearchResult(TypedDict):
    title: str
    url: str
    snippet: str


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._title: tuple[str, str] | None = None
        self._snippet: list[str] | None = None
        self.saw_no_results = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result__a" in classes:
            href = _decode_result_url(attributes.get("href", ""))
            self._title = (href, "")
        elif tag in {"a", "div"} and "result__snippet" in classes:
            self._snippet = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._title is not None:
            href, title = self._title
            if href and title.strip():
                self.results.append(
                    {"title": title.strip(), "url": href, "snippet": ""}
                )
            self._title = None
        elif tag in {"a", "div"} and self._snippet is not None:
            snippet = " ".join("".join(self._snippet).split())
            if self.results:
                self.results[-1]["snippet"] = snippet
            self._snippet = None

    def handle_data(self, data: str) -> None:
        if self._title is not None:
            self._title = (self._title[0], self._title[1] + data)
        if self._snippet is not None:
            self._snippet.append(data)
        if re.search(r"no results", data, re.IGNORECASE):
            self.saw_no_results = True


def _decode_result_url(raw_url: str) -> str:
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urlsplit(raw_url)
    if parsed.path == "/l/":
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        raw_url = unquote(target)
    return raw_url


def parse_search_results(body: str, *, max_results: int) -> list[SearchResult]:
    parser = _DuckDuckGoParser()
    parser.feed(body)
    parser.close()
    if not parser.results and not parser.saw_no_results:
        raise ValueError("search backend failed: could not parse results")
    return parser.results[:max_results]


async def _ddg_search(query: str, max_results: int) -> list[SearchResult]:
    response = await get_response(
        DDG_HTML_ENDPOINT,
        user_agent="zeta/websearch (DuckDuckGo HTML client)",
        max_bytes=MAX_RESPONSE_BYTES,
        params={"q": query},
    )
    return parse_search_results(response_text(response), max_results=max_results)


async def _websearch(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    _abort_signal: AbortSignal,
) -> StructuredToolResult:
    del registry
    query = arguments["query"]
    max_results = arguments.get("max_results", 8)
    results = await _ddg_search(query, max_results)
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    return _success_result(
        output_block(serialized),
        structured_content={"results": results},
    )


def register(registry: ToolRegistry) -> None:
    registry.register(
        "websearch",
        lambda arguments, abort_signal: _websearch(registry, arguments, abort_signal),
        description=(
            "Search the web with DuckDuckGo. Network access requires approval. "
            "The keyless HTML backend may change without notice."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "max_results": {"type": "integer", "minimum": 1},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    )
