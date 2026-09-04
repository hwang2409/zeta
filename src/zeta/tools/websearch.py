"""Search the web through DuckDuckGo's keyless HTML endpoint."""

from __future__ import annotations

import json
from html.parser import HTMLParser
from typing import Any, TypedDict
from urllib.parse import parse_qs, unquote, urlsplit

from ..core.abort import AbortSignal
from ..types import StructuredToolResult
from .fetch import (
    MAX_OUTPUT_BYTES,
    MAX_RESPONSE_BYTES,
    get_response,
    output_block,
    response_text,
)
from .registry import ToolRegistry, _success_result

DDG_HTML_ENDPOINT = "https://html.duckduckgo.com/html/"
DDG_LITE_ENDPOINT = "https://lite.duckduckgo.com/lite/"
DDG_BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)
DDG_BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


class WebsearchError(ValueError):
    """Base class for errors returned by the websearch providers."""


class SearchProviderChallengeError(WebsearchError):
    """The provider returned a homepage or challenge instead of search results."""


class SearchParserError(WebsearchError):
    """The provider response did not match a known result page."""


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
        self._empty_container_depth = 0
        self._empty_marker_seen = False
        self._empty_message_depth = 0
        self._empty_heading_depth = 0
        self._empty_heading_text: list[str] = []
        self.saw_empty_state = False
        self.saw_homepage = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        rels = {rel.lower() for rel in (attributes.get("rel") or "").split()}
        if tag == "link" and "canonical" in rels:
            self.saw_homepage = _is_ddg_homepage(attributes.get("href", ""))
        elif tag == "body" and "body--home" in classes:
            self.saw_homepage = True
        elif tag == "div" and classes == {"no-results__container", "result__title"}:
            self._empty_container_depth = 1
        elif self._empty_container_depth:
            self._empty_container_depth += 1
            if tag == "span" and classes == {"no-results"}:
                self._empty_marker_seen = True
            elif tag == "div" and classes == {"no-results__message"}:
                self._empty_message_depth = 1
            elif self._empty_message_depth:
                self._empty_message_depth += 1
            if tag == "h1" and self._empty_message_depth:
                self._empty_heading_depth = 1
                self._empty_heading_text = []
            elif self._empty_heading_depth:
                self._empty_heading_depth += 1
        elif tag == "a" and "result__a" in classes:
            href = _decode_result_url(attributes.get("href", ""))
            self._title = (href, "")
        elif tag in {"a", "div"} and "result__snippet" in classes:
            self._snippet = []

    def handle_endtag(self, tag: str) -> None:
        if self._empty_heading_depth:
            self._empty_heading_depth -= 1
            if self._empty_heading_depth == 0:
                heading = " ".join("".join(self._empty_heading_text).split())
                self.saw_empty_state = self._empty_marker_seen and heading.startswith(
                    "No results found for "
                )
        if self._empty_message_depth:
            self._empty_message_depth -= 1
        if self._empty_container_depth:
            self._empty_container_depth -= 1
            return
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
        if self._empty_heading_depth:
            self._empty_heading_text.append(data)
            return
        if self._title is not None:
            self._title = (self._title[0], self._title[1] + data)
        if self._snippet is not None:
            self._snippet.append(data)


def _decode_result_url(raw_url: str) -> str:
    if raw_url.startswith("//"):
        raw_url = "https:" + raw_url
    parsed = urlsplit(raw_url)
    if parsed.path == "/l/":
        target = parse_qs(parsed.query).get("uddg", [""])[0]
        raw_url = unquote(target)
    return raw_url


def _is_ddg_homepage(raw_url: str) -> bool:
    parsed = urlsplit(raw_url)
    return (
        parsed.hostname in {"duckduckgo.com", "www.duckduckgo.com"}
        and parsed.path in {"", "/"}
    )


def parse_search_results(body: str, *, max_results: int) -> list[SearchResult]:
    parser = _DuckDuckGoParser()
    parser.feed(body)
    parser.close()
    if parser.results or parser.saw_empty_state:
        return parser.results[:max_results]
    if parser.saw_homepage:
        raise SearchProviderChallengeError(
            "search provider served a no-results/challenge page"
        )
    raise SearchParserError("search backend failed: could not parse results")


class _DuckDuckGoLiteParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._title: tuple[str, list[str]] | None = None
        self._title_depth = 0
        self._snippet: list[str] | None = None
        self._snippet_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = set((attributes.get("class") or "").split())
        if tag == "a" and "result-link" in classes:
            self._title = (attributes.get("href", ""), [])
            self._title_depth = 1
        elif self._title is not None:
            self._title_depth += 1
        if tag in {"a", "td"} and "result-snippet" in classes:
            self._snippet = []
            self._snippet_depth = 1
        elif self._snippet is not None:
            self._snippet_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if self._title is not None:
            self._title_depth -= 1
            if self._title_depth == 0:
                href, title = self._title
                self.results.append(
                    {
                        "title": " ".join("".join(title).split()),
                        "url": href,
                        "snippet": "",
                    }
                )
                self._title = None
        if self._snippet is not None:
            self._snippet_depth -= 1
            if self._snippet_depth == 0:
                if self.results:
                    self.results[-1]["snippet"] = " ".join(
                        "".join(self._snippet).split()
                    )
                self._snippet = None

    def handle_data(self, data: str) -> None:
        if self._title is not None:
            self._title[1].append(data)
        if self._snippet is not None:
            self._snippet.append(data)


def parse_lite_search_results(body: str, *, max_results: int) -> list[SearchResult]:
    parser = _DuckDuckGoLiteParser()
    parser.feed(body)
    parser.close()
    if not parser.results:
        raise SearchParserError("search backend failed: could not parse lite results")
    return parser.results[:max_results]


async def _ddg_search(query: str, max_results: int) -> list[SearchResult]:
    response = await get_response(
        DDG_HTML_ENDPOINT,
        user_agent=DDG_BROWSER_USER_AGENT,
        max_bytes=MAX_RESPONSE_BYTES,
        method="POST",
        data={"q": query},
        headers=DDG_BROWSER_HEADERS,
    )
    try:
        return parse_search_results(response_text(response), max_results=max_results)
    except (SearchProviderChallengeError, SearchParserError) as primary_error:
        try:
            lite_response = await get_response(
                DDG_LITE_ENDPOINT,
                user_agent=DDG_BROWSER_USER_AGENT,
                max_bytes=MAX_RESPONSE_BYTES,
                method="POST",
                data={"q": query},
                headers=DDG_BROWSER_HEADERS,
            )
            return parse_lite_search_results(
                response_text(lite_response), max_results=max_results
            )
        except (SearchProviderChallengeError, SearchParserError):
            raise primary_error


async def _websearch(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    _abort_signal: AbortSignal,
) -> StructuredToolResult:
    query = arguments["query"]
    max_results = arguments.get("max_results", 8)
    results = await _ddg_search(query, max_results)
    serialized = json.dumps(results, ensure_ascii=False, indent=2)
    effective_limit = min(MAX_OUTPUT_BYTES, registry.max_output_chars)
    return _success_result(
        output_block(serialized, limit=effective_limit),
        structured_content={"results": results},
    )


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "websearch",
        _websearch,
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
