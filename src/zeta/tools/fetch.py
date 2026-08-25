"""Fetch a URL and return readable text."""

from __future__ import annotations

import re
from collections.abc import Mapping
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

import httpx

from ..core.abort import AbortSignal
from ..types import StructuredToolResult, ToolTextBlock
from .registry import ToolRegistry, _success_result, text_block

HTTP_TIMEOUT_SECONDS = 15.0
MAX_RESPONSE_BYTES = 2_000_000
MAX_OUTPUT_BYTES = 50_000
MAX_REDIRECTS = 5
OUTPUT_TRUNCATION_MARKER = "\n...[output truncated]"


async def get_response(
    url: str,
    *,
    user_agent: str,
    max_bytes: int = MAX_RESPONSE_BYTES,
    params: Mapping[str, str] | None = None,
) -> httpx.Response:
    """Make one bounded request and turn transport failures into clear errors."""

    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            follow_redirects=True,
            max_redirects=MAX_REDIRECTS,
            headers={"User-Agent": user_agent},
        ) as client:
            response = await client.get(url, params=params)
    except httpx.TooManyRedirects as exc:
        raise ValueError(
            f"request failed: redirect limit exceeded ({MAX_REDIRECTS})"
        ) from exc
    except httpx.TimeoutException as exc:
        raise ValueError("request failed: request timed out") from exc
    except httpx.RequestError as exc:
        raise ValueError(f"request failed: {exc}") from exc

    if response.status_code >= 400:
        reason = response.reason_phrase or "HTTP error"
        raise ValueError(f"request failed: HTTP {response.status_code} {reason}")

    content_length = response.headers.get("content-length")
    if content_length is not None:
        try:
            declared_length = int(content_length)
        except ValueError:
            declared_length = None
        if declared_length is not None and declared_length > max_bytes:
            raise ValueError(
                f"response too large: {declared_length} bytes exceeds {max_bytes}"
            )
    if len(response.content) > max_bytes:
        raise ValueError(
            f"response too large: more than {max_bytes} bytes received"
        )
    return response


def response_text(response: httpx.Response) -> str:
    """Decode a response using its declared charset, with replacement fallback."""

    encoding = response.encoding or "utf-8"
    return response.content.decode(encoding, errors="replace")


def output_block(value: str, *, limit: int = MAX_OUTPUT_BYTES) -> ToolTextBlock:
    """Build a capped MCP text block while retaining the original byte size."""

    encoded = value.encode("utf-8")
    if len(encoded) <= limit:
        return text_block(value, full_size=len(encoded))

    available = max(0, limit - len(OUTPUT_TRUNCATION_MARKER.encode("utf-8")))
    low = 0
    high = len(value)
    while low < high:
        middle = (low + high + 1) // 2
        if len(value[:middle].encode("utf-8")) <= available:
            low = middle
        else:
            high = middle - 1
    shown = value[:low] + OUTPUT_TRUNCATION_MARKER
    return text_block(shown, full_size=len(encoded))


class _ReadableHTMLParser(HTMLParser):
    _block_tags = frozenset({
        "address",
        "article",
        "aside",
        "blockquote",
        "br",
        "div",
        "footer",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "header",
        "li",
        "main",
        "nav",
        "ol",
        "p",
        "pre",
        "section",
        "table",
        "tr",
        "ul",
    })

    def __init__(self, base_url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.parts: list[str] = []
        self._ignored_depth = 0
        self._links: list[tuple[str | None, list[str]]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "a":
            href = dict(attrs).get("href")
            absolute_href = urljoin(self.base_url, href) if href else None
            self._links.append((absolute_href, []))
        if tag in self._block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "a" and self._links:
            href, link_parts = self._links.pop()
            text = "".join(link_parts).strip()
            if text:
                self.parts.append(text)
                if href:
                    self.parts.append(f" ({href})")
        if tag in self._block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._ignored_depth:
            return
        if self._links:
            self._links[-1][1].append(data)
        else:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[ \t\f\v]+", " ", value)
        value = re.sub(r"\n[ \t]+", "\n", value)
        value = re.sub(r"\n{2,}", "\n", value)
        return value.strip()


def _normalize_url(raw_url: str) -> str:
    url = raw_url if "://" in raw_url else f"https://{raw_url}"
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must use http or https and include a host")
    return url


def _readable_content(url: str, content_type: str, body: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if media_type in {"text/html", "application/xhtml+xml"}:
        parser = _ReadableHTMLParser(url)
        parser.feed(body)
        parser.close()
        return parser.text()
    if (
        not media_type
        or media_type.startswith("text/")
        or media_type == "application/json"
        or media_type.endswith("+json")
        or media_type == "application/xml"
        or media_type.endswith("+xml")
    ):
        return body
    if media_type.startswith(("audio/", "video/", "image/")) or media_type in {
        "application/octet-stream",
        "application/pdf",
        "application/zip",
        "application/gzip",
    }:
        raise ValueError(f"refusing binary content-type: {content_type or media_type}")
    raise ValueError(f"refusing binary content-type: {content_type or media_type}")


async def _fetch(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    _abort_signal: AbortSignal,
) -> StructuredToolResult:
    del registry
    url = _normalize_url(arguments["url"])
    max_bytes = arguments.get("max_bytes", MAX_RESPONSE_BYTES)
    response = await get_response(
        url,
        user_agent="zeta/fetch (web tool)",
        max_bytes=max_bytes,
    )
    body = _readable_content(
        url,
        response.headers.get("content-type", ""),
        response_text(response),
    )
    notice = ""
    if urlsplit(url).scheme == "http":
        notice = "notice: http URL is not encrypted\n\n"
    return _success_result(output_block(notice + body))


def register(registry: ToolRegistry) -> None:
    registry.register(
        "fetch",
        lambda arguments, abort_signal: _fetch(registry, arguments, abort_signal),
        description=(
            "Fetch a URL and return readable text. Network access requires approval; "
            "HTTP URLs are allowed with a notice."
        ),
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "minLength": 1},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESPONSE_BYTES,
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    )
