"""Fetch a URL and return readable text."""

from __future__ import annotations

import asyncio
import re
import socket
import zlib
from collections.abc import AsyncIterator, Mapping
from html.parser import HTMLParser
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network
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
_PRIVATE_NETWORKS = (
    ip_network("10.0.0.0/8"),
    ip_network("172.16.0.0/12"),
    ip_network("192.168.0.0/16"),
    ip_network("127.0.0.0/8"),
    ip_network("169.254.0.0/16"),
    ip_network("::1/128"),
    ip_network("fc00::/7"),
    ip_network("fe80::/10"),
)
_CLOUD_METADATA_ADDRESS = ip_address("169.254.169.254")
_PRIVATE_TARGET_EXTENSION = "zeta_private_target"


class _DecompressedResponseTooLarge(ValueError):
    def __init__(self, max_bytes: int) -> None:
        super().__init__(f"response too large: more than {max_bytes} bytes decompressed")
        self.max_bytes = max_bytes


class _CompressedResponseTruncated(ValueError):
    def __init__(self) -> None:
        super().__init__("response truncated: compressed stream ended early")


async def _raw_response_chunks(response: httpx.Response) -> AsyncIterator[bytes]:
    if response.is_stream_consumed:
        yield response.content
        return
    async for chunk in response.aiter_raw():
        yield chunk


def _normalize_address(
    address: IPv4Address | IPv6Address,
) -> IPv4Address | IPv6Address:
    if not isinstance(address, IPv6Address):
        return address
    if address.ipv4_mapped is not None:
        return address.ipv4_mapped
    if address.packed[:12] == b"\x00" * 12 and address not in {
        IPv6Address("::"),
        IPv6Address("::1"),
    }:
        return IPv4Address(int(address))
    return address


def _target_addresses(url: str) -> tuple[IPv4Address | IPv6Address, ...]:
    hostname = urlsplit(url).hostname
    if hostname is None:
        raise ValueError("URL must use http or https and include a host")
    try:
        return (_normalize_address(ip_address(hostname)),)
    except ValueError:
        parsed = urlsplit(url)
        try:
            infos = socket.getaddrinfo(
                hostname,
                parsed.port or (443 if parsed.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except socket.gaierror as exc:
            raise ValueError(f"request failed: could not resolve host {hostname}") from exc
        return tuple(_normalize_address(ip_address(info[4][0])) for info in infos)


def _classify_target(
    addresses: tuple[IPv4Address | IPv6Address, ...],
) -> bool:
    if _CLOUD_METADATA_ADDRESS in addresses:
        raise ValueError("refusing cloud metadata target 169.254.169.254")
    return any(
        address in network for address in addresses for network in _PRIVATE_NETWORKS
    )


def _validate_target(url: str) -> bool:
    return _classify_target(_target_addresses(url))


def _host_header(url: str) -> str:
    return httpx.URL(url).netloc.decode("ascii")


def _pinned_url(url: str, address: IPv4Address | IPv6Address) -> str:
    return str(httpx.URL(url).copy_with(host=str(address)))


async def get_response(
    url: str,
    *,
    user_agent: str,
    max_bytes: int = MAX_RESPONSE_BYTES,
    params: Mapping[str, str] | None = None,
    method: str = "GET",
    data: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
) -> httpx.Response:
    """Make a bounded, manually redirected request.

    Private, loopback, link-local, and RFC1918 targets are allowed because zeta
    is a local-first tool. They produce a notice in the returned fetch output.
    The cloud metadata address 169.254.169.254 is always refused.
    Environment proxies are disabled because they would bypass validated-address
    connection pinning.
    """

    try:
        request_headers = {"User-Agent": user_agent}
        if headers is not None:
            request_headers.update(headers)
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
            headers=request_headers,
        ) as client:
            current_url = url
            current_method = method
            redirects_followed = 0
            while True:
                _validate_url(current_url)
                addresses = _target_addresses(current_url)
                private_target = _classify_target(addresses)
                address = addresses[0]
                async with client.stream(
                    current_method,
                    _pinned_url(current_url, address),
                    params=params if redirects_followed == 0 else None,
                    data=data if redirects_followed == 0 else None,
                    headers={"Host": _host_header(current_url)},
                    extensions={"sni_hostname": urlsplit(current_url).hostname},
                ) as response:
                    if response.is_redirect:
                        location = response.headers.get("location")
                        if not location:
                            raise ValueError(
                                "request failed: redirect response missing location"
                            )
                        if redirects_followed >= MAX_REDIRECTS:
                            raise ValueError(
                                f"request failed: redirect limit exceeded ({MAX_REDIRECTS})"
                            )
                        if response.status_code in {301, 302, 303}:
                            current_method = "GET"
                        current_url = urljoin(current_url, location)
                        redirects_followed += 1
                        continue
                    if response.status_code >= 400:
                        reason = response.reason_phrase or "HTTP error"
                        raise ValueError(
                            f"request failed: HTTP {response.status_code} {reason}"
                        )
                    content_encoding = response.headers.get("content-encoding", "")
                    encodings = tuple(
                        encoding.strip().lower()
                        for encoding in content_encoding.split(",")
                        if encoding.strip()
                    )
                    if len(encodings) > 1:
                        raise ValueError(
                            "request failed: multiple content encodings are not supported"
                        )
                    encoding = encodings[0] if encodings else "identity"
                    decoder = (
                        zlib.decompressobj(wbits=47)
                        if encoding in {"gzip", "deflate"}
                        else None
                    )
                    compressed_prefix = bytearray() if encoding == "deflate" else None
                    chunks: list[bytes] = []
                    received = 0
                    decompressed = 0
                    async for chunk in _raw_response_chunks(response):
                        received += len(chunk)
                        if received > max_bytes:
                            raise ValueError(
                                f"response too large: more than {max_bytes} bytes received"
                            )
                        if decoder is None:
                            chunks.append(chunk)
                            continue
                        if compressed_prefix is not None:
                            compressed_prefix.extend(chunk)
                        pending = chunk
                        while pending:
                            remaining = max_bytes - decompressed
                            try:
                                decoded = decoder.decompress(
                                    pending,
                                    max_length=remaining + 1,
                                )
                            except zlib.error:
                                if compressed_prefix is None:
                                    raise
                                decoder = zlib.decompressobj(wbits=-15)
                                chunks.clear()
                                decompressed = 0
                                pending = bytes(compressed_prefix)
                                compressed_prefix = None
                                continue
                            decompressed += len(decoded)
                            if decompressed > max_bytes:
                                raise _DecompressedResponseTooLarge(max_bytes)
                            if decoded:
                                chunks.append(decoded)
                            pending = decoder.unconsumed_tail
                    if decoder is not None:
                        remaining = max_bytes - decompressed
                        decoded = decoder.flush(remaining + 1)
                        decompressed += len(decoded)
                        if decompressed > max_bytes:
                            raise _DecompressedResponseTooLarge(max_bytes)
                        if decoded:
                            chunks.append(decoded)
                        if not decoder.eof:
                            raise _CompressedResponseTruncated()
                    headers = response.headers.copy()
                    headers.pop("content-encoding", None)
                    headers.pop("content-length", None)
                    return httpx.Response(
                        response.status_code,
                        headers=headers,
                        content=b"".join(chunks),
                        request=httpx.Request(current_method, current_url),
                        extensions={
                            **response.extensions,
                            _PRIVATE_TARGET_EXTENSION: private_target,
                        },
                    )
    except httpx.TooManyRedirects as exc:
        raise ValueError(
            f"request failed: redirect limit exceeded ({MAX_REDIRECTS})"
        ) from exc
    except httpx.TimeoutException as exc:
        raise ValueError("request failed: request timed out") from exc
    except httpx.RequestError as exc:
        raise ValueError(f"request failed: {exc}") from exc
    raise AssertionError("unreachable response loop")


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


def _page_block(
    notice: str,
    body: str,
    *,
    offset: int,
    limit: int,
) -> ToolTextBlock:
    """Build one readable-text page that fits the provider output limit."""

    full_size_chars = len(body)
    full_size = len(body.encode("utf-8"))
    if offset >= full_size_chars:
        page_end = offset
    else:
        page_chars = min(limit, full_size_chars - offset)
        low = 0
        high = page_chars
        while low < high:
            middle = (low + high + 1) // 2
            page = body[offset : offset + middle]
            value = notice + page
            if len(value) <= limit and len(value.encode("utf-8")) <= limit:
                low = middle
            else:
                high = middle - 1
        page_end = offset + low

    page = body[offset:page_end]
    block = text_block(notice + page, full_size=full_size)
    block["full_size_chars"] = full_size_chars
    block["truncated"] = page_end < full_size_chars
    if block["truncated"]:
        block["next_offset"] = page_end
    return block


def _truncated_error_result(
    message: str, *, full_size: int
) -> StructuredToolResult:
    message = f"{message}{OUTPUT_TRUNCATION_MARKER}"
    block = text_block(
        message,
        full_size=max(len(message.encode("utf-8")), full_size),
    )
    block["truncated"] = True
    return {
        "content": [block],
        "isError": True,
        "structuredContent": None,
    }


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


def _validate_url(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("URL must use http or https and include a host")


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
    abort_signal: AbortSignal,
) -> StructuredToolResult:
    url = _normalize_url(arguments["url"])
    max_bytes = arguments.get("max_bytes", MAX_RESPONSE_BYTES)
    offset = arguments.get("offset", 0)
    if abort_signal.is_set():
        raise asyncio.CancelledError()
    try:
        response = await get_response(
            url,
            user_agent="zeta/fetch (web tool)",
            max_bytes=max_bytes,
        )
        if abort_signal.is_set():
            raise asyncio.CancelledError()
    except _DecompressedResponseTooLarge as exc:
        return _truncated_error_result(str(exc), full_size=exc.max_bytes + 1)
    except _CompressedResponseTruncated as exc:
        return _truncated_error_result(str(exc), full_size=max_bytes + 1)
    final_url = str(response.request.url) if response.request is not None else url
    body = _readable_content(
        final_url,
        response.headers.get("content-type", ""),
        response_text(response),
    )
    notices: list[str] = []
    if urlsplit(final_url).scheme == "http":
        notices.append("notice: http URL is not encrypted")
    if response.extensions.get(_PRIVATE_TARGET_EXTENSION, False):
        notices.append("notice: target resolves to a private or loopback address")
    notice = "\n".join(notices)
    if notice:
        notice += "\n\n"
    effective_limit = min(MAX_OUTPUT_BYTES, registry.max_output_chars)
    block = _page_block(notice, body, offset=offset, limit=effective_limit)
    if block["truncated"] and block.get("next_offset", offset) <= offset:
        return _truncated_error_result(
            "output limit too small for notice", full_size=len(body.encode("utf-8"))
        )
    return _success_result(block)


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "fetch",
        _fetch,
        description=(
            "Fetch a URL and return readable text. Network access requires approval; "
            "HTTP URLs are allowed with a notice. Private, loopback, link-local, "
            "and RFC1918 targets are allowed with a notice for local-first use; "
            "the cloud metadata address 169.254.169.254 is refused. If truncated, "
            "call again with offset=next_offset to continue."
        ),
        parallel_safe=True,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "minLength": 1},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_RESPONSE_BYTES,
                },
                "offset": {
                    "type": "integer",
                    "minimum": 0,
                    "description": "Readable-text character offset for continuation.",
                },
            },
            "required": ["url"],
            "additionalProperties": False,
        },
    )
