"""Fetch a URL and return readable text."""

from __future__ import annotations

import asyncio
import codecs
import io
import os
import socket
import zlib
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import asynccontextmanager
from html.parser import HTMLParser
from ipaddress import IPv4Address, IPv6Address, ip_address, ip_network
from typing import Any, BinaryIO
from urllib.parse import urljoin, urlsplit

import httpx

from ...core.abort import AbortSignal
from ...protocol.types import StructuredToolResult
from ..registry import (
    ToolExecutionContext,
    ToolRegistry,
    _success_result,
    text_block,
)

HTTP_TIMEOUT_SECONDS = 15.0
# Kept for websearch callers. The fetch tool has its own larger safety bound.
MAX_RESPONSE_BYTES = 2_000_000
FETCH_SAFETY_MAX_BYTES = 100 * 1024 * 1024
MAX_OUTPUT_BYTES = 50_000
MAX_REDIRECTS = 5
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
_PARTIAL_BODY_NOTICE_EXTENSION = "zeta_partial_body_notice"


def _format_safety_ceiling(size: int) -> str:
    mebibyte = 1024 * 1024
    if size % mebibyte == 0:
        return f"{size // mebibyte} MiB"
    return f"{size} bytes"


class _DecompressionFailed(Exception):
    """Raised when a compressed response body cannot be decoded."""

    def __init__(self, original: str) -> None:
        super().__init__(original)
        self.original = original


async def _run_worker(function: Callable[..., Any], /, *args: Any, **kwargs: Any) -> Any:
    worker = asyncio.create_task(asyncio.to_thread(function, *args, **kwargs))
    try:
        return await asyncio.shield(worker)
    except asyncio.CancelledError:
        await worker
        raise


@asynccontextmanager
async def _temporary_file(spills: Any) -> AsyncIterator[BinaryIO]:
    manager = spills.temporary_file()
    entry = asyncio.create_task(asyncio.to_thread(manager.__enter__))
    try:
        handle = await asyncio.shield(entry)
    except asyncio.CancelledError:
        await entry
        await _run_worker(manager.__exit__, None, None, None)
        raise
    try:
        yield handle
    finally:
        await _run_worker(manager.__exit__, None, None, None)


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


async def _target_addresses(
    url: str,
) -> tuple[IPv4Address | IPv6Address, ...]:
    hostname = urlsplit(url).hostname
    if hostname is None:
        raise ValueError("URL must use http or https and include a host")
    try:
        return (_normalize_address(ip_address(hostname)),)
    except ValueError:
        parsed = urlsplit(url)
        try:
            async with asyncio.timeout(HTTP_TIMEOUT_SECONDS):
                infos = await asyncio.get_running_loop().getaddrinfo(
                    hostname,
                    parsed.port or (443 if parsed.scheme == "https" else 80),
                    type=socket.SOCK_STREAM,
                )
        except TimeoutError as exc:
            raise ValueError("request failed: request timed out") from exc
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
    """Make a bounded, manually redirected request in memory.

    This compatibility interface is used only by callers with small response
    bounds. The fetch tool uses ``_get_response_to_file`` so a large body never
    needs a body-sized allocation.
    """

    with io.BytesIO() as destination:
        response = await _get_response_to_file(
            url,
            destination=destination,
            user_agent=user_agent,
            max_bytes=max_bytes,
            params=params,
            method=method,
            data=data,
            headers=headers,
        )
        return httpx.Response(
            response.status_code,
            headers=response.headers,
            content=destination.getvalue(),
            request=response.request,
            extensions=response.extensions,
        )


def _reset_destinations(
    destination: BinaryIO, received_destination: BinaryIO | None
) -> None:
    destination.seek(0)
    destination.truncate()
    if received_destination is not None:
        received_destination.seek(0)
        received_destination.truncate()


def _flush_destinations(
    destination: BinaryIO, received_destination: BinaryIO | None
) -> None:
    destination.flush()
    if received_destination is not None:
        received_destination.flush()


async def _get_response_to_file(
    url: str,
    *,
    destination: BinaryIO,
    user_agent: str,
    max_bytes: int,
    params: Mapping[str, str] | None = None,
    method: str = "GET",
    data: Mapping[str, str] | None = None,
    headers: Mapping[str, str] | None = None,
    received_destination: BinaryIO | None = None,
) -> httpx.Response:
    """Stream a bounded, decoded response body into a seekable file."""

    request_headers = {"User-Agent": user_agent}
    if headers is not None:
        request_headers.update(headers)
    for attempt in range(2):
        await _run_worker(_reset_destinations, destination, received_destination)
        try:
            return await _request_and_decode(
                url,
                request_headers=request_headers,
                max_bytes=max_bytes,
                params=params,
                method=method,
                data=data,
                destination=destination,
                received_destination=received_destination,
            )
        except _DecompressionFailed as exc:
            if attempt == 0 and not _has_identity_encoding(request_headers):
                request_headers = {**request_headers, "Accept-Encoding": "identity"}
                continue
            raise ValueError(f"request failed: {exc.original}") from exc
    raise AssertionError("unreachable response loop")


def _has_identity_encoding(request_headers: Mapping[str, str]) -> bool:
    for key, value in request_headers.items():
        if key.lower() == "accept-encoding":
            return "identity" in value.lower()
    return False


async def _request_and_decode(
    url: str,
    *,
    request_headers: Mapping[str, str],
    max_bytes: int,
    params: Mapping[str, str] | None,
    method: str,
    data: Mapping[str, str] | None,
    destination: BinaryIO,
    received_destination: BinaryIO | None,
) -> httpx.Response:
    try:
        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SECONDS,
            follow_redirects=False,
            trust_env=False,
            headers=request_headers,
        ) as client:
            current_url = url
            current_method = method
            current_data = data
            redirects_followed = 0
            while True:
                _validate_url(current_url)
                addresses = await _target_addresses(current_url)
                private_target = _classify_target(addresses)
                address = addresses[0]
                async with client.stream(
                    current_method,
                    _pinned_url(current_url, address),
                    params=params if redirects_followed == 0 else None,
                    data=current_data,
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
                            current_data = None
                        current_url = urljoin(current_url, location)
                        redirects_followed += 1
                        continue
                    if response.status_code >= 400:
                        reason = response.reason_phrase or "HTTP error"
                        raise ValueError(
                            f"request failed: HTTP {response.status_code} {reason}"
                        )
                    partial_notice = await _decode_body_to_file(
                        response,
                        destination=destination,
                        received_destination=received_destination,
                        max_bytes=max_bytes,
                    )
                    await _run_worker(
                        _flush_destinations, destination, received_destination
                    )
                    headers = response.headers.copy()
                    headers.pop("content-encoding", None)
                    headers.pop("content-length", None)
                    return httpx.Response(
                        response.status_code,
                        headers=headers,
                        content=b"",
                        request=httpx.Request(current_method, current_url),
                        extensions={
                            **response.extensions,
                            _PRIVATE_TARGET_EXTENSION: private_target,
                            _PARTIAL_BODY_NOTICE_EXTENSION: partial_notice,
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


class _BodyDecoder:
    def __init__(
        self,
        *,
        encoding: str,
        destination: BinaryIO,
        received_destination: BinaryIO | None,
        max_bytes: int,
    ) -> None:
        self.destination = destination
        self.received_destination = received_destination
        self.max_bytes = max_bytes
        self.decoder = (
            zlib.decompressobj(wbits=47)
            if encoding in {"gzip", "deflate"}
            else None
        )
        self.compressed_prefix = bytearray() if encoding == "deflate" else None
        self.received = 0
        self.decompressed = 0
        self.partial_notice: str | None = None

    def consume(self, raw_chunk: bytes) -> None:
        received_remaining = self.max_bytes - self.received
        chunk = raw_chunk[:received_remaining]
        self.received += len(chunk)
        if self.received_destination is not None and chunk:
            self.received_destination.write(chunk)
        received_limited = len(chunk) < len(raw_chunk)
        if self.decoder is None:
            if chunk:
                self.destination.write(chunk)
            if received_limited:
                self.partial_notice = (
                    f"stopped at {_format_safety_ceiling(self.max_bytes)}: response exceeded the "
                    "received safety limit"
                )
            return
        if self.compressed_prefix is not None:
            self.compressed_prefix.extend(chunk)
        pending = chunk
        retried_raw_deflate = False
        produced_output = False
        while pending:
            remaining = self.max_bytes - self.decompressed
            try:
                decoded = self.decoder.decompress(pending, max_length=remaining + 1)
            except zlib.error as exc:
                if self.compressed_prefix is None:
                    raise _DecompressionFailed(str(exc)) from exc
                self.decoder = zlib.decompressobj(wbits=-15)
                self.destination.seek(0)
                self.destination.truncate()
                self.decompressed = 0
                pending = bytes(self.compressed_prefix)
                self.compressed_prefix = None
                retried_raw_deflate = True
                continue
            if len(decoded) > remaining:
                self.destination.write(decoded[:remaining])
                self.decompressed += remaining
                self.partial_notice = (
                    f"stopped at {_format_safety_ceiling(self.max_bytes)}: response exceeded the "
                    "decompressed safety limit"
                )
                break
            self.decompressed += len(decoded)
            if decoded:
                self.destination.write(decoded)
                produced_output = True
            pending = self.decoder.unconsumed_tail
        if (
            self.compressed_prefix is not None
            and not retried_raw_deflate
            and produced_output
        ):
            self.compressed_prefix = None
        if self.partial_notice is None and received_limited:
            self.partial_notice = (
                f"stopped at {_format_safety_ceiling(self.max_bytes)}: response exceeded the "
                "received safety limit"
            )

    def finish(self) -> str | None:
        if self.decoder is not None and self.partial_notice is None:
            remaining = self.max_bytes - self.decompressed
            try:
                decoded = self.decoder.flush(remaining + 1)
            except zlib.error as exc:
                raise _DecompressionFailed(str(exc)) from exc
            if len(decoded) > remaining:
                self.destination.write(decoded[:remaining])
                self.partial_notice = (
                    f"stopped at {_format_safety_ceiling(self.max_bytes)}: response exceeded the "
                    "decompressed safety limit"
                )
            elif decoded:
                self.destination.write(decoded)
            if self.partial_notice is None and not self.decoder.eof:
                self.partial_notice = (
                    "compressed stream ended early; returned decoded prefix"
                )
        return self.partial_notice


async def _decode_body_to_file(
    response: httpx.Response,
    *,
    destination: BinaryIO,
    received_destination: BinaryIO | None,
    max_bytes: int,
) -> str | None:
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
    decoder = _BodyDecoder(
        encoding=encodings[0] if encodings else "identity",
        destination=destination,
        received_destination=received_destination,
        max_bytes=max_bytes,
    )
    async for raw_chunk in _raw_response_chunks(response):
        await _run_worker(decoder.consume, raw_chunk)
        if decoder.partial_notice is not None:
            break
    return await _run_worker(decoder.finish)


def response_text(response: httpx.Response) -> str:
    """Decode a response using its declared charset, with replacement fallback."""

    encoding = response.encoding or "utf-8"
    return response.content.decode(encoding, errors="replace")


class _NormalizedTextWriter:
    """Normalize readable text while retaining only one input chunk."""

    def __init__(self, destination: BinaryIO) -> None:
        self.destination = destination
        self.full_size_chars = 0
        self.full_size = 0
        self._has_output = False
        self._pending_space = False
        self._pending_newline = False

    def write(self, value: str) -> None:
        output: list[str] = []
        for character in value:
            if character == "\n":
                self._pending_newline = True
                self._pending_space = False
                continue
            if character in " \t\f\v":
                if not self._pending_newline:
                    self._pending_space = True
                continue
            if self._has_output:
                if self._pending_newline:
                    output.append("\n")
                elif self._pending_space:
                    output.append(" ")
            output.append(character)
            self._has_output = True
            self._pending_newline = False
            self._pending_space = False
        if not output:
            return
        text = "".join(output)
        encoded = text.encode("utf-8")
        self.destination.write(encoded)
        self.full_size_chars += len(text)
        self.full_size += len(encoded)


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

    def __init__(self, base_url: str, writer: _NormalizedTextWriter) -> None:
        super().__init__(convert_charrefs=True)
        self.base_url = base_url
        self.writer = writer
        self._ignored_depth = 0
        self._link_href: str | None = None
        self._link_parts: list[str] | None = None

    def _emit(self, value: str) -> None:
        if self._link_parts is not None:
            self._link_parts.append(value)
        else:
            self.writer.write(value)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            self._ignored_depth += 1
            return
        if self._ignored_depth:
            return
        if tag == "a" and self._link_parts is None:
            href = dict(attrs).get("href")
            self._link_href = urljoin(self.base_url, href) if href else None
            self._link_parts = []
        if tag in self._block_tags:
            self._emit("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if tag in {"script", "style"}:
            if self._ignored_depth:
                self._ignored_depth -= 1
            return
        if self._ignored_depth:
            return
        if tag == "a" and self._link_parts is not None:
            text = "".join(self._link_parts).strip()
            href = self._link_href
            self._link_parts = None
            self._link_href = None
            if text:
                self.writer.write(text)
                if href:
                    self.writer.write(f" ({href})")
        if tag in self._block_tags:
            self._emit("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self._emit(data)


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


def _validate_readable_content_type(content_type: str) -> str:
    media_type = content_type.split(";", 1)[0].strip().lower()
    if (
        media_type in {"text/html", "application/xhtml+xml"}
        or not media_type
        or media_type.startswith("text/")
        or media_type in {"application/json", "application/xml"}
        or media_type.endswith(("+json", "+xml"))
    ):
        return media_type
    raise ValueError(f"refusing binary content-type: {content_type or media_type}")


def _readable_content(url: str, content_type: str, body: str) -> str:
    """Return readable content for compatibility callers with in-memory text."""

    if _validate_readable_content_type(content_type) not in {
        "text/html",
        "application/xhtml+xml",
    }:
        return body
    with io.BytesIO(body.encode("utf-8")) as source, io.BytesIO() as destination:
        _stream_readable_text(
            source,
            destination,
            url=url,
            content_type=content_type,
            encoding="utf-8",
        )
        return destination.getvalue().decode("utf-8")


def _write_utf8_chunks(destination: BinaryIO, value: str) -> int:
    size = 0
    for start in range(0, len(value), 64 * 1024):
        encoded = value[start : start + 64 * 1024].encode("utf-8")
        destination.write(encoded)
        size += len(encoded)
    return size


def _stream_readable_text(
    source: BinaryIO,
    destination: BinaryIO,
    *,
    url: str,
    content_type: str,
    encoding: str,
) -> tuple[int, int]:
    """Stream readable UTF-8 text and return its character and byte sizes."""

    media_type = _validate_readable_content_type(content_type)
    source.seek(0)
    decoder = codecs.getincrementaldecoder(encoding)(errors="replace")
    if media_type in {"text/html", "application/xhtml+xml"}:
        writer = _NormalizedTextWriter(destination)
        parser = _ReadableHTMLParser(url, writer)
        while chunk := source.read(64 * 1024):
            parser.feed(decoder.decode(chunk))
        parser.feed(decoder.decode(b"", final=True))
        parser.close()
        return writer.full_size_chars, writer.full_size

    full_size_chars = 0
    full_size = 0
    while chunk := source.read(64 * 1024):
        text = decoder.decode(chunk)
        full_size_chars += len(text)
        full_size += _write_utf8_chunks(destination, text)
    final = decoder.decode(b"", final=True)
    full_size_chars += len(final)
    full_size += _write_utf8_chunks(destination, final)
    return full_size_chars, full_size


def _fit_page_prefix(value: str, *, char_limit: int, byte_limit: int) -> str:
    high = min(len(value), max(0, char_limit))
    low = 0
    while low < high:
        middle = (low + high + 1) // 2
        if len(value[:middle].encode("utf-8")) <= byte_limit:
            low = middle
        else:
            high = middle - 1
    return value[:low]


def _read_utf8_page(
    source: BinaryIO,
    *,
    offset: int,
    char_limit: int,
    byte_limit: int,
) -> tuple[str, int]:
    source.seek(0)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="strict")
    position = 0
    parts: list[str] = []
    retained_chars = 0
    retained_bytes = 0
    while chunk := source.read(64 * 1024):
        text = decoder.decode(chunk)
        chunk_end = position + len(text)
        if chunk_end <= offset:
            position = chunk_end
            continue
        candidate = text[max(0, offset - position) :]
        fitted = _fit_page_prefix(
            candidate,
            char_limit=char_limit - retained_chars,
            byte_limit=byte_limit - retained_bytes,
        )
        parts.append(fitted)
        retained_chars += len(fitted)
        retained_bytes += len(fitted.encode("utf-8"))
        position = chunk_end
        if len(fitted) < len(candidate):
            break
    return "".join(parts), offset + retained_chars


async def _fetch(
    registry: ToolRegistry,
    arguments: dict[str, Any],
    abort_signal: AbortSignal,
    *,
    execution_context: ToolExecutionContext | None = None,
) -> StructuredToolResult:
    url = _normalize_url(arguments["url"])
    max_bytes = arguments.get("max_bytes", FETCH_SAFETY_MAX_BYTES)
    offset = arguments.get("offset", 0)
    if abort_signal.is_set():
        raise asyncio.CancelledError()

    async with (
        _temporary_file(registry.spills) as received_body,
        _temporary_file(registry.spills) as raw_body,
    ):
        response = await _get_response_to_file(
            url,
            destination=raw_body,
            user_agent="zeta/fetch (web tool)",
            max_bytes=max_bytes,
            received_destination=received_body,
        )
        if abort_signal.is_set():
            raise asyncio.CancelledError()
        final_url = str(response.request.url) if response.request is not None else url
        async with _temporary_file(registry.spills) as readable_body:
            full_size_chars, full_size = await _run_worker(
                _stream_readable_text,
                raw_body,
                readable_body,
                url=final_url,
                content_type=response.headers.get("content-type", ""),
                encoding=response.encoding or "utf-8",
            )
            await _run_worker(readable_body.flush)

            notices: list[str] = []
            partial_notice = response.extensions.get(
                _PARTIAL_BODY_NOTICE_EXTENSION
            )
            if isinstance(partial_notice, str) and partial_notice:
                notices.append(f"notice: {partial_notice}")
            if urlsplit(final_url).scheme == "http":
                notices.append("notice: http URL is not encrypted")
            if response.extensions.get(_PRIVATE_TARGET_EXTENSION, False):
                notices.append(
                    "notice: target resolves to a private or loopback address"
                )

            effective_limit = min(MAX_OUTPUT_BYTES, registry.max_output_chars)
            spill_path = None
            artifacts: list[dict[str, str | int]] = []
            if (
                full_size_chars > effective_limit
                or full_size > effective_limit
                or partial_notice is not None
            ):
                call_id = (
                    execution_context.tool_call.id
                    if execution_context is not None
                    else "fetch"
                )
                sources: dict[str, list[BinaryIO]] = {"readable": [readable_body]}
                if partial_notice is not None:
                    sources["raw"] = [received_body]
                paths = await registry.spills.awrite_group("fetch", call_id, sources)
                spill_path = paths["readable"]
                artifacts.append(
                    {
                        "name": "readable",
                        "path": str(spill_path),
                        "full_size": full_size,
                    }
                )
                spill_notice = (
                    f"notice: full readable content ({full_size} bytes) "
                    f"is saved at {spill_path}; read it with read using offset/limit"
                )
                existing_notice_size = len("\n".join(notices))
                if existing_notice_size + len(spill_notice) + 3 < effective_limit:
                    notices.append(spill_notice)
                if partial_notice is not None:
                    raw_path = paths["raw"]
                    artifacts.append(
                        {
                            "name": "raw",
                            "path": str(raw_path),
                            "full_size": (
                                await _run_worker(os.fstat, received_body.fileno())
                            ).st_size,
                        }
                    )
                    raw_notice = (
                        "notice: all body bytes received before the safety ceiling "
                        f"are saved at {raw_path}"
                    )
                    if (
                        len("\n".join(notices)) + len(raw_notice) + 3
                        < effective_limit
                    ):
                        notices.append(raw_notice)

            notice = "\n".join(notices)
            if notice:
                notice += "\n\n"
            page, page_end = await _run_worker(
                _read_utf8_page,
                readable_body,
                offset=offset,
                char_limit=max(0, effective_limit - len(notice)),
                byte_limit=max(
                    0, effective_limit - len(notice.encode("utf-8"))
                ),
            )

    block = text_block(notice + page, full_size=full_size)
    block["full_size_chars"] = full_size_chars
    block["truncated"] = page_end < full_size_chars or partial_notice is not None
    if page_end < full_size_chars and page_end > offset:
        block["next_offset"] = page_end
    if spill_path is not None:
        block["spill_path"] = str(spill_path)
    structured_content = {"artifacts": artifacts} if artifacts else None
    return _success_result(block, structured_content=structured_content)


def register(registry: ToolRegistry) -> None:
    registry.register_session_tool(
        "fetch",
        _fetch,
        approval_subject="url",
        description=(
            "Fetch a URL and return readable text. Network access requires approval; "
            "HTTP URLs are allowed with a notice. Private, loopback, link-local, "
            "and RFC1918 targets are allowed with a notice for local-first use; "
            "the cloud metadata address 169.254.169.254 is refused. Except when "
            "the fetch safety ceiling stops network intake, large bodies are saved "
            "in full; use spill_path with read, or call again with "
            "offset=next_offset to continue paging."
        ),
        parallel_safe=True,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "minLength": 1},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": FETCH_SAFETY_MAX_BYTES,
                    "description": (
                        "100 MiB default safety ceiling; lowering this value stops "
                        "network intake early. It applies to received and decompressed "
                        "bytes and returns a successful partial result with a notice."
                    ),
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
