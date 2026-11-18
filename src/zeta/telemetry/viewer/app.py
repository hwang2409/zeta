"""The viewer's one HTTP app: OTLP ingest, read-only JSON, health, and the packaged page."""

from __future__ import annotations

import asyncio
import contextlib
import queue
import re
from collections.abc import AsyncIterator
from importlib import resources
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .process import AUTHORITY, OTLP_URL, URL
from .receiver import MAX_BODY_BYTES, Reject, project_request
from .store import SCHEMA_VERSION, Reader, Writer, open_database, parse_page

_TRACE_ID = re.compile(r"[0-9a-f]{32}\Z")
_ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/viewer.css": ("viewer.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/format.js": ("format.js", "text/javascript; charset=utf-8"),
}
_SECURITY_HEADERS = [
    (b"content-security-policy", (
        b"default-src 'none'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self'; "
        b"base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    )),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"cache-control", b"no-store"),
]


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


class _Guard:
    """Refuses any request not addressed to the exact loopback authority, and adds security headers.

    The exact ``Host`` check stops DNS-rebinding pages; a present ``Origin`` must be the viewer's own.
    """

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        async def send_with_headers(message: Any) -> None:
            if message["type"] == "http.response.start":
                message["headers"] = [*message.get("headers", []), *_SECURITY_HEADERS]
            await send(message)

        headers = Headers(scope=scope)
        origins = headers.getlist("origin")
        if headers.getlist("host") != [AUTHORITY]:
            await _error(400, "invalid host")(scope, receive, send_with_headers)
        elif origins and origins != [URL.rstrip("/")]:
            await _error(403, "invalid origin")(scope, receive, send_with_headers)
        else:
            await self.app(scope, receive, send_with_headers)


def create_app(directory: Path, home_id: str) -> Starlette:
    """Migrate ``directory/telemetry.sqlite3`` and build the app; the writer runs during its lifespan."""
    database = directory / "telemetry.sqlite3"
    open_database(database)
    writer = Writer(database)
    reader = Reader(database)
    accepting = True
    static = resources.files(__package__) / "static"
    assets = {path: ((static / name).read_bytes(), media_type) for path, (name, media_type) in _ASSETS.items()}

    async def ingest(request: Request) -> Response:
        media_type = request.headers.get("content-type", "").split(";", 1)[0].strip().lower()
        if media_type != "application/x-protobuf":
            return _error(415, "content type must be application/x-protobuf")
        if request.headers.get("content-encoding", "identity").strip().lower() != "identity":
            return _error(415, "compressed bodies are not accepted")
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > MAX_BODY_BYTES):
            return _error(413, "body too large")
        if not accepting:
            return _error(503, "receiver stopping")
        body = bytearray()
        async for chunk in request.stream():
            body += chunk
            if len(body) > MAX_BODY_BYTES:
                return _error(413, "body too large")
        try:
            rows = await asyncio.to_thread(project_request, bytes(body))
        except Reject as exc:
            return _error(400, str(exc))
        try:
            committed = writer.submit(rows)
        except queue.Full:
            return _error(503, "receiver busy")
        try:
            await committed
        except Exception:  # noqa: BLE001 - any write failure is a retryable refusal; nothing was committed
            return _error(503, "receiver could not store the batch")
        from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
            ExportTraceServiceResponse,
        )

        return Response(ExportTraceServiceResponse().SerializeToString(), media_type="application/x-protobuf")

    async def sessions(request: Request) -> Response:
        try:
            limit, after = parse_page(request.query_params.get("limit"), request.query_params.get("cursor"))
        except ValueError:
            return _error(400, "invalid pagination")
        return JSONResponse(await asyncio.to_thread(reader.sessions, limit, after))

    async def trace(request: Request) -> Response:
        trace_id = request.path_params["trace_id"]
        result = await asyncio.to_thread(reader.trace, trace_id) if _TRACE_ID.match(trace_id) else None
        return JSONResponse(result) if result is not None else _error(404, "trace not found")

    async def health(request: Request) -> Response:
        return JSONResponse({
            "home_id": home_id,
            "schema_version": SCHEMA_VERSION,
            "protocol": "http/protobuf",
            "url": URL,
            "otlp_url": OTLP_URL,
            "ready": True,
        })

    async def asset(request: Request) -> Response:
        content, media_type = assets[request.url.path]
        return Response(content, media_type=media_type)

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        nonlocal accepting
        writer.start()
        try:
            yield
        finally:
            accepting = False
            await asyncio.to_thread(writer.close)
            await asyncio.to_thread(reader.close)

    routes = [
        Route("/v1/traces", ingest, methods=["POST"]),
        Route("/api/sessions", sessions, methods=["GET"]),
        Route("/api/traces/{trace_id}", trace, methods=["GET"]),
        Route("/health", health, methods=["GET"]),
        *(Route(path, asset, methods=["GET"]) for path in _ASSETS),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(_Guard)], lifespan=lifespan)
    app.state.writer = writer
    return app
