"""The receiver's one HTTP app: OTLP ingest and health. Reads go through ``zeta serve``."""

from __future__ import annotations

import asyncio
import contextlib
import math
import queue
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import uvicorn
from starlette.applications import Starlette
from starlette.datastructures import Headers
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from .process import AUTHORITY, OTLP_URL, URL
from .projector import MAX_BODY_BYTES, Reject, project_request
from .store import SCHEMA_VERSION, Writer

_SECURITY_HEADERS = [
    (b"content-security-policy", (
        b"default-src 'none'; frame-ancestors 'none'"
    )),
    (b"x-content-type-options", b"nosniff"),
    (b"x-frame-options", b"DENY"),
    (b"referrer-policy", b"no-referrer"),
    (b"cross-origin-resource-policy", b"same-origin"),
    (b"cache-control", b"no-store"),
]


class IdleTimer:
    """Time since the last ingest or ``/health`` request; ``seconds=math.inf`` never expires."""

    def __init__(self, seconds: float, clock: Callable[[], float] = time.monotonic) -> None:
        self.seconds, self._clock = seconds, clock
        self._last = clock()

    def touch(self) -> None:
        self._last = self._clock()

    def expired(self) -> bool:
        return self._clock() - self._last >= self.seconds


class IdleExitServer(uvicorn.Server):
    """A uvicorn server that shuts down gracefully, like on SIGTERM, once ``idle`` expires."""

    def __init__(self, config: uvicorn.Config, idle: IdleTimer) -> None:
        super().__init__(config)
        self._idle = idle

    async def on_tick(self, counter: int) -> bool:
        return await super().on_tick(counter) or self._idle.expired()


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


class _Guard:
    """Refuses any request not addressed to the exact loopback authority, and adds security headers.

    The exact ``Host`` check stops DNS-rebinding pages; a present ``Origin`` must be the receiver's own.
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


def create_app(directory: Path, home_id: str, idle: IdleTimer | None = None) -> Starlette:
    """Build the app; during its lifespan the writer thread migrates ``directory/telemetry.sqlite3`` and catches up.

    Each ingest and ``/health`` request touches ``idle``.
    """
    idle = idle or IdleTimer(math.inf)
    writer = Writer(directory / "telemetry.sqlite3")
    accepting = True
    dropped_foreign_home = 0  # spans from another zeta home's exporter; only the count is kept

    async def ingest(request: Request) -> Response:
        nonlocal dropped_foreign_home
        idle.touch()
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
            rows, foreign = await asyncio.to_thread(project_request, bytes(body), home_id)
        except Reject as exc:
            return _error(400, str(exc))
        dropped_foreign_home += foreign
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

        response = ExportTraceServiceResponse()
        if foreign:
            response.partial_success.rejected_spans = foreign
            response.partial_success.error_message = "foreign_home"
        return Response(response.SerializeToString(), media_type="application/x-protobuf")

    async def health(request: Request) -> Response:
        idle.touch()
        return JSONResponse({
            "home_id": home_id,
            "schema_version": SCHEMA_VERSION,
            "protocol": "http/protobuf",
            "otlp_url": OTLP_URL,
            "ready": True,
            "dropped": {"foreign_home": dropped_foreign_home},
        })

    @contextlib.asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        nonlocal accepting
        await asyncio.to_thread(writer.start)  # migration and catch-up block; ingest is accepted only after them
        try:
            yield
        finally:
            accepting = False
            await asyncio.to_thread(writer.close)

    routes = [
        Route("/v1/traces", ingest, methods=["POST"]),
        Route("/health", health, methods=["GET"]),
    ]
    app = Starlette(routes=routes, middleware=[Middleware(_Guard)], lifespan=lifespan)
    app.state.writer = writer
    return app
