"""Authenticated loopback HTTP ingress for durable automation deliveries."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import socket
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .store import PendingWebhookLimitError, SQLiteStore
from .trigger import Webhook

DEFAULT_WEBHOOK_PORT = 8765
MAX_REQUEST_BYTES = 1_048_576
MAX_PROMPT_PAYLOAD_BYTES = 65_536
_TIMESTAMP_TOLERANCE_SECONDS = 300
_ALLOWED_PAYLOAD_HEADERS = {
    "content-type",
    "user-agent",
    "x-github-delivery",
    "x-github-event",
}


def _header(headers: Mapping[str, str], name: str) -> str | None:
    lowered = name.lower()
    return next((value for key, value in headers.items() if key.lower() == lowered), None)


def signature(secret: bytes, body: bytes, *, timestamp: str | None = None) -> str:
    message = body if timestamp is None else timestamp.encode("utf-8") + b"." + body
    return hmac.new(secret, message, hashlib.sha256).hexdigest()


def verify_request(
    trigger: Webhook,
    secret: bytes,
    body: bytes,
    headers: Mapping[str, str],
    *,
    now: float | None = None,
) -> bool:
    """Verify the configured signature before callers interpret request bytes."""
    supplied = _header(headers, trigger.signature_header)
    if supplied is None or not supplied.startswith(trigger.signature_prefix):
        return False
    stamp = None
    if trigger.timestamp_header is not None:
        stamp = _header(headers, trigger.timestamp_header)
        if stamp is None:
            return False
        if (
            not stamp
            or stamp != stamp.strip()
            or not (stamp.isascii() and stamp.removeprefix("-").isdigit())
        ):
            return False
        try:
            stamp_value = int(stamp)
        except ValueError:
            return False
        current = time.time() if now is None else now
        if abs(current - stamp_value) > _TIMESTAMP_TOLERANCE_SECONDS:
            return False
    expected = trigger.signature_prefix + signature(secret, body, timestamp=stamp)
    return hmac.compare_digest(expected, supplied)


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[BaseHTTPRequestHandler],
        *,
        request_timeout: float,
        max_handlers: int,
    ) -> None:
        self.request_timeout = request_timeout
        self._handler_slots = threading.BoundedSemaphore(max_handlers)
        self._active_handlers = 0
        self._handler_sockets: set[socket.socket] = set()
        self._handler_condition = threading.Condition()
        super().__init__(address, handler)

    def get_request(self):
        request, address = super().get_request()
        request.settimeout(self.request_timeout)
        return request, address

    def process_request(self, request, client_address) -> None:
        if not self._handler_slots.acquire(blocking=False):
            try:
                request.sendall(
                    b"HTTP/1.1 503 Service Unavailable\r\n"
                    b"Content-Length: 0\r\nConnection: close\r\n\r\n"
                )
            finally:
                self.shutdown_request(request)
            return
        with self._handler_condition:
            self._active_handlers += 1
            self._handler_sockets.add(request)
        try:
            super().process_request(request, client_address)
        except BaseException:
            self._handler_finished(request)
            raise

    def process_request_thread(self, request, client_address) -> None:
        try:
            super().process_request_thread(request, client_address)
        finally:
            self._handler_finished(request)

    def _handler_finished(self, request: socket.socket) -> None:
        self._handler_slots.release()
        with self._handler_condition:
            self._handler_sockets.discard(request)
            self._active_handlers -= 1
            self._handler_condition.notify_all()

    def shutdown_handler_sockets(self) -> None:
        with self._handler_condition:
            requests = tuple(self._handler_sockets)
        for request in requests:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

    def drain_handlers(self) -> None:
        with self._handler_condition:
            while self._active_handlers:
                self._handler_condition.wait()


class WebhookServer:
    """A dynamically-routed receiver backed by the daemon's automation store."""

    def __init__(
        self,
        store: SQLiteStore,
        *,
        host: str = "127.0.0.1",
        port: int = DEFAULT_WEBHOOK_PORT,
        allow_non_loopback: bool = False,
        max_request_bytes: int = MAX_REQUEST_BYTES,
        rate_limit: tuple[int, float] = (60, 60.0),
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        wake: Callable[[], None] | None = None,
        after_record: Callable[[], None] | None = None,
        request_timeout: float = 10.0,
        max_handlers: int = 16,
        shutdown_timeout: float = 10.0,
    ) -> None:
        if not _is_loopback(host) and not allow_non_loopback:
            raise ValueError(
                "webhook server refuses non-loopback bind without explicit configuration"
            )
        if not 0 <= port <= 65535 or max_request_bytes <= 0:
            raise ValueError("invalid webhook listener configuration")
        if rate_limit[0] <= 0 or rate_limit[1] <= 0:
            raise ValueError("webhook rate limit must be positive")
        if request_timeout <= 0 or max_handlers <= 0 or shutdown_timeout <= 0:
            raise ValueError("webhook resource limits must be positive")
        self.store = store
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.rate_count, self.rate_window = rate_limit
        self.now = now
        self.wake = wake or (lambda: None)
        self.after_record = after_record
        self.request_timeout = request_timeout
        self.max_handlers = max_handlers
        self.shutdown_timeout = shutdown_timeout
        self._requests: dict[str, deque[float]] = defaultdict(deque)
        self._rate_lock = threading.Lock()
        self._httpd: _HTTPServer | None = None
        self._thread: threading.Thread | None = None
        self.closed = False

    @property
    def address(self) -> tuple[str, int]:
        if self._httpd is None:
            return self.host, self.port
        host, port = self._httpd.server_address[:2]
        return str(host), int(port)

    @property
    def url(self) -> str:
        host, port = self.address
        bracketed = f"[{host}]" if ":" in host else host
        return f"http://{bracketed}:{port}"

    def _rate_limited(self, name: str) -> bool:
        current = time.monotonic()
        with self._rate_lock:
            requests = self._requests[name]
            while requests and requests[0] <= current - self.rate_window:
                requests.popleft()
            if len(requests) >= self.rate_count:
                return True
            requests.append(current)
            return False

    def _handler(self) -> type[BaseHTTPRequestHandler]:
        receiver = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, _format: str, *args: Any) -> None:
                return

            def _respond(self, status: int, *, allow: bool = False) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                if allow:
                    self.send_header("Allow", "POST")
                self.end_headers()

            def _method_not_allowed(self) -> None:
                self._respond(405, allow=True)

            def _route(self) -> tuple[Any, Any] | None:
                path = urlsplit(self.path).path
                parts = path.split("/")
                if len(parts) != 3 or parts[:2] != ["", "hooks"] or not parts[2]:
                    return None
                return receiver.store.resolve_webhook_token(parts[2])

            do_GET = _method_not_allowed
            do_HEAD = _method_not_allowed
            do_PUT = _method_not_allowed
            do_DELETE = _method_not_allowed
            do_PATCH = _method_not_allowed
            do_OPTIONS = _method_not_allowed
            do_TRACE = _method_not_allowed
            do_CONNECT = _method_not_allowed

            def __getattr__(self, name: str) -> Any:
                if name.startswith("do_"):
                    return self._method_not_allowed
                raise AttributeError(name)

            def do_POST(self) -> None:
                route = self._route()
                if route is None:
                    self._respond(404)
                    return
                state, credentials = route
                if self.headers.get_all("Transfer-Encoding"):
                    self._respond(400)
                    return
                lengths = self.headers.get_all("Content-Length", [])
                if not lengths:
                    self._respond(411)
                    return
                if len(lengths) != 1:
                    self._respond(400)
                    return
                try:
                    length = int(lengths[0])
                except ValueError:
                    self._respond(400)
                    return
                if length < 0:
                    self._respond(400)
                    return
                if length > receiver.max_request_bytes:
                    self._respond(413)
                    return
                body = self.rfile.read(length)
                if len(body) != length:
                    self._respond(400)
                    return
                trigger = state.job.trigger
                assert isinstance(trigger, Webhook)
                if not verify_request(
                    trigger,
                    credentials.secret,
                    body,
                    self.headers,
                    now=receiver.now().timestamp(),
                ):
                    self._respond(401)
                    return
                if receiver._rate_limited(state.job.name):
                    self._respond(429)
                    return
                delivery_id = (
                    _header(self.headers, trigger.delivery_header)
                    if trigger.delivery_header is not None
                    else None
                )
                if delivery_id is not None and (
                    not delivery_id or len(delivery_id.encode("utf-8")) > 256
                ):
                    self._respond(400)
                    return
                allowed_headers = {
                    key.lower(): value
                    for key, value in self.headers.items()
                    if key.lower() in _ALLOWED_PAYLOAD_HEADERS
                }
                try:
                    inserted = receiver.store.accept_webhook(
                        state.job.name,
                        state.revision,
                        body,
                        allowed_headers,
                        receiver.now(),
                        delivery_id=delivery_id,
                    )
                except PendingWebhookLimitError:
                    self._respond(429)
                    return
                if inserted:
                    receiver.wake()
                if receiver.after_record is not None:
                    receiver.after_record()
                self._respond(202)

        return Handler

    def start(self) -> tuple[str, int]:
        if self.closed:
            raise RuntimeError("webhook server is shut down")
        if self._httpd is not None:
            return self.address
        self._httpd = _HTTPServer(
            (self.host, self.port),
            self._handler(),
            request_timeout=self.request_timeout,
            max_handlers=self.max_handlers,
        )
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name="zeta-webhook",
            daemon=True,
        )
        self._thread.start()
        return self.address

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.shutdown_handler_sockets()
            self._httpd.drain_handlers()
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
