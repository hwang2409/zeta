"""Authenticated loopback HTTP ingress for durable automation deliveries."""

from __future__ import annotations

import hashlib
import hmac
import ipaddress
import threading
import time
from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import urlsplit

from .store import SQLiteStore
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
        try:
            stamp_value = float(stamp)
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
    ) -> None:
        if not _is_loopback(host) and not allow_non_loopback:
            raise ValueError(
                "webhook server refuses non-loopback bind without explicit configuration"
            )
        if not 0 <= port <= 65535 or max_request_bytes <= 0:
            raise ValueError("invalid webhook listener configuration")
        if rate_limit[0] <= 0 or rate_limit[1] <= 0:
            raise ValueError("webhook rate limit must be positive")
        self.store = store
        self.host = host
        self.port = port
        self.max_request_bytes = max_request_bytes
        self.rate_count, self.rate_window = rate_limit
        self.now = now
        self.wake = wake or (lambda: None)
        self.after_record = after_record
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

            def _respond(self, status: int) -> None:
                self.send_response(status)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()

            def _route(self) -> tuple[Any, Any] | None:
                path = urlsplit(self.path).path
                parts = path.split("/")
                if len(parts) != 3 or parts[:2] != ["", "hooks"] or not parts[2]:
                    return None
                return receiver.store.resolve_webhook_token(parts[2])

            def do_GET(self) -> None:
                self._respond(405)

            def do_PUT(self) -> None:
                self._respond(405)

            def do_DELETE(self) -> None:
                self._respond(405)

            def do_POST(self) -> None:
                route = self._route()
                if route is None:
                    self._respond(404)
                    return
                state, credentials = route
                length_text = self.headers.get("Content-Length")
                try:
                    length = int(length_text) if length_text is not None else -1
                except ValueError:
                    length = -1
                if length < 0 or length > receiver.max_request_bytes:
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
                allowed_headers = {
                    key.lower(): value
                    for key, value in self.headers.items()
                    if key.lower() in _ALLOWED_PAYLOAD_HEADERS
                }
                inserted = receiver.store.accept_webhook(
                    state.job.name,
                    state.revision,
                    body,
                    allowed_headers,
                    receiver.now(),
                    delivery_id=delivery_id,
                )
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
        self._httpd = _HTTPServer((self.host, self.port), self._handler())
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
            self._httpd.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2)
