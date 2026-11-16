"""Enabled telemetry implementation. imports from this module stay lazy."""

from __future__ import annotations

import ipaddress
import logging
import os
import threading
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from typing import Self
from urllib.parse import urlsplit

from . import TelemetryError, TelemetryStatus
from .attributes import RESERVED_CONTENT_ATTRIBUTES


# This is the single owner of the loopback endpoint policy.
def validate_loopback_endpoint(endpoint: str) -> str:
    try:
        parsed = urlsplit(endpoint)
        host = parsed.hostname
        address = ipaddress.ip_address(host) if host else None
        port = parsed.port
        if port is not None and not 0 <= port <= 65535:
            raise ValueError
        if (
            parsed.scheme not in {"http", "https"}
            or address is None
            or not address.is_loopback
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
    except (TypeError, ValueError):
        raise TelemetryError("otel.endpoint must be an http(s) URL with a literal loopback host") from None
    return endpoint


class StatusCell:
    """A one-slot handoff from the exporter thread to the event-loop owner."""

    def __init__(self) -> None:
        self._values: deque[str] = deque(maxlen=1)
        self._lock = threading.Lock()

    def publish(self, state: str) -> None:
        with self._lock:
            self._values.append(state)

    def read(self) -> str | None:
        with self._lock:
            return self._values.pop() if self._values else None


class _SpanHandle:
    def __init__(self, scope: object) -> None:
        self._scope = scope
        self._span: object | None = None

    def __enter__(self) -> Self:
        self._span = self._scope.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        self._scope.__exit__(*args)

    def set_attribute(self, name: str, value: object) -> None:
        if name not in RESERVED_CONTENT_ATTRIBUTES:
            self._span.set_attribute(name, value)

    def add_event(self, name: str, attributes: Mapping[str, object] | None = None) -> None:
        safe = {key: value for key, value in (attributes or {}).items() if key not in RESERVED_CONTENT_ATTRIBUTES}
        self._span.add_event(name, attributes=safe)

    def record_exception(self, exception: BaseException) -> None:
        self.set_attribute("error.type", type(exception).__name__)

    def set_error(self, error_type: str | None = None) -> None:
        if error_type is not None:
            self.set_attribute("error.type", error_type)


class _Exporter:
    def __init__(self, delegate: object, cell: StatusCell, session: object) -> None:
        self._delegate = delegate
        self._cell = cell
        self._session = session

    def export(self, spans: object) -> object:
        from opentelemetry.sdk.trace.export import SpanExportResult

        try:
            result = self._delegate.export(spans)
        except Exception:  # noqa: BLE001 - exporter failures must never escape the SDK worker
            self._cell.publish("export_unavailable")
            return SpanExportResult.FAILURE
        if result != SpanExportResult.SUCCESS:
            self._cell.publish("export_unavailable")
        else:
            self._cell.publish("healthy")
        return result

    def shutdown(self) -> None:
        self._delegate.shutdown()
        self._session.close()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return self._delegate.force_flush(timeout_millis)


class _Runtime:
    def __init__(self, provider: object, tracer: object, propagator: object, cell: StatusCell, log_state: list[tuple[logging.Logger, bool, bool, int, list[logging.Handler]]]) -> None:
        self._provider = provider
        self._tracer = tracer
        self._propagator = propagator
        self._cell = cell
        self._log_state = log_state
        self._status = TelemetryStatus(True, "healthy")
        self._closed = False
        self._lock = threading.Lock()

    def status(self) -> TelemetryStatus:
        state = self._cell.read()
        if state is not None:
            self._status = TelemetryStatus(True, "degraded" if state == "export_unavailable" else "healthy", "export_unavailable" if state == "export_unavailable" else None)
        return self._status

    def shutdown(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        try:
            self._provider.shutdown()
        finally:
            for logger, disabled, propagate, level, handlers in self._log_state:
                logger.disabled = disabled
                logger.propagate = propagate
                logger.setLevel(level)
                logger.handlers[:] = handlers

    def span(self, name: str, *, kind: str, attributes: Mapping[str, object] | None) -> object:
        from opentelemetry.trace import SpanKind

        span_kind = {"internal": SpanKind.INTERNAL, "client": SpanKind.CLIENT, "server": SpanKind.SERVER}[kind]
        safe_attributes = {key: value for key, value in (attributes or {}).items() if key not in RESERVED_CONTENT_ATTRIBUTES}
        return _SpanHandle(self._tracer.start_as_current_span(name, kind=span_kind, attributes=safe_attributes))

    def current_trace_context(self) -> dict[str, str]:
        carrier: dict[str, str] = {}
        self.inject(carrier)
        return carrier

    def inject(self, carrier: dict[str, str]) -> None:
        self._propagator.inject(carrier)

    def extract(self, carrier: Mapping[str, str]) -> dict[str, str]:
        context = self._propagator.extract(carrier)
        result: dict[str, str] = {}
        self._propagator.inject(result, context=context)
        return result

    @contextmanager
    def use_trace_context(self, carrier: Mapping[str, str]):
        from opentelemetry import context

        extracted = self._propagator.extract(carrier)
        token = context.attach(extracted)
        try:
            yield
        finally:
            context.detach(token)

    def retry(self, **attributes: object) -> None:
        from opentelemetry import trace

        span = trace.get_current_span()
        span.add_event("gen_ai.client.retry", attributes=attributes)


def configure(endpoint: str) -> _Runtime:
    endpoint = validate_loopback_endpoint(endpoint)
    for name, value in os.environ.items():
        if name.startswith(("OTEL_EXPORTER_OTLP", "OTEL_PYTHON_EXPORTER_OTLP")) and value:
            raise TelemetryError("OTEL exporter environment is not allowed; configure zeta telemetry explicitly")
    if os.environ.get("OTEL_SDK_DISABLED", "").strip().lower() == "true":
        raise TelemetryError("OTEL_SDK_DISABLED cannot disable configured zeta telemetry; remove it or disable [otel]")
    try:
        import requests
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import SpanLimits, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON, ParentBased
        from opentelemetry.trace.propagation.tracecontext import (
            TraceContextTextMapPropagator,
        )
    except ModuleNotFoundError:
        raise TelemetryError("OpenTelemetry support is not installed; install zeta[otel]") from None

    session = requests.Session()
    session.trust_env = False
    cell = StatusCell()
    delegate = OTLPSpanExporter(endpoint=endpoint, headers={}, timeout=5, session=session)
    exporter = _Exporter(delegate, cell, session)
    limits = SpanLimits(
        max_attributes=128,
        max_events=128,
        max_links=128,
        max_span_attributes=128,
        max_event_attributes=128,
        max_link_attributes=128,
        max_attribute_length=4096,
        max_span_attribute_length=4096,
    )
    from importlib.metadata import version

    provider = TracerProvider(
        resource=Resource({"service.name": "zeta", "service.version": version("zeta")}),
        sampler=ParentBased(ALWAYS_ON),
        span_limits=limits,
    )
    provider.add_span_processor(BatchSpanProcessor(exporter, max_queue_size=2048, max_export_batch_size=256, schedule_delay_millis=5000))
    logger_state: list[tuple[logging.Logger, bool, bool, int, list[logging.Handler]]] = []
    for name in ("opentelemetry.exporter", "opentelemetry.sdk._shared_internal"):
        logger = logging.getLogger(name)
        logger_state.append((logger, logger.disabled, logger.propagate, logger.level, list(logger.handlers)))
        logger.disabled = True
        logger.propagate = False
    return _Runtime(provider, provider.get_tracer("zeta"), TraceContextTextMapPropagator(), cell, logger_state)
