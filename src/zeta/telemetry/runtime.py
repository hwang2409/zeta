"""Enabled telemetry implementation. imports from this module stay lazy."""

from __future__ import annotations

import ipaddress
import logging
import os
import threading
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from urllib.parse import urlsplit

from . import SpanHandle, TelemetryError, TelemetryStatus
from .attributes import RESERVED_CONTENT_ATTRIBUTES

_RESERVED_PREFIXES = tuple(RESERVED_CONTENT_ATTRIBUTES)
_OPERATION_ATTRIBUTE_KEYS = {
    "agent": frozenset({"gen_ai.operation.name", "zeta.agent.name", "gen_ai.conversation.id", "gen_ai.request.model", "zeta.agent.depth", "zeta.background"}),
    "turn": frozenset({"gen_ai.conversation.id", "zeta.turn.number", "zeta.notification", "zeta.tool.count"}),
    "inference": frozenset({"gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model", "gen_ai.conversation.id", "zeta.stream"}),
    "tool": frozenset({"gen_ai.operation.name", "zeta.tool.name", "zeta.tool.type", "zeta.tool.call.id", "zeta.agent.name", "gen_ai.conversation.id"}),
    "context": frozenset({"zeta.emergency", "zeta.tokens.before", "zeta.tokens.after", "zeta.marker.count"}),
    "serve": frozenset({"rpc.system", "rpc.method"}),
}
_ALLOWED_ATTRIBUTE_KEYS = frozenset().union(*_OPERATION_ATTRIBUTE_KEYS.values(), {
    "error.type", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.usage.total_tokens",
    "gen_ai.response.model", "gen_ai.response.finish_reasons", "zeta.tool.call.arguments.size", "zeta.tool.call.result.size",
})
_IDENTIFIER = __import__("re").compile(r"[A-Za-z0-9_.:/-]{1,128}\Z")
_MODEL = __import__("re").compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}\Z")
_PROVIDERS = frozenset({"claude", "codex", "ollama"})


def _safe_string(value: str, field: str | None = None) -> str:
    valid = _MODEL if field in {"gen_ai.request.model", "gen_ai.response.model"} else _IDENTIFIER
    if field == "gen_ai.provider.name" and value not in _PROVIDERS:
        return "_invalid"
    if value.startswith(("sk-", "ghp_", "Bearer ")) or not valid.fullmatch(value) or any(name in value for name in _RESERVED_PREFIXES):
        return "_invalid"
    return value


def _safe_attributes(attributes: Mapping[str, object] | None, allowed_keys: frozenset[str]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in (attributes or {}).items():
        if key not in allowed_keys or key not in _ALLOWED_ATTRIBUTE_KEYS:
            continue
        if isinstance(value, str):
            result[key] = _safe_string(value, key)
        elif isinstance(value, (int, bool)):
            result[key] = value
    return result


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


class _SpanHandle(SpanHandle):
    def __init__(self, scope: object) -> None:
        super().__init__(self)
        self._scope = scope
        self._span: object | None = None

    def enter(self) -> None:
        self._span = self._scope.__enter__()

    def exit(self, exc_type: object, exc: object, traceback: object) -> bool:
        if exc_type is not None and isinstance(exc, BaseException):
            self.fail(exc)
        self._scope.__exit__(None, None, None)
        return False

    def _set(self, name: str, value: object) -> None:
        if isinstance(value, str):
            value = _safe_string(value, name)
        self._span.set_attribute(name, value)

    def set_usage(self, *, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self._set("gen_ai.usage.input_tokens", input_tokens)
        self._set("gen_ai.usage.output_tokens", output_tokens)
        self._set("gen_ai.usage.total_tokens", total_tokens)

    def set_response(self, *, model: str, finish_reason: str) -> None:
        self._set("gen_ai.response.model", model)
        self._set("gen_ai.response.finish_reasons", finish_reason)

    def set_sizes(self, *, argument_bytes: int, result_bytes: int) -> None:
        self._set("zeta.tool.call.arguments.size", argument_bytes)
        self._set("zeta.tool.call.result.size", result_bytes)

    def retry(self, *, attempt: int, reason: str, delay: int, after: bool) -> None:
        self._span.add_event("gen_ai.client.retry", attributes={
            "gen_ai.retry.attempt": attempt, "gen_ai.retry.reason": _safe_string(reason, "gen_ai.retry.reason"),
            "gen_ai.retry.delay": delay, "gen_ai.retry.after": after,
        })

    def approval_requested(self, *, decision: str, scope: str) -> None:
        self._span.add_event("zeta.approval.requested", attributes={"decision": _safe_string(decision), "scope": _safe_string(scope)})

    def approval_resolved(self, *, decision: str, scope: str, duration: int) -> None:
        self._span.add_event("zeta.approval.resolved", attributes={"decision": _safe_string(decision), "scope": _safe_string(scope), "duration": duration})

    def fail(self, exc: BaseException | type[BaseException]) -> None:
        error_type = exc.__qualname__ if isinstance(exc, type) else type(exc).__qualname__
        self._set("error.type", _safe_string(error_type))
        from opentelemetry.trace import Status, StatusCode

        self._span.set_status(Status(StatusCode.ERROR))


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

    def _span(self, name: str, *, kind: str, attributes: Mapping[str, object] | None, operation: str) -> object:
        from opentelemetry.trace import SpanKind

        span_kind = {"internal": SpanKind.INTERNAL, "client": SpanKind.CLIENT, "server": SpanKind.SERVER}[kind]
        if " " in name:
            prefix, suffix = name.split(" ", 1)
            name = f"{prefix} {_safe_string(suffix)}"
        safe_attributes = _safe_attributes(attributes, _OPERATION_ATTRIBUTE_KEYS[operation])
        return _SpanHandle(self._tracer.start_as_current_span(name, kind=span_kind, attributes=safe_attributes, record_exception=False, set_status_on_exception=False))

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
