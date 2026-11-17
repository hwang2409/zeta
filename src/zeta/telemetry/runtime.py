"""Enabled telemetry implementation. imports from this module stay lazy."""

from __future__ import annotations

import ipaddress
import logging
import os
import re
import threading
from collections import deque
from collections.abc import Mapping
from contextlib import contextmanager
from urllib.parse import urlsplit

from . import SpanHandle, TelemetryError, TelemetryStatus
from .attributes import GEN_AI_PROVIDER_NAMES, RESERVED_CONTENT_ATTRIBUTES

_OPERATION_ATTRIBUTE_KEYS = {
    "agent": frozenset({"gen_ai.operation.name", "zeta.agent.name", "gen_ai.conversation.id", "gen_ai.request.model", "zeta.agent.depth", "zeta.background"}),
    "turn": frozenset({"gen_ai.conversation.id", "zeta.turn.number", "zeta.notification"}),
    "inference": frozenset({"gen_ai.operation.name", "gen_ai.provider.name", "gen_ai.request.model", "gen_ai.conversation.id", "zeta.stream"}),
    "tool": frozenset({"gen_ai.operation.name", "zeta.tool.name", "zeta.tool.type", "zeta.tool.call.id", "zeta.agent.name", "gen_ai.conversation.id"}),
    "context": frozenset({"zeta.emergency", "zeta.tokens.before", "zeta.tokens.after"}),
    "serve": frozenset({"rpc.system", "rpc.method"}),
}
_EVENT_ATTRIBUTE_KEYS = {"gen_ai.client.retry": frozenset({"gen_ai.retry.attempt", "gen_ai.retry.reason", "gen_ai.retry.delay", "gen_ai.retry.after"}), "zeta.approval.requested": frozenset({"decision", "scope"}), "zeta.approval.resolved": frozenset({"decision", "scope", "duration"}), "zeta.serve.invalid_propagation": frozenset()}
_ALLOWED_ATTRIBUTE_KEYS = frozenset().union(*_OPERATION_ATTRIBUTE_KEYS.values(), {"error.type", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.usage.total_tokens", "gen_ai.response.model", "gen_ai.response.finish_reasons", "zeta.tool.call.arguments.size", "zeta.tool.call.result.size", "zeta.tool.count"})
# Closed sets mirror repository values; tests/telemetry/test_telemetry.py checks each mirror against its source.
# provider names are the GenAI values; the facade maps every catalog provider onto them.
_PROVIDERS = frozenset(GEN_AI_PROVIDER_NAMES.values())
_OPERATION_NAMES = frozenset({"invoke_agent", "chat", "execute_tool"})
# zeta-owned tool families; no repository enum exists yet (slice 3a is the first caller).
_TOOL_TYPES = frozenset({"function", "computer", "mcp", "agent"})
# stop_reason literals set in providers/codex.py and providers/ollama.py, plus the Anthropic API's
# message_delta stop_reason values that providers/anthropic.py passes through unchanged.
_FINISH_REASONS = frozenset({"end_turn", "max_tokens", "tool_use", "content_filter", "refusal", "stop_sequence", "pause_turn", "model_context_window_exceeded"})
# RetryPlan.reason values from CompletionRetryBudget.plan() and retry_reason() in providers/completion.py.
_RETRY_REASONS = frozenset({
    "subscription_limit", "stream_error", "stream_stall", "stream_inconsistent", "overloaded_error",
    "rate_limit_error", "server_error", "http_error", "transport_error", "backend_error", "account_lock_timeout",
})
# core.approval.ApprovalDecision values plus server.protocol.ApprovalDecision values.
_APPROVAL_DECISIONS = frozenset({"allow", "deny", "ask", "approve"})
# server.protocol.APPROVAL_SCOPES.
_APPROVAL_SCOPES = frozenset({"once", "always_tool"})


def _serve_methods() -> frozenset[str]:
    # one source: the served method table, plus hello, which the connection routes before that table
    from ..server.protocol import MethodDescriptor

    return frozenset({"hello", *(descriptor.wire for descriptor in MethodDescriptor)})


_SERVE_METHODS = _serve_methods()
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")
# a session id, or a child agent instance id: the root session id plus ":<n>" per nesting level.
_SESSION_ID = re.compile(r"[0-9a-f]{32}(?::[1-9][0-9]{0,8}){0,5}\Z")
_TOOL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}(?:__[A-Za-z0-9][A-Za-z0-9_-]{0,127})?\Z")
_CALL_ID = re.compile(r"(?=.{1,128}\Z)(?:toolu_[A-Za-z0-9_-]+|call_[A-Za-z0-9_-]+|fc_[A-Za-z0-9_-]+|ollama-[0-9a-f-]+|inline-[0-9a-f-]+)\Z")
# catalog ids and ollama names: [host/][namespace/]model[:tag], e.g. locally-created-model, qwen3:4b, hf.co/user/model:Q4_K_M.
_MODEL = re.compile(r"(?=.{1,128}\Z)(?:[A-Za-z0-9][A-Za-z0-9_.-]*/){0,2}[A-Za-z0-9][A-Za-z0-9_.-]*(?::[A-Za-z0-9_.-]+)?\Z")
_PYTHON_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*\Z")
_STRING_RULES = {
    "gen_ai.operation.name": _OPERATION_NAMES, "gen_ai.provider.name": _PROVIDERS, "gen_ai.conversation.id": _SESSION_ID,
    "gen_ai.request.model": _MODEL, "gen_ai.response.model": _MODEL, "zeta.agent.name": _IDENTIFIER,
    "zeta.tool.name": _TOOL_NAME, "zeta.tool.type": _TOOL_TYPES, "zeta.tool.call.id": _CALL_ID,
    "gen_ai.response.finish_reasons": _FINISH_REASONS, "gen_ai.retry.reason": _RETRY_REASONS,
    "decision": _APPROVAL_DECISIONS, "scope": _APPROVAL_SCOPES, "error.type": _PYTHON_NAME,
    "rpc.system": frozenset({"jsonrpc"}), "rpc.method": _SERVE_METHODS,
}
_INT_KEYS = frozenset({"gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens", "gen_ai.usage.total_tokens", "zeta.tool.call.arguments.size", "zeta.tool.call.result.size", "zeta.agent.depth", "zeta.turn.number", "zeta.tool.count", "zeta.tokens.before", "zeta.tokens.after", "gen_ai.retry.attempt", "gen_ai.retry.delay", "duration"})
_BOOL_KEYS = frozenset({"zeta.background", "zeta.notification", "zeta.stream", "zeta.emergency", "gen_ai.retry.after"})


def _field_value(key: str, value: object) -> object:
    if key in _INT_KEYS:
        return value if type(value) is int and value >= 0 else 0
    if key in _BOOL_KEYS:
        return value if type(value) is bool else False
    rule = _STRING_RULES[key]
    if not isinstance(value, str) or any(name in value for name in RESERVED_CONTENT_ATTRIBUTES):
        return "_invalid"
    if isinstance(rule, frozenset):
        return value if value in rule else "_invalid"
    return value if rule.fullmatch(value) else "_invalid"


def _safe_attributes(attributes: Mapping[str, object] | None, allowed_keys: frozenset[str]) -> dict[str, object]:
    return {key: _field_value(key, value) if isinstance(value, (str, int, bool)) else "_invalid" for key, value in (attributes or {}).items() if key in allowed_keys and value is not None}


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
    except (AttributeError, TypeError, ValueError):
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
        if name not in _ALLOWED_ATTRIBUTE_KEYS:
            return
        self._span.set_attribute(name, _field_value(name, value))

    def set_usage(self, *, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        self._set("gen_ai.usage.input_tokens", input_tokens)
        self._set("gen_ai.usage.output_tokens", output_tokens)
        self._set("gen_ai.usage.total_tokens", total_tokens)

    def set_response(self, *, finish_reason: str | None = None, model: str | None = None) -> None:
        if model is not None:
            self._set("gen_ai.response.model", model)
        if finish_reason is not None:
            self._set("gen_ai.response.finish_reasons", finish_reason)

    def set_tool_count(self, *, tool_count: int) -> None:
        self._set("zeta.tool.count", tool_count)

    def set_sizes(self, *, argument_bytes: int, result_bytes: int) -> None:
        self._set("zeta.tool.call.arguments.size", argument_bytes)
        self._set("zeta.tool.call.result.size", result_bytes)

    def set_compaction(self, *, tokens_after: int) -> None:
        self._set("zeta.tokens.after", tokens_after)

    def retry(self, *, attempt: int, reason: str, delay: int, after: bool) -> None:
        self._span.add_event("gen_ai.client.retry", attributes=_safe_attributes({
            "gen_ai.retry.attempt": attempt, "gen_ai.retry.reason": reason,
            "gen_ai.retry.delay": delay, "gen_ai.retry.after": after,
        }, _EVENT_ATTRIBUTE_KEYS["gen_ai.client.retry"]))

    def approval_requested(self, *, decision: str, scope: str | None = None) -> None:
        self._span.add_event("zeta.approval.requested", attributes=_safe_attributes({"decision": decision, "scope": scope}, _EVENT_ATTRIBUTE_KEYS["zeta.approval.requested"]))

    def approval_resolved(self, *, decision: str | None, duration: int, scope: str | None = None) -> None:
        self._span.add_event("zeta.approval.resolved", attributes=_safe_attributes({"decision": decision, "scope": scope, "duration": duration}, _EVENT_ATTRIBUTE_KEYS["zeta.approval.resolved"]))

    def invalid_propagation(self) -> None:
        self._span.add_event("zeta.serve.invalid_propagation", attributes=_safe_attributes({}, _EVENT_ATTRIBUTE_KEYS["zeta.serve.invalid_propagation"]))

    def fail(self, exc: BaseException | type[BaseException] | None) -> None:
        # _OTHER is the semconv fallback when no exception class names the failure
        error_type = "_OTHER" if exc is None else exc.__qualname__ if isinstance(exc, type) else type(exc).__qualname__
        self._set("error.type", _field_value("error.type", error_type))
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
            key = {"tool": "zeta.tool.name", "inference": "gen_ai.request.model", "serve": "rpc.method"}.get(operation, "zeta.agent.name")
            name = f"{prefix} {_field_value(key, suffix)}"
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
    # the trace-context loggers print received tracestate members, which are client input
    for name in ("opentelemetry.exporter", "opentelemetry.sdk._shared_internal", "opentelemetry.trace.span", "opentelemetry.trace.propagation.tracecontext"):
        logger = logging.getLogger(name)
        logger_state.append((logger, logger.disabled, logger.propagate, logger.level, list(logger.handlers)))
        logger.disabled = True
        logger.propagate = False
    return _Runtime(provider, provider.get_tracer("zeta"), TraceContextTextMapPropagator(), cell, logger_state)
