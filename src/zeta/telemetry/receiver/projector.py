"""The one trust seam: OTLP protobuf in, fixed allowlisted span rows out.

Nothing outside the rows this module returns is retained. Adding a retained key means
changing ``_SPAN_COLUMNS`` or ``_EVENT_COLUMNS`` here and its privacy test.

The resource attribute ``zeta.home.id`` is read only to keep this home's spans: a resource from
another home (or with no home id) is dropped and counted, never stored.
"""

from __future__ import annotations

import importlib.metadata
from typing import Any

from ..runtime import (
    _ALLOWED_ATTRIBUTE_KEYS,
    _EVENT_ATTRIBUTE_KEYS,
    _IDENTIFIER,
    _MCP_METHODS,
    _MODEL,
    _SERVE_METHODS,
    _SESSION_ID,
    _TOOL_NAME,
    _field_value,
)

MAX_BODY_BYTES = 8 * 1024 * 1024
MAX_SPANS = 10_000
MAX_EVENTS_PER_SPAN = 128
MAX_ATTRIBUTES_PER_OBJECT = 128
MAX_STRING_BYTES = 256
_MAX_NS = 2**63 - 1
_INVALID = "_invalid"

# OTLP attribute key -> spans column. Allowlisted keys without a column are dropped.
_SPAN_COLUMNS = {
    "error.type": "error_type",
    "gen_ai.conversation.id": "session_id",
    "zeta.agent.name": "agent_name",
    "zeta.agent.depth": "agent_depth",
    "zeta.background": "background",
    "zeta.turn.number": "turn_number",
    "zeta.notification": "notification",
    "gen_ai.provider.name": "provider",
    "gen_ai.request.model": "request_model",
    "gen_ai.response.model": "response_model",
    "zeta.stream": "stream",
    "gen_ai.usage.input_tokens": "input_tokens",
    "gen_ai.usage.output_tokens": "output_tokens",
    "gen_ai.usage.total_tokens": "total_tokens",
    "gen_ai.response.finish_reasons": "finish_reason",
    "zeta.tool.name": "tool_name",
    "zeta.tool.type": "tool_type",
    "zeta.tool.call.id": "tool_call_id",
    "zeta.tool.call.arguments.size": "tool_argument_bytes",
    "zeta.tool.call.result.size": "tool_result_bytes",
    "zeta.tool.count": "tool_count",
    "zeta.emergency": "emergency",
    "zeta.tokens.before": "tokens_before",
    "zeta.tokens.after": "tokens_after",
    "rpc.method": "rpc_method",
    "mcp.method.name": "mcp_method",
    "zeta.mcp.server.name": "mcp_server",
    "gen_ai.tool.name": "mcp_tool",
    "gen_ai.prompt.name": "mcp_prompt",
    "mcp.protocol.version": "mcp_protocol",
    "jsonrpc.request.id": "request_id",
}
_EVENT_COLUMNS = {
    "gen_ai.retry.attempt": "retry_attempt",
    "gen_ai.retry.reason": "retry_reason",
    "gen_ai.retry.delay": "retry_delay",
    "gen_ai.retry.after": "retry_after",
    "decision": "decision",
    "scope": "scope",
    "duration": "duration_ns",
}
SPAN_COLUMNS = (
    "trace_id", "span_id", "parent_span_id", "operation", "kind", "start_ns", "end_ns",
    "duration_ns", "status", "service_version", "scope_version", *_SPAN_COLUMNS.values(),
)
EVENT_COLUMNS = ("trace_id", "span_id", "event_index", "name", "time_ns", *_EVENT_COLUMNS.values())
_STATUSES = {0: "unset", 1: "ok", 2: "error"}

# "<prefix> <suffix>" span names: operation, suffix rule, and the attribute the suffix must equal.
_NAMED_OPERATIONS = {
    "invoke_agent": ("invoke_agent", _IDENTIFIER, "zeta.agent.name"),
    "chat": ("chat", _MODEL, "gen_ai.request.model"),
    "execute_tool": ("execute_tool", _TOOL_NAME, "zeta.tool.name"),
    "serve": ("serve", _SERVE_METHODS, "rpc.method"),
}
_FIXED_OPERATIONS = {"zeta.turn": "turn", "zeta.context.compact": "context_compact"}


class Reject(ValueError):
    """The whole OTLP request is refused; nothing from it is written."""


HOME_ID_ATTRIBUTE = "zeta.home.id"


def project_request(body: bytes, home_id: str) -> tuple[list[dict[str, Any]], int]:
    """Decode one ExportTraceServiceRequest into this home's projected span rows, or raise ``Reject``.

    Returns the rows and the number of spans dropped because their resource is not ``home_id``'s.
    """
    from google.protobuf.message import DecodeError
    from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import (
        ExportTraceServiceRequest,
    )

    request = ExportTraceServiceRequest()
    try:
        request.ParseFromString(body)
    except (DecodeError, RecursionError, ValueError):
        raise Reject("malformed protobuf") from None
    if sum(len(scope.spans) for resource in request.resource_spans for scope in resource.scope_spans) > MAX_SPANS:
        raise Reject("too many spans")
    zeta_version = importlib.metadata.version("zeta")
    rows: list[dict[str, Any]] = []
    foreign = 0
    for resource_spans in request.resource_spans:
        resource = _attributes(resource_spans.resource.attributes)
        if resource.get(HOME_ID_ATTRIBUTE) != home_id:
            foreign += sum(len(scope.spans) for scope in resource_spans.scope_spans)
            continue
        service_version = _version(resource.get("service.version"), zeta_version)
        for scope_spans in resource_spans.scope_spans:
            scope = scope_spans.scope
            _attributes(scope.attributes)  # enforces the per-object cap; scope attributes are dropped
            if scope.name != "zeta":
                raise Reject("unknown instrumentation scope")
            scope_version = _version(scope.version, zeta_version)
            for span in scope_spans.spans:
                rows.append(_project_span(span, service_version, scope_version))
    return rows, foreign


def _version(value: object, zeta_version: str) -> str:
    return zeta_version if value == zeta_version else _INVALID


def _attributes(items: Any) -> dict[str, object]:
    """Scalar view of OTLP attributes; non-scalar values become ``None`` and later ``_invalid``."""
    if len(items) > MAX_ATTRIBUTES_PER_OBJECT:
        raise Reject("too many attributes")
    result: dict[str, object] = {}
    for item in items:
        kind = item.value.WhichOneof("value")
        result[item.key] = getattr(item.value, kind) if kind in {"string_value", "bool_value", "int_value"} else None
    return result


def _retained(attributes: dict[str, object], allowed: frozenset[str] | set[str]) -> dict[str, object]:
    retained = {}
    for key, value in attributes.items():
        if key not in allowed:
            continue
        if isinstance(value, str) and len(value.encode()) > MAX_STRING_BYTES:
            raise Reject("attribute value too long")
        retained[key] = _field_value(key, value)
    return retained


def _span_id(value: bytes, size: int) -> bytes:
    if len(value) != size or not any(value):
        raise Reject("invalid id")
    return bytes(value)


def _timestamp(value: int) -> int:
    if not 0 <= value <= _MAX_NS:
        raise Reject("invalid timestamp")
    return value


def _operation(name: str, attributes: dict[str, object]) -> str:
    if name in _FIXED_OPERATIONS:
        return _FIXED_OPERATIONS[name]
    prefix, _, suffix = name.partition(" ")
    if prefix in _NAMED_OPERATIONS and suffix:
        operation, rule, key = _NAMED_OPERATIONS[prefix]
        _check_suffix(suffix, rule, attributes.get(key))
        return operation
    if prefix in _MCP_METHODS or (prefix == _INVALID and not suffix):
        _check_suffix(prefix, _MCP_METHODS, attributes.get("mcp.method.name"))
        if suffix:
            target = attributes.get("gen_ai.prompt.name" if prefix == "prompts/get" else "gen_ai.tool.name")
            _check_suffix(suffix, _IDENTIFIER, target)
        return "mcp"
    raise Reject("unknown operation")


def _check_suffix(suffix: str, rule: Any, projected: object) -> None:
    valid = suffix == _INVALID or (suffix in rule if isinstance(rule, frozenset) else rule.fullmatch(suffix))
    if not valid or (projected is not None and projected != suffix):
        raise Reject("unknown operation")


def _project_span(span: Any, service_version: str, scope_version: str) -> dict[str, Any]:
    trace_id = _span_id(span.trace_id, 16)
    span_id = _span_id(span.span_id, 8)
    parent_span_id = _span_id(span.parent_span_id, 8) if span.parent_span_id else None
    start_ns = _timestamp(span.start_time_unix_nano)
    end_ns = _timestamp(span.end_time_unix_nano)
    if end_ns < start_ns:
        raise Reject("invalid timestamp")
    if span.status.code not in _STATUSES or span.kind not in range(6):
        raise Reject("invalid span status or kind")
    if len(span.events) > MAX_EVENTS_PER_SPAN:
        raise Reject("too many events")
    attributes = _retained(_attributes(span.attributes), _ALLOWED_ATTRIBUTE_KEYS)
    row: dict[str, Any] = dict.fromkeys(_SPAN_COLUMNS.values())
    for key, value in attributes.items():
        if key in _SPAN_COLUMNS:
            row[_SPAN_COLUMNS[key]] = int(value) if isinstance(value, bool) else value
    row.update(
        trace_id=trace_id,
        span_id=span_id,
        parent_span_id=parent_span_id,
        operation=_operation(span.name, attributes),
        kind=span.kind,
        start_ns=start_ns,
        end_ns=end_ns,
        duration_ns=end_ns - start_ns,
        status=_STATUSES[span.status.code],
        service_version=service_version,
        scope_version=scope_version,
    )
    row["events"] = _project_events(span.events, trace_id, span_id)
    return row


def _project_events(span_events: Any, trace_id: bytes, span_id: bytes) -> list[dict[str, Any]]:
    events = []
    for index, event in enumerate(span_events):
        event_attributes = _attributes(event.attributes)
        time_ns = _timestamp(event.time_unix_nano)
        if event.name not in _EVENT_ATTRIBUTE_KEYS:
            continue  # unknown event names: drop the event, keep the span
        projected = dict.fromkeys(_EVENT_COLUMNS.values())
        for key, value in _retained(event_attributes, _EVENT_ATTRIBUTE_KEYS[event.name]).items():
            projected[_EVENT_COLUMNS[key]] = int(value) if isinstance(value, bool) else value
        projected.update(trace_id=trace_id, span_id=span_id, event_index=index, name=event.name, time_ns=time_ns)
        events.append(projected)
    return events


def root_session_id(session_id: str | None) -> str | None:
    """The 32-hex root of a valid conversation id, without its ``:<n>`` child segments."""
    if session_id is None or not _SESSION_ID.fullmatch(session_id):
        return None
    return session_id.split(":", 1)[0]
