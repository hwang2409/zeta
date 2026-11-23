"""Optional OpenTelemetry facade for zeta."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Self

from .attributes import GEN_AI_PROVIDER_NAMES


@dataclass(frozen=True)
class TelemetryStatus:
    enabled: bool
    state: str
    reason: str | None = None
    last_export_ns: int | None = None  # when the latest export batch finished; None before the first
    over_cap: bool = False

    @property
    def last_export(self) -> str | None:
        if self.last_export_ns is None:
            return None
        return "error" if self.state == "degraded" else "ok"


@dataclass(frozen=True)
class ReceiverState:
    """One snapshot of the built-in receiver, as seen by this process's supervisor."""

    state: str  # starting | ready | stopped | error
    error: str | None = None


class TelemetryError(RuntimeError):
    """Raised when configured telemetry cannot be constructed safely."""


class SpanHandle:
    """A fixed, metadata-only span interface."""

    def __init__(self, implementation: object | None = None) -> None:
        self._implementation = implementation

    def __enter__(self) -> Self:
        if self._implementation is not None:
            self._implementation.enter()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        if self._implementation is not None:
            return self._implementation.exit(exc_type, exc, traceback)
        return False

    def set_usage(
        self, *, input_tokens: int, output_tokens: int, total_tokens: int,
        cache_read_tokens: int | None = None, cache_write_tokens: int | None = None,
    ) -> None:
        """Cache counts are ``None`` when the provider did not report them; they are part of ``input_tokens``."""
        if self._implementation is not None:
            self._implementation.set_usage(
                input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens,
                cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
            )

    def retry(self, *, attempt: int, reason: str, delay: int, after: bool) -> None:
        """Record one provider retry decision; ``delay`` is in milliseconds."""
        if self._implementation is not None:
            self._implementation.retry(attempt=attempt, reason=reason, delay=delay, after=after)

    def approval_requested(self, *, decision: str, scope: str | None = None) -> None:
        """``scope`` is omitted where the grant scope is unknown (the shared approval gate)."""
        if self._implementation is not None:
            self._implementation.approval_requested(decision=decision, scope=scope)

    def approval_resolved(self, *, decision: str | None, duration: int, scope: str | None = None) -> None:
        """``decision`` is None when the approval ended with no decision (cancelled or failed); ``duration`` is in ms."""
        if self._implementation is not None:
            self._implementation.approval_resolved(decision=decision, scope=scope, duration=duration)

    def set_response(
        self, *, finish_reason: str | None = None, model: str | None = None,
        ttft_ns: int | None = None, retry_count: int | None = None,
    ) -> None:
        """``ttft_ns`` is the span start to the first model output; ``None`` when none came."""
        if self._implementation is not None:
            self._implementation.set_response(finish_reason=finish_reason, model=model, ttft_ns=ttft_ns, retry_count=retry_count)

    def set_outcome(self, outcome: str) -> None:
        """How an ``invoke_agent`` or ``turn`` span ended; a value outside the closed set exports as ``_invalid``."""
        if self._implementation is not None:
            self._implementation.set_outcome(outcome)

    def set_tool_count(self, *, tool_count: int) -> None:
        if self._implementation is not None:
            self._implementation.set_tool_count(tool_count=tool_count)

    def set_sizes(self, *, argument_bytes: int, result_bytes: int) -> None:
        if self._implementation is not None:
            self._implementation.set_sizes(argument_bytes=argument_bytes, result_bytes=result_bytes)

    def set_compaction(self, *, tokens_after: int) -> None:
        if self._implementation is not None:
            self._implementation.set_compaction(tokens_after=tokens_after)

    def invalid_propagation(self) -> None:
        """Record that a received W3C trace context was malformed and ignored; carries no value."""
        if self._implementation is not None:
            self._implementation.invalid_propagation()

    def set_mcp_request(self, *, request_id: int, protocol_version: str | None) -> None:
        """Record the JSON-RPC id and negotiated protocol of one outgoing MCP request."""
        if self._implementation is not None:
            self._implementation.set_mcp_request(request_id=request_id, protocol_version=protocol_version)

    def fail(self, error: object | None, *, code: str | None = None) -> None:
        """Mark the span failed. ``error`` is an exception, an exception class, or an ``ErrorInfo``;
        anything but an exception has error.type ``_OTHER``. ``code`` overrides the mapped ``zeta.error.code``.
        """
        if self._implementation is not None:
            self._implementation.fail(error, code=code)


# the disabled facade hands out this one handle, so a disabled span allocates nothing.
_NOOP = SpanHandle()


class Telemetry:
    """A process-owned, optional telemetry provider."""

    def __init__(self, implementation: object | None = None, receiver: object | None = None) -> None:
        self._implementation = implementation
        self._receiver = receiver

    @classmethod
    def disabled(cls) -> Telemetry:
        return cls()

    @classmethod
    def configure(cls, settings: object, home: Path | None = None) -> Telemetry:
        """With ``home``, the exact built-in endpoint also keeps that home's receiver running."""
        endpoint = getattr(settings, "otel_endpoint", settings if isinstance(settings, str) else None)
        if not endpoint:
            return cls.disabled()
        from .runtime import configure

        receiver = None
        if home is not None:
            from .receiver.autostart import supervisor_for

            receiver = supervisor_for(home, endpoint)
        implementation = configure(endpoint, receiver)
        if receiver is not None:
            receiver.start()
        return cls(implementation, receiver)

    def enabled(self) -> bool:
        return self._implementation is not None

    def status(self) -> TelemetryStatus:
        if self._implementation is None:
            return TelemetryStatus(False, "off")
        return self._implementation.status()

    def receiver(self) -> ReceiverState | None:
        """The latest built-in receiver state, or None when this process does not supervise it; never does I/O."""
        return None if self._receiver is None else self._receiver.snapshot()

    def shutdown(self) -> None:
        if self._receiver is not None:
            self._receiver.stop()  # the receiver itself keeps running for the home's other processes
        if self._implementation is not None:
            self._implementation.shutdown()

    def current_trace_context(self) -> dict[str, str]:
        if self._implementation is None:
            return {}
        return self._implementation.current_trace_context()

    @contextmanager
    def use_trace_context(self, carrier: Mapping[str, str]) -> Iterator[None]:
        if self._implementation is None:
            yield
            return
        with self._implementation.use_trace_context(carrier):
            yield

    def extract(self, carrier: Mapping[str, str]) -> dict[str, str]:
        if self._implementation is None:
            return {}
        return self._implementation.extract(carrier)

    def inject(self, carrier: dict[str, str]) -> None:
        if self._implementation is not None:
            self._implementation.inject(carrier)

    def agent_span(
        self, profile: str, *, agent_name: str, conversation_id: str, model: str, depth: int, background: bool,
        run_id: str | None = None, parent_run_id: str | None = None,
    ) -> SpanHandle:
        """``run_id`` is a child's public run id (its ``child_instance_id``); ``None`` for a top-level agent."""
        if self._implementation is None:
            return _NOOP
        return self._span(f"invoke_agent {profile}", operation="agent", attributes={
            "gen_ai.operation.name": "invoke_agent", "zeta.agent.name": agent_name,
            "gen_ai.conversation.id": conversation_id, "gen_ai.request.model": model,
            "zeta.agent.depth": depth, "zeta.background": background,
            "zeta.agent.run_id": run_id, "zeta.agent.parent_run_id": parent_run_id,
        })

    def turn_span(self, *, conversation_id: str, turn_number: int, notification: bool) -> SpanHandle:
        if self._implementation is None:
            return _NOOP
        return self._span("zeta.turn", operation="turn", attributes={
            "gen_ai.conversation.id": conversation_id, "zeta.turn.number": turn_number,
            "zeta.notification": notification,
        })

    def inference_span(self, model: str, *, provider: str, stream: bool, conversation_id: str) -> SpanHandle:
        """``provider`` is the zeta provider id (claude, codex, ollama); spans carry its GenAI name."""
        if self._implementation is None:
            return _NOOP
        return self._span("chat " + model, kind="client", operation="inference", attributes={
            "gen_ai.operation.name": "chat", "gen_ai.provider.name": GEN_AI_PROVIDER_NAMES.get(provider, provider),
            "gen_ai.request.model": model, "gen_ai.conversation.id": conversation_id,
            "zeta.stream": stream,
        })

    def tool_span(
        self, name: str, *, tool_type: str, call_id: str, agent_name: str, conversation_id: str
    ) -> SpanHandle:
        if self._implementation is None:
            return _NOOP
        return self._span("execute_tool " + name, operation="tool", attributes={
            "gen_ai.operation.name": "execute_tool", "zeta.tool.name": name,
            "zeta.tool.type": tool_type, "zeta.tool.call.id": call_id,
            "zeta.agent.name": agent_name, "gen_ai.conversation.id": conversation_id,
        })

    def context_span(self, *, emergency: bool, tokens_before: int) -> SpanHandle:
        """One compaction; its handle's ``set_compaction`` records the result."""
        if self._implementation is None:
            return _NOOP
        return self._span("zeta.context.compact", operation="context", attributes={
            "zeta.emergency": emergency, "zeta.tokens.before": tokens_before,
        })

    def serve_span(self, method: str) -> SpanHandle:
        """One served JSON-RPC request; ``method`` outside the serve method table exports as ``_invalid``."""
        if self._implementation is None:
            return _NOOP
        return self._span("serve " + method, kind="server", operation="serve", attributes={
            "rpc.system": "jsonrpc", "rpc.method": method,
        })

    def mcp_span(self, method: str, *, server: str, target: str | None = None) -> SpanHandle:
        """One caller's MCP request: enriches the current ``execute_tool`` span, else one ``CLIENT`` span.

        ``target`` is the tool or prompt name; resource URIs are content and are never passed.
        """
        if self._implementation is None:
            return _NOOP
        return self._implementation.mcp_span(method, server=server, target=target)

    def _span(self, name: str, *, kind: str = "internal", operation: str, attributes: Mapping[str, object] | None = None) -> SpanHandle:
        return self._implementation._span(name, kind=kind, operation=operation, attributes=attributes)


__all__ = ["ReceiverState", "SpanHandle", "Telemetry", "TelemetryError", "TelemetryStatus"]
