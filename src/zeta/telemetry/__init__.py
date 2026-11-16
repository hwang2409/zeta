"""Optional OpenTelemetry facade for zeta."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Self


@dataclass(frozen=True)
class TelemetryStatus:
    enabled: bool
    state: str
    reason: str | None = None


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

    def set_usage(self, *, input_tokens: int, output_tokens: int, total_tokens: int) -> None:
        if self._implementation is not None:
            self._implementation.set_usage(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=total_tokens)

    def retry(self, *, attempt: int, reason: str, delay: int, after: bool) -> None:
        if self._implementation is not None:
            self._implementation.retry(attempt=attempt, reason=reason, delay=delay, after=after)

    def approval_requested(self, *, decision: str, scope: str) -> None:
        if self._implementation is not None:
            self._implementation.approval_requested(decision=decision, scope=scope)

    def approval_resolved(self, *, decision: str, scope: str, duration: int) -> None:
        if self._implementation is not None:
            self._implementation.approval_resolved(decision=decision, scope=scope, duration=duration)

    def set_response(self, *, model: str, finish_reason: str) -> None:
        if self._implementation is not None:
            self._implementation.set_response(model=model, finish_reason=finish_reason)

    def set_sizes(self, *, argument_bytes: int, result_bytes: int) -> None:
        if self._implementation is not None:
            self._implementation.set_sizes(argument_bytes=argument_bytes, result_bytes=result_bytes)

    def fail(self, exc: BaseException | type[BaseException]) -> None:
        if self._implementation is not None:
            self._implementation.fail(exc)


class Telemetry:
    """A process-owned, optional telemetry provider."""

    def __init__(self, implementation: object | None = None) -> None:
        self._implementation = implementation

    @classmethod
    def disabled(cls) -> Telemetry:
        return cls()

    @classmethod
    def configure(cls, settings: object) -> Telemetry:
        endpoint = getattr(settings, "otel_endpoint", settings if isinstance(settings, str) else None)
        if not endpoint:
            return cls.disabled()
        from .runtime import configure

        return cls(configure(endpoint))

    def enabled(self) -> bool:
        return self._implementation is not None

    def status(self) -> TelemetryStatus:
        if self._implementation is None:
            return TelemetryStatus(False, "off")
        return self._implementation.status()

    def shutdown(self) -> None:
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
        self, profile: str, *, agent_name: str, conversation_id: str, model: str, depth: int, background: bool
    ) -> SpanHandle:
        return self._span(f"invoke_agent {profile}", attributes={
            "gen_ai.operation.name": "invoke_agent", "zeta.agent.name": agent_name,
            "gen_ai.conversation.id": conversation_id, "gen_ai.request.model": model,
            "zeta.agent.depth": depth, "zeta.background": background,
        })

    def turn_span(self, *, conversation_id: str, turn_number: int, notification: bool, tool_count: int) -> SpanHandle:
        return self._span("zeta.turn", attributes={
            "gen_ai.conversation.id": conversation_id, "zeta.turn.number": turn_number,
            "zeta.notification": notification, "zeta.tool.count": tool_count,
        })

    def inference_span(self, model: str, *, provider: str, stream: bool, conversation_id: str) -> SpanHandle:
        return self._span("chat " + model, kind="client", attributes={
            "gen_ai.operation.name": "chat", "gen_ai.provider.name": provider,
            "gen_ai.request.model": model, "gen_ai.conversation.id": conversation_id,
            "zeta.stream": stream,
        })

    def tool_span(
        self, name: str, *, tool_type: str, call_id: str, agent_name: str, conversation_id: str
    ) -> SpanHandle:
        return self._span("execute_tool " + name, attributes={
            "gen_ai.operation.name": "execute_tool", "zeta.tool.name": name,
            "zeta.tool.type": tool_type, "zeta.tool.call.id": call_id,
            "zeta.agent.name": agent_name, "gen_ai.conversation.id": conversation_id,
        })

    def context_span(self, *, emergency: bool, tokens_before: int, tokens_after: int, marker_count: int) -> SpanHandle:
        return self._span("zeta.context.compact", attributes={
            "zeta.emergency": emergency, "zeta.tokens.before": tokens_before,
            "zeta.tokens.after": tokens_after, "zeta.marker.count": marker_count,
        })

    def serve_span(self, method: str) -> SpanHandle:
        return self._span("serve " + method, kind="server", attributes={"rpc.system": "jsonrpc", "rpc.method": method})

    def _span(self, name: str, *, kind: str = "internal", attributes: Mapping[str, object] | None = None) -> SpanHandle:
        if self._implementation is None:
            return SpanHandle()
        return self._implementation._span(name, kind=kind, attributes=attributes)


__all__ = ["SpanHandle", "Telemetry", "TelemetryError", "TelemetryStatus"]
