"""Optional OpenTelemetry facade for zeta."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Self


@dataclass(frozen=True)
class TelemetryStatus:
    enabled: bool
    state: str
    reason: str | None = None


class TelemetryError(RuntimeError):
    """Raised when configured telemetry cannot be constructed safely."""


class _DisabledSpan:
    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def set_attribute(self, _name: str, _value: object) -> None:
        return None

    def add_event(self, _name: str, _attributes: Mapping[str, object] | None = None) -> None:
        return None

    def record_exception(self, _exception: BaseException) -> None:
        return None

    def set_error(self, _error_type: str | None = None) -> None:
        return None


class Telemetry:
    """A process-owned, optional telemetry provider."""

    def __init__(self, implementation: Any = None) -> None:
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

    def _span(self, name: str, *, kind: str = "internal", attributes: Mapping[str, object] | None = None) -> Any:
        if self._implementation is None:
            return _DisabledSpan()
        return self._implementation._span(name, kind=kind, attributes=attributes)

    def agent_span(self, profile: str, **attributes: object) -> Any:
        return self._span(f"invoke_agent {profile}", attributes=attributes)

    def turn_span(self, **attributes: object) -> Any:
        return self._span("zeta.turn", attributes=attributes)

    def inference_span(self, model: str, **attributes: object) -> Any:
        return self._span(f"chat {model}", kind="client", attributes=attributes)

    def tool_span(self, name: str, **attributes: object) -> Any:
        return self._span(f"execute_tool {name}", attributes=attributes)

    def context_span(self, **attributes: object) -> Any:
        return self._span("zeta.context.compact", attributes=attributes)

    def serve_span(self, method: str, **attributes: object) -> Any:
        return self._span(f"serve {method}", kind="server", attributes=attributes)

    def retry(self, **attributes: object) -> None:
        if self._implementation is not None:
            self._implementation.retry(**attributes)


__all__ = ["Telemetry", "TelemetryError", "TelemetryStatus"]
