from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

telemetry = pytest.importorskip("zeta.telemetry")


def test_disabled_facade_does_not_import_opentelemetry() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from zeta.telemetry import Telemetry; assert Telemetry.disabled().status().state == 'off'; import sys; assert not any(k.startswith('opentelemetry') for k in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    assert result.returncode == 0


def test_hostile_environment_does_not_disable_private_configuration(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace.export import SpanExportResult

    exported = []

    class MemoryExporter:
        def __init__(self, **_kwargs):
            pass

        def export(self, spans):
            exported.extend(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", MemoryExporter)
    for name in ("OTEL_RESOURCE_ATTRIBUTES", "OTEL_SERVICE_NAME", "OTEL_TRACES_SAMPLER", "OTEL_PROPAGATORS"):
        monkeypatch.setenv(name, "secret.key=leak" if name == "OTEL_RESOURCE_ATTRIBUTES" else "always_off")
    for name in ("OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT", "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT"):
        monkeypatch.setenv(name, "0")
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER"):
        monkeypatch.delenv(name, raising=False)

    instance = telemetry.Telemetry.configure(SimpleNamespace(otel_endpoint="http://127.0.0.1:4318/v1/traces"))
    with instance.turn_span():
        carrier = instance.current_trace_context()
    instance.shutdown()

    assert len(exported) == 1
    assert dict(exported[0].resource.attributes) == {"service.name": "zeta", "service.version": "0.1.0"}
    assert set(carrier) <= {"traceparent", "tracestate"}


def test_sdk_disabled_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", " true ")
    with pytest.raises(telemetry.TelemetryError, match="OTEL_SDK_DISABLED cannot disable"):
        telemetry.Telemetry.configure("http://127.0.0.1:4318/v1/traces")


def test_loopback_and_one_slot() -> None:
    from zeta.telemetry.runtime import StatusCell, validate_loopback_endpoint

    assert validate_loopback_endpoint("http://127.0.0.1:4318/v1/traces")
    with pytest.raises(telemetry.TelemetryError):
        validate_loopback_endpoint("http://localhost:4318")
    cell = StatusCell()
    for value in ("healthy", "export_unavailable", "healthy"):
        cell.publish(value)
    assert cell.read() == "healthy"
    assert cell.read() is None
