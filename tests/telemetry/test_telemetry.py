from __future__ import annotations

import http.server
import logging
import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from zeta import telemetry


def test_disabled_facade_does_not_import_opentelemetry() -> None:
    result = subprocess.run(
        [sys.executable, "-c", "from zeta.telemetry import Telemetry; assert Telemetry.disabled().status().state == 'off'; import sys; assert not any(k.startswith('opentelemetry') for k in sys.modules)"],
        check=True,
        capture_output=True,
        text=True,
        env={"PATH": os.environ["PATH"], "PYTHONPATH": os.environ.get("PYTHONPATH", "")},
    )
    assert result.returncode == 0


def test_hostile_environment_does_not_disable_private_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
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
    for name in (
        "OTEL_ATTRIBUTE_VALUE_LENGTH_LIMIT",
        "OTEL_SPAN_ATTRIBUTE_COUNT_LIMIT",
        "OTEL_SPAN_EVENT_COUNT_LIMIT",
        "OTEL_SPAN_LINK_COUNT_LIMIT",
        "OTEL_EVENT_ATTRIBUTE_COUNT_LIMIT",
        "OTEL_LINK_ATTRIBUTE_COUNT_LIMIT",
        "OTEL_ATTRIBUTE_COUNT_LIMIT",
    ):
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("OTEL_PROPAGATORS", "baggage")
    for name in ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_PYTHON_EXPORTER_OTLP_HTTP_TRACES_CREDENTIAL_PROVIDER"):
        monkeypatch.delenv(name, raising=False)

    instance = telemetry.Telemetry.configure(SimpleNamespace(otel_endpoint="http://127.0.0.1:4318/v1/traces"))
    with instance.turn_span():
        carrier = instance.current_trace_context()
    limits = vars(instance._implementation._provider._span_limits)
    instance.shutdown()

    assert limits == {
        "max_events": 128, "max_links": 128, "max_attributes": 128,
        "max_span_attributes": 128, "max_event_attributes": 128,
        "max_link_attributes": 128, "max_attribute_length": 4096,
        "max_span_attribute_length": 4096,
    }
    assert len(exported) == 1
    assert dict(exported[0].resource.attributes) == {"service.name": "zeta", "service.version": __import__("importlib.metadata", fromlist=["version"]).version("zeta")}
    assert set(carrier) <= {"traceparent", "tracestate"}
    limits = exported[0].get_span_context()  # the span proves export was not disabled
    assert limits.is_valid


def test_transport_ignores_proxy_and_does_not_follow_redirects(monkeypatch: pytest.MonkeyPatch) -> None:
    received = threading.Event()
    redirected = threading.Event()

    class RedirectHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            received.set()
            self.send_response(302)
            self.send_header("Location", f"http://127.0.0.1:{redirected_server.server_port}/v1/traces")
            self.end_headers()

        def log_message(self, *_args):
            return

    class RedirectedHandler(http.server.BaseHTTPRequestHandler):
        def _ok(self):
            redirected.set()
            self.send_response(200)
            self.end_headers()

        do_GET = _ok
        do_POST = _ok

        def log_message(self, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    redirected_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectedHandler)
    thread = threading.Thread(target=server.serve_forever)
    redirected_thread = threading.Thread(target=redirected_server.serve_forever)
    thread.start()
    redirected_thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.2:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.2:1")
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
        instance = telemetry.Telemetry.configure(endpoint)
        with instance.turn_span():
            pass
        instance.shutdown()
        assert received.wait(5)
        assert not redirected.wait(0.2)
    finally:
        server.shutdown()
        redirected_server.shutdown()
        server.server_close()
        redirected_server.server_close()
        thread.join(5)
        redirected_thread.join(5)


def test_sdk_disabled_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OTEL_SDK_DISABLED", " true ")
    with pytest.raises(telemetry.TelemetryError, match="OTEL_SDK_DISABLED cannot disable"):
        telemetry.Telemetry.configure("http://127.0.0.1:4318/v1/traces")


def test_shutdown_is_synchronous_and_export_failure_is_not_logged(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace.export import SpanExportResult

    stopped = threading.Event()

    class FailingExporter:
        def __init__(self, **_kwargs):
            pass

        def export(self, _spans):
            logging.getLogger("opentelemetry.exporter").error("internal exporter detail")
            return SpanExportResult.FAILURE

        def shutdown(self):
            stopped.set()

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", FailingExporter)
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.emit = records.append
    app_logger = logging.getLogger("opentelemetry.exporter")
    old_state = (app_logger.disabled, app_logger.propagate, app_logger.level, list(app_logger.handlers))
    app_logger.addHandler(handler)
    try:
        instance = telemetry.Telemetry.configure("http://127.0.0.1:4318/v1/traces")
        with instance.turn_span():
            pass
        instance.shutdown()
        assert stopped.is_set()
        assert not records
    finally:
        app_logger.removeHandler(handler)
        assert (app_logger.disabled, app_logger.propagate, app_logger.level, list(app_logger.handlers)) == old_state


def test_loopback_and_one_slot() -> None:
    from zeta.telemetry.runtime import StatusCell, validate_loopback_endpoint

    assert validate_loopback_endpoint("http://127.0.0.1:4318/v1/traces")
    for endpoint in ("http://localhost:4318", "http://127.0.0.1:notaport", "http://127.0.0.1:99999"):
        with pytest.raises(telemetry.TelemetryError):
            validate_loopback_endpoint(endpoint)
    cell = StatusCell()
    for value in ("healthy", "export_unavailable", "healthy"):
        cell.publish(value)
    assert cell.read() == "healthy"
    assert cell.read() is None


def test_facade_drops_content_and_exception_details(monkeypatch: pytest.MonkeyPatch) -> None:
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
    instance = telemetry.Telemetry.configure("http://127.0.0.1:4318/v1/traces")
    try:
        with instance.tool_span("test", **{name: "secret" for name in (
            "gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions",
            "gen_ai.tool.call.arguments", "gen_ai.tool.call.result",
        )}) as span:
            span.set_attribute("safe", "ok")
            span.add_event("event", {"gen_ai.input.messages": "secret", "safe.event": "ok"})
            span.record_exception(ValueError("secret stack and message"))
    finally:
        instance.shutdown()
    assert len(exported) == 1
    record = exported[0]
    assert not any(key in record.attributes for key in (
        "gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments", "gen_ai.tool.call.result",
    ))
    assert record.attributes["safe"] == "ok"
    assert record.attributes["error.type"] == "ValueError"
    assert all("secret" not in str(event.attributes) for event in record.events)


def test_status_cell_is_bounded_with_concurrent_publishers() -> None:
    from zeta.telemetry.runtime import StatusCell

    cell = StatusCell()
    start = threading.Barrier(8)

    def publish() -> None:
        start.wait(timeout=5)
        cell.publish("export_unavailable")

    threads = [threading.Thread(target=publish) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert all(not thread.is_alive() for thread in threads)
    cell.publish("healthy")
    assert cell.read() == "healthy"
    assert cell.read() is None
