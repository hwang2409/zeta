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
    with instance.turn_span(conversation_id="test", turn_number=1, notification=False, tool_count=0):
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
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.close_connection = True

        def log_message(self, *_args):
            return

    class RedirectedHandler(http.server.BaseHTTPRequestHandler):
        def _ok(self):
            redirected.set()
            self.send_response(200)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self.close_connection = True

        do_GET = _ok
        do_POST = _ok

        def log_message(self, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectHandler)
    redirected_server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), RedirectedHandler)
    server.daemon_threads = True
    redirected_server.daemon_threads = True
    server.block_on_close = False
    redirected_server.block_on_close = False
    thread = threading.Thread(target=server.serve_forever)
    redirected_thread = threading.Thread(target=redirected_server.serve_forever)
    thread.start()
    redirected_thread.start()
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.2:1")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.2:1")
    try:
        endpoint = f"http://127.0.0.1:{server.server_port}/v1/traces"
        instance = telemetry.Telemetry.configure(endpoint)
        with instance.turn_span(conversation_id="test", turn_number=1, notification=False, tool_count=0):
            pass
        instance.shutdown()
        assert received.wait(5)
        assert not redirected.is_set()
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
        with instance.turn_span(conversation_id="test", turn_number=1, notification=False, tool_count=0):
            pass
        instance.shutdown()
        instance.shutdown()
        assert stopped.is_set()
        assert instance.status() == telemetry.TelemetryStatus(True, "degraded", "export_unavailable")
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


def test_facade_table_filters_every_operation_and_handle(monkeypatch: pytest.MonkeyPatch) -> None:
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
    reserved_names = (
        "gen_ai.input.messages", "gen_ai.output.messages", "gen_ai.system_instructions",
        "gen_ai.tool.call.arguments", "gen_ai.tool.call.result",
    )
    leading = tuple(f"{name}.secret-value" for name in reserved_names)
    embedded = tuple(f"safe.{name}.secret-value" for name in reserved_names)
    adversarial = ("multi\nline", "madeup-provider", "sk-secret", "ghp_secret", "Bearer secret", "unicode-☃", "x" * 129)
    operations = (
        lambda value: instance.agent_span(value, agent_name=value, conversation_id=value, model=value, depth=1, background=True),
        lambda value: instance.turn_span(conversation_id=value, turn_number=1, notification=True, tool_count=1),
        lambda value: instance.inference_span(value, provider=adversarial[1], stream=True, conversation_id=value),
        lambda value: instance.tool_span(value, tool_type=value, call_id=value, agent_name=value, conversation_id=value),
        lambda value: instance.context_span(emergency=True, tokens_before=1, tokens_after=2, marker_count=3),
        lambda value: instance.serve_span(value),
    )
    try:
        for operation, value in zip(operations, leading):
            with operation(value) as span:
                for value in embedded:
                    span.set_response(model=value, finish_reason=value)
                    span.retry(attempt=1, reason=value, delay=4, after=True)
                    span.approval_requested(decision=value, scope=value)
                    span.approval_resolved(decision=value, scope=value, duration=5)
                span.set_usage(input_tokens=1, output_tokens=2, total_tokens=3)
                span.set_sizes(argument_bytes=4, result_bytes=5)
                span.fail(ValueError(embedded[0]))
        try:
            with instance.tool_span(leading[0], tool_type=embedded[0], call_id=embedded[1], agent_name=embedded[2], conversation_id=embedded[3]) as span:
                span.fail(LookupError(embedded[4]))
                raise LookupError(embedded[4])
        except LookupError:
            pass
    finally:
        instance.shutdown()

    assert len(exported) == 6
    from zeta.telemetry.runtime import _ALLOWED_ATTRIBUTE_KEYS

    for record in exported:
        assert set(record.attributes) <= _ALLOWED_ATTRIBUTE_KEYS
        exported_text = (record.name, str(record.attributes), *(event.name for event in record.events), *(str(event.attributes) for event in record.events))
        for name in reserved_names:
            assert all(name not in value for value in exported_text)
        for value in adversarial:
            assert all(value not in text for text in exported_text)
        assert record.status.description is None
        assert not any(event.name == "exception" or "exception" in event.name for event in record.events)
        assert record.attributes.get("error.type") == "ValueError" or record.attributes.get("error.type") == "LookupError"

def test_batch_queue_and_shared_logger_are_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace.export import SpanExportResult

    class MemoryExporter:
        def __init__(self, **_kwargs):
            pass

        def export(self, _spans):
            return SpanExportResult.SUCCESS

        def shutdown(self):
            pass

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", MemoryExporter)
    logger = logging.getLogger("opentelemetry.sdk._shared_internal")
    old = (logger.disabled, logger.propagate)
    instance = telemetry.Telemetry.configure("http://127.0.0.1:4318/v1/traces")
    try:
        processor = instance._implementation._provider._active_span_processor._span_processors[0]
        assert processor._batch_processor._max_queue_size == 2048
        assert logger.disabled is True
        assert logger.propagate is False
        records: list[logging.LogRecord] = []
        handler = logging.Handler()
        handler.emit = records.append
        logger.addHandler(handler)
        try:
            batch = processor._batch_processor
            batch._queue.extend([None] * batch._max_queue_size)
            batch.emit(None)
            assert not records
        finally:
            logger.removeHandler(handler)
    finally:
        instance.shutdown()
    assert (logger.disabled, logger.propagate) == old


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
