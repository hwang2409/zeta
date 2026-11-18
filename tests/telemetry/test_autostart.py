"""Built-in receiver auto-start: real receiver processes, the receiver lock, and non-blocking status."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import threading
from collections.abc import Callable
from pathlib import Path

import pytest

pytest.importorskip("opentelemetry.sdk")
pytest.importorskip("starlette")

from tests.support.serve_close import close_serve
from tests.telemetry.test_serve_propagation import (  # noqa: F401 - traced is a fixture
    Wire,
    _server,
    traced,
)
from zeta.server.handlers.telemetry import live_view
from zeta.telemetry import ReceiverState, Telemetry
from zeta.telemetry.receiver import autostart, process
from zeta.telemetry.receiver.process import OTLP_URL, receiver_ready


class Checks:
    """Wake the test after each of the supervisors' health checks, which run on their own threads."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, health: Callable[[], str | None] = autostart._health) -> None:
        self._changed = threading.Condition()
        self._health = health
        monkeypatch.setattr(autostart, "POLL_SECONDS", 0.1)
        # test-contract: seam the health check is where the supervisor observes the receiver process
        monkeypatch.setattr(autostart, "_health", self._check)

    def _check(self) -> str | None:
        try:
            return self._health()
        finally:
            with self._changed:
                self._changed.notify_all()

    def until(self, condition: Callable[[], bool], what: str) -> None:
        with self._changed:
            assert self._changed.wait_for(condition, timeout=30), what


class Launches:
    """Run the real ``_launch`` but send each receiver's output to a file the test can read."""

    def __init__(self, directory: Path, barrier: threading.Barrier | None = None) -> None:
        self.directory, self.barrier = directory, barrier
        self.children: list[tuple[subprocess.Popen, Path]] = []
        self.options: list[dict] = []
        self._popen = subprocess.Popen

    def __call__(self, command: list[str], **options: object) -> subprocess.Popen:
        if self.barrier is not None:
            self.barrier.wait(timeout=20)  # both starters launch together, before either sees a receiver
        self.options.append(dict(options))
        output = self.directory / f"receiver-{len(self.children)}.log"
        with output.open("wb") as file:
            child = self._popen(command, **{**options, "stdout": file, "stderr": file})
        self.children.append((child, output))
        return child

    def alive(self) -> list[subprocess.Popen]:
        return [child for child, _ in self.children if child.poll() is None]

    def close(self) -> None:
        for child, _ in self.children:
            if child.poll() is None:
                child.send_signal(signal.SIGTERM)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=10)


def test_concurrent_starters_produce_one_receiver(tmp_path: Path, builtin_port: None, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    launches = Launches(tmp_path, threading.Barrier(2))
    monkeypatch.setattr(autostart.subprocess, "Popen", launches)
    checks = Checks(monkeypatch)
    starters = [autostart.supervisor_for(home, OTLP_URL), autostart.supervisor_for(home, OTLP_URL)]
    try:
        for starter in starters:
            starter.start()
        checks.until(lambda: all(starter.snapshot() == ReceiverState("ready") for starter in starters), "both starters see the receiver")
        checks.until(lambda: len(launches.alive()) == 1, "the losing launch exits")

        assert len(launches.children) == 2
        assert len(launches.alive()) == 1, "two receivers are running"
        (loser_output,) = [output.read_text() for child, output in launches.children if child.poll() is not None]
        assert "cannot listen" not in loser_output, f"the loser got past the receiver lock: {loser_output!r}"
        assert "already running at http://127.0.0.1:4323" in loser_output or "another receiver holds" in loser_output, loser_output
        assert all(options["start_new_session"] is True and options["stdin"] == subprocess.DEVNULL for options in launches.options)
        assert receiver_ready(home)
    finally:
        for starter in starters:
            starter.stop()
        launches.close()


def test_stale_lock_after_a_crash_is_recovered(tmp_path: Path, builtin_port: None, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    launches = Launches(tmp_path)
    monkeypatch.setattr(autostart.subprocess, "Popen", launches)
    checks = Checks(monkeypatch)
    supervisor = autostart.supervisor_for(home, OTLP_URL)
    try:
        supervisor.start()
        checks.until(lambda: supervisor.snapshot() == ReceiverState("ready"), "the first receiver starts")
        (first,) = launches.alive()
        assert os.getsid(first.pid) == first.pid, "the receiver shares the starter's session and dies with its terminal"

        first.send_signal(signal.SIGKILL)  # a crash: no cleanup runs, the lock file stays behind
        first.wait(timeout=10)
        assert (home / "telemetry" / "receiver.lock").exists()

        checks.until(lambda: len(launches.alive()) == 1 and supervisor.snapshot() == ReceiverState("ready"), "a new receiver takes the stale lock")
        assert receiver_ready(home)
        assert len(launches.children) == 2
        assert launches.alive()[0].pid != first.pid
    finally:
        supervisor.stop()
        launches.close()


@pytest.mark.parametrize("endpoint", [
    "http://127.0.0.1:4318/v1/traces",
    "http://127.0.0.2:4323/v1/traces",
    "http://[::1]:4323/v1/traces",
    "https://127.0.0.1:4323/v1/traces",
    "http://127.0.0.1:4323/v1/traces/",
])
def test_an_external_endpoint_never_starts_a_receiver(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, endpoint: str) -> None:
    launched: list[Path] = []
    # test-contract: seam a launch would start a real process; the test only needs to see that none starts
    monkeypatch.setattr(autostart, "_launch", launched.append)
    instance = Telemetry.configure(endpoint, tmp_path)
    try:
        assert instance.receiver() is None, f"{endpoint} supervises the built-in receiver"
        assert not [thread for thread in threading.enumerate() if thread.name == "zeta-otel-receiver"]
        assert launched == []
    finally:
        instance.shutdown()


def test_status_reads_never_check_health_on_the_caller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home_id = "ab" * 16
    (tmp_path / "telemetry").mkdir()
    (tmp_path / "telemetry" / "home-id").write_text(home_id + "\n")
    callers: list[int] = []

    def health(host: str, port: int) -> str:
        callers.append(threading.get_ident())
        return home_id

    # test-contract: seam the /health request is the I/O a status read must never do
    monkeypatch.setattr(process, "_running_home_id", health)
    # test-contract: seam a ready receiver is never launched again
    monkeypatch.setattr(autostart, "_launch", lambda home: pytest.fail("a ready receiver was launched again"))
    checks = Checks(monkeypatch)
    instance = Telemetry.configure(OTLP_URL, tmp_path)
    try:
        checks.until(lambda: instance.receiver() == ReceiverState("ready"), "the supervisor sees the receiver")
        reads = [instance.receiver(), live_view(instance), instance.receiver()]

        assert threading.get_ident() not in callers, "a status read checked health on the caller's thread"
        assert reads[1] == {"receiver": "ready", "last_export": None, "last_export_ns": None}
    finally:
        instance.shutdown()


class _Fixed:
    def __init__(self, state: ReceiverState) -> None:
        self.state = state

    def snapshot(self) -> ReceiverState:
        return self.state

    def stop(self) -> None:
        pass


@pytest.mark.asyncio
async def test_serve_status_and_telemetry_status_show_the_same_receiver(tmp_path: Path, traced, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: F811
    configured, _memory = traced
    (tmp_path / "settings.toml").write_text(f'[otel]\nendpoint = "{OTLP_URL}"\n')
    telemetry = Telemetry(configured._implementation, _Fixed(ReceiverState("starting")))
    configured._implementation._cell.publish("export_unavailable")
    server = _server(tmp_path, "websocket", telemetry, monkeypatch)
    wire = Wire(server, "websocket")
    await wire.open()
    try:
        await wire.rpc("hello", "hello", {"protocol_version": "1.2", "features": ["telemetry_read_v1"]})
        status = json.loads(await wire.rpc("status", "status"))["result"]
        read = json.loads(await wire.rpc("telemetry", "telemetry_status"))["result"]
    finally:
        await close_serve(server, wire.writer)

    assert status["telemetry"]["receiver"] == read["receiver"] == "starting"
    assert status["telemetry"]["last_export"] == read["last_export"] == "error"
    assert status["telemetry"]["last_export_ns"] == read["last_export_ns"]
    assert isinstance(read["last_export_ns"], int)
    assert read["endpoint"] == "builtin"


def test_a_foreign_home_receiver_is_reported_and_gets_no_spans(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from opentelemetry.exporter.otlp.proto.http import trace_exporter
    from opentelemetry.sdk.trace.export import SpanExportResult

    sent: list[object] = []

    class Recorder:
        def __init__(self, **_options: object) -> None:
            pass

        def export(self, spans: object) -> SpanExportResult:
            sent.append(spans)
            return SpanExportResult.SUCCESS

        def shutdown(self) -> None:
            pass

    monkeypatch.setattr(trace_exporter, "OTLPSpanExporter", Recorder)
    # test-contract: seam a launch would start a real process; none may start while another home holds the port
    monkeypatch.setattr(autostart, "_launch", lambda home: pytest.fail("started a receiver over another home's"))
    checks = Checks(monkeypatch, health=lambda: "e" * 32)
    instance = Telemetry.configure(OTLP_URL, tmp_path)
    try:
        checks.until(lambda: instance.receiver() == ReceiverState("error", autostart.FOREIGN_HOME), "the supervisor reports the other home")
        with instance.turn_span(conversation_id="test", turn_number=1, notification=False):
            pass
        assert instance._implementation._provider.force_flush(10_000)

        assert sent == [], "spans were exported to another home's receiver"
        assert live_view(instance)["last_export"] == "error"
    finally:
        instance.shutdown()
