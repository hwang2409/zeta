"""One owner per process closes configured telemetry exactly once on every exit path."""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path

import pytest
from opentelemetry.sdk.trace import TracerProvider

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.cli.main import build_parser
from zeta.core.session import SessionError
from zeta.runtime.loop import AgentLoop
from zeta.server import ZetaServer, run_server
from zeta.tui import bootstrap
from zeta.tui.bootstrap import create_app

_OWNED_LOGGERS = ("opentelemetry.exporter", "opentelemetry.sdk._shared_internal")


def _logger_state() -> list[tuple[bool, bool, int, list[logging.Handler]]]:
    return [
        (logger.disabled, logger.propagate, logger.level, list(logger.handlers))
        for logger in map(logging.getLogger, _OWNED_LOGGERS)
    ]


def _batch_threads() -> list[threading.Thread]:
    return [thread for thread in threading.enumerate() if thread.name.startswith("OtelBatch")]


class _Failure(Exception):
    pass


async def _serve(home: Path, step: str, monkeypatch: pytest.MonkeyPatch) -> None:
    cwd = home.parent / "launch"
    cwd.mkdir()
    server = ZetaServer(
        home=home,
        cwd=cwd,
        provider="codex",
        port=0,
        backend_factory=lambda provider, model, home: (FakeBackend([]), model or "offline"),
    )
    runtime = server.runtime
    release = threading.Event()
    composing = threading.Event()
    original_compose = runtime._compose

    def compose(**kwargs):
        if step == "compose":
            fail()
        if step == "cancel":
            composing.set()
            assert release.wait(10), "the test never released the blocked compose"
        return original_compose(**kwargs)

    def fail(*_args, **_kwargs):
        assert len(_batch_threads()) == 1, "serve configured telemetry before this step"
        raise _Failure

    runtime._compose = compose
    if step == "prepare":
        monkeypatch.setattr(runtime, "_prepare_new_session", fail)
    if step == "activate":
        monkeypatch.setattr(AgentLoop, "activate", fail)
    if step == "start":
        monkeypatch.setattr(runtime, "initialize_project_registry", fail)
    try:
        if step == "start":
            # run_server is the serve entry point; it must close a server whose start failed
            with pytest.raises(_Failure):
                await await_or_fail(run_server(server), what="serve run with a failing start")
            return
        await server.start()
        assert len(_batch_threads()) == 1, "configured serve owns one exporter thread"
        if step == "cancel":
            setup = asyncio.create_task(runtime.create_session())
            assert await asyncio.to_thread(composing.wait, 10)
            setup.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await await_or_fail(setup, what="canceled session setup")
        elif step == "success":
            await await_or_fail(runtime.create_session(), what="session setup")
        else:
            with pytest.raises(_Failure):
                await await_or_fail(runtime.create_session(), what="failed session setup")
    finally:
        release.set()
        # run_server closes after any start; a second close must not shut down again
        await await_or_fail(server.close(), what="first serve close")
        await await_or_fail(server.close(), what="second serve close")


async def _tui(home: Path, step: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.chdir(home.parent)
    args = build_parser().parse_args(["--provider", "codex", "--no-session"])
    if step == "compose":
        def fail(**_kwargs):
            assert len(_batch_threads()) == 1, "startup configured telemetry before composing"
            raise _Failure

        monkeypatch.setattr(bootstrap, "compose_runtime", fail)
        with pytest.raises(_Failure):
            await asyncio.to_thread(create_app, args)
        return
    app = await asyncio.to_thread(create_app, args)
    assert len(_batch_threads()) == 1, "configured TUI owns one exporter thread"
    await await_or_fail(app.close(), what="first TUI close")
    await await_or_fail(app.close(), what="second TUI close")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("surface", "step"),
    [
        ("serve", "success"),
        ("serve", "start"),
        ("serve", "prepare"),
        ("serve", "compose"),
        ("serve", "activate"),
        ("serve", "cancel"),
        ("tui", "success"),
        ("tui", "compose"),
    ],
    ids=[
        "serve-success",
        "serve-start-fails",
        "serve-prepare-fails",
        "serve-compose-fails",
        "serve-activate-fails",
        "serve-setup-canceled",
        "tui-success",
        "tui-compose-fails",
    ],
)
async def test_telemetry_owner_shuts_down_once_on_every_exit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, surface: str, step: str
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    # nothing listens on the discard port; no span is recorded, so nothing is exported
    (home / "settings.toml").write_text(
        '[otel]\nendpoint = "http://127.0.0.1:9/v1/traces"\n[retention]\ndays = 0\n'
    )
    loggers_before = _logger_state()
    assert _batch_threads() == []
    shutdowns: list[TracerProvider] = []
    original_shutdown = TracerProvider.shutdown

    def counted_shutdown(provider: TracerProvider) -> None:
        shutdowns.append(provider)
        original_shutdown(provider)

    monkeypatch.setattr(TracerProvider, "shutdown", counted_shutdown)

    await (_serve if surface == "serve" else _tui)(home, step, monkeypatch)

    assert len(shutdowns) == 1, "the one provider is shut down exactly once"
    assert _batch_threads() == [], "the exporter thread is joined"
    assert _logger_state() == loggers_before, "exporter loggers are restored"


def test_tui_reports_a_telemetry_startup_error_instead_of_crashing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    (home / "settings.toml").write_text('[otel]\nendpoint = "http://127.0.0.1:9/v1/traces"\n')
    monkeypatch.setenv("ZETA_HOME", str(home))
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://127.0.0.1:4318")
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SessionError, match="OTEL exporter environment is not allowed"):
        create_app(build_parser().parse_args(["--provider", "codex", "--no-session"]))
    assert _batch_threads() == []
