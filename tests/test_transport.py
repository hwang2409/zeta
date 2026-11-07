import asyncio
import gc
import json
import threading
import weakref
from collections.abc import AsyncIterator
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from rich.console import Console

import zeta.providers.stream_diagnostics as stream_diagnostics_module
from zeta.protocol.types import (
    ErrorInfo,
)
from zeta.providers.anthropic_errors import AnthropicStreamError
from zeta.providers.codex_errors import CodexHTTPError, CodexStreamError
from zeta.providers.completion import retryable_provider_error
from zeta.providers.transport import (
    build_response_error,
    cleanup_transport,
    close_transport_resources,
)
from zeta.tui import loop_errors as loop_errors_module
from zeta.tui.loop_errors import run_prompt_toolkit


class _PartialErrorBody(httpx.AsyncByteStream):
    def __init__(self) -> None:
        self.finished = asyncio.Event()
        self.closed = asyncio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        yield b"partial provider error"
        await self.finished.wait()

    async def aclose(self) -> None:
        self.closed.set()


@pytest.mark.asyncio


@pytest.mark.asyncio
async def test_response_error_body_is_bounded_and_keeps_response_metadata() -> None:
    request = httpx.Request("POST", "https://test.invalid")
    response = httpx.Response(
        503,
        headers={"retry-after": "2"},
        content=b"x" * 9000,
        request=request,
    )

    error = await build_response_error(
        response,
        lambda status, body, headers: RuntimeError(
            f"{status}:{len(body)}:{headers['retry-after']}"
        ),
        stall_seconds=5.0,
        on_stall=lambda elapsed: RuntimeError(f"stalled {elapsed}"),
    )

    assert str(error) == "503:8192:2"


@pytest.mark.asyncio
async def test_transport_close_survives_overlapping_cancellation() -> None:
    started = [asyncio.Event(), asyncio.Event()]
    close_tasks: list[asyncio.Task[None]] = []

    async def block(index: int) -> None:
        task = asyncio.current_task()
        assert task is not None
        close_tasks.append(task)
        started[index].set()
        await asyncio.wait_for(asyncio.Event().wait(), 10)

    async def cancel_then_close() -> None:
        try:
            await asyncio.wait_for(asyncio.Event().wait(), 10)
        except asyncio.CancelledError:
            await asyncio.wait_for(close_transport_resources(block(0), block(1)), 10)

    owner = asyncio.create_task(cancel_then_close())
    await asyncio.wait_for(asyncio.sleep(0), 10)
    owner.cancel()
    asyncio.get_running_loop().call_soon(owner.cancel)

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(owner, 10)
    await asyncio.wait_for(asyncio.sleep(0), 10)

    assert all(event.is_set() for event in started)
    assert all(task.done() for task in close_tasks)
    assert not set(close_tasks) & asyncio.all_tasks()
    close_refs = [weakref.ref(task) for task in close_tasks]
    close_tasks.clear()
    gc.collect()
    assert all(
        reference() is None or reference().done()  # type: ignore[union-attr]
        for reference in close_refs
    )


@pytest.mark.asyncio
async def test_detached_close_failure_reaches_loop_diagnostics() -> None:
    loop = asyncio.get_running_loop()
    contexts: list[dict[str, object]] = []
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))
    owner_ready = asyncio.Event()

    async def failed_close() -> None:
        try:
            await asyncio.wait_for(asyncio.Event().wait(), 10)
        except asyncio.CancelledError:
            raise RuntimeError("detached close canary") from None

    async def cancel_then_close() -> None:
        owner_ready.set()
        try:
            await asyncio.wait_for(asyncio.Event().wait(), 10)
        except asyncio.CancelledError:
            await asyncio.wait_for(close_transport_resources(failed_close()), 10)

    try:
        owner = asyncio.create_task(cancel_then_close())
        await asyncio.wait_for(owner_ready.wait(), 10)
        owner.cancel()
        await asyncio.wait_for(owner, 10)
        callbacks_flushed = loop.create_future()
        loop.call_soon(callbacks_flushed.set_result, None)
        await asyncio.wait_for(callbacks_flushed, 10)
    finally:
        loop.set_exception_handler(previous_handler)

    assert len(contexts) == 1
    assert contexts[0] == {
        "exception_type": "builtins.RuntimeError",
        "stage": "background_close",
        "index": 0,
        "primary_outcome": "success",
    }




@pytest.mark.asyncio


class _FailingCloseClient:
    async def aclose(self) -> None:
        raise RuntimeError("cleanup-secret-canary")


@pytest.mark.asyncio
async def test_cleanup_diagnostic_failure_preserves_primary_and_is_secret_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = ValueError("primary-provider-secret-canary")
    diagnostic_canary = "diagnostic-write-secret-canary"
    diagnostics: list[dict[str, object]] = []
    printed: list[object] = []

    async def fail_diagnostic_write(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError(diagnostic_canary)

    monkeypatch.setattr(
        stream_diagnostics_module.StreamDiagnostics,
        "record_cleanup_failure",
        fail_diagnostic_write,
    )
    # test-contract: seam capture the real TUI handler's diagnostic record
    monkeypatch.setattr(
        loop_errors_module,
        "_schedule_diagnostic",
        lambda _path, record: diagnostics.append(record),
    )

    class PromptApplication:
        async def run_async(self, *, set_exception_handler: bool) -> BaseException | None:
            assert not set_exception_handler
            return await asyncio.wait_for(
                cleanup_transport(
                    stream_context=None,
                    entered=False,
                    client=_FailingCloseClient(),
                    owns_client=True,
                    primary_exception=primary,
                    http_error_type=RuntimeError,
                    diagnostics_path=tmp_path / "provider.jsonl",
                    provider="test-provider",
                ),
                10,
            )

    session = SimpleNamespace(app=PromptApplication())
    app = SimpleNamespace(
        _print_unit=printed.append,
        _invalidate_prompt=lambda: None,
    )

    (result,) = await asyncio.wait_for(
        asyncio.gather(run_prompt_toolkit(session, app), return_exceptions=True), 10
    )

    assert result is primary
    assert diagnostic_canary not in repr(result)
    rendered_output = StringIO()
    console = Console(file=rendered_output, color_system=None, width=120)
    for unit in printed:
        console.print(unit)
    rendered = rendered_output.getvalue()
    logged = repr(diagnostics)
    assert printed and diagnostics
    assert diagnostic_canary not in rendered
    assert diagnostic_canary not in logged
    assert "primary-provider-secret-canary" not in rendered
    assert "primary-provider-secret-canary" not in logged


@pytest.mark.asyncio
async def test_cleanup_diagnostic_generator_exit_propagates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contexts: list[dict[str, object]] = []

    async def exit_diagnostic_write(*_args: object, **_kwargs: object) -> None:
        raise GeneratorExit

    monkeypatch.setattr(
        stream_diagnostics_module.StreamDiagnostics,
        "record_cleanup_failure",
        exit_diagnostic_write,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))
    primary = ValueError("primary-provider-secret-canary")
    try:
        with pytest.raises(GeneratorExit) as raised:
            await asyncio.wait_for(
                cleanup_transport(
                    stream_context=None,
                    entered=False,
                    client=_FailingCloseClient(),
                    owns_client=True,
                    primary_exception=primary,
                    http_error_type=RuntimeError,
                    diagnostics_path=tmp_path / "provider.jsonl",
                    provider="test-provider",
                ),
                10,
            )
    finally:
        loop.set_exception_handler(previous_handler)

    assert raised.value.__cause__ is primary
    assert contexts == []


@pytest.mark.asyncio
async def test_cleanup_diagnostic_generator_exit_preserves_primary_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    primary = asyncio.CancelledError()

    async def exit_diagnostic_write(*_args: object, **_kwargs: object) -> None:
        raise GeneratorExit

    monkeypatch.setattr(
        stream_diagnostics_module.StreamDiagnostics,
        "record_cleanup_failure",
        exit_diagnostic_write,
    )

    with pytest.raises(asyncio.CancelledError) as raised:
        await asyncio.wait_for(
            cleanup_transport(
                stream_context=None,
                entered=False,
                client=_FailingCloseClient(),
                owns_client=True,
                primary_exception=primary,
                http_error_type=RuntimeError,
                diagnostics_path=tmp_path / "provider.jsonl",
                provider="test-provider",
            ),
            10,
        )

    assert raised.value is primary


@pytest.mark.asyncio
async def test_unwritable_cleanup_diagnostic_emits_sanitized_fallback(
    tmp_path: Path,
) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file blocks directory creation")
    primary = ValueError("primary-provider-secret-canary")
    contexts: list[dict[str, object]] = []
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))
    try:
        result = await asyncio.wait_for(
            cleanup_transport(
                stream_context=None,
                entered=False,
                client=_FailingCloseClient(),
                owns_client=True,
                primary_exception=primary,
                http_error_type=RuntimeError,
                diagnostics_path=blocked_parent / "provider.jsonl",
                provider="test-provider",
            ),
            10,
        )
    finally:
        loop.set_exception_handler(previous_handler)

    assert result is primary
    assert contexts == [
        {
            "exception_type": "builtins.OSError",
            "stage": "client_diagnostic",
            "index": 0,
            "primary_outcome": "error",
        }
    ]
    assert "primary-provider-secret-canary" not in repr(contexts)
    assert "cleanup-secret-canary" not in repr(contexts)


@pytest.mark.asyncio
async def test_detached_cleanup_diagnostics_are_bounded_when_writes_stall(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    release_reports = asyncio.Event()
    started_reports = 0
    contexts: list[dict[str, object]] = []

    async def stalled_diagnostic_write(*_args: object, **_kwargs: object) -> None:
        nonlocal started_reports
        started_reports += 1
        await asyncio.wait_for(release_reports.wait(), 10)

    monkeypatch.setattr(
        stream_diagnostics_module.StreamDiagnostics,
        "record_cleanup_failure",
        stalled_diagnostic_write,
    )
    loop = asyncio.get_running_loop()
    previous_handler = loop.get_exception_handler()
    loop.set_exception_handler(lambda _loop, context: contexts.append(context))

    async def flush_callbacks() -> None:
        for _ in range(2):
            flushed = loop.create_future()
            loop.call_soon(flushed.set_result, None)
            await asyncio.wait_for(flushed, 10)

    class CancelledFailingCloseClient:
        def __init__(self) -> None:
            self.started = asyncio.Event()

        async def aclose(self) -> None:
            self.started.set()
            try:
                await asyncio.wait_for(asyncio.Event().wait(), 10)
            except asyncio.CancelledError:
                raise RuntimeError("cleanup-secret-canary") from None

    try:
        for index in range(17):
            client = CancelledFailingCloseClient()
            owner = asyncio.create_task(
                cleanup_transport(
                    stream_context=None,
                    entered=False,
                    client=client,
                    owns_client=True,
                    primary_exception=None,
                    http_error_type=RuntimeError,
                    diagnostics_path=tmp_path / f"provider-{index}.jsonl",
                    provider="test-provider",
                )
            )
            await asyncio.wait_for(client.started.wait(), 10)
            owner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await asyncio.wait_for(owner, 10)
            await asyncio.wait_for(flush_callbacks(), 10)

        assert started_reports == 16
        assert len(contexts) == 1
        assert contexts[0]["exception_type"] == "builtins.RuntimeError"
        assert contexts[0]["stage"] == "client_diagnostic_dropped"
        assert contexts[0]["index"] == 1
        assert "cleanup-secret-canary" not in repr(contexts)
    finally:
        release_reports.set()
        await asyncio.wait_for(flush_callbacks(), 10)
        loop.set_exception_handler(previous_handler)


@pytest.mark.asyncio
async def test_cleanup_diagnostic_is_secret_safe_and_written_off_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loop_thread = threading.get_ident()
    write_thread: list[int] = []
    write_started = threading.Event()
    real_write = stream_diagnostics_module.write_stream_diagnostic

    def observed_write(path: Path, record: object, **kwargs: object) -> None:
        write_thread.append(threading.get_ident())
        write_started.set()
        real_write(path, record, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        stream_diagnostics_module, "write_stream_diagnostic", observed_write
    )
    diagnostics_path = tmp_path / "provider.jsonl"
    primary = ValueError("primary-provider-secret-canary")

    outcome = await asyncio.wait_for(
        cleanup_transport(
            stream_context=None,
            entered=False,
            client=_FailingCloseClient(),
            owns_client=True,
            primary_exception=primary,
            http_error_type=RuntimeError,
            diagnostics_path=diagnostics_path,
            provider="test-provider",
        ),
        10,
    )

    assert outcome is primary
    assert write_started.is_set()
    assert write_thread and write_thread[0] != loop_thread
    raw_record = diagnostics_path.read_text()
    record = json.loads(raw_record)
    assert "cleanup-secret-canary" not in raw_record
    assert "primary-provider-secret-canary" not in raw_record
    assert record["exception"] == "builtins.RuntimeError"
    assert record["stage"] == "client"
    assert record["index"] == 0
    assert record["primary_outcome"] == "error"


@pytest.mark.parametrize("status_code", [520, 521, 522, 523, 524, 529])
def test_cloudflare_transient_statuses_are_retryable(status_code: int) -> None:
    error = CodexHTTPError("provider failure", status_code=status_code)

    assert retryable_provider_error(error)


def test_wrapped_request_timeout_is_retryable() -> None:
    error = CodexHTTPError("request failed")
    error.__cause__ = TimeoutError()

    assert retryable_provider_error(error)


@pytest.mark.parametrize("status_code", [525, 526, 527, 528, 530, 400, 401, 404])
def test_non_retryable_statuses_stay_non_retryable(status_code: int) -> None:
    error = CodexHTTPError("provider failure", status_code=status_code)

    assert not retryable_provider_error(error)


def test_http_400_retry_hint_does_not_override_terminal_status() -> None:
    error = CodexHTTPError("overloaded", status_code=400)
    error.retryable = True

    assert not retryable_provider_error(error)


@pytest.mark.parametrize("code", ["auth_error", "permission_denied", "context_length_exceeded"])
def test_terminal_code_ignores_retryable_event_data(code: str) -> None:
    error = RuntimeError("terminal provider error")
    error.code = code

    assert not retryable_provider_error(error, {"retryable": True})


def test_http_429_remains_retryable() -> None:
    error = CodexHTTPError("rate limited", status_code=429)

    assert retryable_provider_error(error)


def test_server_error_code_is_retryable() -> None:
    error = RuntimeError("server_error: stream error")
    error.code = "server_error"

    assert retryable_provider_error(error)


@pytest.mark.parametrize(
    "error",
    [
        ErrorInfo("stream_error", "loop wording can change", retryable=True),
        AnthropicStreamError("anthropic wording can change", retryable=True),
        CodexStreamError("codex wording can change", retryable=True),
    ],
)
def test_early_stream_close_is_retryable(error: object) -> None:
    assert retryable_provider_error(error)


def test_other_stream_errors_are_not_retryable() -> None:
    assert not retryable_provider_error(
        ErrorInfo("stream_error", "provider disconnected")
    )


def test_server_error_with_permanent_4xx_status_is_not_retried() -> None:
    error = CodexHTTPError("server_error", status_code=400)
    error.code = "server_error"

    assert not retryable_provider_error(error)


@pytest.mark.parametrize("status_code", [400, 408, 409, 425, 429])
@pytest.mark.parametrize("code", [None, "server_error"])
def test_retry_precedence_for_4xx_statuses(
    status_code: int, code: str | None
) -> None:
    error = CodexHTTPError("provider failure", status_code=status_code)
    error.code = code

    assert retryable_provider_error(error) is (status_code != 400)

