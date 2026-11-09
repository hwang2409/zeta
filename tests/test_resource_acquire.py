from __future__ import annotations

import asyncio
import errno
import os
import threading
from contextlib import suppress
from pathlib import Path
from typing import Any, Self

import pytest

from zeta.protocol.types import ToolCall
from zeta.skills import SkillCatalog
from zeta.tools._shared.shell import run_shell_macro
from zeta.tools._shared.shell_cwd import bind_shell_cwd_async
from zeta.tools.registry import ToolRegistry


class _TrackedResource:
    def __init__(self, path: Path) -> None:
        self.handle = path.open("w+b")
        self.fd = self.handle.fileno()
        self.close_calls = 0
        self.closed = threading.Event()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.handle, name)

    def close(self) -> None:
        self.close_calls += 1
        self.handle.close()
        self.closed.set()


_CASES = (
    pytest.param("helper", "success", id="helper-success"),
    pytest.param("helper", "single_cancel", id="helper-single-cancel"),
    pytest.param("helper", "repeated_cancel", id="helper-repeated-cancel"),
    pytest.param("helper", "thread_raises", id="helper-thread-raises"),
    pytest.param(
        "helper",
        "cancel_after_finish",
        id="helper-cancel-after-thread-finished",
    ),
    pytest.param("foreground_macro", "repeated_cancel", id="foreground-macro"),
    pytest.param("background_macro", "repeated_cancel", id="background-macro"),
    pytest.param("cwd_bind", "repeated_cancel", id="cwd-bind"),
)


@pytest.mark.parametrize(("route", "scenario"), _CASES)
async def test_off_loop_resource_has_one_owner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    route: str,
    scenario: str,
) -> None:
    started = threading.Event()
    finished = threading.Event()
    release = threading.Event()
    resources: list[_TrackedResource] = []
    registry: ToolRegistry | None = None
    task: asyncio.Task[Any] | None = None
    event_loop = asyncio.get_running_loop()

    def acquire(name: str = "resource") -> _TrackedResource:
        resource = _TrackedResource(tmp_path / name)
        resources.append(resource)
        started.set()
        if scenario != "cancel_after_finish":
            assert release.wait(10)
        if scenario == "thread_raises":
            resource.close()
            raise RuntimeError("acquire failed")
        finished.set()
        return resource

    try:
        if route == "helper":
            from zeta.core.thread_call import run_owned_thread_call

            if scenario == "cancel_after_finish":
                run_in_executor = event_loop.run_in_executor

                def run_controlled(_executor: object, function: Any) -> Any:
                    monkeypatch.setattr(event_loop, "run_in_executor", run_in_executor)
                    function()
                    worker = event_loop.create_future()
                    worker.set_result(None)
                    return worker

                # test-contract: seam finish the worker before its result is used.
                monkeypatch.setattr(event_loop, "run_in_executor", run_controlled)
            task = asyncio.create_task(
                run_owned_thread_call(
                    acquire, _TrackedResource.close, close_is_thread_safe=True
                )
            )
            if scenario == "cancel_after_finish":
                await asyncio.sleep(0)
                task.cancel()
        else:
            registry = ToolRegistry(tmp_path, skill_catalog=SkillCatalog.empty())
            if route == "cwd_bind":
                import zeta.tools._shared.shell_cwd as shell_cwd_module

                def bind(*_args: object) -> tuple[str, int, None]:
                    resource = acquire("cwd")
                    return str(tmp_path), resource.fd, None

                original_acquire = shell_cwd_module.run_owned_thread_call

                async def observed_acquire(
                    acquirer: Any, close: Any, **kwargs: bool
                ) -> Any:
                    def observed_close(result: Any) -> None:
                        close(result)
                        resource = next(
                            item for item in resources if item.fd == result[1]
                        )
                        resource.close_calls += 1
                        resource.closed.set()

                    return await original_acquire(acquirer, observed_close, **kwargs)

                monkeypatch.setattr(
                    shell_cwd_module,
                    "run_owned_thread_call",
                    observed_acquire,
                )

                # test-contract: seam hold and observe the real public cwd binder's worker.
                monkeypatch.setattr(shell_cwd_module, "_bind_shell_cwd", bind)
                task = asyncio.create_task(bind_shell_cwd_async(registry, None))
            else:
                background = route == "background_macro"
                open_calls = 0

                def open_log(_path: str | Path) -> _TrackedResource:
                    nonlocal open_calls
                    open_calls += 1
                    if background and open_calls == 1:
                        resource = _TrackedResource(tmp_path / "initial.log")
                        resources.append(resource)
                        return resource
                    return acquire("owned.log")

                monkeypatch.setattr(registry.background_tasks, "open_log", open_log)
                task = asyncio.create_task(
                    run_shell_macro(
                        registry,
                        ToolCall("ownership", "bash", {"command": "true"}),
                        tmp_path / "macro.log",
                        stream_sink=lambda _event: None,
                        lifecycle_sink=lambda _kind: None,
                        background=background,
                    )
                )

        if scenario == "success":
            release.set()
            result = await asyncio.wait_for(task, 10)
            result.close()
        elif scenario == "thread_raises":
            release.set()
            with pytest.raises(RuntimeError, match="acquire failed"):
                await asyncio.wait_for(task, 10)
        else:
            if scenario == "cancel_after_finish":
                assert finished.is_set()
            else:
                assert await asyncio.to_thread(started.wait, 10)
                task.cancel()
            if scenario == "repeated_cancel":
                await asyncio.sleep(0)
                task.cancel()
                await asyncio.sleep(0)
                task.cancel()
            release.set()
            if route in {"foreground_macro", "background_macro"}:
                result = await asyncio.wait_for(task, 10)
                assert result.is_canceled
            else:
                with pytest.raises(asyncio.CancelledError):
                    await asyncio.wait_for(task, 10)

        assert resources
        for resource in resources:
            assert await asyncio.to_thread(resource.closed.wait, 10)
            assert resource.close_calls == 1
            with pytest.raises(OSError) as error:
                os.fstat(resource.fd)
            assert error.value.errno == errno.EBADF
    finally:
        release.set()
        if task is not None and not task.done():
            task.cancel()
            with suppress(BaseException):
                await asyncio.wait_for(task, 10)
        for resource in resources:
            if not resource.handle.closed:
                with suppress(OSError):
                    resource.handle.close()
        if registry is not None:
            await registry.close()
