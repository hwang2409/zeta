from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from typing import TypeVar

import pytest

from zeta.tui.app import TUIApp

T = TypeVar("T")


async def _cancel_and_drain(
    tasks: list[asyncio.Future[object]],
    *,
    timeout: float = 10,
    what: str = "test tasks",
) -> None:
    unique = list(dict.fromkeys(tasks))
    for task in unique:
        if not task.done():
            task.cancel()
    if not unique:
        return

    done, pending = await asyncio.wait(unique, timeout=timeout)
    await asyncio.gather(*done, return_exceptions=True)
    if pending:
        names = ", ".join(
            sorted(
                task.get_name() if isinstance(task, asyncio.Task) else repr(task)
                for task in pending
            )
        )
        raise AssertionError(f"{what} did not stop within {timeout:g} s: {names}")


async def await_or_fail[T](
    awaitable: Awaitable[T],
    *,
    timeout: float = 10,
    what: str,
    allow_short: bool = False,
) -> T:
    if timeout < 10 and not allow_short:
        raise ValueError("timeout must be at least 10 s")

    future = asyncio.ensure_future(awaitable)
    done, _ = await asyncio.wait([future], timeout=timeout)
    if done:
        return future.result()

    timeout_error = TimeoutError(f"{what} did not finish within {timeout} s")
    try:
        await _cancel_and_drain([future], timeout=10, what=what)
    except AssertionError as cleanup_error:
        timeout_error.add_note(str(cleanup_error))
    try:
        pytest.fail(
            f"hang: {what} did not finish within {timeout} s",
            pytrace=False,
        )
    except pytest.fail.Exception as failure:
        raise failure from timeout_error


class AppOwner:
    def __init__(self, app: TUIApp | None) -> None:
        self._apps = [app] if app is not None else []
        self._tasks: list[asyncio.Task[object]] = []
        self._release: list[Callable[[], object]] = []
        self._contexts = ExitStack()

    def task(self, task: asyncio.Task[T]) -> asyncio.Task[T]:
        self._tasks.append(task)  # type: ignore[arg-type]
        return task

    def release_with(self, callback: Callable[[], object]) -> None:
        self._release.append(callback)

    def enter_context(self, context: AbstractContextManager[T]) -> T:
        return self._contexts.enter_context(context)

    async def close(self) -> None:
        first_error: BaseException | None = None

        async def attempt(action: Callable[[], object]) -> None:
            nonlocal first_error
            try:
                result = action()
                if asyncio.iscoroutine(result):
                    # test-contract: complexity cleanup callbacks own their deadline
                    await result
            except BaseException as exc:  # noqa: BLE001 - cleanup must continue
                if first_error is None:
                    first_error = exc
                else:
                    first_error.add_note(
                        f"additional cleanup error: {type(exc).__name__}: {exc}"
                    )

        for release in self._release:
            await attempt(release)

        tasks = list(self._tasks)
        for app in self._apps:
            # test-contract: seam test cleanup owns the private active provider task.
            if app._active_task is not None:
                tasks.append(app._active_task)
        await attempt(
            lambda: _cancel_and_drain(
                tasks,
                timeout=10,
                what="owned app tasks",
            )
        )
        await attempt(self._contexts.close)
        for app in reversed(self._apps):
            await attempt(app.close)

        if first_error is not None:
            raise first_error


@asynccontextmanager
async def owned_app(app: TUIApp | None = None) -> AsyncIterator[AppOwner]:
    owner = AppOwner(app)
    try:
        yield owner
    except BaseException as primary:
        try:
            await owner.close()
        except BaseException as cleanup:  # noqa: BLE001 - report cleanup second
            primary.add_note(f"cleanup error: {type(cleanup).__name__}: {cleanup}")
            for note in getattr(cleanup, "__notes__", []):
                primary.add_note(note)
        raise
    else:
        await owner.close()
