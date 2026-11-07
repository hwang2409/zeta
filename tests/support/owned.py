from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractContextManager, ExitStack, asynccontextmanager
from typing import Any, TypeVar

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
    # test-contract: seam no public task-drain seam exposes only completed tasks
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
) -> T:
    if timeout < 10:
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


class ResourceOwner:
    def __init__(self) -> None:
        self._stores: list[Any] = []
        self._async_resources: list[tuple[Any, str]] = []
        self._tasks: list[asyncio.Future[object]] = []
        self._release: list[Callable[[], object]] = []
        self._contexts = ExitStack()
        self._closed = False

    def store(self, resource: T) -> T:
        if not any(resource is item for item in self._stores):
            self._stores.append(resource)
        return resource

    def async_resource(self, resource: T, *, what: str) -> T:
        if not any(resource is item for item, _ in self._async_resources):
            self._async_resources.append((resource, what))
        return resource

    def task(self, task: asyncio.Future[T]) -> asyncio.Future[T]:
        if not any(task is item for item in self._tasks):
            self._tasks.append(task)  # type: ignore[arg-type]
        return task

    def release_with(self, callback: Callable[[], object]) -> None:
        self._release.append(callback)

    def enter_context(self, context: AbstractContextManager[T]) -> T:
        return self._contexts.enter_context(context)

    async def _attempt(
        self,
        action: Callable[[], object],
        first_error: BaseException | None,
    ) -> BaseException | None:
        try:
            result = action()
            if asyncio.iscoroutine(result):
                # test-contract: complexity cleanup callbacks own their deadline
                await result
        except BaseException as exc:  # noqa: BLE001 - cleanup must continue
            if first_error is None:
                return exc
            first_error.add_note(
                f"additional cleanup error: {type(exc).__name__}: {exc}"
            )
        return first_error

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        first_error: BaseException | None = None
        for release in reversed(self._release):
            first_error = await self._attempt(release, first_error)
        first_error = await self._attempt(
            lambda: _cancel_and_drain(self._tasks, what="owned test tasks"),
            first_error,
        )
        for resource, what in reversed(self._async_resources):
            first_error = await self._attempt(
                lambda resource=resource, what=what: await_or_fail(
                    resource.close(), what=f"closing {what}"
                ),
                first_error,
            )
        first_error = await self._attempt(self._contexts.close, first_error)
        for resource in reversed(self._stores):
            first_error = await self._attempt(resource.close, first_error)
        if first_error is not None:
            raise first_error


async def _finish_cleanup(owner: ResourceOwner) -> None:
    cleanup = asyncio.create_task(owner.close(), name="owned-resource-cleanup")
    try:
        # test-contract: complexity cleanup shields its own bounded resource closes
        await asyncio.shield(cleanup)
    except asyncio.CancelledError:
        # A second cancellation must not strand resources midway through cleanup.
        # test-contract: complexity cleanup must finish after caller cancellation
        await asyncio.shield(cleanup)
        raise


@asynccontextmanager
async def owned_resources() -> AsyncIterator[ResourceOwner]:
    owner = ResourceOwner()
    try:
        yield owner
    except BaseException as primary:
        try:
            await _finish_cleanup(owner)
        except BaseException as cleanup:  # noqa: BLE001 - report cleanup second
            primary.add_note(f"cleanup error: {type(cleanup).__name__}: {cleanup}")
            for note in getattr(cleanup, "__notes__", []):
                primary.add_note(note)
        raise
    else:
        await _finish_cleanup(owner)


@asynccontextmanager
async def owned_app(app: TUIApp | None = None) -> AsyncIterator[ResourceOwner]:
    async with owned_resources() as owner:
        if app is not None:
            owner.async_resource(app, what="TUI app")
        try:
            yield owner
        finally:
            if app is not None and app._active_task is not None:
                # test-contract: seam test cleanup owns the private active provider task.
                owner.task(app._active_task)
