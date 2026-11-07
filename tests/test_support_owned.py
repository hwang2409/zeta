import asyncio

import pytest

from tests.support.owned import await_or_fail, owned_app


@pytest.mark.asyncio
async def test_await_or_fail_returns_value() -> None:
    async def value() -> str:
        return "done"

    assert await await_or_fail(value(), what="value") == "done"


@pytest.mark.asyncio
async def test_await_or_fail_rejects_short_deadline() -> None:
    async def value() -> str:
        return "done"

    coroutine = value()
    try:
        with pytest.raises(ValueError, match="at least 10 s"):
            await await_or_fail(coroutine, timeout=0, what="value")
    finally:
        coroutine.close()


@pytest.mark.asyncio
async def test_await_or_fail_names_wait_that_never_finishes() -> None:
    never = asyncio.Event()

    with pytest.raises(pytest.fail.Exception, match="hang: readiness did not finish"):
        await await_or_fail(never.wait(), what="readiness")


@pytest.mark.asyncio
async def test_await_or_fail_bounds_cancel_ignoring_task() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def ignore_cancel() -> None:
        started.set()
        while not release.is_set():
            try:
                # test-contract: complexity simulates a task that ignores cancellation
                await release.wait()
            except asyncio.CancelledError:
                continue

    task = asyncio.create_task(ignore_cancel(), name="cancel-ignoring-task")
    try:
        await asyncio.wait_for(started.wait(), timeout=10)
        with pytest.raises(
            pytest.fail.Exception, match="hang: worker did not finish"
        ) as caught:
            await await_or_fail(task, what="worker")
    finally:
        release.set()
        await asyncio.wait_for(task, timeout=10)

    cause = caught.value.__cause__
    assert isinstance(cause, TimeoutError)
    assert any("cancel-ignoring-task" in note for note in cause.__notes__)


@pytest.mark.asyncio
async def test_owned_app_keeps_body_failure_primary_when_task_will_not_cancel() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    task: asyncio.Task[None] | None = None

    async def ignore_cancel() -> None:
        started.set()
        while not release.is_set():
            try:
                # test-contract: complexity simulates a task that ignores cancellation
                await release.wait()
            except asyncio.CancelledError:
                continue

    try:
        with pytest.raises(RuntimeError, match="body failed") as caught:
            async with owned_app() as owner:
                task = owner.task(
                    asyncio.create_task(ignore_cancel(), name="ignores-cancel")
                )
                await asyncio.wait_for(started.wait(), timeout=10)
                raise RuntimeError("body failed")
    finally:
        release.set()
        if task is not None:
            await asyncio.wait_for(task, timeout=10)

    notes = getattr(caught.value, "__notes__", [])
    assert any("owned app tasks" in note for note in notes)
    assert any("ignores-cancel" in note for note in notes)
