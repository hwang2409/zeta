import asyncio

import pytest

from tests.support.owned import ResourceOwner, await_or_fail, owned_app, owned_resources
from zeta.core.store import ConversationStore


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
async def test_await_or_fail_has_no_short_deadline_escape() -> None:
    async def value() -> str:
        return "done"

    explicit = value()
    unpacked = value()
    try:
        with pytest.raises(TypeError, match="allow_short"):
            await await_or_fail(
                explicit, timeout=10, what="value", allow_short=(not False)
            )
        with pytest.raises(TypeError, match="allow_short"):
            await await_or_fail(
                unpacked,
                timeout=10,
                what="value",
                **{"allow_short": True},  # noqa: PIE804 - regression covers unpacking
            )
    finally:
        explicit.close()
        unpacked.close()


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

    async with owned_resources() as owner:
        task = owner.task(
            asyncio.create_task(ignore_cancel(), name="cancel-ignoring-task")
        )
        owner.release_with(release.set)
        await asyncio.wait_for(started.wait(), timeout=10)
        with pytest.raises(
            pytest.fail.Exception, match="hang: worker did not finish"
        ) as caught:
            await await_or_fail(task, what="worker")

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
    assert any("owned test tasks" in note for note in notes)
    assert any("ignores-cancel" in note for note in notes)


class _AsyncResource:
    def __init__(
        self,
        *,
        failure: str | None = None,
        entered: asyncio.Event | None = None,
        release: asyncio.Event | None = None,
    ) -> None:
        self.closed = False
        self.close_count = 0
        self.failure = failure
        self.entered = entered
        self.release = release

    async def close(self) -> None:
        self.close_count += 1
        if self.entered is not None:
            self.entered.set()
        if self.release is not None:
            # test-contract: seam owner applies the public cleanup deadline
            await self.release.wait()
        self.closed = True
        if self.failure is not None:
            raise RuntimeError(self.failure)


class _FailingStore:
    def __init__(self, failure: str) -> None:
        self.failure = failure
        self.closed = False

    def close(self) -> None:
        self.closed = True
        raise RuntimeError(self.failure)


class _FailingCancelFuture(asyncio.Future[object]):
    def cancel(self, msg: object = None) -> bool:
        super().cancel(msg)
        raise RuntimeError("task cleanup failed")


def _assert_store_fd_closed(store: ConversationStore, fd: int) -> None:
    assert store.directory_fd == -1
    with pytest.raises(OSError):
        import os

        os.fstat(fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("body_exit", ["success", "assertion", "cancellation"])
async def test_owned_resources_cleans_every_resource_on_each_body_exit(
    tmp_path, body_exit: str
) -> None:
    gate = asyncio.Event()
    resource = _AsyncResource()
    task_started = asyncio.Event()

    async def worker() -> None:
        task_started.set()
        # test-contract: complexity owner cancellation is the behavior under test
        await asyncio.Event().wait()

    store: ConversationStore | None = None
    store_fd: int | None = None
    task: asyncio.Task[None] | None = None

    async def run() -> None:
        nonlocal store, store_fd, task
        async with owned_resources() as owner:
            store = owner.store(ConversationStore(tmp_path / body_exit))
            store_fd = store.directory_fd
            owner.release_with(gate.set)
            resource_value = owner.async_resource(resource, what="test resource")
            assert resource_value is resource
            task = owner.task(asyncio.create_task(worker()))
            await asyncio.wait_for(task_started.wait(), 10)
            if body_exit == "assertion":
                raise AssertionError("body failed")
            if body_exit == "cancellation":
                raise asyncio.CancelledError

    if body_exit == "assertion":
        with pytest.raises(AssertionError, match="body failed"):
            await run()
    elif body_exit == "cancellation":
        with pytest.raises(asyncio.CancelledError):
            await run()
    else:
        await run()

    assert store is not None and store_fd is not None
    assert task is not None and task.done()
    assert gate.is_set()
    assert resource.closed
    _assert_store_fd_closed(store, store_fd)


@pytest.mark.asyncio
async def test_owned_resources_finishes_cleanup_when_cancelled_at_cleanup_await(
    tmp_path,
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    gate = asyncio.Event()
    resource = _AsyncResource(entered=entered, release=release)
    store: ConversationStore | None = None
    fd: int | None = None

    async def run() -> None:
        nonlocal store, fd
        async with owned_resources() as owner:
            store = owner.store(ConversationStore(tmp_path))
            fd = store.directory_fd
            owner.async_resource(resource, what="blocked resource")
            owner.release_with(gate.set)

    async with owned_resources() as outer_owner:
        task = outer_owner.task(asyncio.create_task(run()))
        await asyncio.wait_for(entered.wait(), 10)
        task.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(task, 10)

        assert task.done()
        assert gate.is_set()
        assert resource.closed
        assert store is not None and fd is not None
        _assert_store_fd_closed(store, fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_at", ["release", "task", "async"])
async def test_owned_resources_finishes_when_cancelled_at_each_cleanup_await(
    tmp_path, cancel_at: str
) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()
    gate = asyncio.Event()
    resource = _AsyncResource(
        entered=entered if cancel_at == "async" else None,
        release=release if cancel_at == "async" else None,
    )
    store: ConversationStore | None = None
    fd: int | None = None
    worker: asyncio.Task[None] | None = None

    async def release_gate() -> None:
        gate.set()
        if cancel_at == "release":
            entered.set()
            # test-contract: seam owner cleanup is bounded by the outer public wait
            await release.wait()

    async def work() -> None:
        try:
            # test-contract: complexity owner cancellation is the behavior under test
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            if cancel_at == "task":
                entered.set()
                # test-contract: seam owner cleanup is bounded by the outer public wait
                await release.wait()
            raise

    async def run() -> None:
        nonlocal store, fd, worker
        async with owned_resources() as owner:
            store = owner.store(ConversationStore(tmp_path / cancel_at))
            fd = store.directory_fd
            owner.async_resource(resource, what="resource")
            worker = owner.task(asyncio.create_task(work()))
            owner.release_with(release_gate)

    async with owned_resources() as outer_owner:
        cleanup = outer_owner.task(asyncio.create_task(run()))
        await asyncio.wait_for(entered.wait(), 10)
        cleanup.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(cleanup, 10)

        assert cleanup.done()
        assert worker is not None and worker.done()
        assert gate.is_set()
        assert resource.closed
        assert store is not None and fd is not None
        _assert_store_fd_closed(store, fd)


@pytest.mark.asyncio
@pytest.mark.parametrize("failing_step", ["release", "task", "async", "store"])
async def test_owned_resources_attempts_every_step_after_cleanup_failure(
    tmp_path, failing_step: str
) -> None:
    resource = _AsyncResource()
    gate = asyncio.Event()
    store: ConversationStore | None = None
    fd: int | None = None
    task: asyncio.Task[bool] | None = None

    with pytest.raises(RuntimeError, match=f"{failing_step} cleanup failed"):
        async with owned_resources() as owner:
            store = owner.store(ConversationStore(tmp_path / failing_step))
            fd = store.directory_fd
            owner.async_resource(resource, what="test resource")
            # test-contract: complexity owner cancellation is the behavior under test
            task = owner.task(asyncio.create_task(asyncio.Event().wait()))
            owner.release_with(gate.set)
            if failing_step == "release":
                owner.release_with(
                    lambda: (_ for _ in ()).throw(
                        RuntimeError("release cleanup failed")
                    )
                )
            elif failing_step == "task":
                owner.task(_FailingCancelFuture())
            elif failing_step == "async":
                owner.async_resource(
                    _AsyncResource(failure="async cleanup failed"),
                    what="failing resource",
                )
            else:
                owner.store(_FailingStore("store cleanup failed"))

    assert task is not None and task.done()
    assert gate.is_set()
    assert resource.closed
    assert store is not None and fd is not None
    _assert_store_fd_closed(store, fd)


@pytest.mark.asyncio
async def test_owned_resources_keeps_first_cleanup_error_and_notes_later_errors() -> (
    None
):
    with pytest.raises(RuntimeError, match="first failure") as caught:
        async with owned_resources() as owner:
            owner.store(_FailingStore("last failure"))
            owner.async_resource(
                _AsyncResource(failure="middle failure"), what="failing resource"
            )
            owner.release_with(
                lambda: (_ for _ in ()).throw(RuntimeError("first failure"))
            )

    assert getattr(caught.value, "__notes__", []) == [
        "additional cleanup error: RuntimeError: middle failure",
        "additional cleanup error: RuntimeError: last failure",
    ]


@pytest.mark.asyncio
async def test_resource_owner_duplicate_registration_and_close_are_idempotent() -> None:
    owner = ResourceOwner()
    resource = _AsyncResource()
    owner.async_resource(resource, what="resource")
    owner.async_resource(resource, what="resource")

    await owner.close()
    await owner.close()

    assert resource.closed
    assert resource.close_count == 1
