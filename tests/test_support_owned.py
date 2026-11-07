import asyncio

import pytest

from tests.support.owned import owned_app


@pytest.mark.asyncio
async def test_owned_app_keeps_body_failure_primary_when_task_will_not_cancel() -> None:
    started = asyncio.Event()
    release = asyncio.Event()
    task: asyncio.Task[None] | None = None

    async def ignore_cancel() -> None:
        started.set()
        while not release.is_set():
            try:
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
