from __future__ import annotations

import asyncio

import pytest

from zeta.protocol.types import StreamEvent, StreamEventType
from zeta.runtime.foreground_events import ForegroundEventPublisher


def _event(index: int) -> StreamEvent:
    return StreamEvent(StreamEventType.MESSAGE_UPDATE, delta=str(index))


async def _source(count: int):
    for index in range(count):
        yield _event(index)


@pytest.mark.asyncio
async def test_closing_wrapper_closes_source_immediately() -> None:
    publisher = ForegroundEventPublisher()
    finalized = asyncio.Event()
    release = asyncio.Event()

    async def source():
        try:
            yield _event(0)
            await asyncio.wait_for(release.wait(), 10)
        finally:
            finalized.set()

    wrapped = publisher.events(source())
    assert (await asyncio.wait_for(wrapped.__anext__(), 10)).delta == "0"
    await asyncio.wait_for(wrapped.aclose(), 10)
    assert finalized.is_set()
    publisher.close()


@pytest.mark.asyncio
async def test_admission_preserves_order_for_primary_and_subscriber() -> None:
    publisher = ForegroundEventPublisher(max_pending=8)
    subscription = publisher.subscribe()

    primary = [event async for event in publisher.events(_source(4))]
    observed = [await asyncio.wait_for(subscription.__anext__(), 10) for _ in range(4)]

    assert [event.delta for event in primary] == ["0", "1", "2", "3"]
    assert [event.delta for event in observed] == ["0", "1", "2", "3"]
    publisher.close()


@pytest.mark.asyncio
async def test_slow_subscriber_never_blocks_producer() -> None:
    publisher = ForegroundEventPublisher(max_pending=1)
    slow = publisher.subscribe()

    primary = [event async for event in publisher.events(_source(3))]

    assert len(primary) == 3
    assert (await asyncio.wait_for(slow.__anext__(), 10)).delta == "0"
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(slow.__anext__(), 10)
    publisher.close()


@pytest.mark.asyncio
async def test_overflow_isolates_slow_subscriber() -> None:
    publisher = ForegroundEventPublisher(max_pending=2)
    slow = publisher.subscribe()
    fast = publisher.subscribe()
    observed: list[StreamEvent] = []

    async def source():
        for index in range(4):
            yield _event(index)
            observed.append(await asyncio.wait_for(fast.__anext__(), 10))

    primary = [event async for event in publisher.events(source())]

    assert len(primary) == 4
    assert [event.delta for event in observed] == ["0", "1", "2", "3"]
    assert [
        (await asyncio.wait_for(slow.__anext__(), 10)).delta
        for _ in range(2)
    ] == ["0", "1"]
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(slow.__anext__(), 10)
    publisher.close()


@pytest.mark.asyncio
async def test_unsubscribe_and_close_wake_waiting_subscribers() -> None:
    publisher = ForegroundEventPublisher()
    subscription = publisher.subscribe()
    waiting = asyncio.create_task(subscription.__anext__())
    asyncio.get_running_loop().call_soon(publisher.unsubscribe, subscription)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(waiting, 10)

    second = publisher.subscribe()
    waiting = asyncio.create_task(second.__anext__())
    asyncio.get_running_loop().call_soon(publisher.close)
    with pytest.raises(StopAsyncIteration):
        await asyncio.wait_for(waiting, 10)
    assert not [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
