from __future__ import annotations

import asyncio
import threading
from types import SimpleNamespace

import pytest

from zeta.agent.conversation_channel import ConversationChannel
from zeta.agent.pending_delivery import (
    NotificationInput,
    ProducerResult,
    PromptInput,
    StepDeliveryFlow,
)
from zeta.protocol.types import MessageOrigin


class _Store:
    def __init__(self) -> None:
        self.calls: list[tuple[object, int]] = []
        self.installed = False

    def _commit_delivery_input(self, item: object) -> ProducerResult:
        self.calls.append((item, threading.get_ident()))
        return ProducerResult((), True, 1)

    async def _to_thread_durable(self, callback, *args):
        result = await asyncio.to_thread(callback, *args)
        self.installed = True
        return result


@pytest.mark.asyncio
async def test_step_delivery_produce_validates_and_commits_before_edge() -> None:
    store = _Store()
    edges: list[str] = []
    flow = StepDeliveryFlow(store, wake_sink=lambda source: edges.append(source))

    result = await flow.produce(
        PromptInput("keep going", MessageOrigin.AGENT_SEND, None)
    )

    assert result.changed
    assert store.calls[0][1] != threading.get_ident()
    assert store.installed
    assert edges == ["prompt"]
    with pytest.raises(ValueError):
        await flow.produce(NotificationInput("task_exit", {}, dedupe_key=None))


@pytest.mark.asyncio
async def test_delivery_registry_routes_same_and_cross_loop() -> None:
    channel = ConversationChannel(SimpleNamespace(), SimpleNamespace())
    local = StepDeliveryFlow(_Store())
    channel.register_delivery_flow("local", local)
    item = PromptInput("local", MessageOrigin.AGENT_SEND, None)
    assert (await channel.produce_to("local", item)).changed

    ready = threading.Event()
    stop = threading.Event()
    holder: dict[str, object] = {}

    def run_loop() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        async def serve() -> None:
            remote = StepDeliveryFlow(_Store())
            holder.update(loop=loop, flow=remote)
            ready.set()
            await asyncio.to_thread(stop.wait)

        loop.run_until_complete(serve())
        loop.close()

    thread = threading.Thread(target=run_loop)
    thread.start()
    assert ready.wait(2)
    channel.register_delivery_flow("remote", holder["flow"], loop=holder["loop"])
    try:
        assert (await channel.produce_to("remote", item)).changed
        await channel.unregister_delivery_flow("remote")
    finally:
        stop.set()
        thread.join(2)
    with pytest.raises(LookupError):
        await channel.produce_to("remote", item)
