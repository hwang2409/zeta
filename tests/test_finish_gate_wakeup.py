from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from tests.support.fake_backend import FakeBackend
from tests.support.owned import await_or_fail
from zeta.agent.background_owner import BackgroundAgentOwner
from zeta.agent.finish_gate import await_owned_work
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop
from zeta.skills import SkillCatalog


@pytest.mark.asyncio
async def test_owned_work_wait_wakes_when_last_owned_child_unregisters(
    tmp_path: Path,
) -> None:
    store = ConversationStore(tmp_path)
    owner = BackgroundAgentOwner(store, finish_gate_timeout=600)
    child = AgentLoop(
        FakeBackend([]),
        store,
        skill_catalog=SkillCatalog.empty(),
        agent_instance_id="root:1",
        background_owner=owner,
    )
    channel = owner.conversation_channel
    channel.register_loop("root:1")
    grandchild_done = asyncio.Event()
    # The parent run must be registered, or the grandchild is canceled at once.
    child_run = asyncio.create_task(asyncio.wait_for(grandchild_done.wait(), 10))
    owner.register("root:1", child_run.cancel, child_run)
    grandchild = asyncio.create_task(asyncio.wait_for(grandchild_done.wait(), 10))
    owner.register(
        "root:1:1", grandchild.cancel, grandchild, parent_instance_id="root:1"
    )
    parked = asyncio.Queue[None]()
    channel_wait = channel.wait

    async def wait_and_report(instance_id: str) -> None:
        parked.put_nowait(None)
        await channel_wait(instance_id)

    channel.wait = wait_and_report  # type: ignore[method-assign]
    waiter = asyncio.create_task(await_owned_work(child))
    try:
        await await_or_fail(parked.get(), what="first owned-work wait")
        # Real order: the completion notice wakes the parent while the
        # grandchild is still registered, and only then does it unregister.
        channel.wake("root:1")
        await await_or_fail(parked.get(), what="re-armed owned-work wait")
        assert not grandchild.done()
        owner.unregister("root:1:1")
        await await_or_fail(
            waiter, what="await_owned_work after the last owned child unregistered"
        )
    finally:
        grandchild_done.set()
        waiter.cancel()
        await asyncio.gather(waiter, grandchild, child_run, return_exceptions=True)
        owner.unregister("root:1")
        await await_or_fail(child.close(), what="child loop close")
        store.close()
