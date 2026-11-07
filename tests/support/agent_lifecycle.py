from __future__ import annotations

import asyncio

import pytest

from zeta.core.approval import ApprovalPolicy, ApprovalRequest
from zeta.core.store import ConversationStore
from zeta.runtime.loop import AgentLoop


# test-contract: seam observe public delegated registration after its durable child write.
def watch_delegated_registrations(
    monkeypatch: pytest.MonkeyPatch,
    policy: ApprovalPolicy,
    *,
    count: int = 1,
) -> asyncio.Event:
    registered = asyncio.Event()
    original = policy.register_delegated
    registrations = 0

    def register_delegated(
        request: ApprovalRequest,
        store: ConversationStore,
        *,
        child_instance_id: str | None = None,
    ) -> None:
        nonlocal registrations
        original(request, store, child_instance_id=child_instance_id)
        registrations += 1
        if registrations >= count:
            registered.set()

    monkeypatch.setattr(policy, "register_delegated", register_delegated)
    return registered


async def wait_background_owner(loop: AgentLoop) -> None:
    # test-contract: seam no public owner-drain event.
    await asyncio.wait_for(loop._background_owner.wait(), timeout=10)
