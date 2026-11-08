"""Lifecycle-owned session grant commit and publication."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal

from ...core.approval import ApprovalPolicy, ApprovalRequest
from ...core.store import ConversationStore
from ..connections import Admission
from ..protocol import EventDescriptor

ApprovalKey = str | tuple[str, str]


@dataclass(frozen=True, slots=True)
class _ApprovalTarget:
    key: ApprovalKey
    store: ConversationStore
    store_request_id: str
    request: ApprovalRequest | None = None
    history_store: ConversationStore | None = None


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    target: _ApprovalTarget
    wire_id: str
    generation: int
    token: int
    admitted: asyncio.Future[dict[str, str]] = field(compare=False, repr=False)
    phase: Literal["starting", "pending", "settling"] = "starting"
    disposition: str = "active"
    request: ApprovalRequest | None = None
    grant: ApprovalRequest | None = None
    grant_committed: bool = False
    grant_event_published: bool = False
    history_published: bool = False
    published: bool = False
    settlement: asyncio.Task[object] | None = None
    publication_owner: asyncio.Task[object] | None = None
    live_end: Admission | None = None
    terminal: tuple[EventDescriptor, str | None, dict[str, object]] | None = None
    terminal_result: Admission | None = None
    terminal_watchdog: asyncio.Task[None] | None = None


class GrantPublication:
    """Commit one grant and publish its first event."""

    def __init__(
        self,
        policy: ApprovalPolicy,
        publish: Callable[[str], Awaitable[None]] | None,
    ) -> None:
        self._policy = policy
        self._publish = publish

    def commit(self, request: ApprovalRequest) -> tuple[str, bool]:
        before = self._policy.session_allow
        rule = self._policy.remember_allow(request)
        return str(rule), rule not in before

    async def publish(self, rule: str) -> None:
        if self._publish is not None:
            await self._publish(rule)

    async def finish(
        self, request: ApprovalRequest, committed: bool, event_published: bool
    ) -> tuple[bool, bool]:
        """Commit a grant and publish its first event, exactly once."""
        if committed:
            return committed, event_published
        rule, changed = self.commit(request)
        if changed and not event_published:
            await self.publish(rule)
        return True, True


async def finish_grant(
    publication: GrantPublication,
    request: ApprovalRequest,
    committed: bool,
    event_published: bool,
) -> tuple[bool, bool, bool]:
    """Finish grant publication while deferring cancellation of the owner."""
    operation = asyncio.create_task(
        publication.finish(request, committed, event_published)
    )
    cancelled = False
    try:
        committed, event_published = await asyncio.shield(operation)
    except asyncio.CancelledError:
        cancelled = True
        committed, event_published = await asyncio.shield(operation)
    return committed, event_published, cancelled
