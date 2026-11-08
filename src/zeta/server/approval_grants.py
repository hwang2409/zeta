"""Lifecycle-owned session grant commit and publication."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..core.approval import ApprovalPolicy, ApprovalRequest


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
