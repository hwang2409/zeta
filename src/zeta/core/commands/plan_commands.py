"""Shared plan-mode slash command behavior."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from ...model_input import ModelInputEnvelope
from ...protocol.types import MessageOrigin


class PlanModeSession(Protocol):
    plan_mode: bool
    active: bool
    pending_approvals: object

    def approval_pending_now(self) -> Awaitable[bool]: ...

    def set_plan_mode(self, enabled: bool) -> Awaitable[None] | None: ...

    def invalidate_prompt(self) -> None: ...

    def plan_mode_background_blocker(self) -> str | None: ...


async def run_plan_command(
    session: PlanModeSession, args: str
) -> str | ModelInputEnvelope:

    requested = args.strip()
    normalized = requested.lower()
    if not requested:
        return await _set_plan_mode(session, not session.plan_mode)
    if normalized == "off":
        return await _set_plan_mode(session, False)
    if normalized == "on":
        return await _set_plan_mode(session, True)
    if normalized == "toggle":
        return await _set_plan_mode(session, not session.plan_mode)
    if session.active or await session.approval_pending_now():
        return (
            "plan mode unchanged: cannot change plan mode while a turn or "
            "approval is active"
        )
    if not session.plan_mode:
        blocked = session.plan_mode_background_blocker()
        if blocked is not None:
            return blocked
        await _change_plan_mode(session, True)
    return ModelInputEnvelope(
        requested, f"/plan {requested}", MessageOrigin.SLASH_EXPANSION
    )


async def run_implement_command(
    session: PlanModeSession, args: str
) -> str | ModelInputEnvelope:

    if args:
        return "implement unchanged: /implement does not accept arguments"
    if not session.plan_mode:
        return "implement unavailable: plan mode is off"
    if session.active or await session.approval_pending_now():
        return (
            "implement unavailable: cannot leave plan mode while a turn or "
            "approval is active"
        )
    await _change_plan_mode(session, False)
    return ModelInputEnvelope(
        "implement the plan you proposed above",
        "/implement",
        MessageOrigin.SLASH_EXPANSION,
    )


async def _set_plan_mode(session: PlanModeSession, enabled: bool) -> str:
    if enabled == session.plan_mode:
        return f"plan mode: {'on' if enabled else 'off'}"
    if session.active or await session.approval_pending_now():
        return (
            "plan mode unchanged: cannot change plan mode while a turn or "
            "approval is active"
        )
    if enabled:
        blocked = session.plan_mode_background_blocker()
        if blocked is not None:
            return blocked
    await _change_plan_mode(session, enabled)
    return (
        "plan mode: on (read-only tools; deliver the plan as your answer)"
        if enabled
        else "plan mode: off"
    )


async def _change_plan_mode(session: PlanModeSession, enabled: bool) -> None:
    persisted = session.set_plan_mode(enabled)
    if persisted is not None:
        await persisted
    session.invalidate_prompt()
