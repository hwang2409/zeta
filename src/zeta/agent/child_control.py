"""Explicit child-run cancellation through the tree transition owner."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from ..core.agent_state import LoadedAgentLifecycle, MissingAgentLifecycle
from ..core.child_runs import TERMINAL_AGENT_STATES, read_child_lifecycle
from ..core.store._child_runs import child_run_segments
from ..runtime.offload import await_joined_offload

if TYPE_CHECKING:
    from ..runtime.loop import AgentLoop


class ChildControlError(Exception):
    def __init__(self, code: str, *, detail: str | None = None) -> None:
        super().__init__(code)
        self.code = code
        self.detail = detail


async def cancel_child_run(loop: AgentLoop, run_id: str) -> str:
    store = loop.store
    try:
        segments = child_run_segments(store.session_id, run_id)
    except ValueError:
        raise ChildControlError("child_run_not_found") from None
    path = store.session_dir.absolute()
    for segment in segments:
        path = path / "agents" / str(segment)

    def state() -> str:
        try:
            lifecycle, _ = read_child_lifecycle(store, path)
        except FileNotFoundError:
            raise ChildControlError("child_run_not_found") from None
        if isinstance(lifecycle, MissingAgentLifecycle):
            raise ChildControlError("child_run_not_found")
        if not isinstance(lifecycle, LoadedAgentLifecycle):
            raise ChildControlError("state_unavailable")
        if lifecycle.value.get("handle") != run_id:
            raise ChildControlError("child_run_not_found")
        return lifecycle.value["state"]

    try:
        current = await await_joined_offload(state)
    except asyncio.CancelledError:
        # An interrupted reply must not withdraw cancellation of a live child.
        loop._background_owner.cancel(run_id)
        raise
    if current in TERMINAL_AGENT_STATES:
        return current
    outcome = await loop._background_owner.cancel_and_wait(run_id)
    if outcome.status == "confirmed":
        return "cancelling"
    current = await await_joined_offload(state)
    if current in TERMINAL_AGENT_STATES:
        return current
    if outcome.status == "failed" or not loop._background_owner.owns_running(run_id):
        raise ChildControlError("state_unavailable")
    raise ChildControlError("operation_busy")
