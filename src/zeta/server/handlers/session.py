"""Session mutation, tree, and transcript repair request bodies."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ...core.checkpoints import list_branches_from_entries
from ...core.session import SessionError
from ...core.transcript_recovery.runtime import (
    TranscriptRepairRequiresUpgrade,
    repair_transcript_for_runtime,
)
from ..protocol import ErrorDescriptor, ProtocolError
from ..runtime import ServerRuntime
from ..session_mutation import rename_session
from . import validation


async def repair_transcript(self, params: dict[str, Any]) -> dict[str, object]:
    runtime = self.server.runtime
    await self.server.control.require_idle()
    if params.get("confirm") is not True:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "repair_transcript requires confirm=true",
        )
    session_id = validation.required_string(params, "session_id")
    child_run_id = params.get("child_run_id")
    if child_run_id is not None and not isinstance(child_run_id, str):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "child_run_id must be a string",
        )
    try:
        result = await repair_transcript_for_runtime(
            runtime.manager,
            session_id,
            child_run_id,
            parent_store=(
                runtime.opened.store if runtime.opened is not None else None
            ),
            memory_reconciler=(
                runtime.loop.memory_reconciler
                if runtime.loop is not None
                else None
            ),
        )
    except TranscriptRepairRequiresUpgrade:
        raise
    except (ValueError, SessionError) as exc:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, str(exc)) from exc
    return {
        "session_id": session_id,
        "child_run_id": child_run_id,
        "quarantine_id": result.quarantine_id,
        "retained_last_sequence": result.retained_last_sequence,
        "memory_review_required": result.memory_review_required,
        "overlapping_memory_versions": list(result.overlapping_memory_versions),
    }


async def delete_session(
    runtime: ServerRuntime,
    session_id: str,
    require_idle: Callable[[], Awaitable[None]],
) -> dict[str, object]:
    session_id = await asyncio.to_thread(runtime.manager.resolve_id, session_id)
    await require_idle()
    if runtime.opened is not None and session_id == runtime.session_id:
        raise ProtocolError(
            ErrorDescriptor.ACTIVE_SESSION,
            "select another session before deleting this session",
        )
    await asyncio.to_thread(runtime.manager.delete, session_id)
    return {"session_id": session_id}


async def rename(
    runtime: ServerRuntime, session_id: str, name: object
) -> dict[str, object]:
    session_id = await asyncio.to_thread(runtime.manager.resolve_id, session_id)
    if not isinstance(name, str):
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "name must be a string")
    metadata = await rename_session(runtime, session_id, name)
    return {"session": metadata.to_dict()}


def tree(runtime: ServerRuntime) -> dict:
    store = runtime.opened.store
    # Read resident entries once as a shallow snapshot instead of the
    # deep-copying ``entries`` property: this read runs off the loop (via
    # run_read) and only inspects ids/parents, so a per-row copy of a 17k-row
    # branch is wasted.
    entries = tuple(store.entry(ref) for ref in store.branch_refs())
    children: dict[str, int] = {}
    for entry in entries:
        if entry.parent_id:
            children[entry.parent_id] = children.get(entry.parent_id, 0) + 1
    depths: dict[str, int] = {}
    for entry in entries:
        depths[entry.id] = depths.get(entry.parent_id, 0) + (
            children.get(entry.parent_id, 0) > 1
        )
    return {
        "branches": [
            {
                "id": branch.head.id,
                "label": branch.preview,
                "depth": depths[branch.head.id],
                "current": branch.is_current,
            }
            for branch in list_branches_from_entries(entries)
        ]
    }
