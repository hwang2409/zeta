"""Session transcript request bodies."""

from __future__ import annotations

from typing import Any

from ...core.session import SessionError
from ...core.transcript_recovery.runtime import (
    TranscriptRepairRequiresUpgrade,
    repair_transcript_for_runtime,
)
from ..protocol import ErrorDescriptor, ProtocolError
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
