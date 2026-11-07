"""Runtime adapters for explicit transcript repair."""

from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from ..store import ConversationStore
from ..store._child_runs import child_run_segments
from . import RepairResult, RepairTarget

if TYPE_CHECKING:
    from ..session import SessionManager


def repair_command_for_child_path(path: Path) -> str:
    """Build the explicit repair command for one child session path."""
    parts = path.parts
    try:
        sessions = parts.index("sessions")
        root = parts[sessions + 1]
        relative = parts[sessions + 2 :]
    except (ValueError, IndexError):
        return ""
    segments = [
        relative[index + 1]
        for index in range(0, len(relative) - 1, 2)
        if relative[index] == "agents"
    ]
    if not root or not segments:
        return ""
    return f"/repair-transcript {root} {root}:{':'.join(segments)}"


def _notification_id(target: RepairTarget, quarantine_id: str) -> str:
    child = target.child_run_id or "root"
    digest = hashlib.sha256(
        f"{target.root_session_id}\0{child}\0{quarantine_id}".encode()
    ).hexdigest()[:32]
    return f"transcript-unavailable:{digest}"


def append_child_repair_notification(
    parent_store: ConversationStore,
    result: RepairResult,
    *,
    child_session_path: Path,
) -> tuple[object, bool]:
    """Publish one parent notice after a confirmed child repair commits."""
    if result.target.child_run_id is None:
        raise ValueError("child target required")
    return parent_store.append_agent_notification_if_absent(
        _notification_id(result.target, result.quarantine_id),
        child_session_path=str(child_session_path),
        description="transcript recovery",
        status="error",
        text=(
            f"child transcript repaired; quarantine {result.quarantine_id}; "
            "reopen the child to continue"
        ),
    )


async def repair_transcript_for_runtime(
    manager: SessionManager,
    session_id: str,
    child_run_id: str | None,
    *,
    parent_store: ConversationStore | None,
) -> RepairResult:
    """Commit one repair, then publish its child lifecycle notice."""
    result = await asyncio.to_thread(
        manager.repair_transcript, session_id, child_run_id
    )
    if (
        child_run_id is not None
        and parent_store is not None
        and parent_store.session_id == session_id
    ):
        child_path = manager.sessions_dir / session_id
        for segment in child_run_segments(session_id, child_run_id):
            child_path = child_path / "agents" / str(segment)
        append_child_repair_notification(
            parent_store, result, child_session_path=child_path
        )
    return result
