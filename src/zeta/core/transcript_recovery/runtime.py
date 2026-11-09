"""Runtime adapters for explicit transcript repair."""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ...storage_format import FEATURE_EPOCHS, StorageFeature, feature_enabled
from ..store import ConversationStore
from ..store._child_runs import child_run_segments
from . import RepairResult, RepairTarget

if TYPE_CHECKING:
    from ...memory.auto import AutoMemoryReconciler
    from ..session import SessionManager


class TranscriptRepairRequiresUpgrade(ValueError):
    """The observed home predates explicit transcript repair."""

    def __init__(self, home_epoch: int) -> None:
        self.home_epoch = home_epoch
        self.required_epoch = FEATURE_EPOCHS[StorageFeature.TRANSCRIPT_REPAIR]
        super().__init__(
            "transcript repair needs the stored-format upgrade "
            f"(home epoch {home_epoch}, needs {self.required_epoch})"
        )


def transcript_repair_available(manager: SessionManager) -> bool:
    return feature_enabled(
        manager.observed_storage_epoch, StorageFeature.TRANSCRIPT_REPAIR
    )


def require_transcript_repair(manager: SessionManager) -> None:
    if not transcript_repair_available(manager):
        raise TranscriptRepairRequiresUpgrade(manager.observed_storage_epoch)


def format_repair_transcript_command(
    session_id: str, child_run_id: str | None = None
) -> str:
    """Build the explicit repair command for a transcript target."""
    target = f" {child_run_id}" if child_run_id is not None else ""
    return f"/repair-transcript {session_id}{target} confirm"


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
    return format_repair_transcript_command(root, f"{root}:{':'.join(segments)}")


def _resident_memory_commit(
    reconciler: AutoMemoryReconciler | None,
    session_id: str,
    child_run_id: str | None,
) -> Callable[[str, int, bool], None] | None:
    if (
        child_run_id is not None
        or reconciler is None
        or reconciler.session_id != session_id
    ):
        return None
    owner_loop = asyncio.get_running_loop()

    def commit(
        quarantine_id: str,
        retained_last_sequence: int,
        memory_review_required: bool,
    ) -> None:
        future = asyncio.run_coroutine_threadsafe(
            reconciler.recovery_committed(
                quarantine_id=quarantine_id,
                retained_last_sequence=retained_last_sequence,
                memory_review_required=memory_review_required,
            ),
            owner_loop,
        )
        future.result()

    return commit


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
    memory_reconciler: AutoMemoryReconciler | None = None,
) -> RepairResult:
    """Commit one repair through its resident owner, then publish success."""
    require_transcript_repair(manager)
    resident_memory_commit = _resident_memory_commit(
        memory_reconciler, session_id, child_run_id
    )
    result = await asyncio.to_thread(
        manager.repair_transcript,
        session_id,
        child_run_id,
        resident_memory_commit=resident_memory_commit,
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
