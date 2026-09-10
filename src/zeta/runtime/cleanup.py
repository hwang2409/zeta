"""Release a frontend's session resources, including partial activation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core.checkpoints.workspace import WorkspaceSnapshotStore
    from ..loop import AgentLoop


async def close_session(
    loop: AgentLoop, snapshots: WorkspaceSnapshotStore | None = None
) -> None:
    """Stop writers before releasing either independent storage lease."""
    try:
        await loop.close()
    finally:
        try:
            await loop.tool_registry.background_tasks.close()
        finally:
            try:
                if snapshots is not None:
                    snapshots.close()
            finally:
                loop.store.close()
