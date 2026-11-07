"""Serve-side session mutations that preserve the resident execution owner."""

from __future__ import annotations

import asyncio

from ...core.session import SessionMetadata
from ..runtime import ServerRuntime


async def rename_session(
    runtime: ServerRuntime, session_id: str, name: str
) -> SessionMetadata:
    """Rename a session without opening a second owner for the active runtime."""
    is_active = runtime.opened is not None and session_id == runtime.session_id
    metadata = await asyncio.to_thread(
        runtime.manager.rename,
        session_id,
        name,
        runtime.opened.store.execution_lease if is_active else None,
    )
    if is_active:
        runtime.manager._copy_metadata(runtime.metadata, metadata)
    return metadata
