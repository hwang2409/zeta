"""Event-loop-safe transcript index refresh."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from .index import TranscriptIndex, TranscriptSource

logger = logging.getLogger(__name__)


async def append_transcript_off_loop(
    index: TranscriptIndex, source: TranscriptSource
) -> None:
    """Refresh one durable transcript without blocking its owner loop."""

    await asyncio.to_thread(index.append, source)


async def refresh_transcript_index(
    projects_root: Path,
    project_id: str,
    session_id: str,
    session_dir: Path,
) -> None:
    """Best-effort refresh for runtime persistence hooks."""

    try:
        index = TranscriptIndex(projects_root / project_id, project_id)
        await append_transcript_off_loop(
            index, TranscriptSource(session_id, session_dir)
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        logger.warning("could not refresh transcript index: %s", exc)
