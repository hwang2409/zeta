"""Best-effort lifecycle cleanup for disposable transcript indexes."""

from __future__ import annotations

import logging
from pathlib import Path

from .index import TranscriptIndex

logger = logging.getLogger(__name__)


def delete_indexed_session(
    projects_root: Path, project_id: str, session_id: str
) -> None:
    try:
        TranscriptIndex(projects_root / project_id, project_id).delete_session(session_id)
    except (OSError, RuntimeError, ValueError) as exc:
        logger.warning("could not remove transcript index session: %s", exc)
