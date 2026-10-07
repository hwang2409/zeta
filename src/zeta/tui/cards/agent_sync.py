"""Read-only active-branch snapshots for live child transcripts."""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass
from typing import Any

from ...core.store import ConversationStore


@dataclass(frozen=True, slots=True)
class AgentTranscriptSnapshot:
    """One immutable active-branch projection from a child store."""

    entry_ids: tuple[str, ...]
    messages: tuple[tuple[str, dict[str, Any]], ...]


def refresh_agent_transcript(store: ConversationStore) -> AgentTranscriptSnapshot:
    """Refresh a read-only store and detach its active message projection."""

    store.refresh()
    branch = store.active_branch_snapshot()
    messages: list[tuple[str, dict[str, Any]]] = []
    for index, entry in enumerate(branch):
        # Detaching can copy megabytes; release the GIL for the TUI event loop.
        if index and index % 64 == 0:
            time.sleep(0.001)
        if entry.type != "message":
            continue
        message = entry.data.get("message")
        if not isinstance(message, dict):
            continue
        metadata = message.get("metadata")
        if (
            isinstance(metadata, dict)
            and metadata.get("zeta_event") == "empty_turn_nudge"
        ):
            continue
        messages.append((entry.id, copy.deepcopy(message)))
    return AgentTranscriptSnapshot(
        tuple(entry.id for entry in branch), tuple(messages)
    )
