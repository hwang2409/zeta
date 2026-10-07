"""Incremental read-only snapshots for every live child-transcript view."""

from __future__ import annotations

import copy
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.store import ConversationStore

_REFRESH_LOCK = threading.Lock()


@dataclass(frozen=True, slots=True)
class AgentTranscriptSnapshot:
    """One immutable active-branch projection from a child store."""

    entry_ids: tuple[str, ...]
    messages: tuple[tuple[str, dict[str, Any]], ...]


@dataclass(frozen=True, slots=True)
class AgentTranscriptTreeSnapshot:
    """A child transcript and every nested child referenced by its branch."""

    root: Path
    transcripts: tuple[tuple[Path, AgentTranscriptSnapshot], ...]

    def transcript(self, path: Path) -> AgentTranscriptSnapshot | None:
        return next(
            (snapshot for candidate, snapshot in self.transcripts if candidate == path),
            None,
        )


class AgentTranscriptSource:
    """Own incremental read-only stores behind one child-transcript seam."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._stores: dict[Path, ConversationStore] = {}
        self._snapshots: dict[Path, AgentTranscriptSnapshot] = {}

    def refresh(self, *, recursive: bool = False) -> AgentTranscriptTreeSnapshot:
        """Refresh this source. Calls are serialized across all TUI sources."""

        with _REFRESH_LOCK:
            return self._refresh_locked(recursive=recursive)

    def close(self) -> None:
        with _REFRESH_LOCK:
            for store in self._stores.values():
                store.close()
            self._stores.clear()
            self._snapshots.clear()

    def _refresh_locked(self, *, recursive: bool) -> AgentTranscriptTreeSnapshot:
        pending = [self.path]
        projected: list[tuple[Path, AgentTranscriptSnapshot]] = []
        seen: set[Path] = set()
        while pending:
            path = pending.pop(0)
            if path in seen:
                continue
            seen.add(path)
            snapshot = self._refresh_path(path)
            projected.append((path, snapshot))
            if recursive:
                pending.extend(
                    child
                    for child in _nested_child_paths(snapshot)
                    if child not in seen
                )
        return AgentTranscriptTreeSnapshot(self.path, tuple(projected))

    def _refresh_path(self, path: Path) -> AgentTranscriptSnapshot:
        store = self._stores.get(path)
        if store is None:
            store = ConversationStore(
                path.parent,
                session_id=path.name,
                _read_only=True,
                _must_exist=True,
            )
            self._stores[path] = store
        else:
            store.refresh()
        branch = store.active_branch_snapshot()
        previous = self._snapshots.get(path)
        previous_by_id = dict(previous.messages) if previous is not None else {}
        messages: list[tuple[str, dict[str, Any]]] = []
        for index, entry in enumerate(branch):
            if index and index % 64 == 0:
                # ConversationStore parsing is incremental. Cooperate while the
                # first snapshot detaches a large resident branch.
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
            retained = previous_by_id.get(entry.id)
            messages.append(
                (entry.id, retained if retained == message else copy.deepcopy(message))
            )
        snapshot = AgentTranscriptSnapshot(
            tuple(entry.id for entry in branch), tuple(messages)
        )
        self._snapshots[path] = snapshot
        return snapshot


def refresh_agent_transcripts(
    sources: Iterable[tuple[AgentTranscriptSource, bool]],
) -> list[AgentTranscriptTreeSnapshot]:
    """Refresh several sources serially in one worker invocation."""

    return [source.refresh(recursive=recursive) for source, recursive in sources]


def _nested_child_paths(snapshot: AgentTranscriptSnapshot) -> list[Path]:
    paths: list[Path] = []
    for _entry_id, message in snapshot.messages:
        result = message.get("tool_result")
        structured = result.get("structured_content") if isinstance(result, dict) else None
        child = structured.get("child_session_path") if isinstance(structured, dict) else None
        if isinstance(child, str) and child:
            paths.append(Path(child))
    return paths
