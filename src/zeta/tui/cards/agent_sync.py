"""Immutable snapshots for every live child-transcript view."""

from __future__ import annotations

import copy
import os
import threading
import time
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...core.checkpoints import (
    ConversationEntry,
    ConversationIntegrityError,
    load_session_json,
)
from ...core.session_files import open_session_file, session_directory
from ...core.store import ConversationStore

_REFRESH_LOCK = threading.Lock()
_REVERSE_READ_BYTES = 64 * 1024
_MAX_REVERSE_SCAN_BYTES = 1024 * 1024
_MAX_FULL_FALLBACK_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True, slots=True)
class AgentTranscriptSnapshot:
    """One immutable active-branch projection from a child transcript."""

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


class _TailNotResolved(Exception):
    """The bounded reverse scan needs the guarded full-read fallback."""


class AgentTranscriptSource:
    """Own child transcript storage and publish detached immutable snapshots.

    A source without a message limit owns incremental read-only stores for the
    selected full child view. A limited source reverse-scans only the active
    branch tail and never retains a store or file descriptor between refreshes.
    Sources have an explicit lifetime: callers must close them and cannot
    refresh them after close.
    """

    def __init__(self, path: Path, *, message_limit: int | None = None) -> None:
        self.path = path
        self._message_limit = message_limit
        self._stores: dict[Path, ConversationStore] = {}
        self._snapshots: dict[Path, AgentTranscriptSnapshot] = {}
        self._closed = False

    def refresh(self, *, recursive: bool = False) -> AgentTranscriptTreeSnapshot:
        """Refresh this source. Calls are serialized across all TUI sources."""

        with _REFRESH_LOCK:
            if self._closed:
                raise RuntimeError("agent transcript source is closed")
            return self._refresh_locked(recursive=recursive)

    def close(self) -> None:
        """Close owned stores and make this source unusable."""

        with _REFRESH_LOCK:
            if self._closed:
                return
            self._closed = True
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
        if self._message_limit is not None:
            entries = _read_active_message_tail(path, self._message_limit)
        else:
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
            entries = store.active_branch_snapshot()
        snapshot = self._project(path, entries)
        self._snapshots[path] = snapshot
        return snapshot

    def _project(
        self, path: Path, entries: Sequence[ConversationEntry]
    ) -> AgentTranscriptSnapshot:
        message_entries = [entry for entry in entries if entry.type == "message"]
        if self._message_limit is not None:
            message_entries = message_entries[-self._message_limit :]
            entry_ids = tuple(entry.id for entry in message_entries)
        else:
            entry_ids = tuple(entry.id for entry in entries)
        previous = self._snapshots.get(path)
        previous_by_id = dict(previous.messages) if previous is not None else {}
        messages: list[tuple[str, dict[str, Any]]] = []
        for index, entry in enumerate(message_entries):
            if index and index % 64 == 0:
                # Full-view parsing is incremental. Cooperate while its first
                # snapshot detaches a large resident branch.
                time.sleep(0.001)
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
        return AgentTranscriptSnapshot(entry_ids, tuple(messages))


def _read_active_message_tail(path: Path, limit: int) -> tuple[ConversationEntry, ...]:
    """Read a bounded active-branch message tail without retaining a store."""

    if limit < 1:
        return ()
    conversation_path = path / "conversation.jsonl"
    with session_directory(path.parent, path.name) as (_, directory_fd):
        try:
            fd = open_session_file(directory_fd, "conversation.jsonl", os.O_RDONLY)
        except FileNotFoundError as exc:
            raise ConversationIntegrityError(
                f"conversation file is missing: {conversation_path}"
            ) from exc
        try:
            size = os.fstat(fd).st_size
            try:
                return _reverse_scan_message_tail(fd, size, limit)
            except _TailNotResolved:
                pass
        finally:
            os.close(fd)

    if size > _MAX_FULL_FALLBACK_BYTES:
        raise ConversationIntegrityError(
            f"conversation tail cannot be resolved within bounded reads: {conversation_path}"
        )
    store = ConversationStore(
        path.parent,
        session_id=path.name,
        _read_only=True,
        _must_exist=True,
    )
    try:
        messages = [
            entry for entry in store.active_branch_snapshot() if entry.type == "message"
        ]
        return tuple(messages[-limit:])
    finally:
        store.close()


def _reverse_scan_message_tail(
    fd: int, size: int, limit: int
) -> tuple[ConversationEntry, ...]:
    expected_id: str | None = None
    messages: list[ConversationEntry] = []
    saw_entry = False
    for raw in _reverse_lines(fd, size, _MAX_REVERSE_SCAN_BYTES):
        row = load_session_json(raw)
        if not isinstance(row, dict):
            raise ConversationIntegrityError("conversation row is not an object")
        if row.get("type") == "header":
            if not saw_entry:
                return ()
            if expected_id is not None:
                raise _TailNotResolved
            break
        entry = ConversationEntry.from_dict(row)
        if not saw_entry:
            saw_entry = True
            expected_id = entry.id
        if entry.id != expected_id:
            continue
        if entry.type == "message":
            messages.append(entry)
            if len(messages) == limit:
                return tuple(reversed(messages))
        expected_id = entry.parent_id
        if expected_id is None:
            return tuple(reversed(messages))
    if expected_id is not None:
        raise _TailNotResolved
    if not saw_entry:
        raise ConversationIntegrityError("conversation file is empty")
    return tuple(reversed(messages))


def _reverse_lines(fd: int, size: int, max_bytes: int) -> Iterator[bytes]:
    position = size
    suffix = b""
    scanned = 0
    while position:
        read_size = min(_REVERSE_READ_BYTES, position)
        position -= read_size
        chunk = os.pread(fd, read_size, position)
        if len(chunk) != read_size:
            raise ConversationIntegrityError("conversation file changed while reading")
        scanned += len(chunk)
        if scanned > max_bytes:
            raise _TailNotResolved
        parts = (chunk + suffix).split(b"\n")
        suffix = parts[0]
        for line in reversed(parts[1:]):
            if line:
                yield line
    if suffix:
        yield suffix


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


async def refresh_agent_cards(cards: Iterable[Any]) -> list[bool]:
    """Refresh eligible card snapshots in one serialized worker call."""

    card_list = list(cards)
    eligible = [
        card
        for card in card_list
        if card.refresh_eligible and card.transcript_source is not None
    ]
    if not eligible:
        return [False] * len(card_list)
    import asyncio

    snapshots = await asyncio.to_thread(
        refresh_agent_transcripts,
        [(card.transcript_source, True) for card in eligible],
    )
    refreshed = iter(
        card.apply_transcript_snapshot(snapshot)
        for card, snapshot in zip(eligible, snapshots)
    )
    return [next(refreshed) if card in eligible else False for card in card_list]
