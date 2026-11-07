"""Loop-scoped coordination for transcript index refreshes."""

from __future__ import annotations

import asyncio
import logging
import sqlite3
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

from zeta.core.store._log import MAX_PENDING_APPEND_RECEIPTS, PersistedAppend
from zeta.transcript_search.index import (
    TranscriptIndex,
    TranscriptIndexError,
    TranscriptSource,
)

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _RefreshState:
    pending: dict[str, TranscriptSource] = field(default_factory=dict)
    waiters: list[asyncio.Future[None]] = field(default_factory=list)
    task: asyncio.Task[None] | None = None
    failure: str | None = None


_REFRESH_STATES: weakref.WeakKeyDictionary[
    asyncio.AbstractEventLoop, dict[tuple[Path, str], _RefreshState]
] = weakref.WeakKeyDictionary()


def transcript_index_refresh_failure(
    projects_root: Path, project_id: str, session_id: str
) -> str | None:
    """Return this loop's refresh failure for one indexed session."""
    try:
        states = _REFRESH_STATES.get(asyncio.get_running_loop(), {})
    except RuntimeError:
        return None
    key = (Path(projects_root), project_id)
    state = states.get(key)
    if state is None or session_id not in state.pending:
        return None
    return state.failure


async def refresh_transcript_index(
    projects_root: Path,
    project_id: str,
    session_id: str,
    session_dir: Path,
    append_receipts: tuple[PersistedAppend, ...] | None = None,
) -> None:
    """Coalesce project refreshes and perform every filesystem operation off-loop."""

    loop = asyncio.get_running_loop()
    key = (Path(projects_root).resolve(), project_id)
    states = _REFRESH_STATES.setdefault(loop, {})
    state = states.setdefault(key, _RefreshState())
    incoming = TranscriptSource(session_id, session_dir, project_id, append_receipts)
    existing = state.pending.get(session_id)
    state.pending[session_id] = (
        _merge_refresh_sources(existing, incoming) if existing is not None else incoming
    )
    waiter = loop.create_future()
    state.waiters.append(waiter)
    if state.task is None:
        state.task = loop.create_task(_drain_refreshes(key, state, states))
    await waiter


async def _drain_refreshes(
    key: tuple[Path, str],
    state: _RefreshState,
    states: dict[tuple[Path, str], _RefreshState],
) -> None:
    failure: TranscriptIndexError | None = None
    try:
        while state.pending:
            pending = tuple(state.pending.values())
            state.pending.clear()
            try:
                await asyncio.to_thread(_refresh_sources, key[0], key[1], pending)
            except (
                OSError,
                sqlite3.Error,
                TranscriptIndexError,
                TypeError,
                ValueError,
            ) as exc:
                logger.warning("could not refresh transcript index: %s", exc)
                for source in pending:
                    newer = state.pending.get(source.session_id)
                    state.pending[source.session_id] = (
                        _merge_refresh_sources(source, newer)
                        if newer is not None
                        else source
                    )
                state.failure = f"{exc}; pending changes will retry after the next transcript refresh"
                failure = TranscriptIndexError(
                    f"could not refresh transcript index: {exc}"
                )
                break
            else:
                state.failure = None
    finally:
        waiters, state.waiters = state.waiters, []
        state.task = None
        if not state.pending:
            states.pop(key, None)
        for waiter in waiters:
            if waiter.done():
                continue
            if failure is None:
                waiter.set_result(None)
            else:
                waiter.set_exception(failure)


def _merge_refresh_sources(
    earlier: TranscriptSource, later: TranscriptSource
) -> TranscriptSource:
    receipts = (
        earlier.append_receipts + later.append_receipts
        if earlier.append_receipts is not None and later.append_receipts is not None
        else None
    )
    receipts = (
        None
        if receipts is not None and len(receipts) >= MAX_PENDING_APPEND_RECEIPTS
        else receipts
    )
    return replace(later, append_receipts=receipts)


def _refresh_sources(
    projects_root: Path,
    project_id: str,
    sources: Sequence[TranscriptSource],
) -> None:
    index = TranscriptIndex(projects_root / project_id, project_id)
    for source in sources:
        index.append(source)
