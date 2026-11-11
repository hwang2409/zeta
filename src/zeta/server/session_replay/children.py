"""The serve loop's projection of live child lifecycles.

The lifecycle owner (``core.agent_progress``) posts each fact here. This adapter keeps
the latest row and revision of every active run, publishes the lifecycle events to
the run's immediate-parent replay scope, and supplies a join's admission baseline.
Replay itself keeps no per-run state.
"""

from __future__ import annotations

import asyncio
import weakref
from typing import TYPE_CHECKING, Any

from ..protocol import EventDescriptor
from . import ReplayScopeId

if TYPE_CHECKING:
    from .output import ReplayOutput


def parent_scope(row: dict[str, object]) -> ReplayScopeId:
    """The immediate parent's scope, where a run's lifecycle events go."""
    parent = row["parent_run_id"]
    if type(parent) is str:
        return ReplayScopeId.child(parent)
    return ReplayScopeId.top(str(row["session_id"]))


class ChildRunProjection:
    def __init__(self, loop: asyncio.AbstractEventLoop, output: ReplayOutput) -> None:
        self._loop = loop
        self._output = output
        # run_id -> (row, revision, start row sequence)
        self._rows: dict[str, tuple[dict[str, object], int, int | None]] = {}
        self._stores: dict[str, weakref.ReferenceType[object]] = {}

    def sink(
        self,
        kind: str,
        run_id: str,
        row: dict[str, object],
        number: int,
        store: object | None,
    ) -> None:
        """Called by the lifecycle owner on any thread; applied in order on the loop."""
        try:
            self._loop.call_soon_threadsafe(
                self._apply, kind, run_id, row, number, store
            )
        except RuntimeError:
            if not self._loop.is_closed():
                raise

    def _apply(
        self,
        kind: str,
        run_id: str,
        row: dict[str, object],
        number: int,
        store: object | None,
    ) -> None:
        session_id = str(row["session_id"])
        current = self._rows.get(run_id)
        if kind == "progress":
            if current is not None and current[1] >= number:
                return
            sequence = None if current is None else current[2]
            self._rows[run_id] = (row, number, sequence)
            if store is not None:
                self._stores[run_id] = weakref.ref(store)
            self._output.publish_child_lifecycle(
                EventDescriptor.CHILD_PROGRESS,
                parent_scope(row),
                session_id,
                {"run_id": run_id, "sequence": sequence, "revision": number, "child": row},
                None,
            )
            return
        if kind == "started":
            if current is None:
                self._rows[run_id] = (row, 0, number)
            event = EventDescriptor.CHILD_STARTED
        else:
            self._rows.pop(run_id, None)
            self._stores.pop(run_id, None)
            event = EventDescriptor.CHILD_ENDED
        self._output.publish_child_lifecycle(
            event,
            parent_scope(row),
            session_id,
            {"run_id": run_id, "sequence": number, "child": row},
            number,
        )

    def live_store(self, run_id: str) -> object | None:
        ref = self._stores.get(run_id)
        return None if ref is None else ref()

    def current(self, run_id: str) -> tuple[dict[str, object], int] | None:
        state = self._rows.get(run_id)
        return None if state is None else (state[0], state[1])

    def with_baseline(self, result: dict[str, Any]) -> dict[str, Any]:
        """Add each active run's latest row after its ``child_started`` page row.

        O(page rows); no I/O, lock, or await. Runs already ended or not running in
        this process have no snapshot row.
        """
        messages = result.get("messages")
        if not isinstance(messages, list):
            return result
        with_rows: list[object] = []
        for message in messages:
            with_rows.append(message)
            if not isinstance(message, dict) or message.get("event") != "child_started":
                continue
            state = self.current(str(message["run_id"]))
            if state is not None:
                with_rows.append(
                    {
                        "event": "child_progress",
                        "session_id": message["session_id"],
                        "run_id": message["run_id"],
                        "sequence": message["sequence"],
                        "revision": state[1],
                        "child": state[0],
                    }
                )
        return {**result, "messages": with_rows}

    def clear(self) -> None:
        self._rows.clear()
