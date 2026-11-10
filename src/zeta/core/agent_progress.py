"""Process-local snapshots for live child-agent progress."""

from __future__ import annotations

import copy
import logging
import threading
import weakref
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .checkpoints import load_session_json
from .child_lifecycle import select_legacy_child_lifecycle

if TYPE_CHECKING:
    from .store import ConversationStore

_LOGGER = logging.getLogger(__name__)
_lock = threading.Lock()
_lifecycles: dict[Path, dict[str, Any]] = {}
# serve's lifecycle projection sinks: sink(kind, run_id, row, number), where kind is
# "progress" (number = revision), "started" or "ended" (number = parent row seq)
ChildRunSink = Callable[[str, str, dict[str, object], int], None]
_sinks: list[ChildRunSink] = []
_live_stores: dict[str, weakref.ref[ConversationStore]] = {}
# process-local lifecycle freshness; never persisted or compared across processes
_revision = 0


def register_child_run_sink(sink: ChildRunSink) -> Callable[[], None]:
    """Receive child lifecycle facts until the returned function is called."""
    with _lock:
        _sinks.append(sink)

    def unregister() -> None:
        with _lock:
            if sink in _sinks:
                _sinks.remove(sink)

    return unregister


def live_child_store(run_id: str) -> ConversationStore | None:
    """The store of a child this process runs, by public run id."""
    with _lock:
        ref = _live_stores.get(run_id)
    return None if ref is None else ref()


def post_child_progress(store: ConversationStore, lifecycle: dict[str, Any]) -> None:
    """Post one semantic lifecycle change with the next revision to every sink."""
    global _revision
    from .child_runs import project_child_run_row

    run_id = lifecycle.get("handle")
    if type(run_id) is not str:
        return
    run_id = str(lifecycle["handle"])
    row = project_child_run_row(lifecycle, session_id=store.session_id, parent_run_id=run_id.rpartition(":")[0] or None, accepts_follow_ups=store.accepts_follow_ups)
    with _lock:
        _live_stores[run_id] = weakref.ref(store)
        _revision += 1
        # posted under the lock, so sinks see revisions in order
        for sink in tuple(_sinks):
            sink("progress", run_id, row, _revision)


def publish_child_started(
    run_id: str, sequence: int, data: dict[str, Any], *, accepts_follow_ups: bool
) -> None:
    """Publish a durable start row installed in the immediate parent's store."""
    from .child_runs import project_child_run_row

    run_id = str(data["run_id"])
    row = project_child_run_row({"handle": run_id, "description": data.get("task"), "agent_type": data.get("name") or "general", "state": "running", "turns_used": 0, "tool_calls": 0, "current_step": "starting", "started_at": data.get("started_at"), "finished_at": None, "elapsed": 0.0}, session_id=run_id.split(":", 1)[0], parent_run_id=run_id.rpartition(":")[0] or None, accepts_follow_ups=accepts_follow_ups)
    with _lock:
        for sink in tuple(_sinks):
            sink("started", run_id, row, sequence)


def publish_child_ended(run_id: str, receipt_sequence: int, lifecycle: object) -> None:
    """The one publisher of ``child_ended``: call it only after the immediate
    parent durably installed a new receipt for a run whose lifecycle is terminal."""
    from .child_runs import project_child_run_row

    if not isinstance(lifecycle, dict) or lifecycle.get("handle") != run_id:
        return
    if lifecycle.get("finished_at") is None:
        _LOGGER.error("child_ended for %s before its lifecycle is final", run_id)
        return
    row = project_child_run_row(lifecycle, session_id=run_id.split(":", 1)[0], parent_run_id=run_id.rpartition(":")[0] or None, accepts_follow_ups=False)
    with _lock:
        for sink in tuple(_sinks):
            sink("ended", run_id, row, receipt_sequence)


def publish_live_agent_lifecycle(path: Path, lifecycle: dict[str, Any]) -> None:
    """Publish a detached lifecycle snapshot without touching disk."""

    resolved = path.resolve()
    with _lock:
        _lifecycles[resolved] = copy.deepcopy(lifecycle)


def live_agent_lifecycle(path: str | Path) -> dict[str, Any] | None:
    """Return live progress for this process, when it owns the child."""

    resolved = Path(path).resolve()
    with _lock:
        lifecycle = _lifecycles.get(resolved)
        return copy.deepcopy(lifecycle) if lifecycle is not None else None


def read_agent_lifecycle(path: str | Path) -> object:
    """Read one legacy projection with durable terminal precedence."""

    lifecycle_path = Path(path)
    live = live_agent_lifecycle(lifecycle_path)
    try:
        durable = load_session_json(lifecycle_path)
    except (OSError, ValueError):
        if live is not None:
            return live
        raise
    selected = select_legacy_child_lifecycle(durable, live)
    return selected if selected is not None else {}


def remove_live_agent_lifecycle(path: str | Path) -> None:
    """Stop publishing a child after its terminal transition is durable."""

    resolved = Path(path).resolve()
    with _lock:
        _lifecycles.pop(resolved, None)
        for run_id, ref in tuple(_live_stores.items()):
            store = ref()
            if store is None or store.agent_lifecycle_path.resolve() == resolved:
                del _live_stores[run_id]
