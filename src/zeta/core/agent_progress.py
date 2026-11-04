"""Process-local snapshots for live child-agent progress."""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .checkpoints import load_session_json
from .child_lifecycle import select_legacy_child_lifecycle

_lock = threading.Lock()
_lifecycles: dict[Path, tuple[int, dict[str, Any]]] = {}
_projections: dict[Path, tuple[int, dict[str, Any]]] = {}


@dataclass(frozen=True)
class AgentLifecycleProjection:
    generation: int
    lifecycle: dict[str, Any]


def publish_live_agent_lifecycle(path: Path, lifecycle: dict[str, Any]) -> None:
    """Publish a detached lifecycle snapshot without touching disk."""

    resolved = path.resolve()
    with _lock:
        generation = _lifecycles.get(resolved, (0, {}))[0] + 1
        detached = copy.deepcopy(lifecycle)
        _lifecycles[resolved] = (generation, detached)
        _projections.pop(resolved, None)


def live_agent_lifecycle(path: str | Path) -> dict[str, Any] | None:
    """Return live progress for this process, when it owns the child."""

    entry = live_agent_lifecycle_snapshot(path)
    return entry[1] if entry is not None else None


def live_agent_lifecycle_snapshot(
    path: str | Path,
) -> tuple[int, dict[str, Any]] | None:
    resolved = Path(path).resolve()
    with _lock:
        entry = _lifecycles.get(resolved)
        return (entry[0], copy.deepcopy(entry[1])) if entry is not None else None


def publish_agent_lifecycle_projection(
    path: str | Path, lifecycle: object, generation: int
) -> None:
    """Publish a worker projection only if its live snapshot is still current."""

    if type(lifecycle) is not dict:
        return
    resolved = Path(path).resolve()
    with _lock:
        current_generation = _lifecycles.get(resolved, (0, {}))[0]
        if generation != current_generation:
            return
        current = _projections.get(resolved)
        if current is None or generation >= current[0]:
            _projections[resolved] = (generation, copy.deepcopy(lifecycle))


def agent_lifecycle_projection(path: str | Path) -> AgentLifecycleProjection | None:
    resolved = Path(path).resolve()
    with _lock:
        entry = _projections.get(resolved)
        if entry is None:
            return None
        return AgentLifecycleProjection(entry[0], copy.deepcopy(entry[1]))


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
        _projections.pop(resolved, None)
