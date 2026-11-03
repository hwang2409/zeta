"""Process-local snapshots for live child-agent progress."""

from __future__ import annotations

import copy
import threading
from pathlib import Path
from typing import Any

from .checkpoints import load_session_json
from .child_lifecycle import select_legacy_child_lifecycle

_lock = threading.Lock()
_lifecycles: dict[Path, dict[str, Any]] = {}
_terminal_lifecycles: dict[Path, dict[str, Any]] = {}


def publish_live_agent_lifecycle(path: Path, lifecycle: dict[str, Any]) -> None:
    """Publish a detached lifecycle snapshot without touching disk."""

    resolved = path.resolve()
    with _lock:
        _terminal_lifecycles.pop(resolved, None)
        _lifecycles[resolved] = copy.deepcopy(lifecycle)


def live_agent_lifecycle(path: str | Path) -> dict[str, Any] | None:
    """Return live progress for this process, when it owns the child."""

    resolved = Path(path).resolve()
    with _lock:
        lifecycle = _lifecycles.get(resolved)
        return copy.deepcopy(lifecycle) if lifecycle is not None else None


def take_terminal_agent_lifecycle(path: str | Path) -> dict[str, Any] | None:
    """Take a detached terminal projection handed off by a worker read."""

    resolved = Path(path).resolve()
    with _lock:
        lifecycle = _terminal_lifecycles.pop(resolved, None)
        return copy.deepcopy(lifecycle) if lifecycle is not None else None


def remember_terminal_agent_lifecycle(path: str | Path, lifecycle: object) -> None:
    """Retain an immutable terminal projection for event-loop readers."""

    if type(lifecycle) is not dict or lifecycle.get("state") not in {
        "completed",
        "failed",
        "canceled",
    }:
        return
    resolved = Path(path).resolve()
    with _lock:
        _terminal_lifecycles[resolved] = copy.deepcopy(lifecycle)


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
    if live is not None:
        remember_terminal_agent_lifecycle(lifecycle_path, selected)
    return selected if selected is not None else {}


def remove_live_agent_lifecycle(path: str | Path) -> None:
    """Stop publishing a child after its terminal transition is durable."""

    resolved = Path(path).resolve()
    with _lock:
        _lifecycles.pop(resolved, None)
