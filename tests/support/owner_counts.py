from __future__ import annotations

import inspect
import sys
import threading
import weakref
from collections import defaultdict
from functools import wraps
from pathlib import Path
from typing import Any

CLOSE_GUIDANCE = {
    "AgentLoop": "await loop.close()",
    "ConversationStore": "store.close()",
    "AutoMemoryReconciler": "await reconciler.close()",
    "WorkspaceSnapshotStore": "workspace.close()",
    "ToolRegistry": "await registry.close()",
    "BackgroundTaskRegistry": "await registry.close()",
    "ZetaServer": "await server.close()",
    "TUIApp": "async with owned_app(app)",
}

OwnerRow = tuple[str, str, int]


class OwnerCounts:
    def __init__(self) -> None:
        self.active_node: str | None = None
        # Weak refs only: a test may rely on an unclosed owner being collected.
        self._records: dict[int, tuple[weakref.ref[Any], str, str]] = {}
        self._counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
        self._rows: list[OwnerRow] = []
        self._originals: list[tuple[type[Any], str, Any]] = []
        self._lock = threading.Lock()

    def node_start(self, nodeid: str) -> None:
        self.active_node = nodeid

    def node_finish(self, nodeid: str) -> None:
        with self._lock:
            for (owner_node, kind), (constructed, explicit) in self._counts.items():
                if owner_node == nodeid:
                    self._rows.append((nodeid, kind, constructed - explicit))
            self._counts = defaultdict(
                lambda: [0, 0],
                {key: value for key, value in self._counts.items() if key[0] != nodeid},
            )
            self._records = {
                key: value for key, value in self._records.items() if value[1] != nodeid
            }
        self.active_node = None

    def constructed(self, obj: Any, kind: str) -> None:
        nodeid = self.active_node
        if nodeid is None:
            return
        with self._lock:
            self._records[id(obj)] = (weakref.ref(obj), nodeid, kind)
            self._counts[(nodeid, kind)][0] += 1

    def closed(self, obj: Any) -> None:
        with self._lock:
            record = self._records.get(id(obj))
            if record is None or record[0]() is not obj or record[1] != self.active_node:
                return
            if not _closed_by_cleanup_fixture():
                self._counts[record[1:]][1] += 1
            self._records.pop(id(obj), None)

    def rows(self) -> list[OwnerRow]:
        return sorted(self._rows)

    def install(self) -> None:
        from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
        from zeta.core.store import ConversationStore
        from zeta.memory.auto import AutoMemoryReconciler
        from zeta.runtime.loop import AgentLoop
        from zeta.server import ZetaServer
        from zeta.tools import ToolRegistry
        from zeta.tools._shared.process import BackgroundTaskRegistry
        from zeta.tui.app import TUIApp

        owners = (
            (AgentLoop, "AgentLoop", "close"),
            (ConversationStore, "ConversationStore", "close"),
            # SessionManager joins the gate when r7 adds SessionManager.close.
            (AutoMemoryReconciler, "AutoMemoryReconciler", "close"),
            (WorkspaceSnapshotStore, "WorkspaceSnapshotStore", "close"),
            (ToolRegistry, "ToolRegistry", "close"),
            (BackgroundTaskRegistry, "BackgroundTaskRegistry", "close"),
            (ZetaServer, "ZetaServer", "close"),
            (TUIApp, "TUIApp", "close"),
        )
        for cls, kind, close_name in owners:
            original_init = cls.__init__
            self._originals.append((cls, "__init__", original_init))

            @wraps(original_init)
            def init(
                obj: Any,
                *args: Any,
                _original: Any = original_init,
                _kind: str = kind,
                **kwargs: Any,
            ) -> None:
                _original(obj, *args, **kwargs)
                self.constructed(obj, _kind)

            cls.__init__ = init
            original_close = getattr(cls, close_name)
            self._originals.append((cls, close_name, original_close))
            if inspect.iscoroutinefunction(original_close):

                @wraps(original_close)
                async def close(
                    obj: Any,
                    *args: Any,
                    _original: Any = original_close,
                    **kwargs: Any,
                ) -> Any:
                    result = await _original(obj, *args, **kwargs)
                    self.closed(obj)
                    return result

            else:

                @wraps(original_close)
                def close(
                    obj: Any,
                    *args: Any,
                    _original: Any = original_close,
                    **kwargs: Any,
                ) -> Any:
                    result = _original(obj, *args, **kwargs)
                    self.closed(obj)
                    return result

            setattr(cls, close_name, close)

    def uninstall(self) -> None:
        for cls, name, original in reversed(self._originals):
            setattr(cls, name, original)


def _closed_by_cleanup_fixture() -> bool:
    """Closes done by conftest's safety-net fixture do not count as test closes."""
    frame = sys._getframe(2)
    while frame is not None:
        if frame.f_code.co_name == "close_direct_test_runtime_owners":
            return True
        frame = frame.f_back
    return False


def merge_rows(groups: list[list[OwnerRow]]) -> list[OwnerRow]:
    merged: dict[tuple[str, str], int] = {}
    for rows in groups:
        for nodeid, kind, unclosed in rows:
            key = (nodeid, kind)
            if key in merged:
                raise ValueError(f"duplicate owner-count row: {nodeid}::{kind}")
            merged[key] = unclosed
    return [(nodeid, kind, count) for (nodeid, kind), count in sorted(merged.items())]


def read_baseline(path: Path) -> dict[tuple[str, str], int]:
    if not path.exists():
        return {}
    baseline = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        label, raw_count = raw_line.rsplit("=", 1)
        nodeid, kind = label.rsplit("::", 1)
        baseline[(nodeid, kind)] = int(raw_count)
    return baseline


def gate_failures(rows: list[OwnerRow], baseline_path: Path) -> list[str]:
    baseline = read_baseline(baseline_path)
    failures = []
    for nodeid, kind, unclosed in rows:
        allowed = baseline.get((nodeid, kind), 0)
        if unclosed <= allowed:
            continue
        guidance = CLOSE_GUIDANCE[kind]
        failures.append(
            f"owner leak ratchet: {nodeid} left {unclosed} {kind} owner(s) unclosed "
            f"(baseline {allowed}); close the owner with {guidance}"
        )
    return failures


def write_rows(path: Path, rows: list[OwnerRow]) -> None:
    """Write every observed row, zeros included, so a refresh can lower to 0."""
    path.write_text(
        "".join(f"{nodeid}::{kind}={unclosed}\n" for nodeid, kind, unclosed in rows),
        encoding="utf-8",
    )


def refresh_baseline(baseline_path: Path, observed_path: Path) -> int:
    """Lower baseline counts to the observed ones; never raise or add a count."""
    observed = read_baseline(observed_path)
    refreshed = {
        key: min(count, observed.get(key, count))
        for key, count in read_baseline(baseline_path).items()
    }
    lines = [
        f"{nodeid}::{kind}={count}"
        for (nodeid, kind), count in sorted(refreshed.items())
        if count > 0
    ]
    baseline_path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return len(lines)
