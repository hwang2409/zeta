from __future__ import annotations

import inspect
import threading
from collections import defaultdict
from pathlib import Path
from typing import Any

BASELINE_START = "owner-leak-baseline-start"
BASELINE_END = "owner-leak-baseline-end"
CLOSE_GUIDANCE = {
    "AgentLoop": "await loop.close()",
    "ConversationStore": "store.close() or use the conftest owner helper",
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
        self._records: dict[int, tuple[Any, str, str]] = {}
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
            self._records[id(obj)] = (obj, nodeid, kind)
            self._counts[(nodeid, kind)][0] += 1

    def closed(self, obj: Any) -> None:
        with self._lock:
            record = self._records.get(id(obj))
            if record is None or record[1] != self.active_node:
                return
            if not any(
                frame.function == "close_direct_test_runtime_owners"
                for frame in inspect.stack(context=0)
            ):
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


def baseline_lines(rows: list[OwnerRow]) -> list[str]:
    return [
        f"{nodeid}::{kind}={unclosed}"
        for nodeid, kind, unclosed in rows
        if unclosed > 0
    ]


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


def print_record(rows: list[OwnerRow]) -> None:
    print(BASELINE_START, flush=True)
    for nodeid, kind, unclosed in rows:
        print(f"{nodeid}::{kind}={unclosed}", flush=True)
    print(BASELINE_END, flush=True)


def refresh_baseline(baseline_path: Path, observed_path: Path) -> int:
    observed = _read_observed(observed_path)
    if not baseline_path.exists():
        refreshed = {key: count for key, count in observed.items() if count > 0}
    else:
        current = read_baseline(baseline_path)
        refreshed = {
            key: min(count, observed.get(key, count))
            for key, count in current.items()
            if min(count, observed.get(key, count)) > 0
        }
    lines = [
        f"{nodeid}::{kind}={count}"
        for (nodeid, kind), count in sorted(refreshed.items())
    ]
    baseline_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def _read_observed(path: Path) -> dict[tuple[str, str], int]:
    inside = False
    observed = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line == BASELINE_START:
            inside = True
            continue
        if line == BASELINE_END:
            break
        if not inside or "::" not in line or "=" not in line:
            continue
        label, raw_count = line.rsplit("=", 1)
        nodeid, kind = label.rsplit("::", 1)
        observed[(nodeid, kind)] = int(raw_count)
    if not observed and not inside:
        raise ValueError(f"no {BASELINE_START} marker in {path}")
    return observed
