from __future__ import annotations

import inspect
import json
import threading
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class OwnerCounts:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.active_node: str | None = None
        self._records: dict[int, tuple[str, str, str, str | None]] = {}
        self._counts: dict[tuple[str, str, str], list[int]] = defaultdict(
            lambda: [0, 0, 0]
        )
        self._originals: list[tuple[type[Any], str, Any]] = []
        self._lock = threading.Lock()

    def node_start(self, nodeid: str) -> None:
        self.active_node = nodeid

    def node_finish(self, nodeid: str) -> None:
        rows = []
        with self._lock:
            for (owner_node, kind, site), values in self._counts.items():
                if owner_node == nodeid:
                    constructed, explicit, forced = values
                    rows.append(
                        {
                            "nodeid": nodeid,
                            "kind": kind,
                            "site": site,
                            "constructed": constructed,
                            "explicit": explicit,
                            "forced": forced,
                            "unclosed": constructed - explicit,
                        }
                    )
            self._counts = defaultdict(lambda: [0, 0, 0], {
                key: value
                for key, value in self._counts.items()
                if key[0] != nodeid
            })
            self._records = {
                key: value
                for key, value in self._records.items()
                if value[3] != nodeid
            }
        with self.path.open("a", encoding="utf-8") as output:
            for row in sorted(rows, key=lambda row: (row["nodeid"], row["kind"], row["site"])):
                output.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        self.active_node = None

    def constructed(self, obj: Any, kind: str) -> None:
        nodeid = self.active_node
        if nodeid is None:
            return
        frame = inspect.currentframe()
        assert frame is not None
        caller = frame.f_back.f_back
        site = f"{caller.f_code.co_filename}:{caller.f_lineno}"
        key = (nodeid, kind, site)
        with self._lock:
            self._records[id(obj)] = (nodeid, kind, site, nodeid)
            self._counts[key][0] += 1

    def closed(self, obj: Any) -> None:
        record = self._records.get(id(obj))
        if record is None or record[0] != self.active_node:
            return
        key = record[:3]
        forced = _FORCED or any(
            frame.function == "close_direct_test_runtime_owners"
            for frame in inspect.stack(context=0)
        )
        with self._lock:
            self._counts[key][1 if not forced else 2] += 1
            self._records.pop(id(obj), None)

    @contextmanager
    def forced_cleanup(self):
        global _FORCED
        old = _FORCED
        _FORCED = True
        try:
            yield
        finally:
            _FORCED = old

    def install(self) -> None:
        from zeta.core.checkpoints.workspace import WorkspaceSnapshotStore
        from zeta.core.session import SessionManager
        from zeta.core.store import ConversationStore
        from zeta.memory.auto import AutoMemoryReconciler
        from zeta.runtime.loop import AgentLoop
        from zeta.server import ZetaServer
        from zeta.tools import ToolRegistry
        from zeta.tools._shared.process import BackgroundTaskRegistry
        from zeta.tui.app import TUIApp

        owners = (
            (AgentLoop, "AgentLoop", "__init__", None),
            (ConversationStore, "ConversationStore", "__init__", "close"),
            (SessionManager, "SessionManager", "__init__", "_release_storage_lease"),
            (AutoMemoryReconciler, "AutoMemoryReconciler", "__init__", "close"),
            (WorkspaceSnapshotStore, "WorkspaceSnapshotStore", "__init__", "close"),
            (ToolRegistry, "ToolRegistry", "__init__", "close"),
            (BackgroundTaskRegistry, "BackgroundTaskRegistry", "__init__", "close"),
            (ZetaServer, "ZetaServer", "__init__", "close"),
            (TUIApp, "TUIApp", "__init__", "close"),
        )
        for cls, kind, init_name, close_name in owners:
            original_init = getattr(cls, init_name)
            self._originals.append((cls, init_name, original_init))

            def init(obj: Any, *args: Any, _original=original_init, _kind=kind, _close_name=close_name, **kwargs: Any) -> None:
                _original(obj, *args, **kwargs)
                self.constructed(obj, _kind)
                if _close_name == "_release_storage_lease":
                    original_release = getattr(obj, _close_name)

                    def release(*release_args: Any, **release_kwargs: Any) -> Any:
                        result = original_release(*release_args, **release_kwargs)
                        self.closed(obj)
                        return result

                    setattr(obj, _close_name, release)

            setattr(cls, init_name, init)
            if close_name is not None and close_name != "_release_storage_lease":
                original_close = getattr(cls, close_name)
                self._originals.append((cls, close_name, original_close))
                if inspect.iscoroutinefunction(original_close):
                    async def close(obj: Any, *args: Any, _original=original_close, **kwargs: Any) -> Any:
                        result = await _original(obj, *args, **kwargs)
                        self.closed(obj)
                        return result
                else:
                    def close(obj: Any, *args: Any, _original=original_close, **kwargs: Any) -> Any:
                        result = _original(obj, *args, **kwargs)
                        self.closed(obj)
                        return result
                setattr(cls, close_name, close)

    def uninstall(self) -> None:
        for cls, name, original in reversed(self._originals):
            setattr(cls, name, original)


_FORCED = False
_CAPTURE: OwnerCounts | None = None


def capture() -> OwnerCounts | None:
    return _CAPTURE


def start(path: str) -> OwnerCounts:
    global _CAPTURE
    recorder = OwnerCounts(path)
    recorder.path.parent.mkdir(parents=True, exist_ok=True)
    recorder.install()
    _CAPTURE = recorder
    return recorder


def stop() -> None:
    global _CAPTURE
    if _CAPTURE is not None:
        _CAPTURE.uninstall()
    _CAPTURE = None
