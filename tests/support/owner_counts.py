from __future__ import annotations

import inspect
import threading
import weakref
from collections import defaultdict
from contextvars import ContextVar
from functools import wraps
from pathlib import Path
from typing import Any

# conftest's safety net sets this for its cleanup phase. Tasks and asyncio.to_thread
# copy it, so closes it starts there are not test closes. shortcut: a close the
# safety net reaches through a plain thread or run_in_executor would count as a
# test close; no gated close runs that way today.
SAFETY_NET_CLOSING: ContextVar[bool] = ContextVar("safety_net_closing", default=False)

CLOSE_GUIDANCE = {
    "AgentLoop": "await loop.close()",
    "ConversationStore": "store.close()",
    "AutoMemoryReconciler": "await reconciler.close()",
    "WorkspaceSnapshotStore": "workspace.close()",
    "ToolRegistry": "await registry.close()",
    "BackgroundTaskRegistry": "await registry.close()",
    "ZetaServer": "await server.close()",
    "TUIApp": "async with owned_app(app) (tests/support/owned.py)",
}

OwnerRow = tuple[str, str, int]

_installed: OwnerCounts | None = None


def dropped_for_gc(owner: Any, reason: str) -> Any:
    """Stop counting this one owner: the test drops it on purpose so the GC
    finalizes it. Other owners in the same test still count. Returns the owner."""
    assert reason, "say why the test needs the collector to free this owner"
    if _installed is not None:
        _installed.forget(owner)
    return owner


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
            if not SAFETY_NET_CLOSING.get():
                self._counts[record[1:]][1] += 1
            self._records.pop(id(obj), None)

    def forget(self, obj: Any) -> None:
        with self._lock:
            record = self._records.get(id(obj))
            if record is None or record[0]() is not obj:
                return
            self._counts[record[1:]][0] -= 1
            self._records.pop(id(obj))

    def rows(self) -> list[OwnerRow]:
        return sorted(self._rows)

    def install(self) -> None:
        global _installed
        _installed = self
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
        global _installed
        _installed = None
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


def parse_baseline(text: str) -> dict[tuple[str, str], int]:
    baseline = {}
    for raw_line in text.splitlines():
        label, raw_count = raw_line.rsplit("=", 1)
        nodeid, kind = label.rsplit("::", 1)
        baseline[(nodeid, kind)] = int(raw_count)
    return baseline


def read_baseline(path: Path) -> dict[tuple[str, str], int]:
    if not path.exists():
        return {}
    return parse_baseline(path.read_text(encoding="utf-8"))


def baseline_growth(base_text: str, head_text: str) -> list[str]:
    """Rows the head baseline adds or raises over the base; only lowering is allowed."""
    base = parse_baseline(base_text)
    return [
        f"owner leak baseline: {nodeid}::{kind}={count} (base "
        f"{base.get((nodeid, kind), 'has no row')}); do not add or raise baseline "
        f"rows: close the owner in the test ({CLOSE_GUIDANCE[kind]})"
        for (nodeid, kind), count in sorted(parse_baseline(head_text).items())
        if count > base.get((nodeid, kind), 0)
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
            f"(baseline {allowed}); close the owner with {guidance}; "
            "do not add rows to tests/owner_leak_ratchet.txt; if the test drops "
            "it on purpose for the GC, pass it through dropped_for_gc(owner, reason) "
            "(tests/support/owner_counts.py)"
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
