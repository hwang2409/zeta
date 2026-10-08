"""Shared user-facing project-memory slash command behavior."""

from __future__ import annotations

from typing import Protocol

from .memory.reconciliation_state import TerminalReceipt
from .project_registry import ProjectRegistryError


class MemoryReconciler(Protocol):
    def terminal_receipts(self) -> tuple[TerminalReceipt, ...]: ...
    def retry_terminal(self, key: str) -> bool: ...


class MemoryRegistry(Protocol):
    def ensure_memory_supported(self, project_id: str) -> None: ...
    def memory_log(self, project_id: str, *, limit: int = 100) -> list[dict[str, object]]: ...
    def undo_memory(self, project_id: str) -> list[tuple[str, str]]: ...
    def accept_memory(self, project_id: str, name: str) -> list[tuple[str, str]]: ...


def run_memory_command(
    registry: MemoryRegistry,
    project_id: str,
    args: str,
    reconciler: MemoryReconciler | None = None,
) -> str:
    """Run ``/memory`` actions shared by TUI and serve clients."""
    action = args.strip()
    usage = "usage: /memory [log|retry [receipt]|undo|accept <file>]"
    try:
        registry.ensure_memory_supported(project_id)
        if action == "retry":
            if reconciler is None:
                return "memory retry: unavailable for this session"
            receipts = reconciler.terminal_receipts()
            if not receipts:
                return "memory retry: no terminal receipts"
            return "\n".join(
                f"{item.key} seq {item.seq_start}-{item.seq_end} "
                f"attempts={item.attempt_count} {item.validation_summary}"
                for item in receipts
            )
        if action.startswith("retry "):
            prefix = action.removeprefix("retry ").strip()
            if reconciler is None or not prefix or " " in prefix:
                return usage
            matches = [
                item.key
                for item in reconciler.terminal_receipts()
                if item.key.startswith(prefix)
            ]
            if len(matches) != 1:
                return "memory retry: receipt not found or prefix is ambiguous"
            if not reconciler.retry_terminal(matches[0]):
                return "memory retry: receipt not found"
            return f"memory retry queued: {matches[0]}"
        if action == "undo":
            restored = registry.undo_memory(project_id)
            names = ", ".join(name for name, _ in restored)
            return f"memory undo complete: {names}" if names else "memory undo complete"
        if action.startswith("accept "):
            name = action.removeprefix("accept ").strip()
            if not name or " " in name:
                return usage
            registry.accept_memory(project_id, name)
            return f"memory accepted: {name}"
        if action in {"", "log"}:
            records = registry.memory_log(project_id, limit=20)
            if not records:
                return "memory log: empty"
            return "\n".join(
                f"{item.get('created_at', '?')} {item.get('kind', '?')} "
                f"{', '.join(item.get('files', []))} "
                f"[{item.get('provenance', {})}]"
                for item in records
            )
    except ProjectRegistryError as exc:
        return f"memory: {exc}"
    return usage


__all__ = ["run_memory_command"]
