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
    def memory_format(self, project_id: str) -> int: ...
    def memory_log(self, project_id: str, *, limit: int = 100) -> list[dict[str, object]]: ...
    def undo_memory(self, project_id: str) -> list[tuple[str, str]]: ...
    def accept_memory(self, project_id: str, name: str) -> list[tuple[str, str]]: ...
    def entry_memory_log(
        self, project_id: str, *, entry_id: str | None = None, limit: int = 100
    ) -> list[dict[str, object]]: ...
    def _undo_entry_transaction(self, project_id: str, target_id: str | None = None): ...
    def _accept_memory_entry(self, project_id: str, entry_id: str): ...


def run_memory_command(
    registry: MemoryRegistry,
    project_id: str,
    args: str,
    reconciler: MemoryReconciler | None = None,
) -> str:
    """Run ``/memory`` actions shared by TUI and serve clients."""
    action = args.strip()
    format_one_usage = "usage: /memory [log|retry [receipt]|undo|accept <file>]"
    try:
        memory_format = registry.memory_format(project_id)
        usage = (
            format_one_usage
            if memory_format == 1
            else "usage: /memory [log [kind|entry-id]|retry [receipt]|undo [entry-id|version-id]|accept <entry-id>]"
        )
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
        if action == "undo" or action.startswith("undo "):
            if memory_format == 2:
                parts = action.split()
                if len(parts) > 2:
                    return usage
                target = parts[1] if len(parts) == 2 else None
                result = registry._undo_entry_transaction(project_id, target)
                ids = ", ".join(result.receipts[0].target_ids)
                return f"memory undo complete: {ids}" if ids else "memory undo complete"
            if action != "undo":
                return usage
            restored = registry.undo_memory(project_id)
            names = ", ".join(name for name, _ in restored)
            return f"memory undo complete: {names}" if names else "memory undo complete"
        if action.startswith("accept "):
            name = action.removeprefix("accept ").strip()
            if not name or " " in name:
                return usage
            if memory_format == 2:
                registry._accept_memory_entry(project_id, name)
            else:
                registry.accept_memory(project_id, name)
            return f"memory accepted: {name}"
        if action == "accept":
            return usage
        if memory_format == 2 and (action in {"", "log"} or action.startswith("log ")):
            parts = action.split()
            if len(parts) > 2:
                return usage
            target = parts[1] if len(parts) == 2 else None
            records = registry.entry_memory_log(project_id, entry_id=target, limit=20)
            if not records:
                return "memory log: empty"
            lines = []
            for record in records:
                for operation in record.get("operations", []):
                    ids = [*operation.get("target_ids", []), *operation.get("result_ids", [])]
                    lines.append(
                        f"{record.get('created_at', '?')} {record.get('version', '?')} "
                        f"{operation.get('type', '?')} {', '.join(dict.fromkeys(ids))} "
                        f"automatic={operation.get('automatic', '?')}"
                    )
            return "\n".join(lines) if lines else "memory log: empty"
        if action in {"", "log"}:
            registry.ensure_memory_supported(project_id)
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
