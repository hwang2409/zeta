"""Shared user-facing project-memory slash command behavior."""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Protocol

from .memory.reconciliation_state import TerminalReceipt
from .memory.user_authorization import MemoryMutationAuthorization
from .project_registry import ProjectRegistryError

SUPPORTED_MEMORY_FORMATS = frozenset({1, 2})


class MemoryReconciler(Protocol):
    @property
    def scanner_failure(self) -> tuple[int, str] | None: ...
    def terminal_receipts(self) -> tuple[TerminalReceipt, ...]: ...
    def retry_terminal(self, key: str) -> bool: ...
    def resume_recovery(self, root_session_id: str, quarantine_id: str) -> Awaitable[str]: ...


class MemoryRegistry(Protocol):
    def ensure_memory_supported(self, project_id: str) -> None: ...
    def memory_format(self, project_id: str) -> int: ...
    def memory_log(self, project_id: str, *, limit: int = 100) -> list[dict[str, object]]: ...
    def undo_memory(
        self, project_id: str, *, expected_current: str | None = None
    ) -> list[tuple[str, str]]: ...
    def accept_memory(self, project_id: str, name: str) -> list[tuple[str, str]]: ...
    def _entry_memory_log(
        self,
        project_id: str,
        *,
        entry_id: str | None = None,
        kind: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]: ...
    def _undo_entry_transaction(self, project_id: str, target_id: str | None = None): ...
    def _accept_memory_entry(self, project_id: str, entry_id: str): ...
    def _resolve_memory_entries(
        self, project_id: str, entry_ids: tuple[str, ...], *, reason: str = "resolved by user"
    ): ...


def _run_recovery_command(
    action: str,
    usage: str,
    reconciler: MemoryReconciler | None,
    authorization: MemoryMutationAuthorization | None,
) -> str | Awaitable[str]:
    parts = action.split()
    if len(parts) != 4 or parts[3] != "resume" or reconciler is None:
        return usage
    root_session_id, quarantine_id = parts[1:3]
    _authorize(authorization, "recovery", f"{root_session_id} {quarantine_id}")
    return reconciler.resume_recovery(root_session_id, quarantine_id)


def _run_retry_command(
    action: str,
    usage: str,
    reconciler: MemoryReconciler | None,
    authorization: MemoryMutationAuthorization | None,
) -> str:
    if reconciler is None:
        return "memory retry: unavailable for this session" if action == "retry" else usage
    if action == "retry":
        receipts = reconciler.terminal_receipts()
        if not receipts:
            return "memory retry: no terminal receipts"
        return "\n".join(
            f"{item.key} seq {item.seq_start}-{item.seq_end} "
            f"attempts={item.attempt_count} {item.validation_summary}"
            for item in receipts
        )
    prefix = action.removeprefix("retry ").strip()
    if not prefix or " " in prefix:
        return usage
    matches = [
        item.key for item in reconciler.terminal_receipts() if item.key.startswith(prefix)
    ]
    if len(matches) != 1:
        return "memory retry: receipt not found or prefix is ambiguous"
    scanner_failure = reconciler.scanner_failure
    if scanner_failure is not None:
        seq, cause = scanner_failure
        return (
            "memory retry refused: automatic project-memory reconciliation "
            f"is blocked by {cause} (seq {seq}); repair the transcript first"
        )
    _authorize(authorization, "retry", matches[0])
    if not reconciler.retry_terminal(matches[0]):
        return "memory retry: receipt not found"
    return f"memory retry queued: {matches[0]}"


def run_memory_command(
    registry: MemoryRegistry,
    project_id: str,
    args: str,
    reconciler: MemoryReconciler | None = None,
    authorization: MemoryMutationAuthorization | None = None,
) -> str | Awaitable[str]:
    """Run ``/memory`` actions shared by TUI and serve clients."""
    action = args.strip()
    format_one_usage = "usage: /memory [log|retry [receipt]|undo [version-id]|accept <file>|recovery <session-id> <quarantine-id> resume]"
    try:
        memory_format = registry.memory_format(project_id)
        usage = (
            format_one_usage
            if memory_format == 1
            else "usage: /memory [log [kind|entry-id]|retry [receipt]|undo [entry-id|version-id]|accept <entry-id>|resolve <entry-id>...]"
        )
        if action.startswith("recovery "):
            return _run_recovery_command(action, usage, reconciler, authorization)
        if action == "retry" or action.startswith("retry "):
            return _run_retry_command(action, usage, reconciler, authorization)
        if action == "undo" or action.startswith("undo "):
            if memory_format == 2:
                parts = action.split()
                if len(parts) > 2:
                    return usage
                target = parts[1] if len(parts) == 2 else None
                _authorize(authorization, "undo", target or "latest")
                result = registry._undo_entry_transaction(project_id, target)
                ids = ", ".join(result.receipts[0].target_ids)
                return f"memory undo complete: {ids}" if ids else "memory undo complete"
            parts = action.split()
            if len(parts) > 2:
                return usage
            expected_current = parts[1] if len(parts) == 2 else None
            _authorize(authorization, "undo", expected_current or "latest")
            restored = (
                registry.undo_memory(project_id, expected_current=expected_current)
                if expected_current is not None
                else registry.undo_memory(project_id)
            )
            names = ", ".join(name for name, _ in restored)
            return f"memory undo complete: {names}" if names else "memory undo complete"
        if action.startswith("accept "):
            name = action.removeprefix("accept ").strip()
            if not name or " " in name:
                return usage
            _authorize(authorization, "accept", name)
            if memory_format == 2:
                registry._accept_memory_entry(project_id, name)
            else:
                registry.accept_memory(project_id, name)
            return f"memory accepted: {name}"
        if action == "accept":
            return usage
        if memory_format == 2 and action.startswith("resolve "):
            entry_ids = tuple(action.removeprefix("resolve ").split())
            if not entry_ids:
                return usage
            target = ", ".join(entry_ids)
            _authorize(authorization, "resolve", target)
            registry._resolve_memory_entries(project_id, entry_ids)
            return f"memory resolved: {target}"
        if action == "resolve":
            return usage
        if memory_format == 2 and (action in {"", "log"} or action.startswith("log ")):
            parts = action.split()
            if len(parts) > 2:
                return usage
            target = parts[1] if len(parts) == 2 else None
            records = registry._entry_memory_log(
                project_id,
                entry_id=target if target and target.startswith("m_") else None,
                kind=target if target and not target.startswith("m_") else None,
                limit=20,
            )
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


def _authorize(
    authorization: MemoryMutationAuthorization | None, action: str, target: str
) -> None:
    if authorization is None:
        raise ProjectRegistryError(
            f"memory {action} is unavailable without direct user authorization"
        )
    authorization.authorize(action=action, target=target)


__all__ = ["run_memory_command"]
