"""Shared user-facing project-memory slash command behavior."""

from __future__ import annotations

from typing import Protocol

from .project_registry import ProjectRegistryError


class MemoryRegistry(Protocol):
    def memory_log(self, project_id: str, *, limit: int = 100) -> list[dict[str, object]]: ...
    def undo_memory(self, project_id: str) -> list[tuple[str, str]]: ...
    def accept_memory(self, project_id: str, name: str) -> list[tuple[str, str]]: ...


def run_memory_command(registry: MemoryRegistry, project_id: str, args: str) -> str:
    """Run ``/memory`` actions shared by TUI and serve clients."""
    action = args.strip()
    try:
        if action == "undo":
            restored = registry.undo_memory(project_id)
            names = ", ".join(name for name, _ in restored)
            return f"memory undo complete: {names}" if names else "memory undo complete"
        if action.startswith("accept "):
            name = action.removeprefix("accept ").strip()
            if not name or " " in name:
                return "usage: /memory [log|undo|accept <file>]"
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
    return "usage: /memory [log|undo|accept <file>]"


__all__ = ["run_memory_command"]
