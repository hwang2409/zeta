"""Direct-user authorization for every explicit project-memory mutation.

CLI and slash adapters provide the interaction mechanism. This module owns the
rule that tool subprocesses and untrusted/non-interactive callers cannot write
memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import IO, Literal

from zeta.project_errors import ProjectRegistryError

TOOL_SUBPROCESS_ENV = "ZETA_TOOL_SUBPROCESS"
MutationChannel = Literal["terminal", "slash"]


@dataclass(frozen=True, slots=True)
class MemoryMutationAuthorization:
    """Authorize memory writes through one direct-user seam."""

    channel: MutationChannel
    stdin: IO[str] | None = None
    stdout: IO[str] | None = None

    @classmethod
    def terminal(
        cls, *, stdin: IO[str], stdout: IO[str]
    ) -> MemoryMutationAuthorization:
        return cls("terminal", stdin, stdout)

    @classmethod
    def direct_slash(cls) -> MemoryMutationAuthorization:
        """Mark a slash command already submitted through a user-facing client."""
        return cls("slash")

    def authorize(
        self,
        *,
        action: str,
        target: str,
        preview: str = "",
        noun: str = "memory",
    ) -> None:
        """Reject indirect callers and confirm terminal mutations before writes."""
        if os.environ.get(TOOL_SUBPROCESS_ENV):
            raise ProjectRegistryError(
                f"memory {action} is unavailable from a Zeta tool subprocess"
            )
        if self.channel == "slash":
            return
        if (
            self.stdin is None
            or self.stdout is None
            or not self.stdin.isatty()
            or not self.stdout.isatty()
        ):
            raise ProjectRegistryError(
                f"memory {action} requires an interactive terminal"
            )
        heading = (
            f"Automatic memory {noun}: {target}"
            if action == "accept"
            else f"Memory {action} {noun}: {target}"
        )
        print(heading, file=self.stdout)
        print(f"Preview: {preview[:240] or '(empty)'}", file=self.stdout)
        prompt = (
            f"Type '{target}' or 'accept' to accept this {noun}: "
            if action == "accept"
            else f"Type '{target}' or '{action}' to confirm: "
        )
        print(prompt, end="", file=self.stdout, flush=True)
        answer = self.stdin.readline().strip()
        if answer not in {target, action}:
            raise ProjectRegistryError(
                f"memory {action} cancelled; {noun} was not changed"
            )


def memory_accept_preview(registry: object, project_id: str, name: str) -> tuple[str, str]:
    """Return the noun and bounded preview for one acceptance target."""
    memory_format = getattr(registry, "memory_format", lambda _project_id: 1)(project_id)
    if memory_format == 2:
        value = registry._entry_memory_view(project_id)
        preview = next(
            (
                str(entry.get("text", ""))[:240]
                for entry in value.get("entries", [])
                if entry.get("id") == name
            ),
            "",
        )
        return "entry", preview
    preview = ""
    loader = getattr(registry, "load_memory_for_context", None)
    if loader is not None:
        for entry in loader(project_id):
            if getattr(entry, "name", None) == name:
                preview = getattr(entry, "content", "")[:240]
                break
    return "file", preview


__all__ = ["MemoryMutationAuthorization", "memory_accept_preview"]
