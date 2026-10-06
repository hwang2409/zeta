"""Guards for CLI actions that require direct user authorization."""

from __future__ import annotations

import os
from typing import IO

from ..project_registry import ProjectRegistryError

TOOL_SUBPROCESS_ENV = "ZETA_TOOL_SUBPROCESS"


def confirm_memory_accept(
    registry: object,
    project_id: str,
    name: str,
    *,
    stdin: IO[str],
    stdout: IO[str],
) -> None:
    """Require an interactive human confirmation before accepting memory."""
    if os.environ.get(TOOL_SUBPROCESS_ENV) == "1":
        raise ProjectRegistryError("memory accept is unavailable from a Zeta tool subprocess")
    if not stdin.isatty() or not stdout.isatty():
        raise ProjectRegistryError("memory accept requires an interactive terminal")
    preview = ""
    loader = getattr(registry, "load_memory_for_context", None)
    if loader is not None:
        for entry in loader(project_id):
            if getattr(entry, "name", None) == name:
                preview = getattr(entry, "content", "")[:240]
                break
    print(f"Automatic memory file: {name}", file=stdout)
    print(f"Preview: {preview or '(empty)'}", file=stdout)
    print(f"Type '{name}' or 'accept' to accept this file: ", end="", file=stdout, flush=True)
    answer = stdin.readline().strip()
    if answer not in {name, "accept"}:
        raise ProjectRegistryError("memory accept cancelled; file was not changed")


__all__ = ["confirm_memory_accept"]
