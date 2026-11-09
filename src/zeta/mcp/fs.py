"""Filesystem operations for MCP-owned directories."""

from __future__ import annotations

from pathlib import Path


def ensure_directory_below_existing_root(root: Path, directory: Path) -> None:
    """Create directory below root without creating root or its ancestors."""

    root = root.expanduser()
    directory = directory.expanduser()
    relative = directory.relative_to(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    current = root
    for component in relative.parts:
        current /= component
        current.mkdir(mode=0o700, parents=False, exist_ok=True)
