"""Short, process-unique Unix socket paths for tests."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

_CREATED_PATHS: set[Path] = set()


def short_socket_path(root: Path, tag: str = "") -> Path:
    """Return and record a short socket path unique to this process and test."""
    identity = f"{os.getpid()}\0{root.resolve()}\0{tag}".encode()
    token = hashlib.sha256(identity).hexdigest()[:20]
    socket_path = Path("/tmp") / f"zeta-test-{token}.sock"
    _CREATED_PATHS.update((socket_path, Path(f"{socket_path}.lock")))
    return socket_path


def cleanup_short_socket_paths() -> None:
    """Remove only paths handed out in this process."""
    while _CREATED_PATHS:
        _CREATED_PATHS.pop().unlink(missing_ok=True)
