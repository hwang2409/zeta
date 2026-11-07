from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.approval import ApprovedCwdExecution
from .cwd_state import SessionCwdRemoved, open_verified_directory

if TYPE_CHECKING:
    from ..registry import ToolRegistry


def bind_shell_cwd(
    registry: ToolRegistry,
    requested_cwd: object,
    approved_cwd: ApprovedCwdExecution | None = None,
) -> tuple[str, int]:
    """Resolve and bind one shell cwd descriptor for process launch."""

    start_cwd = (
        approved_cwd.cwd
        if approved_cwd is not None
        else registry.resolve_shell_cwd(requested_cwd)
    )
    registry.verify_cwd_identity(allow_removed=True)
    if approved_cwd is not None:
        return start_cwd, open_verified_directory(start_cwd, approved_cwd.identity)
    if Path(start_cwd) == registry.cwd:
        try:
            return start_cwd, registry._open_cwd()
        except SessionCwdRemoved:
            if requested_cwd is not None:
                raise
            start_cwd = registry.recover_shell_cwd(start_cwd)
    bound_cwd = os.path.realpath(start_cwd)
    try:
        identity = os.lstat(bound_cwd)
    except OSError as exc:
        raise ValueError(f"could not execute command: {exc}") from exc
    return start_cwd, open_verified_directory(
        bound_cwd, (identity.st_dev, identity.st_ino)
    )
