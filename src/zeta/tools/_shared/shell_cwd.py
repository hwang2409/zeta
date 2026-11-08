from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.approval import ApprovedCwdExecution
from ...core.thread_call import run_owned_thread_call
from .cwd_state import (
    SessionCwdRemoved,
    nearest_existing_directory,
    open_verified_directory,
)
from .sandbox import resolve_shell_cwd

if TYPE_CHECKING:
    from ..registry import ToolRegistry


def _bind_shell_cwd(
    registry: ToolRegistry,
    requested_cwd: object,
    approved_cwd: ApprovedCwdExecution | None,
) -> tuple[str, int, tuple[str, str] | None]:
    """Resolve and bind a cwd off-loop without changing registry state."""

    start_cwd = (
        approved_cwd.cwd
        if approved_cwd is not None
        else resolve_shell_cwd(registry.bash_cwd, requested_cwd)
    )
    recovery: tuple[str, str] | None = None
    if approved_cwd is not None:
        fd = open_verified_directory(start_cwd, approved_cwd.identity)
    elif Path(start_cwd) == registry.cwd:
        try:
            fd = registry._open_cwd()
        except SessionCwdRemoved:
            if requested_cwd is not None:
                raise
            fallback = nearest_existing_directory(start_cwd)
            recovery = (start_cwd, fallback)
            start_cwd = fallback
            identity = os.lstat(fallback)
            fd = open_verified_directory(fallback, (identity.st_dev, identity.st_ino))
    else:
        if requested_cwd is None and not Path(start_cwd).is_dir():
            fallback = nearest_existing_directory(start_cwd)
            recovery = (start_cwd, fallback)
            start_cwd = fallback
        bound_cwd = os.path.realpath(start_cwd)
        try:
            identity = os.lstat(bound_cwd)
        except OSError as exc:
            raise ValueError(f"could not execute command: {exc}") from exc
        fd = open_verified_directory(bound_cwd, (identity.st_dev, identity.st_ino))
    return start_cwd, fd, recovery


async def bind_shell_cwd_async(
    registry: ToolRegistry,
    requested_cwd: object,
    approved_cwd: ApprovedCwdExecution | None = None,
) -> tuple[str, int]:
    """Bind a shell cwd off-loop and commit recovery only after its fd is bound."""

    start_cwd, fd, recovery = await run_owned_thread_call(
        lambda: _bind_shell_cwd(registry, requested_cwd, approved_cwd),
        lambda result: os.close(result[1]),
    )
    if recovery is not None:
        registry.commit_cwd_recovery(*recovery)
    return start_cwd, fd
