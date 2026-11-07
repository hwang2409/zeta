from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.approval import ApprovedCwdExecution
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
    if approved_cwd is None and requested_cwd is None:
        removed = False
        if Path(start_cwd) == registry.cwd:
            try:
                registry.verify_cwd_identity()
            except SessionCwdRemoved:
                removed = True
        else:
            removed = not Path(start_cwd).is_dir()
        if removed:
            fallback = nearest_existing_directory(start_cwd)
            recovery = (start_cwd, fallback)
            start_cwd = fallback

    if approved_cwd is not None:
        fd = open_verified_directory(start_cwd, approved_cwd.identity)
    elif Path(start_cwd) == registry.cwd:
        fd = registry._open_cwd()
    else:
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

    worker = asyncio.create_task(
        asyncio.to_thread(_bind_shell_cwd, registry, requested_cwd, approved_cwd)
    )
    try:
        start_cwd, fd, recovery = await asyncio.shield(worker)
    except BaseException:
        try:
            result = await asyncio.shield(worker)
        except BaseException as cleanup_error:  # noqa: BLE001 - drain every worker outcome
            cleanup_error.add_note("shell cwd binding failed during cancellation")
        else:
            os.close(result[1])
        raise
    if recovery is not None:
        registry.commit_cwd_recovery(*recovery)
    return start_cwd, fd
