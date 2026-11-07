from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from ...core.approval import ApprovedCwdExecution
from .cwd_state import SessionCwdRemoved, open_verified_directory
from .sandbox import resolve_shell_cwd

if TYPE_CHECKING:
    from ..registry import ToolRegistry


def _resolve_start_cwd(
    registry: ToolRegistry,
    requested_cwd: object,
    approved_cwd: ApprovedCwdExecution | None,
    recover: bool,
) -> str:
    if approved_cwd is not None:
        return approved_cwd.cwd
    if requested_cwd is None and registry.bash_cwd == str(registry.cwd):
        try:
            registry.verify_cwd_identity()
        except SessionCwdRemoved:
            if not recover:
                raise
            return registry.recover_shell_cwd(registry.bash_cwd)
    if recover:
        return registry.resolve_shell_cwd(requested_cwd)
    resolved = resolve_shell_cwd(registry.bash_cwd, requested_cwd)
    if requested_cwd is None and not Path(resolved).is_dir():
        raise SessionCwdRemoved(resolved)
    return resolved


def bind_shell_cwd(
    registry: ToolRegistry,
    requested_cwd: object,
    approved_cwd: ApprovedCwdExecution | None = None,
    *,
    recover: bool = True,
) -> tuple[str, int]:
    """Resolve and bind one shell cwd descriptor for process launch."""

    start_cwd = _resolve_start_cwd(registry, requested_cwd, approved_cwd, recover)
    if (
        approved_cwd is not None
        or requested_cwd is not None
        or Path(start_cwd) == registry.cwd
    ):
        registry.verify_cwd_identity(allow_removed=True)
    if approved_cwd is not None:
        return start_cwd, open_verified_directory(start_cwd, approved_cwd.identity)
    if Path(start_cwd) == registry.cwd:
        try:
            return start_cwd, registry._open_cwd()
        except SessionCwdRemoved:
            if requested_cwd is not None or not recover:
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


async def bind_shell_cwd_async(
    registry: ToolRegistry,
    requested_cwd: object,
    approved_cwd: ApprovedCwdExecution | None = None,
) -> tuple[str, int]:
    """Bind a shell cwd without blocking the runtime event loop."""

    recovery = await asyncio.to_thread(
        _find_removed_cwd, registry, requested_cwd, approved_cwd
    )
    if recovery is not None:
        registry.recover_shell_cwd(recovery)
    binding = asyncio.create_task(
        asyncio.to_thread(
            bind_shell_cwd,
            registry,
            requested_cwd,
            approved_cwd,
            recover=False,
        )
    )
    try:
        return await asyncio.shield(binding)
    except BaseException:
        if binding.done():
            _close_binding(binding)
        else:
            binding.add_done_callback(_close_binding)
        raise


def _find_removed_cwd(
    registry: ToolRegistry,
    requested_cwd: object,
    approved_cwd: ApprovedCwdExecution | None,
) -> str | None:
    if approved_cwd is not None or requested_cwd is not None:
        return None
    if registry.bash_cwd == str(registry.cwd):
        try:
            registry.verify_cwd_identity()
        except SessionCwdRemoved:
            return registry.bash_cwd
        if not registry.cwd.is_dir():
            return registry.bash_cwd
        return None
    resolved = resolve_shell_cwd(registry.bash_cwd, requested_cwd)
    if not Path(resolved).is_dir():
        return resolved
    return None


def _close_binding(binding: asyncio.Task[tuple[str, int]]) -> None:
    if binding.cancelled() or binding.exception() is not None:
        return
    os.close(binding.result()[1])
