"""Serve adapters for the active session's background work."""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from ..tools._shared.run_work import background_task_row, run_work_snapshot
from . import ergonomics
from .protocol import ErrorDescriptor, ProtocolError
from .runtime import ServerRuntime


async def list_run_work(
    runtime: ServerRuntime,
    params: dict[str, object],
    run_blocking: Callable[[Callable[[], object]], Awaitable[object]],
) -> dict[str, object]:
    store = ergonomics.active(runtime, params)
    loop = runtime.loop
    if loop is None:
        raise ProtocolError(ErrorDescriptor.STATE_UNAVAILABLE, "session runtime is unavailable")
    result = await run_blocking(
        lambda: run_work_snapshot(
            loop.tool_registry.background_tasks,
            store.todo_items(),
        )
    )
    assert isinstance(result, dict)
    return result


async def cancel_background_task(
    runtime: ServerRuntime, params: dict[str, object]
) -> dict[str, object]:
    ergonomics.active(runtime, params)
    loop = runtime.loop
    if loop is None:
        raise ProtocolError(ErrorDescriptor.STATE_UNAVAILABLE, "session runtime is unavailable")
    task_id = params.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "task_id must be a nonempty string"
        )
    registry = loop.tool_registry.background_tasks
    try:
        await registry.kill(task_id, cause="serve client requested cancellation")
        task = next(item for item in registry.snapshot() if item.task_id == task_id)
    except (StopIteration, ValueError) as exc:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, str(exc)) from exc
    return {"task": background_task_row(task)}
