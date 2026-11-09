"""Serve handlers for background work and todo snapshots."""

from __future__ import annotations

from ...tools._shared.run_work import background_task_row, run_work_snapshot
from ..protocol import ErrorDescriptor, EventDescriptor, ProtocolError
from . import validation


async def list_run_work(client, request_id, params):
    del request_id
    runtime = client.server.runtime
    store = validation.active(runtime, params)
    if runtime.loop is None:
        raise ProtocolError(ErrorDescriptor.STATE_UNAVAILABLE, "session runtime is unavailable")
    result = await client.server.control._run_blocking(
        lambda: run_work_snapshot(runtime.loop.tool_registry.background_tasks, store.todo_items())
    )
    assert isinstance(result, dict)
    return result


async def cancel_background_task(client, request_id, params):
    del request_id
    runtime = client.server.runtime
    validation.active(runtime, params)
    if runtime.loop is None:
        raise ProtocolError(ErrorDescriptor.STATE_UNAVAILABLE, "session runtime is unavailable")
    task_id = params.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "task_id must be a nonempty string")
    registry = runtime.loop.tool_registry.background_tasks
    task = next((item for item in registry.snapshot() if item.task_id == task_id), None)
    if task is None:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "unknown background task")
    client.server.control._publish_run_work_event(
        runtime.session_id,
        EventDescriptor.BACKGROUND_TASK_UPDATED,
        {"task": background_task_row(task, phase="stopping")},
    )
    try:
        await registry.kill(task_id, cause="serve client requested cancellation")
        task = next(item for item in registry.snapshot() if item.task_id == task_id)
    except (StopIteration, ValueError) as exc:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, str(exc)) from exc
    return {"task": background_task_row(task)}
