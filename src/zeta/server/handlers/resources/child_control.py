"""Controller commands for the currently attached agent tree."""

from __future__ import annotations

from typing import Any

from ...resource_protocol import Access, ResourceMethod, require_params

FEATURE = "child_run_control_v1"


def _string(params: dict[str, Any], key: str, max_bytes: int) -> str:
    from ...protocol import invalid_resource_params

    value = params[key]
    if type(value) is not str or not value.strip() or "\0" in value:
        raise invalid_resource_params()
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError:
        raise invalid_resource_params() from None
    if size > max_bytes:
        raise invalid_resource_params()
    return value


def _owner_error(code: str):
    from ...protocol import ErrorDescriptor, invalid_resource_params, resource_error

    if code == "invalid_params":
        return invalid_resource_params()
    return resource_error({
        "child_run_not_found": ErrorDescriptor.CHILD_RUN_NOT_FOUND,
        "operation_busy": ErrorDescriptor.OPERATION_BUSY,
        "state_unavailable": ErrorDescriptor.STATE_UNAVAILABLE,
    }[code])


def _active(runtime, params):
    from ...protocol import ErrorDescriptor, resource_error

    session_id = _string(params, "session_id", 256)
    if runtime.opened is None:
        raise resource_error(ErrorDescriptor.NO_ACTIVE_SESSION)
    if runtime.session_id != session_id:
        raise resource_error(ErrorDescriptor.SESSION_MISMATCH)
    if runtime.loop is None:
        raise resource_error(ErrorDescriptor.NO_ACTIVE_SESSION)
    return runtime.loop


async def child_run_spawn(client, request_id, params, controller):
    from ....agent.runner import ChildControlError, spawn_child
    from ...protocol import ErrorDescriptor, invalid_resource_params, resource_error

    require_params(params, "session_id", "agent_type", "task", optional=("cwd", "background"))
    loop = _active(controller.runtime, params)
    agent_type = _string(params, "agent_type", 256)
    task = _string(params, "task", 64 * 1024)
    cwd = _string(params, "cwd", 4096) if "cwd" in params else None
    if params.get("background", True) is not True:
        raise invalid_resource_params()
    control = controller.control
    control._reject_if_unhealthy()
    if control.client_turn_detached():
        raise resource_error(ErrorDescriptor.NO_ACTIVE_SESSION)
    if control.state_change_busy():
        raise resource_error(ErrorDescriptor.OPERATION_BUSY)
    try:
        run_id = await spawn_child(loop, agent_type=agent_type, task=task, cwd=cwd)
    except ChildControlError as exc:
        raise _owner_error(exc.code) from None
    return {"data": {"accepted": True, "run_id": run_id}}


async def child_run_cancel(client, request_id, params, controller):
    from ....agent.runner import ChildControlError, cancel_child_run
    from ...protocol import invalid_resource_params

    require_params(params, "session_id", "run_id", optional=("reason",))
    loop = _active(controller.runtime, params)
    run_id = _string(params, "run_id", 256)
    if params.get("reason", "user_cancelled") != "user_cancelled":
        raise invalid_resource_params()
    try:
        state = await cancel_child_run(loop, run_id)
    except ChildControlError as exc:
        raise _owner_error(exc.code) from None
    return {"data": {"accepted": True, "run_id": run_id, "state": state}}


METHODS = (
    ResourceMethod("child_run_spawn", Access.CONTROLLER_WRITE, child_run_spawn),
    ResourceMethod("child_run_cancel", Access.CONTROLLER_WRITE, child_run_cancel),
)
