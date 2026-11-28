from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum, auto
from functools import partial
from typing import Any

from ...mcp import MCPManagementError, MCPToolRefreshError
from .. import abort_scope, accounts, model_selection, slash_commands
from .. import control as session_grants
from ..protocol import (
    APPROVAL_SCOPES,
    RESOURCE_METHODS,
    Access,
    ApprovalDecision,
    ErrorDescriptor,
    MethodDescriptor,
    ProtocolError,
    feature_error,
    protocol_error,
)
from ..resource_protocol import ResourceDescriptor, require_access
from ..session_controllers import SessionUse
from ..session_replay import join_history_and_subscribe
from . import (
    child_runs,
    lifecycle,
    path_completion,
    session,
    settings,
    validation,
)
from .run_work import cancel_background_task, list_run_work
from .state import (
    state_accounts,
    state_agents,
    state_attention,
    state_background_tasks,
    state_catalog,
    state_inbox,
    state_inbox_detail,
    state_memory,
    state_memory_detail,
    state_projects,
    state_retention,
    state_session,
    state_sessions,
    state_settings,
    state_storage,
)
from .telemetry import (
    telemetry_agent_tree,
    telemetry_errors,
    telemetry_models,
    telemetry_overview,
    telemetry_series,
    telemetry_sessions,
    telemetry_status,
    telemetry_tools,
    telemetry_trace,
)

Handler = Callable[
    [object, str | int, dict[str, Any], SessionUse | None], Awaitable[object]
]


class Resolve(Enum):
    """Where a request resolves its one record."""

    # session mutations: base runs the owned control's health check before the
    # version and feature checks, so this resolve point keeps that error order
    OWNED_FIRST = auto()
    OWNED = auto()  # after version and feature checks
    OBSERVER_VIEW = auto()  # owned, except for an observer in multi mode
    SELECTED = auto()  # once, at the handler's validated session_id
    NONE = auto()  # global or sessionless: the handler gets None


_O, _S, _N, _F = Resolve.OWNED, Resolve.SELECTED, Resolve.NONE, Resolve.OWNED_FIRST
_V = Resolve.OBSERVER_VIEW
_M = MethodDescriptor
RESOLVE: dict[MethodDescriptor | tuple[str, str], Resolve] = {
    _M.LIST_SESSIONS: _N,
    _M.NEW_SESSION: _O,
    _M.RESUME: _O,
    _M.SEND: _O,
    _M.STEER: _O,
    _M.APPROVE: _F,
    _M.DENY: _F,
    _M.ABORT: _F,
    _M.STATUS: _V,
    _M.RENAME_SESSION: _F,
    _M.DELETE_SESSION: _O,
    _M.REPAIR_TRANSCRIPT: _F,
    _M.SESSION_TREE: _S,
    _M.SESSION_HISTORY: _S,
    _M.SESSION_HISTORY_AND_SUBSCRIBE: _S,
    _M.CHILD_RUN_HISTORY: _S,
    _M.CHILD_RUN_EVENTS: _S,
    _M.CONTENT_READ: _V,
    _M.SWITCH_BRANCH: _F,
    _M.FORK_MESSAGE: _F,
    _M.MODEL_CATALOG: _V,
    _M.SESSION_SETTINGS: _V,
    _M.SET_SETTINGS: _F,
    _M.SET_COMPACTION: _F,
    _M.SEND_IMAGES: _O,
    _M.PATH_COMPLETE: _S,
    _M.SLASH_LIST: _O,
    _M.SLASH_RUN: _F,
    _M.ACCOUNT_LIST: _O,
    _M.ACCOUNT_LOGOUT: _O,
    _M.ACCOUNT_REFRESH: _O,
    _M.ACCOUNT_SWITCH: _O,
    # controller writes take the owned control's wake hold, as on base
    _M.LOGIN_START: _O,
    _M.LOGIN_STATUS: _N,
    _M.LOGIN_CANCEL: _O,
    _M.LOGIN_PROVIDERS: _N,
    _M.PING: _N,
    _M.SET_TURN_CONTEXT: _F,
    _M.LIST_PROJECTS: _N,
    _M.PROJECT_SHOW: _N,
    _M.PROJECT_MEMORY_LOG: _N,
    _M.PROJECT_INBOX: _N,
    _M.CLEAR_STEERING: _F,
    _M.DELIVERY_STATUS: _V,
    _M.SESSION_GRANTS: _V,
    _M.REVOKE_SESSION_GRANTS: _F,
    _M.RUN_WORK_LIST: _O,
    _M.BACKGROUND_TASK_CANCEL: _F,
    _M.CHILD_RUN_LIST: _S,
    _M.CHILD_RUN_SEND: _O,
    _M.MCP_REFRESH_TOOLS: _F,
    _M.STATE_SESSIONS: _N,
    _M.STATE_SESSION: _N,
    _M.STATE_AGENTS: _N,
    _M.STATE_BACKGROUND_TASKS: _N,
    _M.STATE_CATALOG: _N,
    _M.STATE_PROJECTS: _N,
    _M.STATE_MEMORY: _N,
    _M.STATE_MEMORY_DETAIL: _N,
    _M.STATE_INBOX: _N,
    _M.STATE_INBOX_DETAIL: _N,
    _M.STATE_ATTENTION: _N,
    _M.STATE_SETTINGS: _N,
    _M.STATE_ACCOUNTS: _N,
    _M.STATE_STORAGE: _N,
    _M.STATE_RETENTION: _N,
    _M.TELEMETRY_STATUS: _N,
    _M.TELEMETRY_SESSIONS: _N,
    _M.TELEMETRY_TRACE: _N,
    _M.TELEMETRY_OVERVIEW: _N,
    _M.TELEMETRY_MODELS: _N,
    _M.TELEMETRY_TOOLS: _N,
    _M.TELEMETRY_ERRORS: _N,
    _M.TELEMETRY_SERIES: _N,
    _M.TELEMETRY_AGENT_TREE: _N,
    # account_list_idle_v1
    ("account_list_idle_v1", "account_list"): _O,
    ("account_list_idle_v1", "login_start"): _O,
    # agent_run_catalog_v1
    ("agent_run_catalog_v1", "agent_run_catalog"): _N,
    # child_run_control_v1
    ("child_run_control_v1", "child_run_spawn"): _O,
    ("child_run_control_v1", "child_run_cancel"): _O,
    # home_settings_v1
    ("home_settings_v1", "home_settings_get"): _N,
    ("home_settings_v1", "home_settings_update"): _O,
    # library_read_v1
    ("library_read_v1", "skill_catalog"): _V,
    ("library_read_v1", "skill_read"): _V,
    ("library_read_v1", "agent_profile_catalog"): _V,
    ("library_read_v1", "agent_profile_read"): _V,
    # library_write_v1
    ("library_write_v1", "skill_write"): _O,
    ("library_write_v1", "skill_delete"): _O,
    ("library_write_v1", "agent_profile_write"): _O,
    ("library_write_v1", "agent_profile_delete"): _O,
    # mcp_management_v1
    ("mcp_management_v1", "mcp_server_list"): _V,
    ("mcp_management_v1", "mcp_server_add"): _O,
    ("mcp_management_v1", "mcp_server_remove"): _O,
    ("mcp_management_v1", "mcp_server_set_enabled"): _O,
    ("mcp_management_v1", "mcp_server_set_trusted"): _O,
    ("mcp_management_v1", "mcp_server_test"): _O,
    ("mcp_management_v1", "mcp_auth_start"): _O,
    ("mcp_management_v1", "mcp_auth_status"): _N,
    ("mcp_management_v1", "mcp_auth_cancel"): _O,
    ("mcp_management_v1", "mcp_auth_logout"): _O,
    # project_writes_v1
    ("project_writes_v1", "project_memory_update"): _O,
    ("project_writes_v1", "project_inbox_send"): _O,
    # session_history_read_v1
    ("session_history_read_v1", "session_history_read"): _V,
}


def _policy_key(descriptor: MethodDescriptor | ResourceDescriptor) -> MethodDescriptor | tuple[str, str]:
    if isinstance(descriptor, ResourceDescriptor):
        return descriptor.feature, descriptor.wire
    return descriptor


if RESOLVE.keys() != {_policy_key(item) for item in (*MethodDescriptor, *RESOURCE_METHODS)} or any(
    (RESOLVE[_policy_key(item)] is Resolve.OWNED_FIRST) != (item.access is Access.SESSION_MUTATION)
    or (
        item.access in {Access.DELIVERY, Access.CONTROLLER_WRITE}
        and RESOLVE[_policy_key(item)] is not Resolve.OWNED
    )
    or (item.access is Access.READ and RESOLVE[_policy_key(item)] is Resolve.OWNED)
    or (item.access is not Access.READ and RESOLVE[_policy_key(item)] is Resolve.OBSERVER_VIEW)
    for item in (*MethodDescriptor, *RESOURCE_METHODS)
):
    raise RuntimeError("serve method resolve policies are not exhaustive")


async def list_sessions(client, request_id, params, controller):
    if client.role == "observer" and "project_id" in params:
        raise ProtocolError(
            ErrorDescriptor.OBSERVER_READ_ONLY,
            "method is unavailable to read-only observers",
            {"method": "list_sessions"},
        )
    result, _ = await client.connections.run_read(
        client, lambda: lifecycle.list_sessions(client, request_id, params)
    )
    return result


async def new_session(client, request_id, params, controller):
    return await lifecycle.new_session(client, params, controller)


async def resume(client, request_id, params, controller):
    return await lifecycle.resume(client, params, controller)


async def send(client, request_id, params, controller):
    return await controller.control.deliveries.send(controller, params, client.features)


async def steer(client, request_id, params, controller):
    return await controller.control.deliveries.steer(controller, params, client.features)


async def _resolve_approval(client, params, decision, controller):
    scope = params.get("scope", "once")
    if not isinstance(scope, str) or scope not in APPROVAL_SCOPES:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "scope must be 'once' or 'always_tool'"
        )
    if scope == "always_tool" and decision is not ApprovalDecision.APPROVE:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "scope 'always_tool' requires approve"
        )
    return await controller.control.resolve_approval(
        controller, decision, validation.required_string(params, "request_id"), scope
    )


async def approve(client, request_id, params, controller):
    return await _resolve_approval(client, params, ApprovalDecision.APPROVE, controller)


async def deny(client, request_id, params, controller):
    return await _resolve_approval(client, params, ApprovalDecision.DENY, controller)


async def abort(client, request_id, params, controller):
    if controller.runtime.opened is None:
        raise ProtocolError(ErrorDescriptor.INTERNAL_ERROR, "server has no active session")
    return await controller.control.abort(
        abort_scope.parse_abort_scope(params, client.features)
    )


async def status(client, request_id, params, controller):
    runtime = controller.runtime
    if client.role == "observer":
        fallback_config = (
            await client.connections.run_blocking(lambda: runtime._config(None, None))
            if runtime.loop is None
            else None
        )
        snapshot = await lifecycle.status(
            client, fallback_config=fallback_config, controller=controller
        )
        if runtime.opened is None:
            snapshot.pop("project_context_health", None)
    else:
        snapshot = await lifecycle.status(
            client,
            include_connection_setup_counts=(
                params.get("include_connection_setup_counts") is True
            ),
            controller=controller,
        )
    return await client.connections.read_status(client, snapshot, controller)


async def rename_session(client, request_id, params, controller):
    return await session.rename(
        controller.runtime,
        validation.required_string(params, "session_id"),
        params.get("name"),
    )


async def delete_session(client, request_id, params, controller):
    operation = lambda: session.delete_session(
        controller.runtime,
        validation.required_string(params, "session_id"),
        controller.control.require_idle,
    )
    return await operation()


async def repair_transcript(client, request_id, params, controller):
    return await session.repair_transcript(params, controller)


async def session_tree(client, request_id, params, controller):
    return await client.connections.read_tree(client, params)


async def session_history(client, request_id, params, controller):
    return await client.connections.read_history(client, params)


async def session_history_and_subscribe(client, request_id, params, controller):
    return await join_history_and_subscribe(client, params)


async def content_read(client, request_id, params, controller):
    return await client.connections.read_content(client, params, controller)


async def child_run_history(client, request_id, params, controller):
    return await client.connections.read_child_run_history(client, params)


async def child_run_events(client, request_id, params, controller):
    from ..session_replay import ReplayScopeId
    connections = client.connections
    if set(params) != {"session_id", "run_id"} or not all(type(params.get(key)) is str for key in ("session_id", "run_id")):
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "session_id and run_id are required child event parameters")
    with client.server.controllers.selected(client, params["session_id"]) as selected:
        if selected.runtime.session_id != params["session_id"]:
            raise ProtocolError(ErrorDescriptor.NO_ACTIVE_SESSION, "session is not active")
        try:
            await connections.begin_replay_join(
                client, ReplayScopeId.child(params["run_id"]), selected
            )
            history, frontier_seq, row = await connections.read_child_run_events(
                client, params, selected
            )
            result = connections.finish_replay_join(client, history, frontier_seq, None)
            current = connections.child_runs.current(params["run_id"])
            return {**result, "child": current[0] if current is not None else row, "revision": current[1] if current is not None else 0}
        except BaseException:
            connections.invalidate_replay_connection(client, reason="rejoined")
            raise


async def _mutable_store(controller, params):
    runtime = controller.runtime
    store = validation.active(runtime, params)
    await controller.control.require_idle()
    validation.require_mutable(runtime, controller.control.has_pending_approvals())
    return runtime, store


async def switch_branch(client, request_id, params, controller):
    _runtime, store = await _mutable_store(controller, params)
    head = validation.required_string(params, "head_id")
    if store.active_branch_head_id() != head:
        await store.switch_to_branch_async(head)
    return await client.connections.read_tree(client, params, controller)


async def fork_message(client, request_id, params, controller):
    _runtime, store = await _mutable_store(controller, params)
    await store.append_message_fork_async(
        validation.required_string(params, "message_id")
    )
    return await client.connections.read_tree(client, params, controller)


async def model_catalog(client, request_id, params, controller):
    return await client.connections.read_catalog(client, controller)


async def session_settings(client, request_id, params, controller):
    if client.role == "observer":
        return await client.connections.read_settings(client, params, controller)
    validation.active(controller.runtime, params)
    return settings.settings(
        controller.runtime, include_plan_mode=True, telemetry=settings.telemetry(client)
    )


async def set_settings(client, request_id, params, controller):
    runtime, _store = await _mutable_store(controller, params)
    model = validation.required_string(params, "model")
    mode = validation.required_string(params, "approval_mode")
    model_selection.validate(model, mode)
    await model_selection.apply_rpc(runtime, model, mode)
    return settings.settings(runtime, telemetry=settings.telemetry(client))


async def set_compaction(client, request_id, params, controller):
    await _mutable_store(controller, params)
    validation.required_string(params, "mode")
    raise ProtocolError(
        ErrorDescriptor.INVALID_PARAMS,
        "compaction mode selection was removed; eviction is always used",
    )


async def send_images(client, request_id, params, controller):
    validation.active(controller.runtime, params)
    return await controller.control.deliveries.send_images(controller, params, client.features)


async def path_complete(client, request_id, params, controller):
    with client.server.controllers.selected(
        client, validation.required_string(params, "session_id")
    ) as selected:
        store = validation.active(selected.runtime, params)
        return await path_completion.complete(
            params,
            base_dir=store.cwd,
            run_blocking=selected.control._run_blocking,
        )


async def slash_list(client, request_id, params, controller):
    validation.active(controller.runtime, params)
    include_unavailable = params.get("include_unavailable", False)
    if not isinstance(include_unavailable, bool):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "include_unavailable must be a boolean",
        )
    return await slash_commands.list_commands(
        controller.runtime, include_unavailable=include_unavailable
    )


async def slash_run(client, request_id, params, controller):
    runtime = controller.runtime
    validation.active(runtime, params)
    text = validation.required_string(params, "text")
    registry = await slash_commands.registry_for_request(runtime)
    control = controller.control
    busy = control.state_change_busy()
    if busy and not slash_commands.can_run_while_busy(registry, text):
        raise ProtocolError(ErrorDescriptor.OPERATION_BUSY, "a turn is already running")
    return await slash_commands.run_command(
        runtime,
        text,
        register_model_input=control.model_input_registrar(client.features),
        approval_control=control,
        slash_lock=control.slash_lock,
        busy=busy,
        registry=registry,
    )


async def account_list(client, request_id, params, controller):
    validation.active(controller.runtime, params)
    return await accounts.list_accounts(controller.runtime)


async def account_logout(client, request_id, params, controller):
    validation.active(controller.runtime, params)
    return await client.server.account_operations.run("logout", controller, params)


async def account_refresh(client, request_id, params, controller):
    validation.active(controller.runtime, params)
    return await client.server.account_operations.run("refresh", controller, params)


async def account_switch(client, request_id, params, controller):
    validation.active(controller.runtime, params)
    return await client.server.account_operations.run("switch", controller, params)


async def login_start(client, request_id, params, controller):
    return await client.logins.login_start(params)


async def login_status(client, request_id, params, controller):
    return await client.logins.login_status(params)


async def login_cancel(client, request_id, params, controller):
    return await client.logins.login_cancel(params)


async def login_providers(client, request_id, params, controller):
    return await client.logins.login_providers(params)


async def ping(client, request_id, params, controller):
    return {"pong": True}


async def set_turn_context(client, request_id, params, controller):
    return client.server.turn_contexts.set_request(
        client.features, controller.runtime, params
    )


async def _project(client, request_id, params, method, **kwargs):
    return await asyncio.to_thread(
        method, partial(client.codec.response_fits, request_id), params, **kwargs,
    )


async def list_projects(client, request_id, params, controller):
    return await _project(client, request_id, params, client.server.projects.list_projects)


async def project_show(client, request_id, params, controller):
    return await _project(
        client, request_id, params, client.server.projects.show,
        entry_response="projects-memory-v2" in client.features,
    )


async def project_memory_log(client, request_id, params, controller):
    return await _project(
        client, request_id, params, client.server.projects.memory_log,
        entry_response="projects-memory-v2" in client.features,
    )


async def project_inbox(client, request_id, params, controller):
    return await _project(client, request_id, params, client.server.projects.inbox_view)


async def clear_steering(client, request_id, params, controller):
    return abort_scope.clear_pending_steering(
        client.features, controller.runtime.loop
    )


async def delivery_status(client, request_id, params, controller):
    return controller.control.deliveries.status(params, client.features, lookup=not controller.is_view)


async def session_grants_read(client, request_id, params, controller):
    session_grants.require_feature(client.features)
    runtime = controller.runtime
    if runtime.opened is None:
        raise ProtocolError(ErrorDescriptor.INTERNAL_ERROR, "server has no active session")
    return session_grants.read(runtime.policy, runtime.session_id, params)


async def revoke_session_grants(client, request_id, params, controller):
    session_grants.require_feature(client.features)
    runtime = controller.runtime
    if runtime.opened is None:
        raise ProtocolError(ErrorDescriptor.INTERNAL_ERROR, "server has no active session")
    result, change = session_grants.revoke(
        runtime.policy, runtime.session_id, params
    )
    if change.changed:
        await controller.control.publish_session_grant_change(change)
    return result


async def child_run_list(client, request_id, params, controller):
    with client.server.controllers.selected(
        client, validation.required_string(params, "session_id")
    ) as selected:
        return await child_runs.list_runs(client, params, selected)


async def child_run_send(client, request_id, params, controller):
    return await child_runs.send(params, controller)


async def mcp_refresh_tools(client, request_id, params, controller):
    runtime = controller.runtime
    validation.active(runtime, params)
    if set(params) - {"session_id", "server"}:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "mcp_refresh_tools accepts only session_id and optional server",
        )
    server = validation.optional_string(params, "server")
    try:
        servers = await runtime.loop.refresh_mcp_tools(server)
    except ValueError as exc:
        raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, str(exc)) from exc
    except (MCPManagementError, MCPToolRefreshError) as exc:
        data: dict[str, object] = {
            "session_id": runtime.session_id,
            "reason": str(exc),
        }
        if server is not None:
            data["server"] = server
        raise ProtocolError(
            ErrorDescriptor.MCP_TOOL_REFRESH_FAILED,
            "MCP tool refresh failed",
            data,
        ) from exc
    return {"session_id": runtime.session_id, "servers": list(servers)}


HANDLERS: dict[MethodDescriptor, Handler] = {
    MethodDescriptor.LIST_SESSIONS: list_sessions,
    MethodDescriptor.NEW_SESSION: new_session,
    MethodDescriptor.RESUME: resume,
    MethodDescriptor.SEND: send,
    MethodDescriptor.STEER: steer,
    MethodDescriptor.APPROVE: approve,
    MethodDescriptor.DENY: deny,
    MethodDescriptor.ABORT: abort,
    MethodDescriptor.STATUS: status,
    MethodDescriptor.RENAME_SESSION: rename_session,
    MethodDescriptor.DELETE_SESSION: delete_session,
    MethodDescriptor.REPAIR_TRANSCRIPT: repair_transcript,
    MethodDescriptor.SESSION_TREE: session_tree,
    MethodDescriptor.SESSION_HISTORY: session_history,
    MethodDescriptor.SESSION_HISTORY_AND_SUBSCRIBE: session_history_and_subscribe,
    MethodDescriptor.CONTENT_READ: content_read,
    MethodDescriptor.CHILD_RUN_HISTORY: child_run_history,
    MethodDescriptor.CHILD_RUN_EVENTS: child_run_events,
    MethodDescriptor.SWITCH_BRANCH: switch_branch,
    MethodDescriptor.FORK_MESSAGE: fork_message,
    MethodDescriptor.MODEL_CATALOG: model_catalog,
    MethodDescriptor.SESSION_SETTINGS: session_settings,
    MethodDescriptor.SET_SETTINGS: set_settings,
    MethodDescriptor.SET_COMPACTION: set_compaction,
    MethodDescriptor.SEND_IMAGES: send_images,
    MethodDescriptor.PATH_COMPLETE: path_complete,
    MethodDescriptor.SLASH_LIST: slash_list,
    MethodDescriptor.SLASH_RUN: slash_run,
    MethodDescriptor.ACCOUNT_LIST: account_list,
    MethodDescriptor.ACCOUNT_LOGOUT: account_logout,
    MethodDescriptor.ACCOUNT_REFRESH: account_refresh,
    MethodDescriptor.ACCOUNT_SWITCH: account_switch,
    MethodDescriptor.LOGIN_START: login_start,
    MethodDescriptor.LOGIN_STATUS: login_status,
    MethodDescriptor.LOGIN_CANCEL: login_cancel,
    MethodDescriptor.LOGIN_PROVIDERS: login_providers,
    MethodDescriptor.PING: ping,
    MethodDescriptor.SET_TURN_CONTEXT: set_turn_context,
    MethodDescriptor.LIST_PROJECTS: list_projects,
    MethodDescriptor.PROJECT_SHOW: project_show,
    MethodDescriptor.PROJECT_MEMORY_LOG: project_memory_log,
    MethodDescriptor.PROJECT_INBOX: project_inbox,
    MethodDescriptor.CLEAR_STEERING: clear_steering,
    MethodDescriptor.DELIVERY_STATUS: delivery_status,
    MethodDescriptor.SESSION_GRANTS: session_grants_read,
    MethodDescriptor.REVOKE_SESSION_GRANTS: revoke_session_grants,
    MethodDescriptor.RUN_WORK_LIST: list_run_work,
    MethodDescriptor.BACKGROUND_TASK_CANCEL: cancel_background_task,
    MethodDescriptor.CHILD_RUN_LIST: child_run_list,
    MethodDescriptor.CHILD_RUN_SEND: child_run_send,
    MethodDescriptor.MCP_REFRESH_TOOLS: mcp_refresh_tools,
    MethodDescriptor.STATE_SESSIONS: state_sessions,
    MethodDescriptor.STATE_SESSION: state_session,
    MethodDescriptor.STATE_AGENTS: state_agents,
    MethodDescriptor.STATE_BACKGROUND_TASKS: state_background_tasks,
    MethodDescriptor.STATE_CATALOG: state_catalog,
    MethodDescriptor.STATE_PROJECTS: state_projects,
    MethodDescriptor.STATE_MEMORY: state_memory,
    MethodDescriptor.STATE_MEMORY_DETAIL: state_memory_detail,
    MethodDescriptor.STATE_INBOX: state_inbox,
    MethodDescriptor.STATE_INBOX_DETAIL: state_inbox_detail,
    MethodDescriptor.STATE_ATTENTION: state_attention,
    MethodDescriptor.STATE_SETTINGS: state_settings,
    MethodDescriptor.STATE_ACCOUNTS: state_accounts,
    MethodDescriptor.STATE_STORAGE: state_storage,
    MethodDescriptor.STATE_RETENTION: state_retention,
    MethodDescriptor.TELEMETRY_STATUS: telemetry_status,
    MethodDescriptor.TELEMETRY_SESSIONS: telemetry_sessions,
    MethodDescriptor.TELEMETRY_TRACE: telemetry_trace,
    MethodDescriptor.TELEMETRY_OVERVIEW: telemetry_overview,
    MethodDescriptor.TELEMETRY_MODELS: telemetry_models,
    MethodDescriptor.TELEMETRY_TOOLS: telemetry_tools,
    MethodDescriptor.TELEMETRY_ERRORS: telemetry_errors,
    MethodDescriptor.TELEMETRY_SERIES: telemetry_series,
    MethodDescriptor.TELEMETRY_AGENT_TREE: telemetry_agent_tree,
}
if HANDLERS.keys() != set(MethodDescriptor):
    raise RuntimeError("serve method handlers are not exhaustive")


# Keep the complete v1 contract explicit: missing descriptors or handlers must not negotiate.
STATE_READ_V1 = frozenset({
    "state_projects", "state_memory", "state_memory_detail", "state_inbox", "state_inbox_detail",
    "state_sessions", "state_session", "state_agents", "state_background_tasks", "state_attention",
    "state_catalog", "state_settings", "state_accounts", "state_storage", "state_retention",
})


def state_read_available() -> bool:
    return STATE_READ_V1 == {method.wire for method in HANDLERS if method.feature == "state_read_v1"}


def _require_attached_session(runtime: Any, params: dict[str, Any]) -> None:
    if "session_id" not in params:
        return
    session_id = params["session_id"]
    if not isinstance(session_id, str) or not session_id:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "session_id must be a nonempty string"
        )
    attached = runtime.session_id if runtime.opened is not None else None
    if attached is not None and session_id != attached:
        raise ProtocolError(
            ErrorDescriptor.SESSION_MISMATCH,
            "session is not active",
            {"attached_session_id": attached, "session_id": session_id},
        )


async def dispatch(
    client: object,
    request_id: str | int,
    method: str,
    params: dict[str, Any],
) -> object:
    handled, result = await client._route_request(request_id, method, params)  # type: ignore[attr-defined]
    if handled:
        return result
    descriptor = MethodDescriptor.for_wire(method, client.features)  # type: ignore[attr-defined]
    require_access(
        getattr(client, "role", None), method, descriptor.access if descriptor is not None else None,
        # home_settings_v1 shipped on main before the shared resource codes and keeps main's reply
        resource=isinstance(descriptor, ResourceDescriptor) and descriptor.feature != "home_settings_v1",
    )
    if descriptor is None:
        raise ProtocolError(
            ErrorDescriptor.METHOD_NOT_FOUND, f"method not found: {method}"
        )
    policy = RESOLVE[_policy_key(descriptor)]
    handler = descriptor.handler if isinstance(descriptor, ResourceDescriptor) else HANDLERS[descriptor]
    if policy is not Resolve.OWNED_FIRST:
        _require_negotiated(client, descriptor)
    if (
        policy is Resolve.OBSERVER_VIEW
        and client.role == "observer"
        and client.server.controllers.controller_mode == "multi"
    ):
        with client.server.controllers.observer_view(client) as controller:
            return await handler(client, request_id, params, controller)
    if policy in {Resolve.NONE, Resolve.SELECTED}:
        return await handler(client, request_id, params, None)
    with client.server.controllers.owned(client) as controller:
        if policy is Resolve.OWNED_FIRST:
            # legacy order: a mutation's health error precedes its version/feature error
            controller.control._reject_if_unhealthy()
            _require_negotiated(client, descriptor)
        if descriptor.access is Access.DELIVERY:
            _require_attached_session(controller.runtime, params)
        if descriptor.access in {Access.SESSION_MUTATION, Access.CONTROLLER_WRITE}:
            # a queued wake may not start its setup while a state change runs
            with controller.control.hold_wake_start():
                return await handler(client, request_id, params, controller)
        return await handler(client, request_id, params, controller)


def _require_negotiated(client: object, descriptor: MethodDescriptor | ResourceDescriptor) -> None:
    if descriptor.since is not None and tuple(
        int(part) for part in client.protocol_version.split(".")  # type: ignore[attr-defined]
    ) < tuple(int(part) for part in descriptor.since.split(".")):
        raise protocol_error(descriptor)
    if descriptor.feature is not None and descriptor.feature not in client.features:  # type: ignore[attr-defined]
        raise feature_error(descriptor)
