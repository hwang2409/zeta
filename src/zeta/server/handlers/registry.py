from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from .. import abort_scope, accounts, model_selection, slash_commands
from .. import control as session_grants
from ..protocol import (
    Access,
    ApprovalDecision,
    ErrorDescriptor,
    MethodDescriptor,
    ProtocolError,
    feature_error,
    protocol_error,
)
from . import child_runs, lifecycle, path_completion, session, settings, validation

Handler = Callable[[object, str | int, dict[str, Any]], Awaitable[object]]


async def list_sessions(client, request_id, params):
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


async def new_session(client, request_id, params):
    return await lifecycle.new_session(client, params)


async def resume(client, request_id, params):
    return await lifecycle.resume(client, params)


async def send(client, request_id, params):
    return await client.server.control.deliveries.send(params, client.features)


async def steer(client, request_id, params):
    return await client.server.control.deliveries.steer(params, client.features)


async def _resolve_approval(client, params, decision):
    scope = params.get("scope", "once")
    if not isinstance(scope, str) or scope not in {"once", "always_tool"}:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "scope must be 'once' or 'always_tool'"
        )
    if scope == "always_tool" and decision is not ApprovalDecision.APPROVE:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "scope 'always_tool' requires approve"
        )
    return await client.server.control.resolve_approval(
        decision, validation.required_string(params, "request_id"), scope
    )


async def approve(client, request_id, params):
    return await _resolve_approval(client, params, ApprovalDecision.APPROVE)


async def deny(client, request_id, params):
    return await _resolve_approval(client, params, ApprovalDecision.DENY)


async def abort(client, request_id, params):
    return await client.server.control.abort(
        abort_scope.parse_abort_scope(params, client.features)
    )


async def status(client, request_id, params):
    runtime = client.server.runtime
    session_id = runtime.session_id if runtime.opened is not None else None
    if client.role == "observer":
        fallback_config = (
            await client.connections.run_blocking(lambda: runtime._config(None, None))
            if runtime.loop is None
            else None
        )
        snapshot = await lifecycle.status(client, fallback_config=fallback_config)
        if runtime.opened is None:
            snapshot.pop("project_context_health", None)
    else:
        snapshot = await lifecycle.status(
            client,
            include_connection_setup_counts=(
                params.get("include_connection_setup_counts") is True
            ),
        )
    return await client.connections.read_status(client, snapshot, session_id)


async def rename_session(client, request_id, params):
    return await session.rename(
        client.server.runtime,
        validation.required_string(params, "session_id"),
        params.get("name"),
    )


async def delete_session(client, request_id, params):
    return await session.delete_session(
        client.server.runtime,
        validation.required_string(params, "session_id"),
        client.server.control.require_idle,
    )


async def repair_transcript(client, request_id, params):
    return await session.repair_transcript(client, params)


async def session_tree(client, request_id, params):
    return await client.connections.read_tree(client, params)


async def session_history(client, request_id, params):
    return await client.connections.read_history(client, params)


async def content_read(client, request_id, params):
    return await client.connections.read_content(client, params)


async def child_run_history(client, request_id, params):
    return await client.connections.read_child_run_history(client, params)


async def _mutable_store(client, params):
    runtime = client.server.runtime
    store = validation.active(runtime, params)
    await client.server.control.require_idle()
    validation.require_mutable(
        runtime, client.server.control.has_pending_approvals()
    )
    return runtime, store


async def switch_branch(client, request_id, params):
    runtime, store = await _mutable_store(client, params)
    head = validation.required_string(params, "head_id")
    current = store.replay()
    if not current or current[-1].id != head:
        store.switch_to_branch(head)
    return session.tree(runtime)


async def fork_message(client, request_id, params):
    runtime, store = await _mutable_store(client, params)
    store.append_message_fork(validation.required_string(params, "message_id"))
    return session.tree(runtime)


async def model_catalog(client, request_id, params):
    return await client.connections.read_catalog(client)


async def session_settings(client, request_id, params):
    if client.role == "observer":
        return await client.connections.read_settings(client, params)
    validation.active(client.server.runtime, params)
    return settings.settings(client.server.runtime, include_plan_mode=True)


async def set_settings(client, request_id, params):
    runtime, _store = await _mutable_store(client, params)
    model = validation.required_string(params, "model")
    mode = validation.required_string(params, "approval_mode")
    model_selection.validate(model, mode)
    await model_selection.apply_rpc(runtime, model, mode)
    return settings.settings(runtime)


async def set_compaction(client, request_id, params):
    await _mutable_store(client, params)
    validation.required_string(params, "mode")
    raise ProtocolError(
        ErrorDescriptor.INVALID_PARAMS,
        "compaction mode selection was removed; eviction is always used",
    )


async def send_images(client, request_id, params):
    validation.active(client.server.runtime, params)
    return await client.server.control.deliveries.send_images(params, client.features)


async def path_complete(client, request_id, params):
    store = validation.active(client.server.runtime, params)
    return await path_completion.complete(
        params,
        base_dir=store.cwd,
        run_blocking=client.server.control._run_blocking,
    )


async def slash_list(client, request_id, params):
    validation.active(client.server.runtime, params)
    include_unavailable = params.get("include_unavailable", False)
    if not isinstance(include_unavailable, bool):
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "include_unavailable must be a boolean",
        )
    return await slash_commands.list_commands(
        client.server.runtime, include_unavailable=include_unavailable
    )


async def slash_run(client, request_id, params):
    runtime = client.server.runtime
    validation.active(runtime, params)
    text = validation.required_string(params, "text")
    registry = await slash_commands.registry_for_request(runtime)
    if client.server.control.turn_busy() and not slash_commands.can_run_while_busy(
        registry, text
    ):
        await client.server.control.require_idle()
    register = client.server.control.model_input_registrar(client.features)
    return await slash_commands.run_command(
        runtime,
        text,
        register_model_input=register,
        approval_control=client.server.control,
        slash_lock=client.server.control.slash_lock,
        busy=client.server.control.turn_busy(),
        registry=registry,
    )


async def account_list(client, request_id, params):
    validation.active(client.server.runtime, params)
    return await accounts.list_accounts(client.server.runtime)


async def account_logout(client, request_id, params):
    validation.active(client.server.runtime, params)
    return await accounts.logout(
        client.server.runtime, client.server.control.require_idle, params
    )


async def account_refresh(client, request_id, params):
    validation.active(client.server.runtime, params)
    await client.server.control.require_idle()
    return await accounts.refresh(client.server.runtime)


async def account_switch(client, request_id, params):
    validation.active(client.server.runtime, params)
    operation = accounts.switch(
        client.server.runtime, client.server.control.require_idle, params
    )
    return await accounts.run_owned_switch(client, operation)


async def login_start(client, request_id, params):
    return await client.logins.login_start(params)


async def login_status(client, request_id, params):
    return await client.logins.login_status(params)


async def login_cancel(client, request_id, params):
    return await client.logins.login_cancel(params)


async def login_providers(client, request_id, params):
    return await client.logins.login_providers(params)


async def ping(client, request_id, params):
    return {"pong": True}


async def set_turn_context(client, request_id, params):
    return client.server.turn_contexts.set_request(
        client.features, client.server.runtime, params
    )


async def _project(client, request_id, params, operation):
    assert client.projects is not None
    return await asyncio.to_thread(operation, request_id, params)


async def list_projects(client, request_id, params):
    return await _project(client, request_id, params, client.projects.list_projects)


async def project_show(client, request_id, params):
    return await asyncio.to_thread(
        client.projects.show,
        request_id,
        params,
        entry_response="projects-memory-v2" in client.features,
    )


async def project_memory_log(client, request_id, params):
    return await asyncio.to_thread(
        client.projects.memory_log,
        request_id,
        params,
        entry_response="projects-memory-v2" in client.features,
    )


async def project_inbox(client, request_id, params):
    return await _project(client, request_id, params, client.projects.inbox_view)


async def clear_steering(client, request_id, params):
    return abort_scope.clear_pending_steering(
        client.features, client.server.runtime.loop
    )


async def delivery_status(client, request_id, params):
    return client.server.control.deliveries.status(params, client.features)


async def session_grants_read(client, request_id, params):
    session_grants.require_feature(client.features)
    runtime = client.server.runtime
    return session_grants.read(runtime.policy, runtime.session_id, params)


async def revoke_session_grants(client, request_id, params):
    session_grants.require_feature(client.features)
    runtime = client.server.runtime
    result, change = session_grants.revoke(
        runtime.policy, runtime.session_id, params
    )
    if change.changed:
        await client.server.control.publish_session_grant_change(change)
    return result


async def child_run_list(client, request_id, params):
    return await child_runs.list_runs(client, params)


async def child_run_send(client, request_id, params):
    return await child_runs.send(client, params)


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
    MethodDescriptor.CONTENT_READ: content_read,
    MethodDescriptor.CHILD_RUN_HISTORY: child_run_history,
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
    MethodDescriptor.CHILD_RUN_LIST: child_run_list,
    MethodDescriptor.CHILD_RUN_SEND: child_run_send,
}
if HANDLERS.keys() != set(MethodDescriptor):
    raise RuntimeError("serve method handlers are not exhaustive")


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
    descriptor = MethodDescriptor.for_wire(method)
    if getattr(client, "role", None) == "observer" and (
        descriptor is None or descriptor.access is not Access.READ
    ):
        raise ProtocolError(
            ErrorDescriptor.OBSERVER_READ_ONLY,
            "method is unavailable to read-only observers",
            {"method": method},
        )
    if descriptor is None:
        raise ProtocolError(
            ErrorDescriptor.METHOD_NOT_FOUND, f"method not found: {method}"
        )
    if descriptor.access is Access.SESSION_MUTATION:
        client.server.control._reject_if_unhealthy()  # type: ignore[attr-defined]
    if descriptor.since is not None and tuple(
        int(part) for part in client.protocol_version.split(".")  # type: ignore[attr-defined]
    ) < tuple(int(part) for part in descriptor.since.split(".")):
        raise protocol_error(descriptor)
    if descriptor.feature is not None and descriptor.feature not in client.features:  # type: ignore[attr-defined]
        raise feature_error(descriptor)
    if descriptor.access is Access.DELIVERY:
        _require_attached_session(client.server.runtime, params)  # type: ignore[attr-defined]
    return await HANDLERS[descriptor](client, request_id, params)
