from __future__ import annotations

import asyncio
from typing import Any

from .. import abort_scope, model_selection, slash_commands
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
from . import lifecycle, session, settings, validation

_LOGIN_METHODS = frozenset(MethodDescriptor[name] for name in ("LOGIN_START", "LOGIN_STATUS", "LOGIN_CANCEL", "LOGIN_PROVIDERS"))
_ERGONOMICS_METHODS = frozenset(MethodDescriptor[name] for name in ("RENAME_SESSION", "DELETE_SESSION", "REPAIR_TRANSCRIPT", "SESSION_TREE", "SESSION_HISTORY", "CONTENT_READ", "MODEL_CATALOG", "SESSION_SETTINGS", "SET_SETTINGS", "SET_COMPACTION", "SEND_IMAGES", "SWITCH_BRANCH", "FORK_MESSAGE", "SLASH_LIST", "SLASH_RUN", "ACCOUNT_LIST", "ACCOUNT_LOGOUT", "ACCOUNT_REFRESH", "ACCOUNT_SWITCH"))
_PROJECT_METHODS = frozenset(MethodDescriptor[name] for name in ("LIST_PROJECTS", "PROJECT_SHOW", "PROJECT_MEMORY_LOG", "PROJECT_INBOX"))


async def _dispatch_descriptor(
    client: object,
    request_id: str | int,
    descriptor: MethodDescriptor,
    params: dict[str, Any],
) -> object:
    method = descriptor.wire
    if descriptor in _LOGIN_METHODS:
        return await client.logins.request(method, params)  # type: ignore[attr-defined]
    if descriptor in _ERGONOMICS_METHODS:
        return await _ergonomics(client, descriptor, params)
    if descriptor is MethodDescriptor.PING:
        return {"pong": True}
    if descriptor is MethodDescriptor.SET_TURN_CONTEXT:
        return client.server.turn_contexts.set_request(  # type: ignore[attr-defined]
            client.features, client.server.runtime, params  # type: ignore[attr-defined]
        )
    if descriptor in _PROJECT_METHODS:
        assert client.projects is not None  # type: ignore[attr-defined]
        return await asyncio.to_thread(  # type: ignore[attr-defined]
            client.projects.dispatch,
            request_id,
            method,
            params,
            features=client.features,
        )
    if descriptor is MethodDescriptor.LIST_SESSIONS:
        result, _ = await client.connections.run_read(  # type: ignore[attr-defined]
            client, lambda: lifecycle.list_sessions(client, request_id, params)
        )
        return result
    if descriptor is MethodDescriptor.NEW_SESSION:
        return await lifecycle.new_session(client, params)
    if descriptor is MethodDescriptor.RESUME:
        return await lifecycle.resume(client, params)
    if descriptor in {MethodDescriptor.SESSION_GRANTS, MethodDescriptor.REVOKE_SESSION_GRANTS}:
        session_grants.require_feature(client.features)  # type: ignore[attr-defined]
        runtime = client.server.runtime  # type: ignore[attr-defined]
        if descriptor is MethodDescriptor.SESSION_GRANTS:
            return session_grants.read(runtime.policy, runtime.session_id, params)
        result, change = session_grants.revoke(runtime.policy, runtime.session_id, params)
        if change.changed:
            await client.server.control.publish_session_grant_change(change)  # type: ignore[attr-defined]
        return result
    if descriptor is MethodDescriptor.SEND:
        return await client.server.control.deliveries.send(params, client.features)  # type: ignore[attr-defined]
    if descriptor is MethodDescriptor.STEER:
        return await client.server.control.deliveries.steer(params, client.features)  # type: ignore[attr-defined]
    if descriptor is MethodDescriptor.DELIVERY_STATUS:
        return client.server.control.deliveries.status(params, client.features)  # type: ignore[attr-defined]
    if descriptor is MethodDescriptor.CLEAR_STEERING:
        return abort_scope.clear_pending_steering(client.features, client.server.runtime.loop)  # type: ignore[attr-defined]
    if descriptor in {MethodDescriptor.APPROVE, MethodDescriptor.DENY}:
        scope = params.get("scope", "once")
        if not isinstance(scope, str) or scope not in {"once", "always_tool"}:
            raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "scope must be 'once' or 'always_tool'")
        if scope == "always_tool" and descriptor is not MethodDescriptor.APPROVE:
            raise ProtocolError(ErrorDescriptor.INVALID_PARAMS, "scope 'always_tool' requires approve")
        decision = ApprovalDecision.APPROVE if descriptor is MethodDescriptor.APPROVE else ApprovalDecision.DENY
        return await client.server.control.resolve_approval(  # type: ignore[attr-defined]
            decision, validation.required_string(params, "request_id"), scope
        )
    if descriptor is MethodDescriptor.ABORT:
        return await client.server.control.abort(  # type: ignore[attr-defined]
            abort_scope.parse_abort_scope(params, client.features)  # type: ignore[attr-defined]
        )
    if descriptor is MethodDescriptor.STATUS:
        session_id = client.server.runtime.session_id if client.server.runtime.opened is not None else None  # type: ignore[attr-defined]
        return await client.connections.read_status(  # type: ignore[attr-defined]
            client,
            await lifecycle.status(client, include_connection_setup_counts=params.get("include_connection_setup_counts") is True),
            session_id,
        )
    raise ProtocolError(ErrorDescriptor.METHOD_NOT_FOUND, f"method not found: {method}")


async def _observer_read(client: object, request_id: str | int, descriptor: MethodDescriptor, params: dict[str, Any]) -> object:
    if descriptor is MethodDescriptor.LIST_SESSIONS:
        result, _ = await client.connections.run_read(  # type: ignore[attr-defined]
            client, lambda: lifecycle.list_sessions(client, request_id, params)
        )
        return result
    if descriptor in {MethodDescriptor.SESSION_TREE, MethodDescriptor.SESSION_HISTORY, MethodDescriptor.CONTENT_READ, MethodDescriptor.MODEL_CATALOG, MethodDescriptor.SESSION_SETTINGS}:
        return await _observer_operation(client, descriptor, params)  # type: ignore[attr-defined]
    return await _dispatch_descriptor(client, request_id, descriptor, params)


async def _dispatch_session_grants(
    client, descriptor: MethodDescriptor, params: dict[str, Any]
) -> object:
    session_grants.require_feature(client.features)
    runtime = client.server.runtime
    if descriptor is MethodDescriptor.SESSION_GRANTS:
        return session_grants.read(runtime.policy, runtime.session_id, params)
    result, change = session_grants.revoke(
        runtime.policy, runtime.session_id, params
    )
    if change.changed:
        await client.server.control.publish_session_grant_change(change)
    return result


async def _ergonomics(
    client, descriptor: MethodDescriptor, params: dict[str, Any]
) -> object:
    if isinstance(descriptor, str):
        descriptor = MethodDescriptor.for_wire(descriptor)
        if descriptor is None:
            raise ProtocolError(
                ErrorDescriptor.METHOD_NOT_FOUND, "method not found"
            )
    runtime = client.server.runtime
    method = descriptor.wire
    if descriptor is MethodDescriptor.REPAIR_TRANSCRIPT:
        return await session.repair_transcript(client, params)
    if descriptor in {
        MethodDescriptor.RENAME_SESSION,
        MethodDescriptor.DELETE_SESSION,
    }:

        return await session.mutate_session(
            runtime,
            method,
            validation.required_string(params, "session_id"),
            params.get("name"),
            client.server.control.require_idle,
        )
    if descriptor in {
        MethodDescriptor.SESSION_TREE,
        MethodDescriptor.SESSION_HISTORY,
    }:
        assert client.server.connections is not None
        return await _observer_operation(client, descriptor, params)
    if descriptor is MethodDescriptor.CONTENT_READ:
        if "content_read" not in client.features:
            raise ProtocolError(
                ErrorDescriptor.METHOD_NOT_FOUND,
                "content_read requires the negotiated content_read feature",
            )
        assert client.server.connections is not None
        return await _observer_operation(client, descriptor, params)
    if descriptor is MethodDescriptor.MODEL_CATALOG:
        return settings.catalog(runtime)
    store = validation.active(runtime, params)
    if descriptor is MethodDescriptor.SEND_IMAGES:
        return await client.server.control.deliveries.send_images(
            params, client.features
        )
    if descriptor is MethodDescriptor.SESSION_SETTINGS:
        return settings.settings(runtime, include_plan_mode=True)
    if descriptor in {
        MethodDescriptor.ACCOUNT_LIST,
        MethodDescriptor.ACCOUNT_LOGOUT,
        MethodDescriptor.ACCOUNT_REFRESH,
        MethodDescriptor.ACCOUNT_SWITCH,
        MethodDescriptor.SLASH_LIST,
    }:
        return await slash_commands.read_request(
            method, runtime, client.server.control.require_idle, params, client
        )
    if descriptor is MethodDescriptor.SLASH_RUN:
        text = validation.required_string(params, "text")
        if (
            client.server.control.turn_busy()
            and not await slash_commands.can_run_while_busy(runtime, text)
        ):
            await client.server.control.require_idle()
        register = client.server.control.model_input_registrar(client.features)
        return await slash_commands.run_command(
            runtime,
            text,
            register_model_input=register,
            approval_control=client.server.control,
            run_blocking=client.server.control._run_blocking,
            busy=client.server.control.turn_busy(),
        )
    await client.server.control.require_idle()
    validation.require_mutable(runtime, client.server.control.has_pending_approvals())
    if descriptor is MethodDescriptor.SET_SETTINGS:
        model = validation.required_string(params, "model")
        mode = validation.required_string(params, "approval_mode")
        model_selection.validate(model, mode)
        await model_selection.apply_rpc(runtime, model, mode)
        return settings.settings(runtime)
    if descriptor is MethodDescriptor.SET_COMPACTION:
        validation.required_string(params, "mode")
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS,
            "compaction mode selection was removed; eviction is always used",
        )
    if descriptor is MethodDescriptor.FORK_MESSAGE:
        store.append_message_fork(validation.required_string(params, "message_id"))
    elif descriptor is MethodDescriptor.SWITCH_BRANCH:
        head = validation.required_string(params, "head_id")
        current = store.replay()
        if not current or current[-1].id != head:
            store.switch_to_branch(head)
    return session.tree(runtime)


async def _observer_operation(client: object, descriptor: MethodDescriptor, params: dict[str, Any]) -> object:
    connections = client.server.connections
    if descriptor is MethodDescriptor.SESSION_TREE:
        return await connections.read_tree(client, params)
    if descriptor is MethodDescriptor.SESSION_HISTORY:
        return await connections.read_history(client, params)
    if descriptor is MethodDescriptor.CONTENT_READ:
        return await connections.read_content(client, params)
    if descriptor is MethodDescriptor.MODEL_CATALOG:
        return await connections.read_catalog(client)
    if descriptor is MethodDescriptor.SESSION_SETTINGS:
        return await connections.read_settings(client, params)
    raise ProtocolError(ErrorDescriptor.METHOD_NOT_FOUND, "method not found")


def _descriptor_handler(descriptor: MethodDescriptor):
    async def handle(client: object, request_id: str | int, params: dict[str, Any]) -> object:
        if getattr(client, "role", None) == "observer":
            return await _observer_read(client, request_id, descriptor, params)
        return await _dispatch_descriptor(client, request_id, descriptor, params)

    return handle


HANDLERS = {descriptor: _descriptor_handler(descriptor) for descriptor in MethodDescriptor}
if HANDLERS.keys() != set(MethodDescriptor):
    raise RuntimeError("serve method handlers are not exhaustive")


async def dispatch(client: object, request_id: str | int, method: str, params: dict[str, Any]) -> object:
    handled, result = await client._route_request(request_id, method, params)  # type: ignore[attr-defined]
    if handled:
        return result
    descriptor = MethodDescriptor.for_wire(method)
    if getattr(client, "role", None) == "observer" and (
        descriptor is None
        or descriptor.access is not Access.READ
        or (descriptor is MethodDescriptor.LIST_SESSIONS and "project_id" in params)
    ):
        raise ProtocolError(
            ErrorDescriptor.OBSERVER_READ_ONLY,
            "method is unavailable to read-only observers",
            {"method": method},
        )
    if descriptor is None:
        raise ProtocolError(ErrorDescriptor.METHOD_NOT_FOUND, f"method not found: {method}")
    if descriptor.access is Access.SESSION_MUTATION:
        client.server.control._reject_if_unhealthy()  # type: ignore[attr-defined]
    if descriptor.since is not None and tuple(int(part) for part in client.protocol_version.split(".")) < tuple(int(part) for part in descriptor.since.split(".")):  # type: ignore[attr-defined]
        raise protocol_error(descriptor)
    if descriptor.feature is not None and descriptor.feature not in client.features:  # type: ignore[attr-defined]
        raise feature_error(descriptor)
    return await HANDLERS[descriptor](client, request_id, params)
