"""Read-only methods negotiated under ``state_read_v1``."""

from ...read_page import InvalidParams, TooLarge
from ..protocol import ErrorDescriptor, MethodDescriptor, ProtocolError
from ..read_home_state import (
    accounts_page,
    retention_result,
    settings_page,
    storage_page,
)
from ..read_sessions import read_session_page


async def state_catalog(client, request_id, params, controller):
    try:
        return await client.connections.state_reader.catalog(
            client.server.home, params, lambda result: client.codec.response_fits(request_id, result),
        )
    except InvalidParams:
        raise ProtocolError(ErrorDescriptor.INVALID_STATE_PARAMS, "invalid params") from None
    except TooLarge:
        raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit") from None


def _project_state_handler(collection):
    async def read(client, request_id, params, controller):
        try:
            return await client.connections.state_reader.project_state(
                client.server.home, collection, params, lambda result: client.codec.response_fits(request_id, result),
            )
        except InvalidParams:
            raise ProtocolError(ErrorDescriptor.INVALID_STATE_PARAMS, "invalid params") from None
        except TooLarge:
            raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit") from None

    return read


state_projects = _project_state_handler("projects")
state_memory = _project_state_handler("memory")
state_memory_detail = _project_state_handler("memory_detail")
state_inbox = _project_state_handler("inbox")
state_inbox_detail = _project_state_handler("inbox_detail")
state_attention = _project_state_handler("attention")


async def _session_read(client, request_id, params, method, *, collection=None, detail=False):
    server = client.server
    snapshot = server.session_statuses()
    try:
        return await client.connections.run_blocking(lambda: read_session_page(
            server.home, method, params, lambda result: client.codec.response_fits(request_id, result), snapshot,
            collection=collection, detail=detail,
        ))
    except InvalidParams:
        raise ProtocolError(ErrorDescriptor.INVALID_STATE_PARAMS, "invalid params") from None
    except TooLarge:
        raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit") from None


async def state_sessions(client, request_id, params, controller):
    return await _session_read(client, request_id, params, MethodDescriptor.STATE_SESSIONS)


async def state_session(client, request_id, params, controller):
    return await _session_read(client, request_id, params, MethodDescriptor.STATE_SESSION, detail=True)


async def state_agents(client, request_id, params, controller):
    return await _session_read(client, request_id, params, MethodDescriptor.STATE_AGENTS, collection="agents")


async def state_background_tasks(client, request_id, params, controller):
    return await _session_read(client, request_id, params, MethodDescriptor.STATE_BACKGROUND_TASKS, collection="background_tasks")


def _home_state_handler(owner_read):
    async def read(client, request_id, params, controller):
        try:
            return await client.connections.run_blocking(lambda: owner_read(
                client.server.home, params, lambda result: client.codec.response_fits(request_id, result),
            ))
        except InvalidParams:
            raise ProtocolError(ErrorDescriptor.INVALID_STATE_PARAMS, "invalid params") from None
        except TooLarge:
            raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit") from None

    return read


state_settings = _home_state_handler(settings_page)
state_accounts = _home_state_handler(accounts_page)
state_storage = _home_state_handler(storage_page)
state_retention = _home_state_handler(retention_result)
