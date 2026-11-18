"""Read-only methods negotiated under ``state_read_v1``."""

from ...read_page import InvalidParams, TooLarge
from ..protocol import ErrorDescriptor, ProtocolError


async def state_catalog(client, request_id, params):
    try:
        return await client.connections.state_reader.catalog(
            client.server.home, params, lambda result: client.codec.response_fits(request_id, result),
        )
    except InvalidParams:
        raise ProtocolError(ErrorDescriptor.INVALID_STATE_PARAMS, "invalid params") from None
    except TooLarge:
        raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit") from None


def _project_state_handler(collection):
    async def read(client, request_id, params):
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
