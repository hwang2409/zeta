"""Read-only methods negotiated under ``state_read_v1``."""

from ...read_page import InvalidParams, TooLarge
from ..protocol import ErrorDescriptor, ProtocolError


async def state_catalog(client, request_id, params):
    try:
        return await client.connections.read_catalog.catalog(
            client.server.home, params, lambda result: client.codec.response_fits(request_id, result),
        )
    except InvalidParams:
        raise ProtocolError(ErrorDescriptor.INVALID_STATE_PARAMS, "invalid params") from None
    except TooLarge:
        raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit") from None
