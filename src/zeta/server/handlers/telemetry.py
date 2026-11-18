"""``telemetry_read_v1``: read-only telemetry methods over the built-in receiver's database."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ...config.settings import load_settings
from ..protocol import ErrorDescriptor, FrameCodec, ProtocolError

_CODEC = FrameCodec()


async def telemetry_status(client, request_id, params):
    if params:
        raise ProtocolError(ErrorDescriptor.INVALID_TELEMETRY_PARAMS, "invalid params")
    home = client.server.home
    return await _read(client, request_id, lambda reader, fits: reader.status(home, load_settings(home=home).settings.otel_endpoint))


async def telemetry_sessions(client, request_id, params):
    return await _read(client, request_id, lambda reader, fits: reader.sessions(client.server.home, params))


async def telemetry_trace(client, request_id, params):
    return await _read(client, request_id, lambda reader, fits: reader.trace(client.server.home, params, fits))


async def _read(client, request_id, read: Callable[[Any, Callable[[object], bool]], dict[str, Any]]) -> dict[str, Any]:
    def fits(result: object) -> bool:
        return _CODEC.response_fits(request_id, result)

    def run() -> dict[str, Any]:
        # imported here: the projector imports serve's method table, so a top-level import is circular
        from ...telemetry.receiver import reader

        try:
            result = read(reader, fits)
        except reader.InvalidParams:
            raise ProtocolError(ErrorDescriptor.INVALID_TELEMETRY_PARAMS, "invalid params") from None
        except reader.TraceNotFound as exc:
            raise ProtocolError(ErrorDescriptor.TRACE_NOT_FOUND, "resource not found", {"id": exc.trace_id}) from None
        except reader.Unavailable:
            raise ProtocolError(ErrorDescriptor.TELEMETRY_OFF, "state unavailable", {"retryable": True}) from None
        except reader.SchemaMismatch:
            raise ProtocolError(ErrorDescriptor.TELEMETRY_SCHEMA_MISMATCH, "telemetry schema is unsupported") from None
        except reader.TooLarge:
            raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit") from None
        if not fits(result):  # sessions and status pages are bounded far below the frame; this is the backstop
            raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit")
        return result

    return await client.connections.run_read(client, run)
