"""``telemetry_read_v1``: read-only telemetry methods over the built-in receiver's database."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import Any

from ...config.settings import load_settings
from ..protocol import ErrorDescriptor, FrameCodec, ProtocolError

_CODEC = FrameCodec()


async def telemetry_status(client, request_id, params):
    if params:
        raise ProtocolError(ErrorDescriptor.INVALID_TELEMETRY_PARAMS, "invalid params: telemetry_status takes no params")
    return await _read(client, request_id, lambda reader, home, endpoint: reader.status(home, endpoint))


async def telemetry_sessions(client, request_id, params):
    return await _read(client, request_id, lambda reader, home, endpoint: reader.sessions(home, endpoint, params))


async def telemetry_trace(client, request_id, params):
    return await _read(client, request_id, lambda reader, home, endpoint: reader.trace(home, endpoint, params))


async def _read(client, request_id, read: Callable[[Any, Path, str | None], dict[str, Any]]) -> dict[str, Any]:
    home = client.server.home

    def run() -> dict[str, Any]:
        # imported here: the projector imports serve's method table, so a top-level import is circular
        from ...telemetry.receiver import reader

        try:
            result = read(reader, home, load_settings(home=home).settings.otel_endpoint)
        except reader.InvalidParams as exc:
            raise ProtocolError(ErrorDescriptor.INVALID_TELEMETRY_PARAMS, f"invalid params: {exc}") from None
        except reader.TraceNotFound as exc:
            raise ProtocolError(ErrorDescriptor.TRACE_NOT_FOUND, "resource not found", {"id": exc.trace_id}) from None
        except reader.Unavailable as exc:
            descriptor = (
                ErrorDescriptor.TELEMETRY_OFF if exc.code == "telemetry_off" else ErrorDescriptor.TELEMETRY_RECEIVER_UNAVAILABLE
            )
            raise ProtocolError(descriptor, "state unavailable", {"retryable": True}) from None
        except sqlite3.OperationalError:
            raise  # busy or unreadable file: an internal error, not corrupt data
        except (reader.CorruptState, sqlite3.DatabaseError):
            raise ProtocolError(ErrorDescriptor.CORRUPT_STATE, "stored state is invalid", {"resource": "telemetry"}) from None
        # sized off the loop: a page with many large spans must fail with a typed error, not a dropped frame
        if not _CODEC.response_fits(request_id, result):
            raise ProtocolError(ErrorDescriptor.RESULT_TOO_LARGE, "result exceeds frame limit")
        return result

    return await client.connections.run_read(client, run)
