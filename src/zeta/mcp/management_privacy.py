"""Suppress transport diagnostics during secret-bearing management I/O."""

import logging
from contextlib import contextmanager
from contextvars import ContextVar

_PRIVATE = ContextVar("mcp_private_management", default=False)


def _public_record(_record: logging.LogRecord) -> bool:
    return not _PRIVATE.get()


@contextmanager
def private_management_io():
    # Logger filters run at the source, before handlers (including telemetry).
    # The HTTP transports are imported before this call and the scope follows their tasks.
    for name, logger in logging.Logger.manager.loggerDict.copy().items():
        if isinstance(logger, logging.Logger) and (
            name in ("asyncio", "httpx", "httpcore") or name.startswith(("httpcore.", "zeta.mcp."))
        ):
            logger.addFilter(_public_record)
    token = _PRIVATE.set(True)
    try:
        yield
    finally:
        _PRIVATE.reset(token)
