"""Named-account requests for the native frontend protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable

from ..accounts import AccountUsage
from ..accounts.usage import usage_json
from .protocol import ProtocolError
from .runtime import ServerRuntime

REQUEST_METHODS = {"account_list", "account_refresh"}


def list_accounts(runtime: ServerRuntime) -> dict[str, object]:
    context = getattr(runtime.loop, "account_context", None)
    if context is None:
        raise ProtocolError(-32005, "accounts unavailable for this session")
    selected = context.snapshot()
    displays = context.store.list_display()
    return {
        "accounts": [
            {
                "account": str(item.account),
                "label": item.label,
                "detail": item.detail,
                "current": selected.get(item.account.provider) == item.account,
                "default": context.store.default(item.account.provider) == item.account,
                "usage": usage_json(item.usage),
            }
            for item in displays
        ]
    }


async def request(
    method: str,
    runtime: ServerRuntime,
    require_idle: Callable[[], Awaitable[None]],
) -> dict[str, object]:
    if method == "account_list":
        return list_accounts(runtime)
    await require_idle()
    return await refresh_accounts(runtime)


async def refresh_accounts(runtime: ServerRuntime) -> dict[str, object]:
    await asyncio.to_thread(
        lambda: asyncio.run(
            AccountUsage(runtime.loop.account_context.store).refresh_stale()
        )
    )
    return list_accounts(runtime)
