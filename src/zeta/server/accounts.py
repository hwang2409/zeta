"""Named-account requests for the native frontend protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..accounts import AccountId, AccountStore, AccountUsage
from ..accounts.usage import usage_json
from .protocol import ProtocolError
from .runtime import ServerRuntime

REQUEST_METHODS = {"account_list", "account_refresh", "account_switch"}


async def migrate(home: Path) -> None:
    await asyncio.to_thread(AccountStore(home).migrate)


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
    params: dict[str, object],
) -> dict[str, object]:
    if method == "account_list":
        return await asyncio.to_thread(list_accounts, runtime)
    await require_idle()
    if method == "account_refresh":
        return await refresh_accounts(runtime)
    raw = params.get("account")
    if not isinstance(raw, str) or ":" not in raw:
        raise ProtocolError(-32602, "account must be provider:name")
    account = AccountId(*raw.split(":", 1))
    loop = runtime.loop
    entries = loop.store.replay()
    boundary = entries[-1].seq if entries else 0
    loop.account_context.switch_active(
        loop, account, runtime.metadata.provider, loop.backend.model, boundary=boundary
    )
    return {"account": str(account), "replay": False}


async def refresh_accounts(runtime: ServerRuntime) -> dict[str, object]:
    await asyncio.to_thread(
        lambda: asyncio.run(
            AccountUsage(runtime.loop.account_context.store).refresh_stale()
        )
    )
    return await asyncio.to_thread(list_accounts, runtime)
