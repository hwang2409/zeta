"""Named-account requests for the native frontend protocol."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..accounts import AccountId, AccountStore, AccountUsage, account_provider
from ..accounts.usage import UsageState, usage_json
from ..core.session import account_home
from ..providers.factory import build_backend
from .protocol import ErrorDescriptor, ProtocolError
from .runtime import ServerRuntime

REQUEST_METHODS = {
    "account_list",
    "account_logout",
    "account_refresh",
    "account_switch",
}


async def run_owned_switch(owner: Any, operation: Awaitable[object]) -> object:
    task = asyncio.create_task(operation)
    owner._account_switch_task = task
    try:
        return await task
    finally:
        owner._account_switch_task = None


async def cancel_owned_switch(owner: Any) -> None:
    task = getattr(owner, "_account_switch_task", None)
    if task is not None and not task.done():
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
    owner._account_switch_task = None


async def migrate(home: Path) -> tuple[str, ...]:
    return await asyncio.to_thread(AccountStore(account_home(home)).migrate)


async def list_accounts(
    runtime: ServerRuntime,
    refreshed: dict[AccountId, UsageState] | None = None,
) -> dict[str, object]:
    context = getattr(runtime.loop, "account_context", None)
    if context is None:
        raise ProtocolError(
            ErrorDescriptor.STATE_UNAVAILABLE, "accounts unavailable for this session"
        )
    snapshot = await context.list_snapshot()
    return {
        "accounts": [
            {
                "account": str(item.account),
                "label": item.label,
                "detail": item.detail,
                "current": snapshot.selected.get(item.account.provider) == item.account,
                "default": snapshot.defaults.get(item.account.provider) == item.account,
                "usage": usage_json(
                    refreshed.get(item.account, item.usage) if refreshed else item.usage
                ),
            }
            for item in snapshot.displays
        ]
    }


async def request(
    method: str,
    runtime: ServerRuntime,
    require_idle: Callable[[], Awaitable[None]],
    params: dict[str, object],
) -> dict[str, object]:
    if method == "account_list":
        return await list_accounts(runtime)
    await require_idle()
    if method == "account_refresh":
        return await refresh_accounts(runtime)
    raw = params.get("account")
    if not isinstance(raw, str) or ":" not in raw:
        raise ProtocolError(
            ErrorDescriptor.INVALID_PARAMS, "account must be provider:name"
        )
    account = AccountId(*raw.split(":", 1))
    loop = runtime.loop
    if method == "account_logout":
        try:
            message = await asyncio.to_thread(
                loop.account_context.store.logout,
                account,
                sessions_root=runtime.home / "sessions",
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise ProtocolError(ErrorDescriptor.STATE_UNAVAILABLE, str(exc)) from exc
        return {"account": str(account), "message": message}
    context = loop.account_context
    provider = runtime.metadata.provider
    current_provider = account_provider(provider)

    def prepare_backend(selected: AccountId) -> object | None:
        if selected.provider != current_provider:
            return None
        backend, _ = build_backend(
            provider,
            loop.backend.model,
            home=context.home,
            account=selected,
            token_budget=loop.context_assembler.token_budget,
        )
        return backend

    await context.switch_active(
        loop,
        account,
        boundary=loop.store.active_branch_head_sequence(),
        provider=provider,
        build_backend=prepare_backend,
    )
    return {
        "account": str(account),
        "replay": False,
        "account_state": "unknown" if context.account_state_unknown else "ready",
    }


async def refresh_accounts(runtime: ServerRuntime) -> dict[str, object]:
    refreshed = await asyncio.to_thread(
        lambda: asyncio.run(
            AccountUsage(runtime.loop.account_context.store).refresh_stale()
        )
    )
    return await list_accounts(runtime, refreshed)
