"""Shared named-account login entry point for CLI and interactive surfaces."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from ..core.login_flow import run_login
from ..core.session import account_home
from ..providers.login import build_login_provider, pkce_values
from .store import AccountError, AccountId, AccountStore


def login_named_account(
    home: Path,
    provider: str,
    label: str,
    *,
    replace_existing: bool = False,
    login: Callable[..., Awaitable[Any]] = run_login,
) -> str | None:
    store = AccountStore(account_home(home))
    with store.provider_login_lock(provider):
        store.migrate()
        try:
            account = store.resolve_reference(provider, label)
        except AccountError:
            account = AccountId(provider, label)
        else:
            if not replace_existing:
                raise AccountError(
                    f"account {provider}:{label} already exists; use --replace"
                )
        return asyncio.run(
            login(build_login_provider(provider, store.home, account.name), pkce_values)
        )
